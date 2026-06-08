from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Keywords that indicate a plan doc contains actionable task definitions.
_TASK_KEYWORDS = frozenset({
    "task", "tasks", "package", "packages", "implementation", "implement",
    "test", "tests", "fixture", "fixtures", "step", "steps", "checklist",
    "requirement", "requirements", "deliverable", "deliverables",
})

# Filenames (lowercased) whose docs are status/tracking only, not task definitions.
_STATUS_ONLY_NAMES = frozenset({
    "state.md", "ledger.md", "status.md", "readme.md", "changelog.md",
})

# Conversation summary substrings that indicate the user was doing real work.
_TASK_SUMMARY_MARKERS = (
    "fix", "add", "implement", "create", "build", "refactor", "update",
    "remove", "delete", "test", "debug", "migrate", "integrate", "deploy",
    "run", "write", "change", "review", "score", "intake", "prepare",
)


@dataclass(frozen=True)
class EvidenceRecord:
    source_type: str
    path: Path
    summary: str
    confidence: str = "medium"
    metadata: dict[str, str] = field(default_factory=dict)
    relevance_score: float = 0.0
    relevance_reasons: tuple[str, ...] = ()
    task_signal: str = ""
    recency_rank: int | None = None


@dataclass(frozen=True)
class EvidenceSourceConfig:
    codex_history_roots: tuple[Path, ...] = ()
    claude_history_roots: tuple[Path, ...] = ()


def default_codex_history_roots() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".codex" / "sessions",
        home / ".codex" / "archived_sessions",
    )


def default_claude_history_roots() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".claude" / "projects",
        home / ".claude" / "sessions",
        home / ".claude" / "conversation-logs",
    )


def discover_evidence_sources(
    project_path: str | Path,
    config: EvidenceSourceConfig | None = None,
    *,
    warnings: list[str] | None = None,
) -> list[EvidenceRecord]:
    project = Path(project_path).resolve()
    cfg = config or (
        EvidenceSourceConfig()
        if _is_temporary_project(project)
        else EvidenceSourceConfig(
            codex_history_roots=default_codex_history_roots(),
            claude_history_roots=default_claude_history_roots(),
        )
    )
    scan_warnings: list[str] | None = [] if warnings is not None else None
    records: list[EvidenceRecord] = []
    records.extend(scan_plan_records(project))
    records.extend(scan_codex_history(project, cfg.codex_history_roots, warnings=scan_warnings))
    records.extend(scan_claude_history(project, cfg.claude_history_roots, warnings=scan_warnings))
    if warnings is not None and scan_warnings:
        warnings.extend(scan_warnings)
    return records


def _is_temporary_project(project: Path) -> bool:
    text = project.as_posix()
    return (
        text.startswith("/tmp/")
        or text.startswith("/private/tmp/")
        or "/pytest-" in text
        or "/pytest-of-" in text
    )


def scan_plan_records(project_path: str | Path) -> list[EvidenceRecord]:
    project = Path(project_path).resolve()
    records = []
    for path in sorted((project / "docs" / "plans").glob("**/*.md")):
        if _is_metadata_path(path):
            continue
        title = _first_heading(path) or path.stem
        records.append(
            EvidenceRecord(
                source_type="plan",
                path=path,
                summary=title,
                confidence="high",
                metadata={"relative_path": _relative(path, project)},
            )
        )
    return records


def scan_codex_history(project_path: str | Path, roots: Iterable[str | Path], *, warnings: list[str] | None = None) -> list[EvidenceRecord]:
    return _scan_history(project_path, roots, source_type="codex_history", warnings=warnings)


def scan_claude_history(project_path: str | Path, roots: Iterable[str | Path], *, warnings: list[str] | None = None) -> list[EvidenceRecord]:
    return _scan_history(project_path, roots, source_type="claude_history", warnings=warnings)


def _project_aliases(project: Path) -> list[tuple[str, str]]:
    """Generate matchable aliases for a project path.

    Returns list of (alias_text, alias_type) ordered from most-specific to
    least-specific so callers can prefer the best match.
    """
    aliases: list[tuple[str, str]] = []
    project_text = str(project)
    project_name = project.name

    # Full resolved path (most specific)
    aliases.append((project_text, "resolved_path"))

    # Path with ~ if under home directory
    home = str(Path.home())
    if project_text.startswith(home):
        aliases.append((project_text.replace(home, "~", 1), "home_relative"))

    # Common path fragments: last 2-3 segments (more specific than basename alone)
    parts = project.parts
    for n in range(min(3, len(parts)), 1, -1):
        fragment = str(Path(*parts[-n:]))
        if fragment != project_name:
            aliases.append((fragment, f"path_fragment_{n}"))

    # Basename (least specific match, checked after path fragments)
    if project_name and len(project_name) >= 3:
        aliases.append((project_name, "basename"))

    # Normalized variants: replace underscores/dashes with space variants
    normalized = project_name.replace("_", " ").replace("-", " ")
    if normalized != project_name and len(normalized) >= 3:
        aliases.append((normalized, "normalized_name"))

    return aliases


def _extract_mentioned_files(text: str) -> list[str]:
    """Extract file-like paths mentioned in conversation text.

    Returns relative-looking paths and short filenames the user may have
    discussed.  Only extracts fragments — this is NOT a robust parser and
    should never be treated as one.
    """
    mentioned: list[str] = []
    # Relative paths like src/foo.py or docs/plans/INDEX.md
    for m in re.finditer(r'(?:^|\s|["\'`])([\w./-]+\.\w{1,10})(?:\s|$|["\'`:,])', text):
        path_frag = m.group(1).strip("./")
        if "/" in path_frag or "." in path_frag:
            mentioned.append(path_frag)
    # Python module references like replay_checker.sources
    for m in re.finditer(r'(?:^|\s|["\'`])([\w]+(?:\.[\w]+)+)(?:\s|$|["\'`:,])', text):
        mod_frag = m.group(1)
        # Convert module paths to file paths
        mentioned.append(mod_frag.replace(".", "/") + ".py")
    return list(dict.fromkeys(mentioned))  # preserve order, dedupe


def _extract_timestamp(file_path: Path, text: str) -> str:
    """Extract the best timestamp for a history file.

    Tries embedded JSONL timestamps first, falls back to file mtime.
    Returns ISO-format string or empty string.
    """
    # Try to find a timestamp in the first few JSONL lines
    for raw_line in text.splitlines()[:5]:
        parsed = _try_json(raw_line.strip())
        if isinstance(parsed, dict):
            ts = parsed.get("timestamp") or parsed.get("ts") or parsed.get("created_at") or parsed.get("createdAt")
            if ts:
                try:
                    # Accept Unix timestamps (int/float) and ISO strings
                    if isinstance(ts, (int, float)):
                        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    return str(ts)[:26]
                except (ValueError, OSError):
                    pass

    # Fall back to file modification time
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _score_alias_match_quality(alias_type: str) -> float:
    """Return a quality multiplier (0.0-1.0) for how specific an alias match is."""
    quality = {
        "resolved_path": 1.0,
        "home_relative": 0.95,
        "path_fragment_3": 0.7,
        "path_fragment_2": 0.6,
        "basename": 0.4,
        "normalized_name": 0.3,
    }
    return quality.get(alias_type, 0.5)


def _scan_history(
    project_path: str | Path,
    roots: Iterable[str | Path],
    *,
    source_type: str,
    warnings: list[str] | None = None,
) -> list[EvidenceRecord]:
    project = Path(project_path).resolve()
    project_text = str(project)
    aliases = _project_aliases(project)
    records = []
    for file_path in _iter_history_files(roots):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            if warnings is not None:
                warnings.append(f"skipped {file_path}: {exc}")
            continue

        # Check all aliases; prefer the most specific match
        best_alias = ""
        best_alias_type = ""
        for alias, alias_type in aliases:
            if alias in text:
                best_alias = alias
                best_alias_type = alias_type
                break

        if not best_alias:
            continue

        summary = _history_summary(text)
        if not summary:
            summary = f"Conversation mentions {project.name}"
        if len(summary) > 500:
            summary = summary[:497] + "..."

        mentioned_files = _extract_mentioned_files(text)
        timestamp = _extract_timestamp(file_path, text)
        alias_quality = _score_alias_match_quality(best_alias_type)

        metadata = {
            "project_path": project_text,
            "matched_alias": best_alias,
            "matched_alias_type": best_alias_type,
            "alias_match_quality": f"{alias_quality:.2f}",
        }
        if mentioned_files:
            metadata["mentioned_files"] = ",".join(mentioned_files[:20])
        if timestamp:
            metadata["timestamp"] = timestamp

        confidence = "high" if best_alias_type in ("resolved_path", "home_relative") else "medium"

        records.append(
            EvidenceRecord(
                source_type=source_type,
                path=file_path,
                summary=summary,
                confidence=confidence,
                metadata=metadata,
            )
        )
    return records


def _iter_history_files(
    roots: Iterable[str | Path],
    *,
    max_files: int = 500,
) -> Iterable[Path]:
    suffixes = {".jsonl", ".json", ".md", ".txt", ".log"}
    yielded = 0
    for root_like in roots:
        if yielded >= max_files:
            return
        root = Path(root_like).expanduser()
        if root.is_file() and root.suffix.lower() in suffixes and not _is_metadata_path(root):
            yield root
            yielded += 1
            continue
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames
                if not _is_metadata_path(Path(dirpath) / d)
            )
            for filename in sorted(filenames):
                if yielded >= max_files:
                    return
                path = Path(dirpath) / filename
                if path.suffix.lower() not in suffixes or _is_metadata_path(path):
                    continue
                yield path
                yielded += 1


def _history_summary(text: str) -> str:
    snippets: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = _try_json(line)
        candidate = _extract_user_text(parsed) if parsed is not None else line
        if candidate:
            snippets.append(_compact(candidate))
        if len(snippets) >= 3:
            break
    return " | ".join(snippets)[:500]


def _try_json(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _extract_user_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_extract_user_text(item) for item in value]
        return " ".join(part for part in parts if part)
    if not isinstance(value, dict):
        return ""

    role = str(value.get("role") or value.get("type") or "").lower()

    # Codex content fragments: {"type": "input_text", "text": "..."}
    if role in ("input_text", "output_text"):
        return str(value.get("text") or "")

    if role == "user":
        return _extract_user_text(value.get("content") or value.get("message") or value.get("text") or "")

    payload = value.get("payload")
    if isinstance(payload, dict):
        payload_role = str(payload.get("role") or "").lower()
        if payload_role == "user":
            return _extract_user_text(payload.get("content") or payload.get("message") or payload.get("text") or "")

    message = value.get("message")
    if isinstance(message, dict):
        msg_role = str(message.get("role") or "").lower()
        if msg_role == "user":
            return _extract_user_text(message.get("content") or message.get("text") or "")
    elif isinstance(message, str):
        return message

    return ""


def _first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _compact(text: str) -> str:
    return " ".join(text.split())


def _is_metadata_path(path: Path) -> bool:
    return any(part.startswith("._") or part in {".DS_Store", "__MACOSX"} for part in path.parts)


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------

def score_evidence_records(
    records: list[EvidenceRecord],
    *,
    project_path: str | Path | None = None,
) -> list[EvidenceRecord]:
    """Return *new* EvidenceRecord instances with relevance_score populated."""
    project_files: set[str] | None = None
    if project_path is not None and any(
        r.source_type in ("codex_history", "claude_history")
        and bool(r.metadata.get("mentioned_files", ""))
        for r in records
    ):
        project_files = _collect_project_files(Path(project_path))
    return [
        score_single_record(
            r,
            index,
            len(records),
            project_path=project_path,
            project_files=project_files,
        )
        for index, r in enumerate(records)
    ]


def score_single_record(
    record: EvidenceRecord,
    index: int,
    total: int,
    *,
    project_path: str | Path | None = None,
    project_files: set[str] | None = None,
) -> EvidenceRecord:
    score = 0.0
    reasons: list[str] = []
    task_signal = "none"

    if record.source_type == "plan":
        score, reasons, task_signal = _score_plan_record(record)
    elif record.source_type in ("codex_history", "claude_history"):
        score, reasons, task_signal = _score_history_record(
            record,
            project_path=project_path,
            project_files=project_files,
        )
    else:
        reasons.append(f"source_type={record.source_type}")

    # recency bonus from timestamp metadata or index fallback
    recency_rank: int | None = None
    recency_bonus = _compute_recency_bonus(record, index, total)
    if recency_bonus > 0:
        score += recency_bonus
        reasons.append("recency bonus")
    if total > 1:
        recency_rank = index

    return EvidenceRecord(
        source_type=record.source_type,
        path=record.path,
        summary=record.summary,
        confidence=record.confidence,
        metadata=record.metadata,
        relevance_score=round(min(score, 1.0), 3),
        relevance_reasons=tuple(reasons),
        task_signal=task_signal,
        recency_rank=recency_rank,
    )


def _score_plan_record(record: EvidenceRecord) -> tuple[float, list[str], str]:
    score = 0.5  # baseline for any plan doc
    reasons: list[str] = ["plan doc"]
    task_signal = "plan_task"

    # Check if status-only
    lower_name = record.path.name.lower()
    if lower_name in _STATUS_ONLY_NAMES:
        score = 0.2
        reasons = ["plan doc but status-only"]
        task_signal = "plan_status_only"
        return score, reasons, task_signal

    # Check for task keywords in heading/summary
    text = record.summary.lower()
    has_task_kw = any(kw in text for kw in _TASK_KEYWORDS)
    if has_task_kw:
        score += 0.3
        reasons.append("task/package keyword in title")
    else:
        task_signal = "plan_status_only"
        score -= 0.15
        reasons.append("no task keyword in title")

    # Boost if path contains sub-packages (indicates structured task breakdown)
    rel = record.metadata.get("relative_path", "")
    if "/" in rel:
        score += 0.05
        reasons.append("nested plan path")

    return score, reasons, task_signal


def _score_history_record(
    record: EvidenceRecord,
    *,
    project_path: str | Path | None = None,
    project_files: set[str] | None = None,
) -> tuple[float, list[str], str]:
    summary = record.summary.lower()
    reasons: list[str] = []
    task_signal = "conversation_mention"

    # Assess alias match quality — basename-only matches are inherently weaker
    alias_type = record.metadata.get("matched_alias_type", "")
    alias_quality = _score_alias_match_quality(alias_type)

    # Check path-only / alias-only mention (no task content)
    if _is_path_only_mention(summary, alias_type):
        score = 0.08
        reasons.append(f"path-only mention (alias: {alias_type})")
        task_signal = "conversation_mention"
        return max(score, 0.0), reasons, task_signal

    # Baseline: non-trivial conversation, scaled by alias quality
    score = 0.35 + (0.15 * alias_quality)

    # Check for task-specific action verbs
    has_task_marker = any(marker in summary for marker in _TASK_SUMMARY_MARKERS)
    if has_task_marker:
        score += 0.2 + (0.05 * alias_quality)
        reasons.append("task action verbs in summary")
        task_signal = "conversation_task"

    # File co-occurrence boost: mentioned files that look like real project files
    mentioned_files = record.metadata.get("mentioned_files", "")
    if mentioned_files and project_path:
        file_inventory = project_files
        if file_inventory is None:
            file_inventory = _collect_project_files(Path(project_path))
        overlap = _file_overlap_score(mentioned_files.split(","), file_inventory)
        if overlap > 0:
            file_bonus = 0.1 * min(overlap, 3)
            score += file_bonus
            reasons.append(f"file co-occurrence with project (+{file_bonus:.2f})")
    elif mentioned_files:
        # Without project context, check if files look like source code paths
        if _has_source_file_mentions(mentioned_files):
            score += 0.05
            reasons.append("source file mentions in conversation")

    # Penalize dependency/lockfile chatter
    if _is_low_value_conversation(summary):
        score -= 0.2
        reasons.append("low-value dependency/lockfile mention")
        if task_signal == "conversation_task":
            task_signal = "conversation_low_value"

    return max(score, 0.03), reasons, task_signal


def _compute_recency_bonus(record: EvidenceRecord, index: int, total: int) -> float:
    """Compute recency bonus from timestamp metadata or index fallback."""
    ts_str = record.metadata.get("timestamp", "")
    if ts_str:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_days = (now - ts).days
            # Fresher history gets a higher bonus
            if age_days < 1:
                return 0.1
            elif age_days < 7:
                return 0.07
            elif age_days < 30:
                return 0.04
            elif age_days < 90:
                return 0.02
            else:
                return 0.0
        except (ValueError, TypeError):
            pass

    # Fall back to index-based recency if timestamp not available
    if total > 1:
        return 0.05 * (1.0 - index / max(total - 1, 1))
    return 0.0


def _collect_project_files(project_path: Path, max_files: int = 200) -> set[str]:
    """Collect relative file paths from a project for co-occurrence checking."""
    files: set[str] = set()
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".tox", "work", "venv", ".claude"}
    if not project_path.is_dir():
        return files
    for root, dirs, filenames in os.walk(str(project_path)):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            try:
                rel = str(Path(root) / fname).replace(str(project_path) + os.sep, "")
                files.add(rel)
            except ValueError:
                pass
        if len(files) >= max_files:
            break
    return files


def _file_overlap_score(mentioned: list[str], project_files: set[str]) -> float:
    """Score file co-occurrence: how many mentioned files exist in the project."""
    if not project_files:
        return 0.0
    matches = 0
    for mf in mentioned:
        mf_clean = mf.strip().lstrip("./")
        # Exact match
        if mf_clean in project_files:
            matches += 1
            continue
        # Suffix match (file basename exists in project)
        mf_name = Path(mf_clean).name
        if any(pf.endswith(mf_name) for pf in project_files):
            matches += 0.5
    return matches


def _has_source_file_mentions(mentioned_files: str) -> bool:
    """Check if mentioned files include source code extensions."""
    source_exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb", ".c", ".cpp"}
    for mf in mentioned_files.split(","):
        ext = Path(mf.strip()).suffix.lower()
        if ext in source_exts:
            return True
    return False


def _is_path_only_mention(summary: str, alias_type: str = "") -> bool:
    """True if the summary is basically just a file path or project name with no task content."""
    stripped = summary.strip().lower()
    # Contains path-like separators and no space-separated content
    if "/" in stripped and " " not in stripped:
        return True
    # Very short, no action words
    if len(stripped) < 20 and not any(m in stripped for m in _TASK_SUMMARY_MARKERS):
        return True
    # Basename-only alias match with no additional context words
    if alias_type in ("basename", "normalized_name") and not any(
        m in stripped for m in _TASK_SUMMARY_MARKERS
    ):
        if len(stripped.split()) <= 3:
            return True
    return False


def _is_low_value_conversation(summary: str) -> bool:
    low_markers = (
        "lockfile", "lock file", "package-lock", "yarn.lock", "poetry.lock",
        "dependency", "dependencies", "bump", "version bump",
    )
    return any(m in summary for m in low_markers)
