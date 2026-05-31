from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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
    records: list[EvidenceRecord] = []
    records.extend(scan_plan_records(project))
    records.extend(scan_codex_history(project, cfg.codex_history_roots))
    records.extend(scan_claude_history(project, cfg.claude_history_roots))
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


def scan_codex_history(project_path: str | Path, roots: Iterable[str | Path]) -> list[EvidenceRecord]:
    return _scan_history(project_path, roots, source_type="codex_history")


def scan_claude_history(project_path: str | Path, roots: Iterable[str | Path]) -> list[EvidenceRecord]:
    return _scan_history(project_path, roots, source_type="claude_history")


def _scan_history(
    project_path: str | Path,
    roots: Iterable[str | Path],
    *,
    source_type: str,
) -> list[EvidenceRecord]:
    project = Path(project_path).resolve()
    project_text = str(project)
    records = []
    for file_path in _iter_history_files(roots):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if project_text not in text:
            continue
        summary = _history_summary(text)
        if not summary:
            summary = f"Conversation mentions {project.name}"
        records.append(
            EvidenceRecord(
                source_type=source_type,
                path=file_path,
                summary=summary,
                confidence="high",
                metadata={"project_path": project_text},
            )
        )
    return records


def _iter_history_files(roots: Iterable[str | Path]) -> Iterable[Path]:
    suffixes = {".jsonl", ".json", ".md", ".txt", ".log"}
    for root_like in roots:
        root = Path(root_like).expanduser()
        if root.is_file() and root.suffix.lower() in suffixes and not _is_metadata_path(root):
            yield root
            continue
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes and not _is_metadata_path(path):
                yield path


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

def score_evidence_records(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Return *new* EvidenceRecord instances with relevance_score populated."""
    return [score_single_record(r, index, len(records)) for index, r in enumerate(records)]


def score_single_record(
    record: EvidenceRecord,
    index: int,
    total: int,
) -> EvidenceRecord:
    score = 0.0
    reasons: list[str] = []
    task_signal = "none"

    if record.source_type == "plan":
        score, reasons, task_signal = _score_plan_record(record)
    elif record.source_type in ("codex_history", "claude_history"):
        score, reasons, task_signal = _score_history_record(record)
    else:
        reasons.append(f"source_type={record.source_type}")

    # recency bonus: later records in the list get a small bump (index-based)
    recency_rank: int | None = None
    if total > 1:
        recency_rank = index
        recency_bonus = 0.05 * (1.0 - index / max(total - 1, 1))
        score += recency_bonus
        if recency_bonus > 0:
            reasons.append("recency bonus")

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


def _score_history_record(record: EvidenceRecord) -> tuple[float, list[str], str]:
    summary = record.summary.lower()
    reasons: list[str] = []
    task_signal = "conversation_mention"

    # Penalize if only the project name appears with no task content
    if _is_path_only_mention(summary):
        score = 0.1
        reasons.append("path-only mention")
        task_signal = "conversation_mention"
        return score, reasons, task_signal

    score = 0.45  # baseline for non-trivial conversation

    # Check for task-specific action verbs
    has_task_marker = any(marker in summary for marker in _TASK_SUMMARY_MARKERS)
    if has_task_marker:
        score += 0.25
        reasons.append("task action verbs in summary")
        task_signal = "conversation_task"

    # Penalize dependency/lockfile chatter
    if _is_low_value_conversation(summary):
        score -= 0.25
        reasons.append("low-value dependency/lockfile mention")
        task_signal = "conversation_low_value"

    return max(score, 0.05), reasons, task_signal


def _is_path_only_mention(summary: str) -> bool:
    """True if the summary is basically just a file path or project name."""
    stripped = summary.strip()
    # Contains no spaces and looks like a path
    if "/" in stripped and " " not in stripped:
        return True
    # Very short, no action words
    if len(stripped) < 20 and not any(m in stripped.lower() for m in _TASK_SUMMARY_MARKERS):
        return True
    return False


def _is_low_value_conversation(summary: str) -> bool:
    low_markers = (
        "lockfile", "lock file", "package-lock", "yarn.lock", "poetry.lock",
        "dependency", "dependencies", "bump", "version bump",
    )
    return any(m in summary for m in low_markers)
