from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import stable_hash
from .candidates import CandidateCase, build_case_candidates, select_case_candidate
from .evaluation import Recommendation, read_recommendation
from .git_utils import DEFAULT_GIT_TIMEOUT, git_output, git_run
from .packages import ExecutionPackageRenderer, compile_execution_package, _extract_goal_from_plan
from .scoring import (
    CaseProvenance,
    EvidenceGate,
    build_case_provenance,
    compute_score_ceilings,
    evaluate_evidence_gates,
    parse_rubric,
)
from .sources import EvidenceRecord, EvidenceSourceConfig, discover_evidence_sources
from .yaml_lite import coerce_nested_int, emit_yaml, parse_yaml_file

_GIT_TIMEOUT = DEFAULT_GIT_TIMEOUT

FORBIDDEN_PATH_PATTERNS = (
    "_reference/", "_reference\\",
    ".git/", ".git\\",
    "cases/", "cases\\",
    "runs/", "runs\\",
)
SUSPICIOUS_PATH_PATTERNS = (
    ".env", "credentials", "secret", "token", "password", "key",
)


def _git_run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = _GIT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with a timeout."""
    return git_run(args, cwd=cwd, timeout=timeout)


def _git_output(args: list[str], *, cwd: Path) -> str:
    """Run a git command and return stdout (empty string on failure/timeout)."""
    return git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


@dataclass(frozen=True)
class PlanPackage:
    plan_path: Path
    title: str
    package_count: int


@dataclass(frozen=True)
class ReplayCase:
    id: str
    root: Path
    project_path: Path
    plan_path: Path
    base_commit: str
    verification_commands: tuple[str, ...] = ()
    base_source: str = ""
    base_confidence: str = "unknown"
    source_type: str = "manual"
    source_path: str = ""
    selection_reason: str = ""
    synthetic_case: bool = False
    evidence_sources: tuple[str, ...] = ()
    selected_candidate_id: str = ""
    candidate_score: float = 0.0
    candidate_source_type: str = ""
    reference_target: str = ""


@dataclass(frozen=True)
class ReplayRun:
    id: str
    root: Path
    case: ReplayCase
    runner_label: str
    workspace: Path


@dataclass(frozen=True)
class Evidence:
    run_id: str
    status: str
    changed_files: tuple[str, ...] = field(default_factory=tuple)
    diff_path: Path | None = None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Telemetry:
    changed_file_count: int
    added_lines: int
    deleted_lines: int
    forbidden_path_touches: tuple[str, ...]
    suspicious_path_touches: tuple[str, ...]
    untracked_file_count: int
    commit_count: int
    rollback_signal: str


@dataclass(frozen=True)
class RunHealth:
    run_id: str
    status: str
    raw_status: str
    workspace: Path
    workspace_exists: bool
    completion_report_exists: bool
    has_result_diff: bool
    changed_files: tuple[str, ...]
    reason: str


def parse_simple_yaml(path: str | Path) -> dict[str, object]:
    """Parse the tiny YAML subset this project writes: scalar keys, string lists, nested blocks."""
    data = parse_yaml_file(
        path,
        scalar_parser=coerce_nested_int,
        parse_list_item_dicts=False,
    )
    return data if isinstance(data, dict) else {}


def _write_simple_yaml(path: Path, data: dict[str, object], *, indent: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(emit_yaml(data, indent=indent) + "\n", encoding="utf-8")


def _first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.parent.name


def discover_plan_packages(project_path: str | Path) -> list[PlanPackage]:
    root = Path(project_path)
    plans = []
    for index in sorted((root / "docs" / "plans").glob("**/INDEX.md")):
        package_dir = index.parent / "packages"
        package_count = len(list(package_dir.glob("*.md"))) if package_dir.exists() else 0
        plans.append(PlanPackage(index, _first_heading(index), package_count))
    return plans


def _detect_all_sources(project_path: Path) -> list[tuple[int, str, Path, str]]:
    """Return all plan candidates sorted by score descending.

    Each entry is (score, reasons_string, path, source_type).
    """
    plans_dir = project_path / "docs" / "plans"
    if not plans_dir.exists():
        return []

    candidates: list[tuple[int, str, Path, str]] = []

    for index in sorted(plans_dir.glob("**/INDEX.md")):
        # Skip root INDEX.md (meta-index, not a kit)
        if index.parent == plans_dir:
            continue
        candidate_dir = index.parent
        score, reasons = _score_plan_candidate(candidate_dir, project_path)
        candidates.append((score, reasons, index, "orchestration_kit"))

    seen_dirs = {idx.parent for _, _, idx, _ in candidates}
    for md in sorted(plans_dir.glob("**/*.md")):
        if md.parent in seen_dirs:
            continue
        text = md.read_text(encoding="utf-8", errors="ignore").lower()
        if "acceptance criteria" in text and ("goal" in text or "steps" in text):
            candidate_dir = md.parent
            score, reasons = _score_plan_candidate(candidate_dir, project_path)
            candidates.append((score, reasons, md, "handoff_plan"))

    candidates.sort(key=lambda x: (-x[0], str(x[2])))
    return candidates


def _detect_source(project_path: Path) -> dict[str, object]:
    """Detect the best task source in a project.

    Collects all plan candidates, scores them, and returns the highest-scoring one.
    Falls back to git_history when no plan candidates are found.

    Returns a dict with keys: source_type, source_path, plan_path, title,
    selection_reason, verification_commands.
    """
    plans_dir = project_path / "docs" / "plans"

    if plans_dir.exists():
        candidates: list[tuple[int, str, Path, str]] = []

        # Collect INDEX.md candidates (orchestration kits)
        for index in sorted(plans_dir.glob("**/INDEX.md")):
            # Skip root INDEX.md (meta-index, not a kit)
            if index.parent == plans_dir:
                continue
            candidate_dir = index.parent
            score, reasons = _score_plan_candidate(candidate_dir, project_path)
            candidates.append((score, reasons, index, "orchestration_kit"))

        # Collect handoff plan candidates (non-INDEX.md with acceptance criteria)
        seen_dirs = {idx.parent for _, _, idx, _ in candidates}
        for md in sorted(plans_dir.glob("**/*.md")):
            if md.parent in seen_dirs:
                continue
            text = md.read_text(encoding="utf-8", errors="ignore").lower()
            if "acceptance criteria" in text and ("goal" in text or "steps" in text):
                candidate_dir = md.parent
                score, reasons = _score_plan_candidate(candidate_dir, project_path)
                candidates.append((score, reasons, md, "handoff_plan"))

        if candidates:
            # Sort by score descending, then by path for determinism
            candidates.sort(key=lambda x: (-x[0], str(x[2])))
            best_score, best_reasons, best_path, best_type = candidates[0]

            return {
                "source_type": best_type,
                "source_path": str(best_path),
                "plan_path": str(best_path),
                "title": _first_heading(best_path),
                "selection_reason": f"Scored {best_score} ({best_reasons}) — {best_type}",
                "verification_commands": _extract_verification(project_path, best_path.parent),
                **_plan_source_metadata(project_path, best_path.parent),
            }

    # No plan found — return signal for git history fallback
    return {
        "source_type": "git_history",
        "source_path": "",
        "plan_path": "",
        "title": "",
        "selection_reason": "No plan package found, will generate synthetic case from git history",
        "verification_commands": [],
    }


_COMMAND_PREFIXES = (
    "rtk ", "python ", "python3 ", "pytest", "make", "bash ", "sh ",
    "git ", "npm ", "pnpm ", "yarn ", "uv ", "cargo ", "go ", "gradle ",
    "./gradlew", "ruby ", "bundle ", "ls ", "cat ", "echo ",
)


def _looks_like_command(line: str) -> bool:
    """Return True if *line* starts with a recognised executable prefix."""
    return any(line.startswith(p) for p in _COMMAND_PREFIXES)


def _extract_verification(project_path: Path, plan_dir: Path) -> list[str]:
    """Extract verification commands from plan markdown files.

    Only lines that look like executable commands are kept.  Prose bullets
    (e.g. policy instructions) are ignored.  Nested backtick artifacts are
    never emitted.
    """
    commands = []
    for md in plan_dir.rglob("*.md"):
        in_verification = False
        in_code_block = False
        for line in md.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "verification" in line.lower() and "command" in line.lower():
                in_verification = True
                in_code_block = False
                continue
            if in_verification:
                stripped = line.strip()
                if stripped.startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    if stripped and not stripped.startswith("#") and _looks_like_command(stripped):
                        commands.append(stripped)
                    continue
                if stripped.startswith("#"):
                    break
                if stripped.startswith("- "):
                    cmd = stripped[2:].strip()
                    # Strip surrounding Markdown backticks.
                    if cmd.startswith("`"):
                        end = cmd.rfind("`", 1)
                        cmd = cmd[1:end] if end > 0 else cmd[1:]
                    if cmd and _looks_like_command(cmd):
                        commands.append(cmd)
                elif stripped and not stripped.startswith("-"):
                    # Non-bullet, non-empty line ends the section.
                    break
    return commands


def _plan_source_metadata(project_path: Path, plan_dir: Path) -> dict[str, str]:
    """Extract per-kit git bounds from orchestration status metadata."""
    base, target = _read_plan_state_bounds(project_path, plan_dir)
    data: dict[str, str] = {}
    if base:
        data["base_commit"] = base
        data["base_source"] = "plan_state_base"
        data["base_confidence"] = "high"
    if target:
        data["reference_target"] = target
    return data


def _read_plan_state_bounds(project_path: Path, plan_dir: Path) -> tuple[str, str]:
    """Read base and target commits from a generated kit's status/state.tsv."""
    state_path = plan_dir / "status" / "state.tsv"
    if not state_path.is_file():
        return ("", "")
    try:
        rows = state_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ("", "")
    if not rows:
        return ("", "")

    header = rows[0].split("\t")
    try:
        base_idx = header.index("base_commit")
    except ValueError:
        base_idx = -1
    try:
        target_idx = header.index("commit_hash")
    except ValueError:
        target_idx = -1
    if base_idx < 0 and target_idx < 0:
        return ("", "")

    bases: list[str] = []
    targets: list[str] = []
    for row in rows[1:]:
        cols = row.split("\t")
        if base_idx >= 0 and base_idx < len(cols):
            base = cols[base_idx].strip()
            resolved = _resolve_commit(project_path, base)
            if resolved:
                bases.append(resolved)
        if target_idx >= 0 and target_idx < len(cols):
            target = cols[target_idx].strip()
            resolved = _resolve_commit(project_path, target)
            if resolved:
                targets.append(resolved)

    return (bases[0] if bases else "", targets[-1] if targets else "")


def _commit_exists(project_path: Path, sha: str) -> bool:
    return bool(_resolve_commit(project_path, sha))


def _resolve_commit(project_path: Path, sha: str) -> str:
    if not sha:
        return ""
    try:
        output = _git_output(["rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=project_path)
    except subprocess.TimeoutExpired:
        return ""
    return output.strip()


def _git_recent_activity(project_path: Path, plan_dir: Path) -> bool:
    """Check if plan files have been modified in the last 30 days."""
    output = _git_output(
        ["log", "--oneline", "--since=30.days", "--", str(plan_dir)],
        cwd=project_path,
    )
    return bool(output.strip())


def _score_plan_candidate(candidate_dir: Path, project_path: Path) -> tuple[int, str]:
    """Score a plan candidate directory. Returns (score, reasons_string).

    Prefers orchestration kits with packages, package graph, status ledger,
    final report, verification commands, and recent git activity.
    """
    score = 0
    reasons: list[str] = []

    package_dir = candidate_dir / "packages"
    if package_dir.exists():
        pkg_count = len(list(package_dir.glob("*.md")))
        if pkg_count > 0:
            score += 3
            reasons.append(f"packages({pkg_count})")

    if (candidate_dir / "launchers" / "package-graph.tsv").exists():
        score += 2
        reasons.append("pkg-graph")

    if (candidate_dir / "status" / "state.tsv").exists():
        score += 2
        reasons.append("status")

    if (candidate_dir / "FINAL_REPORT.md").exists():
        score += 2
        reasons.append("final-report")

    vcommands = _extract_verification(project_path, candidate_dir)
    if vcommands:
        score += 2
        reasons.append(f"verification({len(vcommands)})")

    if _git_recent_activity(project_path, candidate_dir):
        score += 1
        reasons.append("recent")

    return score, ", ".join(reasons) if reasons else "no-signals"


def _infer_base_commit(project_path: Path, source: dict[str, object]) -> tuple[str, str, str]:
    """Infer the base commit for a case.

    Returns (commit_sha, base_source, base_confidence).
    """
    is_git = (project_path / ".git").exists()
    if not is_git:
        return ("", "no_git", "low")

    source_base = str(source.get("base_commit", "")).strip()
    if source_base:
        return (
            source_base,
            str(source.get("base_source", "plan_state_base") or "plan_state_base"),
            str(source.get("base_confidence", "high") or "high"),
        )

    # Check for dirty working tree
    status_output = _git_output(["status", "--porcelain"], cwd=project_path)
    if status_output.strip():
        head = _git_head(project_path)
        return (head, "head_dirty", "low")

    source_type = source["source_type"]

    # Check for empty history (no commits at all)
    log_output = _git_output(["log", "--oneline", "-1"], cwd=project_path)
    if not log_output.strip():
        return ("", "empty_history", "low")

    if source_type in ("orchestration_kit", "handoff_plan"):
        return _base_from_plan(project_path, source)

    if source_type == "git_history":
        return _base_from_synthetic(project_path)

    head = _git_head(project_path)
    return (head, "head_fallback", "low")


def _git_head(project_path: Path) -> str:
    return _git_output(["rev-parse", "HEAD"], cwd=project_path).strip()


def _base_from_plan(project_path: Path, source: dict[str, object]) -> tuple[str, str, str]:
    """Find the base commit for a plan-based case.

    Strategy: find the first commit that introduced the plan file, use its parent.
    """
    plan_path = source.get("source_path", "")
    if not plan_path:
        head = _git_head(project_path)
        return (head, "head_fallback", "low")

    # Find first commit that added the plan file
    output = _git_output(
        ["log", "--diff-filter=A", "--format=%H", "--", plan_path],
        cwd=project_path,
    )
    commits = [c for c in output.strip().splitlines() if c]
    if not commits:
        head = _git_head(project_path)
        return (head, "head_fallback", "low")

    first_commit = commits[-1]  # oldest commit that added the file

    # Get parent of that commit
    parent_output = _git_output(["rev-parse", f"{first_commit}^"], cwd=project_path)
    if parent_output:
        return (parent_output.strip(), "plan_first_commit_parent", "high")

    # First commit has no parent — use it as base
    return (first_commit, "plan_first_commit", "medium")


def _base_from_synthetic(project_path: Path) -> tuple[str, str, str]:
    """Find the best target commit for synthetic case generation."""
    output = _git_output(["log", "--no-merges", "--format=%H %s", "--"], cwd=project_path)
    if not output.strip():
        return ("", "no_history", "low")

    _SKIP_PREFIXES = ("format:", "style:", "lint:", "bump:", "chore: format", "chore: style", "chore: lint")
    _SKIP_KEYWORDS = ("dependabot", "renovate", "lock file", "lockfile", "auto-format", "auto-style")

    commits = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        sha, message = line.split(" ", 1)
        msg_lower = message.lower()
        if any(msg_lower.startswith(p) for p in _SKIP_PREFIXES):
            continue
        if any(kw in msg_lower for kw in _SKIP_KEYWORDS):
            continue
        commits.append((sha, message))

    if not commits:
        head = _git_head(project_path)
        return (head, "head_fallback", "low")

    # Pick first meaningful commit (target)
    target_sha, _ = commits[0]

    # Get parent as base
    parent_output = _git_output(["rev-parse", f"{target_sha}^"], cwd=project_path)
    if parent_output:
        return (parent_output.strip(), "synthetic_target_parent", "high")

    return (target_sha, "synthetic_first_commit", "medium")


def _generate_case_id(project: Path, source: dict[str, object]) -> str:
    """Generate a deterministic case ID from project name and source."""
    from .core import sanitize_slug, stable_hash

    project_name = sanitize_slug(project.name)
    source_hash = stable_hash(str(source.get("source_path", "")) + str(source.get("title", "")))
    return f"{project_name}-{source_hash}"


def _write_case_yaml(
    case: ReplayCase,
    *,
    reconstruction_risk: str = "",
    reconstruction_confidence: str = "",
    case_fingerprint: str = "",
    duplicate_status: str = "",
    duplicate_of: str = "",
    duplicate_reasons: tuple[str, ...] | list[str] = (),
) -> None:
    """Write case.yaml with full metadata for audit trail."""
    commands = list(case.verification_commands)
    data: dict[str, object] = {
        "id": case.id,
        "project_path": case.project_path,
        "plan_path": case.plan_path,
        "base_commit": case.base_commit,
        "base_source": case.base_source,
        "base_confidence": case.base_confidence,
        "source_type": case.source_type,
        "source_path": case.source_path,
        "selection_reason": case.selection_reason,
        "synthetic_case": str(case.synthetic_case).lower(),
        "evidence_sources": list(case.evidence_sources),
        "verification_commands": commands,
    }
    if case.selected_candidate_id:
        data["selected_candidate_id"] = case.selected_candidate_id
    if case.candidate_score > 0:
        data["candidate_score"] = str(case.candidate_score)
    if case.candidate_source_type:
        data["candidate_source_type"] = case.candidate_source_type
    if case.synthetic_case:
        data["reconstruction_risk"] = reconstruction_risk or "(none)"
        data["reconstruction_confidence"] = reconstruction_confidence or "unknown"
    if case_fingerprint:
        data["case_fingerprint"] = case_fingerprint
    if duplicate_status:
        data["duplicate_status"] = duplicate_status
    if duplicate_of:
        data["duplicate_of"] = duplicate_of
    if duplicate_reasons:
        data["duplicate_reasons"] = list(duplicate_reasons)
    if case.reference_target:
        data["reference_target"] = case.reference_target
    _write_simple_yaml(case.root / "case.yaml", data)



def _apply_duplicate_check(
    case: ReplayCase,
    cases_root: Path,
    *,
    allow_duplicate: bool = False,
    reconstruction_risk: str = "",
    reconstruction_confidence: str = "",
) -> tuple[ReplayCase, str]:
    """Check for duplicates after case creation and rewrite case.yaml."""
    from .case_dedupe import build_case_fingerprint, find_duplicates, scan_all_fingerprints

    fp = build_case_fingerprint(case.root)
    existing = scan_all_fingerprints(cases_root)
    existing_dirs: dict[str, Path] = {
        cid: cases_root / cid for cid in existing if cid != case.id
    }
    existing_filtered = {k: v for k, v in existing.items() if k != case.id}

    if not existing_filtered:
        _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                         reconstruction_confidence=reconstruction_confidence,
                         case_fingerprint=fp.fingerprint, duplicate_status="unique")
        return case, "unique"

    report = find_duplicates(case.root, existing_filtered, existing_case_dirs=existing_dirs)
    exact_matches = [m for m in report.matches if m.relation == "exact"]
    likely_matches = [m for m in report.matches if m.relation == "likely_duplicate"]

    if exact_matches and not allow_duplicate:
        best = exact_matches[0]
        shutil.rmtree(case.root, ignore_errors=True)
        return load_case(cases_root, best.case_id), "exact_duplicate"

    if exact_matches:
        best = exact_matches[0]
        _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                         reconstruction_confidence=reconstruction_confidence,
                         case_fingerprint=fp.fingerprint, duplicate_status="exact_duplicate",
                         duplicate_of=best.case_id, duplicate_reasons=best.reasons)
        return case, "exact_duplicate"

    if likely_matches:
        best = likely_matches[0]
        _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                         reconstruction_confidence=reconstruction_confidence,
                         case_fingerprint=fp.fingerprint, duplicate_status="likely_duplicate",
                         duplicate_of=best.case_id, duplicate_reasons=best.reasons)
        return case, "likely_duplicate"

    _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                     reconstruction_confidence=reconstruction_confidence,
                     case_fingerprint=fp.fingerprint, duplicate_status="unique")
    return case, "unique"


def _discover_case_evidence(
    project: Path,
    *,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> list[EvidenceRecord]:
    if codex_history_roots is None and claude_history_roots is None:
        return discover_evidence_sources(project)
    return discover_evidence_sources(
        project,
        EvidenceSourceConfig(
            codex_history_roots=tuple(Path(path) for path in (codex_history_roots or ())),
            claude_history_roots=tuple(Path(path) for path in (claude_history_roots or ())),
        ),
    )


def _evidence_record_label(record: EvidenceRecord) -> str:
    return f"{record.source_type}: {record.summary}"


def _write_evidence_sources(case: ReplayCase, records: list[EvidenceRecord]) -> None:
    lines = [
        f"# Evidence Sources: {case.id}",
        "",
        "These are bounded summaries. Raw history logs are not copied into the case.",
        "",
    ]
    if not records:
        lines.append("- none")
    for record in records:
        lines.append(f"- `{record.source_type}` confidence={record.confidence}: {record.summary}")
        lines.append(f"  - path: `{record.path}`")
    (case.root / "evidence_sources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_candidate_report(
    case_dir: Path,
    case_id: str,
    candidates: list[CandidateCase],
    selected: CandidateCase | None,
) -> None:
    """Write candidate_report.md into the case directory."""
    lines = [
        f"# Candidate Report: {case_id}",
        "",
        f"Total candidates: {len(candidates)}",
    ]

    if not candidates:
        lines.append("")
        lines.append("No evidence candidates were found.")
        (case_dir / "candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(["", "## Evidence Summary", ""])

    for idx, candidate in enumerate(candidates):
        tag = " **[selected]**" if selected and candidate.candidate_id == selected.candidate_id else ""
        lines.append(f"{idx + 1}. `{candidate.candidate_id}`{tag}")
        lines.append(f"   - Source type: `{candidate.source_type}`")
        lines.append(f"   - Primary: `{candidate.primary_source}`")
        if candidate.supporting_sources:
            lines.append(f"   - Supporting: {len(candidate.supporting_sources)} other source(s)")
        lines.append(f"   - Relevance score: {candidate.relevance_score:.3f}")
        if candidate.selection_reasons:
            lines.append(f"   - Reasons: {'; '.join(candidate.selection_reasons)}")
        if candidate.risks:
            lines.append(f"   - Risks: {'; '.join(candidate.risks)}")
        lines.append("")

    if selected:
        lines.extend([
            "## Selected Candidate",
            "",
            f"- ID: `{selected.candidate_id}`",
            f"- Source type: `{selected.source_type}`",
            f"- Score: {selected.relevance_score:.3f}",
            "",
        ])

    (case_dir / "candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def intake(
    *,
    cases_root: str | Path,
    project_path: str | Path,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    allow_duplicate: bool = False,
    _force_source: dict[str, object] | None = None,
    _cached_evidence: list[EvidenceRecord] | None = None,
) -> ReplayCase:
    """Auto-generate a replay case from a project directory.

    Scans for orchestration kits, handoff plans, or falls back to git history.
    Builds evidence candidates, selects the best one, and writes candidate_report.md.
    Checks for content-aware duplicates unless allow_duplicate is True.
    """
    project = Path(project_path).resolve()
    cases = Path(cases_root)

    source = _force_source if _force_source is not None else _detect_source(project)

    # Handle no-git and empty-history as source types
    is_git = (project / ".git").exists()
    if not is_git:
        source = {**source, "source_type": "no_git", "selection_reason": "No .git directory found"}

    base_commit, base_source, base_confidence = _infer_base_commit(project, source)
    if base_source == "empty_history" and source["source_type"] == "git_history":
        source = {**source, "source_type": "empty_history", "selection_reason": "Git repository has no commits"}

    case_id = _generate_case_id(project, source)
    case_dir = cases / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # A case is synthetic only when we actually found a meaningful target commit.
    # base_source "head_fallback" or "no_history" means no usable target was identified.
    _SYNTHETIC_BASE_SOURCES = {"synthetic_target_parent", "synthetic_first_commit"}
    is_synthetic = (
        source["source_type"] == "git_history"
        and base_source in _SYNTHETIC_BASE_SOURCES
    )

    # Discover evidence and build candidates
    evidence_records = _cached_evidence if _cached_evidence is not None else _discover_case_evidence(
        project,
        codex_history_roots=codex_history_roots,
        claude_history_roots=claude_history_roots,
    )
    candidates = build_case_candidates(evidence_records, project_path=project)
    selected_candidate = select_case_candidate(candidates)

    # Write candidate report
    _write_candidate_report(case_dir, case_id, candidates, selected_candidate)

    case = ReplayCase(
        id=case_id,
        root=case_dir,
        project_path=project,
        plan_path=Path(source["plan_path"]) if source["plan_path"] else Path(""),
        base_commit=base_commit,
        verification_commands=tuple(source.get("verification_commands", [])),
        base_source=base_source,
        base_confidence=base_confidence,
        source_type=source["source_type"],
        source_path=source["source_path"],
        selection_reason=source["selection_reason"],
        synthetic_case=is_synthetic,
        evidence_sources=tuple(_evidence_record_label(record) for record in evidence_records),
        selected_candidate_id=selected_candidate.candidate_id if selected_candidate else "",
        candidate_score=selected_candidate.relevance_score if selected_candidate else 0.0,
        candidate_source_type=selected_candidate.source_type if selected_candidate else "",
        reference_target=str(source.get("reference_target", "")),
    )

    # Compute reconstruction metadata before writing case.yaml for synthetic cases
    reconstruction_risk = ""
    reconstruction_confidence = ""
    if is_synthetic:
        recon_ctx = _build_reconstruction_context(case, project, list(evidence_records))
        risk_notes = recon_ctx.get("risk_notes", [])
        if isinstance(risk_notes, list) and risk_notes:
            reconstruction_risk = "; ".join(risk_notes)
            # derive confidence from risk presence
            if any("Low confidence" in r or "No verification hints" in r or "Commit message only" in r for r in risk_notes):
                reconstruction_confidence = "low"
            else:
                reconstruction_confidence = "medium"
        else:
            reconstruction_confidence = "high"

    _write_case_yaml(
        case,
        reconstruction_risk=reconstruction_risk,
        reconstruction_confidence=reconstruction_confidence,
    )
    _write_evidence_sources(case, evidence_records)

    _save_reference_evidence(case, project)

    if case.synthetic_case:
        _write_synthetic_task(case, project, evidence_records)
    else:
        _write_case_task(case, evidence_records)

    # Duplicate check after all files are written
    case, _dup_status = _apply_duplicate_check(
        case,
        cases,
        allow_duplicate=allow_duplicate,
        reconstruction_risk=reconstruction_risk,
        reconstruction_confidence=reconstruction_confidence,
    )

    return case


def batch_intake(
    *,
    cases_root: str | Path,
    project_path: str | Path,
    min_score: int = 3,
    max_cases: int = 50,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    allow_duplicate: bool = False,
) -> list[ReplayCase]:
    """Extract cases from all orchestration kits in a project.

    Scans all plan candidates, filters by minimum score, and creates a case
    for each qualifying kit.  Returns the list of created cases.
    """
    project = Path(project_path).resolve()
    all_sources = _detect_all_sources(project)

    if not all_sources:
        return []

    # Pre-compute evidence once (expensive I/O)
    cached_evidence = _discover_case_evidence(
        project,
        codex_history_roots=codex_history_roots,
        claude_history_roots=claude_history_roots,
    )

    # Build force-source dicts for candidates that meet the threshold.
    # max_cases is applied after quality filtering so low-quality candidates
    # cannot crowd out later usable cases.
    qualified: list[dict[str, object]] = []
    for score, reasons, path, source_type in all_sources:
        if score < min_score:
            continue
        qualified.append({
            "source_type": source_type,
            "source_path": str(path),
            "plan_path": str(path),
            "title": _first_heading(path),
            "selection_reason": f"Scored {score} ({reasons}) — {source_type}",
            "verification_commands": _extract_verification(project, path.parent),
            **_plan_source_metadata(project, path.parent),
        })

    created: list[ReplayCase] = []
    seen_case_ids: set[str] = set()
    for src in qualified:
        if len(created) >= max_cases:
            break
        expected_case_id = _generate_case_id(project, src)
        try:
            case = intake(
                cases_root=cases_root,
                project_path=project_path,
                allow_duplicate=allow_duplicate,
                _force_source=src,
                _cached_evidence=cached_evidence,
            )
            if case.id in seen_case_ids:
                continue
            quality_issues = _case_quality_gate_issues(case)
            if quality_issues:
                if case.id == expected_case_id and case.root.exists():
                    shutil.rmtree(case.root, ignore_errors=True)
                continue
            created.append(case)
            seen_case_ids.add(case.id)
        except Exception:
            continue

    return created


def _case_quality_gate_issues(case: ReplayCase) -> list[str]:
    """Return blocking quality issues for batch-extracted cases."""
    issues: list[str] = []
    issues.extend(validate_case(case))
    issues.extend(validate_case_task(case))

    if not case.verification_commands:
        issues.append("missing_verification_commands")

    ref_dir = case.root / "_reference"
    diff_path = ref_dir / "diff.patch"
    if not diff_path.exists() or diff_path.stat().st_size == 0:
        issues.append("missing_reference_diff")

    metadata_path = ref_dir / "reference_metadata.yaml"
    metadata = parse_simple_yaml(metadata_path) if metadata_path.exists() else {}
    changed_files = metadata.get("changed_files") or []
    if not isinstance(changed_files, list) or not changed_files:
        issues.append("missing_reference_changed_files")

    return issues


def create_case(
    *,
    cases_root: str | Path,
    project_path: str | Path,
    plan_path: str | Path,
    base_commit: str,
    case_id: str,
    verification_commands: list[str] | tuple[str, ...] | None = None,
    allow_duplicate: bool = False,
) -> ReplayCase:
    root = Path(cases_root) / case_id
    root.mkdir(parents=True, exist_ok=True)
    project = Path(project_path).resolve()
    raw_plan = Path(plan_path)
    plan = raw_plan.resolve() if raw_plan.is_absolute() else (project / raw_plan).resolve()
    cases = Path(cases_root)

    # Auto-extract verification commands from plan when not explicitly provided
    commands = tuple(verification_commands or ())
    if not commands and plan.exists():
        commands = tuple(_extract_verification(project, plan.parent))

    # Try to discover local evidence sources
    evidence_records = _discover_case_evidence(project)
    evidence_labels = tuple(_evidence_record_label(r) for r in evidence_records)

    auto_reason = ""
    if commands:
        auto_reason += f"; auto-extracted {len(commands)} verification command(s)"
    if evidence_records:
        auto_reason += f"; {len(evidence_records)} evidence source(s) found"

    case = ReplayCase(
        id=case_id,
        root=root,
        project_path=project,
        plan_path=plan,
        base_commit=base_commit,
        verification_commands=commands,
        base_source="manual",
        base_confidence="user_supplied",
        source_type="manual",
        source_path=str(plan),
        selection_reason=f"Manual case created from explicit plan and base commit{auto_reason}",
        synthetic_case=False,
        evidence_sources=evidence_labels,
        reference_target=_git_head(project),
    )
    _write_case_yaml(case)
    _write_evidence_sources(case, evidence_records)
    _write_case_task(case, evidence_records)
    _save_reference_evidence(case, project)

    # Duplicate check after all files are written
    case, _dup_status = _apply_duplicate_check(
        case,
        cases,
        allow_duplicate=allow_duplicate,
    )

    return case


def load_case(cases_root: str | Path, case_id: str) -> ReplayCase:
    root = Path(cases_root) / case_id
    data = parse_simple_yaml(root / "case.yaml")
    if not data.get("id"):
        raise ValueError(f"case.yaml at {root} is missing required field 'id'")
    commands = tuple(data.get("verification_commands") or [])
    return ReplayCase(
        id=str(data["id"]),
        root=root,
        project_path=Path(str(data.get("project_path", ""))),
        plan_path=Path(str(data.get("plan_path", ""))),
        base_commit=str(data.get("base_commit", "")),
        verification_commands=commands,
        base_source=str(data.get("base_source", "")),
        base_confidence=str(data.get("base_confidence", "unknown")),
        source_type=str(data.get("source_type", "manual")),
        source_path=str(data.get("source_path", "")),
        selection_reason=str(data.get("selection_reason", "")),
        synthetic_case=str(data.get("synthetic_case", "false")).lower() == "true",
        evidence_sources=tuple(data.get("evidence_sources") or []),
        selected_candidate_id=str(data.get("selected_candidate_id", "")),
        candidate_score=float(str(data.get("candidate_score", "0"))),
        candidate_source_type=str(data.get("candidate_source_type", "")),
        reference_target=str(data.get("reference_target", "")),
    )


def _write_case_task(
    case: ReplayCase,
    evidence_records: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
) -> None:
    """Write task.md — the contract that drives agent execution."""
    lines = [
        f"# Replay Case: {case.id}",
        "",
        "This case captures a historical project situation for agent evaluation.",
        "",
        "Do not launch or control an agent automatically from this case.",
        "Replay Checker prepares the contract and evidence paths; the user chooses the agent platform and model.",
        "",
        "## Source",
        f"- Project: `{case.project_path}`",
    ]

    if case.base_commit:
        lines.append(f"- Base commit: `{case.base_commit}`")
    else:
        lines.append("- Base commit: (none — working from current project state)")

    if case.synthetic_case:
        lines.extend([
            f"- Source type: `synthetic` (generated from git history, not a human-written plan)",
            f"- Selection reason: {case.selection_reason}",
            "",
            "## Note",
            "This is a synthetic case reconstructed from git history.",
            "Reference evidence (commit diff) is stored separately and not shown to the executing agent.",
        ])
    elif case.source_type in ("no_git", "empty_history"):
        lines.extend([
            f"- Source type: `{case.source_type}`",
            f"- Selection reason: {case.selection_reason}",
            "",
            "## Note",
            "This case has no git history baseline. The agent should work from the current project state.",
        ])
    else:
        lines.extend([
            f"- Plan package: `{case.plan_path}`",
            f"- Source type: {case.source_type}",
        ])

        if case.plan_path:
            goal_text = _extract_goal_section(Path(case.plan_path)) or _extract_goal_from_plan(str(case.plan_path))
            if goal_text:
                lines.extend(["", "## Goal", goal_text])

    if case.verification_commands:
        lines.extend([
            "",
            "## Verification Commands",
            *[f"- `{command}`" for command in case.verification_commands],
        ])

    _append_conversation_evidence(lines, evidence_records)

    (case.root / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _extract_goal_section(plan_path: Path) -> str:
    """Extract the Goal section from an orchestration kit INDEX.md."""
    if not plan_path.exists():
        return ""
    try:
        text = plan_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    in_goal = False
    goal_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Goal"):
            in_goal = True
            continue
        if in_goal:
            if line.startswith("## "):
                break
            goal_lines.append(line)

    # Trim leading/trailing blank lines
    while goal_lines and not goal_lines[0].strip():
        goal_lines.pop(0)
    while goal_lines and not goal_lines[-1].strip():
        goal_lines.pop()

    return "\n".join(goal_lines).strip()


def _find_reference_target(project_path: Path, base_commit: str, base_source: str) -> str | None:
    """Find the target commit whose diff serves as oracle evidence.

    Supports synthetic cases (synthetic_target_parent, synthetic_first_commit),
    plan-based cases (plan_first_commit_parent, plan_first_commit), and manual
    or fallback cases (user_supplied, head_fallback).
    """
    if not base_commit:
        return None

    # Synthetic: base is parent of target
    if base_source == "synthetic_target_parent":
        output = _git_output(
            ["log", "--no-merges", "--format=%H", f"{base_commit}..HEAD", "--"],
            cwd=project_path,
        )
        commits = [c for c in output.strip().splitlines() if c]
        return commits[-1] if commits else None

    # Synthetic: base IS the target (first commit)
    if base_source == "synthetic_first_commit":
        return base_commit

    # Plan-based: base is parent of first plan-introducing commit; that commit is the target
    if base_source == "plan_first_commit_parent":
        output = _git_output(
            ["log", "--no-merges", "--format=%H", f"{base_commit}..HEAD", "--"],
            cwd=project_path,
        )
        commits = [c for c in output.strip().splitlines() if c]
        return commits[-1] if commits else None

    # Plan-based: base is the first plan commit; use HEAD as target
    if base_source == "plan_first_commit":
        head = _git_head(project_path)
        return head if head and head != base_commit else None

    # Manual or fallback: use HEAD as target
    if base_source in ("user_supplied", "manual", "head_fallback", "head_dirty", "commit", "plan_state_base"):
        head = _git_head(project_path)
        return head if head and head != base_commit else None

    return None


def _save_reference_evidence(case: ReplayCase, project_path: Path) -> None:
    """Persist reference/oracle evidence for a case outside agent-visible paths.

    Generates a diff from base_commit to the inferred target commit plus metadata.
    The _reference/ directory must never be accessible to executing agents.
    """
    if not case.base_commit:
        return
    target = case.reference_target or _find_reference_target(project_path, case.base_commit, case.base_source)
    if not target or target == case.base_commit:
        return

    log_output = _git_output(["log", "--format=%s%n%b", "-1", target], cwd=project_path)
    commit_message = _compact_reference_commit_message(log_output)
    diff_output = _git_output(
        ["diff", case.base_commit, target, "--binary"],
        cwd=project_path,
    )

    changed_output = _git_output(
        ["diff", "--name-only", case.base_commit, target],
        cwd=project_path,
    )
    changed_files = [f for f in changed_output.strip().splitlines() if f]

    ref_dir = case.root / "_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    (ref_dir / "diff.patch").write_text(diff_output, encoding="utf-8")

    _write_simple_yaml(
        ref_dir / "reference_metadata.yaml",
        {
            "target_commit": target,
            "base_commit": case.base_commit,
            "commit_message": commit_message,
            "changed_files": changed_files,
        },
    )


def _compact_reference_commit_message(log_output: str) -> str:
    """Keep reference metadata YAML parseable when commit messages are multiline."""
    return " ".join(line.strip() for line in log_output.splitlines() if line.strip())


# ---------------------------------------------------------------------------
# Synthetic task reconstruction helpers
# ---------------------------------------------------------------------------

_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java",
    ".rb", ".c", ".cpp", ".h", ".hpp", ".swift", ".kt", ".scala", ".cs",
    ".php", ".r", ".m", ".mm",
})

_TEST_NAMES: frozenset[str] = frozenset({"test", "tests", "spec", "__tests__", "testing"})

_CONFIG_NAMES: frozenset[str] = frozenset({
    "pyproject.toml", "setup.cfg", "setup.py", "package.json",
    "cargo.toml", "go.mod", "makefile", "tox.ini", ".gitignore",
    "dockerfile", "docker-compose.yml", "cmakelists.txt",
})

_DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".rst", ".txt", ".adoc"})


def _classify_changed_files(files: list[str]) -> dict[str, list[str]]:
    """Classify file paths into source/test/config/doc/other categories."""
    categories: dict[str, list[str]] = {"source": [], "test": [], "config": [], "doc": [], "other": []}
    for f in files:
        lower = f.lower()
        ext = Path(f).suffix.lower()
        parts_lower = tuple(p.lower() for p in Path(f).parts)

        if any(ind in parts_lower for ind in _TEST_NAMES):
            categories["test"].append(f)
        elif Path(f).name.lower() in _CONFIG_NAMES:
            categories["config"].append(f)
        elif ext in _DOC_EXTENSIONS:
            categories["doc"].append(f)
        elif ext in _SOURCE_EXTENSIONS:
            categories["source"].append(f)
        else:
            categories["other"].append(f)
    return {k: v for k, v in categories.items() if v}


def _is_test_path(path: str) -> bool:
    """Check if a path looks like a test file or lives in a test directory."""
    lower = path.lower()
    name = Path(path).name.lower()
    parts_lower = tuple(p.lower() for p in Path(path).parts)
    if name.startswith("test_") or name.endswith("_test.py") or ".test." in name or ".spec." in name:
        return True
    return any(ind in parts_lower for ind in _TEST_NAMES)


def _is_code_like(text: str) -> bool:
    """Heuristic: return True if *text* looks like code rather than prose.

    Checks each line independently.  A line is code-like if it starts with
    a language keyword, diff marker, brace/bracket, or looks like an
    assignment with length > 10.
    """
    lines = text.strip().splitlines()
    if not lines:
        return False
    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("def ", "class ", "function ", "import ", "from ",
                                "const ", "let ", "var ", "fn ", "pub ")):
            code_indicators += 1
        elif stripped.startswith(("+", "-", "@@")) and len(stripped) > 1:
            code_indicators += 2
        elif stripped.startswith(("{", "}", "```", "<!--")):
            code_indicators += 1
        elif re.search(r"^[a-zA-Z_]\w*\s*[=:]\s*", stripped) and len(stripped) > 10:
            code_indicators += 1
    return code_indicators >= 2


def _body_contains_code(lines: list[str]) -> bool:
    """Check if any line in the body looks like source code."""
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("def ", "class ", "function ", "import ", "from ",
                                "const ", "let ", "var ", "fn ", "pub ")):
            code_lines += 1
        elif re.search(r"^[a-zA-Z_]\w*\s*[=:]\s*", stripped) and len(stripped) > 15:
            code_lines += 1
    return code_lines >= 2


def _find_verification_hints(
    project_path: Path,
    changed_files: list[str],
) -> list[str]:
    """Find verification-related signals from project structure and changed files."""
    hints: list[str] = []

    # Check if any changed file is itself a test
    test_files = [f for f in changed_files if _is_test_path(f)]
    if test_files:
        if len(test_files) == 1:
            hints.append(f"Test file is part of the change: `{test_files[0]}`")
        else:
            hints.append(f"Test files are part of the change ({len(test_files)} files)")

    # Check for test directories in the project
    for test_dir_name in ("tests", "test", "spec"):
        test_dir = project_path / test_dir_name
        if test_dir.is_dir():
            hints.append(f"Test directory `{test_dir_name}/` exists in project")
            break

    # Check for build/test config files
    for config_name in ("Makefile", "pyproject.toml", "package.json", "Cargo.toml"):
        if (project_path / config_name).exists():
            content = ""
            try:
                content = (project_path / config_name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
            if "pytest" in content or "test" in content.lower():
                hints.append(f"Build config `{config_name}` contains test configuration")

    # Check for CI config with test commands
    for ci_path in (".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"):
        full = project_path / ci_path
        if full.exists():
            hints.append(f"CI configuration found: `{ci_path}`")
            break

    return hints


def _find_doc_nearby(
    project_path: Path,
    changed_files: list[str],
) -> list[str]:
    """Find documentation headings or README files near changed paths."""
    hints: list[str] = []
    seen_dirs: set[str] = set()

    # Check for README/docs near changed files
    for f in changed_files:
        parent = str(Path(f).parent)
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for doc_name in ("README.md", "README", "README.rst"):
            doc_path = project_path / parent / doc_name
            if doc_path.is_file():
                hints.append(f"Documentation: `{parent}/{doc_name}`")
                break

    # Check overarching plans directory
    plans_dir = project_path / "docs" / "plans"
    if plans_dir.is_dir():
        index_files = list(plans_dir.glob("**/INDEX.md"))
        if index_files:
            hints.append("Project contains orchestration plan documentation")

    return hints


def _build_reconstruction_context(
    case: ReplayCase,
    project_path: Path,
    evidence_records: list[EvidenceRecord],
) -> dict[str, object]:
    """Build deterministic context for synthetic task reconstruction.

    Gathers commit metadata, changed file paths (names only, no content),
    file classification, verification hints, and conversation summaries.
    Does NOT copy diff hunks, target implementation code, or raw logs.
    """
    target = _find_reference_target(project_path, case.base_commit, case.base_source)
    ctx: dict[str, object] = {
        "target_commit": target or "",
        "commit_subject": "",
        "commit_body": "",
        "changed_files": [],
        "file_categories": {},
        "test_signals": [],
        "verification_hints": [],
        "doc_hints": [],
        "risk_notes": [],
        "scope_summary": "unknown scope",
    }

    if not target:
        ctx["risk_notes"] = ["No target commit identified"]
        return ctx

    # Commit subject + body
    full_msg = _git_output(["log", "--format=%B", "-1", target], cwd=project_path)
    msg_lines = full_msg.strip().splitlines()
    ctx["commit_subject"] = msg_lines[0] if msg_lines else ""
    body = "\n".join(msg_lines[1:]).strip() if len(msg_lines) > 1 else ""

    # Filter out diff dumps, git signatures, and code-like bodies
    if body:
        safe_body_lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Signed-off-by:", "Co-authored-by:", "Reviewed-by:", "Acked-by:", "Tested-by:", "Change-Id:", "Closes #", "Fixes #", "Refs #", "See also:", "CR:", "Differential Revision:")):
                continue
            if stripped.startswith(("diff ", "index ", "---", "+++", "@@")):
                break
            safe_body_lines.append(stripped)
        if safe_body_lines and not _body_contains_code(safe_body_lines):
            ctx["commit_body"] = " ".join(safe_body_lines)[:500]
        else:
            ctx["commit_body"] = ""

    if not ctx["commit_body"]:
        ctx["risk_notes"] = ["Commit message only — no detailed body available"]

    # Changed files (names only)
    changed_output = _git_output(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", target],
        cwd=project_path,
    )
    changed_files = [f for f in changed_output.strip().splitlines() if f]
    ctx["changed_files"] = changed_files

    if not changed_files:
        ctx["risk_notes"] = [*(ctx["risk_notes"] if isinstance(ctx["risk_notes"], list) else []), "No changed files detected"]
    else:
        categories = _classify_changed_files(changed_files)
        ctx["file_categories"] = categories
        scope_parts: list[str] = []
        if categories.get("source"):
            scope_parts.append(f"{len(categories['source'])} source file(s)")
        if categories.get("test"):
            scope_parts.append(f"{len(categories['test'])} test file(s)")
        if categories.get("config"):
            scope_parts.append(f"{len(categories['config'])} config file(s)")
        if categories.get("doc"):
            scope_parts.append(f"{len(categories['doc'])} doc file(s)")
        if categories.get("other"):
            scope_parts.append(f"{len(categories['other'])} other file(s)")
        ctx["scope_summary"] = ", ".join(scope_parts) if scope_parts else "unknown scope"

    # Test signals from changed files
    ctx["test_signals"] = [f for f in changed_files if _is_test_path(f)]

    # Verification hints from project structure
    ctx["verification_hints"] = _find_verification_hints(project_path, changed_files)

    # Doc hints
    ctx["doc_hints"] = _find_doc_nearby(project_path, changed_files)

    # Bounded conversation summaries
    conversation = [r for r in evidence_records if r.source_type in ("codex_history", "claude_history")]
    if conversation:
        ctx["conversation_summaries"] = [r.summary[:500] for r in conversation[:3]]

    # Risk assessment
    risk_notes: list[str] = list(ctx.get("risk_notes", []))
    if not risk_notes and not ctx.get("test_signals") and not ctx.get("verification_hints"):
        risk_notes.append("No verification hints found — task scope is inferred from commit message only")
    if len(changed_files) == 1:
        risk_notes.append("Single file change — limited scope signals")
    ctx["risk_notes"] = risk_notes

    return ctx


def _write_synthetic_task(
    case: ReplayCase,
    project_path: Path,
    evidence_records: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
) -> None:
    """Write a richer task.md for synthetic cases using safe local evidence.

    Includes: objective (commit message), scope hints (changed file paths
    with categories), verification hints (test files, build config), and
    risk metadata.  Does NOT copy diff hunks, target implementation code,
    or raw conversation logs into the execution-facing task file.
    """
    ctx = _build_reconstruction_context(case, project_path, list(evidence_records))

    lines = [
        f"# Replay Case: {case.id}",
        "",
        "This case captures a historical project situation for agent evaluation.",
        "",
        "Do not launch or control an agent automatically from this case.",
        "Replay Checker prepares the contract and evidence paths; the user chooses the agent platform and model.",
        "",
        "## Source",
        f"- Project: `{case.project_path}`",
    ]
    if case.base_commit:
        lines.append(f"- Base commit: `{case.base_commit}`")
    else:
        lines.append("- Base commit: (none \u2014 working from current project state)")
    lines.extend([
        f"- Source type: `synthetic` (generated from git history, not a human-written plan)",
        f"- Selection reason: {case.selection_reason}",
        "",
        "## Reconstructed Task",
    ])

    # Objective
    commit_subject = str(ctx.get("commit_subject", ""))
    commit_body = str(ctx.get("commit_body", ""))
    if commit_subject:
        lines.extend([
            "",
            "### Objective",
            "",
            f"The historical commit message was: **{commit_subject}**",
        ])
        if commit_body:
            lines.extend([
                "",
                f"Additional context: _{commit_body}_",
            ])
        lines.extend([
            "",
            f"Starting from base commit `{case.base_commit}`, implement the changes described by this commit message.",
            "The original implementation exists as a later commit — produce equivalent changes independently.",
        ])
    else:
        lines.extend([
            "",
            "### Objective",
            "",
            "No commit message available. Review the project state at the base commit and identify meaningful improvements.",
        ])

    # Scope Hints
    changed_files: list[str] = list(ctx.get("changed_files", []))
    file_categories: dict[str, list[str]] = dict(ctx.get("file_categories", {}))
    if changed_files:
        lines.extend([
            "",
            "### Scope Hints",
            "",
            f"The target change affects {ctx.get('scope_summary', 'unknown scope')}:",
            "",
        ])
        # Show files grouped by category
        for cat, files in file_categories.items():
            cat_label = {"source": "Source", "test": "Test", "config": "Config", "doc": "Documentation", "other": "Other"}.get(cat, cat.title())
            lines.append(f"**{cat_label}**:")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    # Verification Hints
    verification_hints: list[str] = list(ctx.get("verification_hints", []))
    test_signals: list[str] = list(ctx.get("test_signals", []))
    doc_hints: list[str] = list(ctx.get("doc_hints", []))
    if verification_hints or test_signals:
        lines.extend([
            "### Verification Hints",
            "",
        ])
        for hint in verification_hints:
            lines.append(f"- {hint}")
        if test_signals:
            if len(test_signals) == 1:
                lines.append(f"- Changed file includes test: `{test_signals[0]}`")
            else:
                lines.append(f"- {len(test_signals)} test files are part of the change")
        lines.append("")

    if doc_hints:
        lines.extend([
            "### Documentation Nearby",
            "",
        ])
        for hint in doc_hints:
            lines.append(f"- {hint}")
        lines.append("")

    # Risk Metadata
    risk_notes: list[str] = list(ctx.get("risk_notes", []))
    if risk_notes:
        lines.extend([
            "### Reconstruction Confidence",
            "",
        ])
        for risk in risk_notes:
            lines.append(f"- {risk}")
        lines.append("")

    _append_conversation_evidence(lines, evidence_records)

    (case.root / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_conversation_evidence(
    lines: list[str],
    records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
) -> None:
    conversation = [
        record for record in records
        if record.source_type in {"codex_history", "claude_history"}
    ]
    if not conversation:
        return
    lines.extend([
        "",
        "## Conversation Evidence",
        "Bounded summaries from local agent history. Raw logs are not included.",
    ])
    for record in conversation[:5]:
        lines.append(f"- `{record.source_type}`: {record.summary}")


def _next_run_id(case_id: str, runs_root: Path) -> str:
    prefix = f"{case_id}-"
    highest = 0
    for path in runs_root.glob(f"{case_id}-*"):
        if not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{case_id}-{highest + 1:03d}"


def prepare_run(case: ReplayCase, *, runs_root: str | Path, runner_label: str) -> ReplayRun:
    missing = validate_case(case)
    if missing:
        raise ValueError(
            f"Case {case.id} is incomplete — missing: {', '.join(missing)}. "
            "Required fields: base_source, base_confidence, source_type, source_path, selection_reason, evidence_sources.md"
        )
    runs = Path(runs_root)
    runs.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = _next_run_id(case.id, runs)
        run_root = runs / run_id
        workspace = run_root / "workspace"
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    _create_worktree(case, workspace)
    run = ReplayRun(run_id, run_root, case, runner_label, workspace)
    _write_simple_yaml(
        run_root / "run.yaml",
        {
            "id": run.id,
            "case_id": case.id,
            "runner_label": runner_label,
            "anonymous_runner_id": _stable_anonymous_runner_id(run.id),
            "workspace": workspace,
            "base_commit": case.base_commit,
            "status": "prepared",
        },
    )
    _write_completion_template(run)
    _write_run_task(run)
    return run


def _write_completion_template(run: ReplayRun) -> None:
    lines = [
        "# Completion Report",
        "",
        "Fill this out after the agent run. The status line is required for evidence collection.",
        "",
        "status: ",
        "",
        "## Summary",
        "",
        "What was accomplished?",
        "",
        "## Changes Made",
        "",
        "List files changed and why.",
        "Run `git add -A` in the workspace before finishing so that new files appear in the diff.",
        "",
        "## Verification",
        "",
        "What verification commands were run and what was their output?",
        "",
        "## Issues / Notes",
        "",
        "Any blockers, risks, or notes for the scorer.",
    ]
    (run.root / "completion_report_template.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_worktree(case: ReplayCase, workspace: Path) -> None:
    # When base_commit is empty (no-git, empty history, etc.), skip git worktree
    # and copy the project tree directly, excluding .git and forbidden paths.
    if not case.base_commit.strip():
        _copy_project_tree(case.project_path, workspace)
        return

    # Try git worktree first; fall back to git archive when sandbox blocks .git/worktrees/.
    try:
        completed = _git_run(
            ["worktree", "add", "--detach", str(workspace), "--", case.base_commit],
            cwd=case.project_path,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("git worktree add timed out")
    if completed.returncode == 0:
        return
    # Worktree failed — try git archive fallback
    if _extract_commit_fallback(case, workspace):
        return
    # Last resort: copy project tree directly
    _copy_project_tree(case.project_path, workspace)


def _extract_commit_fallback(case: ReplayCase, workspace: Path) -> bool:
    """Extract base_commit files via git archive when worktree creation fails.

    Returns True on success, False if the fallback also fails.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    # Use subprocess pipe directly: git archive | tar -x
    try:
        archive_env = {**os.environ, "COPYFILE_DISABLE": "1"}
        archive_proc = subprocess.Popen(
            ["git", "archive", case.base_commit],
            cwd=str(case.project_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=archive_env,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-x", "-C", str(workspace)],
            stdin=archive_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        archive_proc.stdout.close()  # allow archive_proc to receive SIGPIPE
        _, tar_stderr = tar_proc.communicate(timeout=600)
        archive_proc.wait(timeout=600)
    except Exception:
        return False
    if archive_proc.returncode != 0 or tar_proc.returncode != 0:
        return False
    # Init a fresh git repo so collect_run can produce a diff
    _git_run(["init"], cwd=workspace)
    _git_run(["add", "-A"], cwd=workspace)
    _git_run(["commit", "-m", "initial snapshot at base_commit", "--allow-empty"], cwd=workspace)
    return True


def _copy_project_tree(project_path: Path, workspace: Path) -> None:
    """Copy project tree to workspace, excluding .git and forbidden paths.

    Initializes a fresh git repo in the workspace so that collect_run can
    produce a meaningful diff even when the source project has no git history.
    """
    shutil.copytree(
        project_path,
        workspace,
        ignore=shutil.ignore_patterns(".git", ".gitignore", "_reference", ".gradle", ".idea", ".claude", ".worktrees", "build"),
        dirs_exist_ok=True,
    )
    # Init a fresh repo and commit the baseline so agent changes show up as diffs.
    try:
        git_run(["init"], cwd=workspace)
        git_run(["add", "."], cwd=workspace)
        git_run(["commit", "-m", "baseline: copied from project", "--allow-empty"], cwd=workspace)
    except (subprocess.TimeoutExpired, OSError):
        pass  # Workspace is still usable without git; diff collection will produce empty results.


def _write_run_task(run: ReplayRun) -> None:
    pkg = compile_execution_package(
        run_id=run.id,
        case_id=run.case.id,
        case_task_path=str(run.case.root / "task.md"),
        plan_path=str(run.case.plan_path),
        workspace=str(run.workspace),
        completion_template_path=str(run.root / "completion_report_template.md"),
        verification_commands=run.case.verification_commands,
    )
    renderer = ExecutionPackageRenderer()
    text = renderer.render(pkg)
    (run.root / "TASK.md").write_text(text, encoding="utf-8")


def load_run(runs_root: str | Path, run_id: str, *, cases_root: str | Path) -> ReplayRun:
    root = Path(runs_root) / run_id
    run_yaml = root / "run.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"run.yaml not found at {run_yaml}")
    data = parse_simple_yaml(run_yaml)
    case_id = str(data.get("case_id", ""))
    if not case_id:
        raise ValueError(f"run.yaml at {root} is missing required field 'case_id'")
    case = load_case(cases_root, case_id)
    workspace = _resolve_workspace_path(root, data)
    return ReplayRun(
        id=str(data.get("id", run_id)),
        root=root,
        case=case,
        runner_label=str(data.get("runner_label", "")),
        workspace=workspace,
    )


def inspect_run_dir(run_root: str | Path) -> RunHealth:
    """Inspect a run directory and derive a canonical health status from evidence."""
    root = Path(run_root)
    data = parse_simple_yaml(root / "run.yaml") if (root / "run.yaml").exists() else {}
    run_id = str(data.get("id", root.name))
    workspace = _resolve_workspace_path(root, data)
    workspace_exists = workspace.is_dir()
    completion_ws = workspace / "completion_report.md"
    completion_legacy = root / "completion_report.md"
    completion = completion_ws if completion_ws.exists() else completion_legacy
    raw_status = _completion_status(completion)

    raw_base = data.get("base_commit", "")
    base = "" if isinstance(raw_base, list) else str(raw_base).strip()
    if base == "[]":
        base = ""
    diff = ""
    changed_files: tuple[str, ...] = ()
    if workspace_exists:
        # When base is empty, try HEAD as reference (for copytree-with-git-init workspaces).
        diff_ref = base
        if not diff_ref:
            head = _git_capture(workspace, ["rev-parse", "HEAD"]).strip()
            diff_ref = head if head else ""
        diff = _git_capture(workspace, ["diff", diff_ref, "--binary"]) if diff_ref else _git_capture(workspace, ["diff", "--binary"])
        changed_output = _git_capture(workspace, ["diff", "--name-only", diff_ref]) if diff_ref else _git_capture(workspace, ["diff", "--name-only"])
        changed_files = tuple(f for f in changed_output.strip().splitlines() if f)

    status, reason = _derive_canonical_run_status(
        raw_status=raw_status,
        workspace_exists=workspace_exists,
        completion_report_exists=completion.exists(),
        has_result_diff=bool(diff.strip()) and bool(changed_files),
        completion_text=_read_text_if_exists(completion),
    )
    return RunHealth(
        run_id=run_id,
        status=status,
        raw_status=raw_status,
        workspace=workspace,
        workspace_exists=workspace_exists,
        completion_report_exists=completion.exists(),
        has_result_diff=bool(diff.strip()) and bool(changed_files),
        changed_files=changed_files,
        reason=reason,
    )


def doctor_runs(runs_root: str | Path, *, runner_label: str | None = None) -> tuple[RunHealth, ...]:
    root = Path(runs_root)
    if not root.is_dir():
        return ()
    reports: list[RunHealth] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "run.yaml").exists()):
        data = parse_simple_yaml(run_dir / "run.yaml")
        if runner_label is not None and str(data.get("runner_label", "")) != runner_label:
            continue
        reports.append(inspect_run_dir(run_dir))
    return tuple(reports)


def _resolve_workspace_path(run_root: Path, data: dict[str, object]) -> Path:
    raw = str(data.get("workspace", "workspace")).strip() or "workspace"
    workspace = Path(raw)
    if workspace.is_absolute():
        return workspace
    if raw.startswith("runs/") or raw.startswith("runs\\"):
        return run_root.parent.parent / workspace
    return run_root / workspace


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------


def _gather_telemetry(
    run: ReplayRun,
    base_commit: str,
    changed_files: tuple[str, ...],
    diff: str,
) -> Telemetry:
    workspace = run.workspace

    added_lines, deleted_lines = _parse_diff_stat(workspace, base_commit)
    forbidden = _detect_forbidden_paths(changed_files)
    suspicious = _detect_suspicious_paths(changed_files)
    untracked_count = _count_untracked(workspace)
    commit_count = _count_workspace_commits(workspace, base_commit)
    rollback = _detect_rollback_signal(workspace)

    return Telemetry(
        changed_file_count=len(changed_files),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        forbidden_path_touches=tuple(forbidden),
        suspicious_path_touches=tuple(suspicious),
        untracked_file_count=untracked_count,
        commit_count=commit_count,
        rollback_signal=rollback,
    )


def _parse_diff_stat(workspace: Path, base_commit: str) -> tuple[int, int]:
    stat = _git_capture(workspace, ["diff", "--stat", base_commit]) if base_commit else _git_capture(workspace, ["diff", "--stat"])
    if not stat.strip():
        return (0, 0)
    added = 0
    deleted = 0
    for part in stat.split(","):
        part = part.strip()
        if "insertion" in part:
            try:
                added = int(part.split()[0])
            except (ValueError, IndexError):
                pass
        elif "deletion" in part:
            try:
                deleted = int(part.split()[0])
            except (ValueError, IndexError):
                pass
    return (added, deleted)


def _detect_forbidden_paths(changed_files: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for f in changed_files:
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern in f:
                hits.append(f)
                break
    return hits


def _detect_suspicious_paths(changed_files: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for f in changed_files:
        fname = f.split("/")[-1].lower()
        for pattern in SUSPICIOUS_PATH_PATTERNS:
            if pattern in fname:
                hits.append(f)
                break
    return hits


def _count_untracked(workspace: Path) -> int:
    output = _git_capture(workspace, ["ls-files", "--others", "--exclude-standard"])
    return len([line for line in output.strip().splitlines() if line])


def _count_workspace_commits(workspace: Path, base_commit: str) -> int:
    if not base_commit:
        return 0
    output = _git_capture(
        workspace, ["rev-list", "--count", f"{base_commit}..HEAD"]
    )
    try:
        return int(output.strip()) if output.strip() else 0
    except ValueError:
        return 0


def _detect_rollback_signal(workspace: Path) -> str:
    reflog = _git_capture(workspace, ["reflog", "--oneline", "-5"])
    if not reflog.strip():
        return ""
    signals: list[str] = []
    for line in reflog.splitlines():
        lower = line.lower()
        if "reset" in lower:
            signals.append("reset")
        if "checkout" in lower:
            signals.append("checkout")
        if "rebase" in lower:
            signals.append("rebase")
    return ", ".join(sorted(set(signals))) if signals else ""


def _write_telemetry_block(
    evidence_data: dict[str, object],
    telemetry: Telemetry,
) -> None:
    evidence_data["telemetry"] = {
        "changed_file_count": telemetry.changed_file_count,
        "added_lines": telemetry.added_lines,
        "deleted_lines": telemetry.deleted_lines,
        "forbidden_path_touches": list(telemetry.forbidden_path_touches),
        "suspicious_path_touches": list(telemetry.suspicious_path_touches),
        "untracked_file_count": telemetry.untracked_file_count,
        "commit_count": telemetry.commit_count,
        "rollback_signal": telemetry.rollback_signal,
    }


# ---------------------------------------------------------------------------
# Core run operations
# ---------------------------------------------------------------------------


def collect_run(run: ReplayRun) -> Evidence:
    evidence_dir = run.root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Prefer the original base_commit saved at prepare-run time;
    # the case-level base_commit may have drifted.
    saved_base = _read_run_base(run) or run.case.base_commit
    base = saved_base

    # Determine effective diff reference.
    # When base is empty (no-git / empty-git projects), the copytree fallback
    # initializes a fresh repo with a baseline commit.  Use HEAD so that both
    # staged and unstaged agent changes appear in the diff.
    if not base:
        head = _git_capture(run.workspace, ["rev-parse", "HEAD"]).strip()
        diff_ref = head if head else ""
    else:
        diff_ref = base

    # Collect diff: first the standard tracked diff, then check for untracked
    # files that the agent may have created but not staged.  When untracked
    # files are present, stage them so the full result evidence is captured.
    # The evidence requirements tell agents to `git add -A`, but this is not
    # enforced — many agents skip it.  We recover the evidence here.
    diff = _git_capture(run.workspace, ["diff", diff_ref, "--binary"]) if diff_ref else _git_capture(run.workspace, ["diff", "--binary"])
    changed_output = _git_capture(run.workspace, ["diff", "--name-only", diff_ref]) if diff_ref else _git_capture(run.workspace, ["diff", "--name-only"])

    untracked_raw = _git_capture(run.workspace, ["ls-files", "--others", "--exclude-standard"])
    untracked_files = [f for f in untracked_raw.strip().splitlines() if f]
    if untracked_files:
        _git_capture(run.workspace, ["add", "-A"])
        diff = _git_capture(run.workspace, ["diff", "--cached", diff_ref, "--binary"]) if diff_ref else _git_capture(run.workspace, ["diff", "--cached", "--binary"])
        changed_output = _git_capture(run.workspace, ["diff", "--cached", "--name-only", diff_ref]) if diff_ref else _git_capture(run.workspace, ["diff", "--cached", "--name-only"])

    changed_files = tuple(f for f in changed_output.strip().splitlines() if f)
    diff_path = evidence_dir / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8")
    # Prefer workspace root (post-2026-06-04 contract), fall back to run root.
    completion_ws = run.workspace / "completion_report.md"
    completion_legacy = run.root / "completion_report.md"
    completion = completion_ws if completion_ws.exists() else completion_legacy
    run_status = _completion_status(completion)

    missing: list[str] = []
    if run_status == "missing-completion-report":
        missing.append("completion_report.md")
    if not diff.strip():
        missing.append("diff.patch (empty)")
    if not changed_files:
        missing.append("changed_files (none)")

    telemetry = _gather_telemetry(run, diff_ref, changed_files, diff)

    evidence_data: dict[str, object] = {
        "run_id": run.id,
        "status": run_status,
        "changed_files": list(changed_files),
        "diff_path": diff_path,
        "missing_fields": missing,
    }
    _write_telemetry_block(evidence_data, telemetry)
    _write_simple_yaml(evidence_dir / "evidence.yaml", evidence_data)
    _update_run_yaml_status(run.root, inspect_run_dir(run.root))
    return Evidence(run.id, run_status, changed_files, diff_path, tuple(missing))


def _git_capture(cwd: Path, args: list[str]) -> str:
    return git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


def _parse_changed_files(status: str) -> list[str]:
    files = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


def _read_run_base(run: ReplayRun) -> str:
    """Read the original base_commit saved at prepare-run time."""
    run_yaml = run.root / "run.yaml"
    if run_yaml.exists():
        data = parse_simple_yaml(run_yaml)
        base = data.get("base_commit", "")
        # Handle YAML roundtrip: empty string may parse as empty list [].
        if isinstance(base, list):
            return ""
        base = str(base).strip()
        if base and base != "[]":
            return base
    return ""


def _completion_status(path: Path) -> str:
    if not path.exists():
        return "missing-completion-report"
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^status:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "reported"


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _derive_canonical_run_status(
    *,
    raw_status: str,
    workspace_exists: bool,
    completion_report_exists: bool,
    has_result_diff: bool,
    completion_text: str,
) -> tuple[str, str]:
    status = _normalize_status(raw_status)
    if not workspace_exists:
        return ("failed", "workspace missing")
    if status in {"failed", "invalid", "stale"}:
        return (status, f"completion report status is {status}")
    if status in {"partial", "blocked"}:
        return ("partial", f"completion report status is {status}")
    if not completion_report_exists or status == "missing-completion-report":
        return ("failed", "completion_report.md missing")
    if status == "reported":
        return ("partial", "completion report has no explicit status line")
    if status in {"completed", "solved"}:
        if has_result_diff:
            return ("completed", "completed report and non-empty result diff")
        if _completion_declares_noop(completion_text):
            return ("completed", "completed report declares no-op/base already satisfied")
        return ("failed", "completed report has no result diff")
    return ("partial", f"unrecognized completion status: {raw_status}")


def _normalize_status(raw_status: str) -> str:
    """Map common variant status labels to canonical forms.

    Agents use inconsistent vocabulary (complete, completed, pass, etc.).
    This normalizes known variants so the state machine doesn't miscategorize
    valid completion reports as unrecognized/partial.
    """
    s = raw_status.lower().strip()
    # Accept 'complete' as a valid completion signal — it is the most common
    # variant across agents (21/42 reports in audit).
    if s == "complete":
        return "completed"
    # 'pass' is used by some agents as a completion marker.
    if s == "pass":
        return "completed"
    # 'complete-already-implemented' is a no-op variant.
    if s == "complete-already-implemented":
        return "completed"
    return s


def _completion_declares_noop(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "no-op",
        "no changes needed",
        "already implemented",
        "already contained",
        "already had",
        "base commit already",
        "base commit 已",
        "已含完整实现",
    )
    return any(pattern in lowered for pattern in patterns)


def _update_run_yaml_status(run_root: Path, health: RunHealth) -> None:
    run_yaml = run_root / "run.yaml"
    if not run_yaml.exists():
        return
    data = parse_simple_yaml(run_yaml)
    data["status"] = health.status
    data["raw_status"] = health.raw_status
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data["workspace_exists"] = "yes" if health.workspace_exists else "no"
    data["completion_report_exists"] = "yes" if health.completion_report_exists else "no"
    data["has_result_diff"] = "yes" if health.has_result_diff else "no"
    data["changed_file_count"] = len(health.changed_files)
    data["status_reason"] = health.reason
    _write_simple_yaml(run_yaml, data)


def validate_case(case: ReplayCase) -> list[str]:
    """Return a list of missing/incomplete fields for a case.

    An empty list means the case is complete.
    """
    missing: list[str] = []
    if not case.base_source:
        missing.append("base_source")
    if not case.base_confidence or case.base_confidence == "unknown":
        missing.append("base_confidence")
    # base_commit may be empty for no_git / empty_history cases — that is valid.
    # Only flag it when the case claims a git-based source but has no commit.
    if not case.base_commit and case.source_type not in ("no_git", "empty_history", "manual"):
        missing.append("base_commit")
    if not case.source_type:
        missing.append("source_type")
    if not case.source_path and case.source_type not in ("git_history", "no_git", "empty_history"):
        missing.append("source_path")
    if not case.selection_reason:
        missing.append("selection_reason")
    if not case.plan_path and case.source_type in ("orchestration_kit", "handoff_plan"):
        missing.append("plan_path")
    evidence_md = case.root / "evidence_sources.md"
    if not evidence_md.exists():
        missing.append("evidence_sources.md")
    # Quality warnings are logged separately, not added to blocking missing list.
    return missing


_MAX_EVIDENCE_SOURCES = 10
_MAX_PLAN_LINES = 200


def _warn_case_quality(case: ReplayCase) -> list[str]:
    """Return quality warnings for a case (non-blocking)."""
    warnings: list[str] = []

    # Warn if evidence_sources list is too long (likely over-scoped)
    evidence_md = case.root / "evidence_sources.md"
    if evidence_md.exists():
        content = evidence_md.read_text(encoding="utf-8")
        source_count = sum(1 for line in content.splitlines() if line.strip().startswith("-"))
        if source_count > _MAX_EVIDENCE_SOURCES:
            warnings.append(
                f"evidence_sources 有 {source_count} 项（上限 {_MAX_EVIDENCE_SOURCES}），"
                "任务范围可能过大"
            )

    # Warn if orchestration kit plan is too large
    if case.plan_path and case.source_type == "orchestration_kit":
        plan = Path(case.plan_path)
        if plan.exists():
            line_count = len(plan.read_text(encoding="utf-8").splitlines())
            if line_count > _MAX_PLAN_LINES:
                warnings.append(
                    f"plan 文件 {plan.name} 有 {line_count} 行（建议 ≤{_MAX_PLAN_LINES}），"
                    "任务可能过于复杂"
                )

    return warnings


def validate_case_task(case: ReplayCase) -> list[str]:
    """Check case task.md content quality for agent executability."""
    task_md = case.root / "task.md"
    if not task_md.exists():
        return ["missing case task.md"]

    content = task_md.read_text(encoding="utf-8")
    issues: list[str] = []

    # Goal must exist and be substantive
    if "## Goal" not in content:
        issues.append("case task.md 缺少 ## Goal 段落")
    else:
        goal_text = content.split("## Goal")[1].split("##")[0].strip()
        if len(goal_text) < 20:
            issues.append(f"case task.md Goal 过于简略（{len(goal_text)} 字符，建议 ≥20）")

    # git_history cases must have a Change Summary
    if case.source_type == "git_history" and "## Change Summary" not in content:
        issues.append("git_history case 缺少 ## Change Summary")

    return issues


def score_run(run: ReplayRun, *, rubric_path: str | Path) -> Path:
    evidence_yaml = run.root / "evidence" / "evidence.yaml"
    if not evidence_yaml.exists():
        raise FileNotFoundError(
            f"Evidence not found at {evidence_yaml}. "
            "Run collect-run before score-run."
        )
    rubric = parse_rubric(Path(rubric_path))
    anonymous = _anonymous_runner_id(run)
    evidence = parse_simple_yaml(run.root / "evidence" / "evidence.yaml")
    missing = list(evidence.get("missing_fields") or [])
    changed_files = list(evidence.get("changed_files") or [])
    diff_path = run.root / "evidence" / "diff.patch"
    completion_path = run.root / "completion_report.md"
    evidence_status = evidence.get("status", "unknown")

    lines = [
        f"# Scoring Package: {run.id}",
        "",
        "Protocol Version: replay-checker-score-v2",
        "",
        f"Anonymous runner: {anonymous}",
        f"Case: {run.case.id}",
        "",
        "## Required Inputs",
        f"- Evidence YAML: `{run.root / 'evidence' / 'evidence.yaml'}`",
        f"- Diff patch: `{diff_path}`",
        f"- Completion report: `{completion_path}`",
        f"- Case metadata: `{run.case.root / 'case.yaml'}`",
        "",
        "## Rubric Weights",
        f"- Result score weight: {rubric.get('result_weight', '80')}",
        f"- Process score weight: {rubric.get('process_weight', '20')}",
        "",
        "## Evidence Inventory",
        f"- Status: {evidence_status}",
        f"- Changed files ({len(changed_files)}): {', '.join(changed_files) if changed_files else '(none)'}",
        f"- Diff present: {diff_path.exists() and diff_path.stat().st_size > 0}",
        f"- Completion report: {completion_path.exists()}",
    ]

    # Include telemetry section
    telemetry_block = evidence.get("telemetry", None)
    if isinstance(telemetry_block, dict):
        lines.extend(_format_telemetry_scoring(telemetry_block))

    if missing:
        lines.extend([
            "",
            "## Missing Evidence",
            *[f"- {item}" for item in missing],
        ])
    else:
        lines.extend([
            "",
            "## Missing Evidence",
            "- (none)",
        ])

    minimum = rubric.get("minimum_evidence", [])
    if minimum:
        lines.extend([
            "",
            "## Minimum Evidence Check (from rubric)",
            *[f"- {item}: {'found' if _evidence_item_found(item, completion_path, diff_path, evidence_status, changed_files) else 'MISSING'}" for item in minimum],
        ])

    lines.extend([
        "",
        "## Bias Warnings",
        "- Do not infer quality from agent identity, model family, verbosity, or style similarity.",
        "- Score observable result evidence first; award process points only for cited process evidence.",
        "- The runner label is intentionally excluded from this package to prevent identity bias.",
        "- Mark the score invalid if evidence is missing or uncited.",
        "- Compare only evidence packs, not assumptions about the runner's capabilities.",
    ])

    # Case provenance — scorer-facing trust and risk metadata
    provenance = build_case_provenance(
        source_type=run.case.source_type,
        base_source=run.case.base_source,
        base_confidence=run.case.base_confidence,
        candidate_score=run.case.candidate_score,
        is_synthetic=run.case.synthetic_case,
        evidence_sources=run.case.evidence_sources,
        candidate_source_type=run.case.candidate_source_type,
    )
    lines.extend(_render_provenance_section(provenance))

    lines.extend([
        "",
        "## Process Scoring Guidance",
        "- Telemetry is a risk/scope signal, not an automatic quality score.",
        "- Do not penalize agents for low commit counts (commits are not required by default).",
        "- Do not penalize agents for having untracked files unless they contain secrets or forbidden paths.",
        "- Forbidden path touches (e.g. `_reference/`, `.git/`) should cap or warn on scope discipline.",
        "- Suspicious path touches (e.g. `.env`, `credentials`) warrant a note but are not automatic failures.",
        "- Assess rollback/reflog signals in context; iterative refinement is normal development practice.",
    ])

    lines.extend([
        "",
        "## Invalid Score Conditions",
        "- The score cites runner/model identity instead of evidence.",
        "- The score uses uncited assumptions about unavailable files or hidden context.",
        "- Required evidence is missing and the score does not mark the affected category as unverified.",
        "- The scoring agent reads `_reference/` oracle material while evaluating an executing agent's output.",
    ])

    lines.extend([
        "",
        "## Evidence References",
        f"- Diff: `{diff_path}`",
        f"- Evidence YAML: `{run.root / 'evidence' / 'evidence.yaml'}`",
        f"- Completion report: `{completion_path}`",
        f"- Case: `{run.case.root / 'case.yaml'}`",
    ])

    # Evidence gate assessment
    # Skip runner identity gate: run.yaml is internal metadata, not exposed to the scorer.
    gate_eval = evaluate_evidence_gates(
        evidence_root=run.root,
        run_id=run.id,
        runner_label=None,
        changed_files=changed_files if isinstance(changed_files, list) else [],
    )

    display_results = [r for r in gate_eval.results if r.gate != EvidenceGate.RUNNER_IDENTITY_HIDDEN]

    lines.extend([
        "",
        "## Evidence Gate Assessment",
    ])
    for result in display_results:
        status = "PASS" if result.passed else "FAIL"
        detail = f" — {result.detail}" if result.detail else ""
        lines.append(f"- [{status}] {result.gate.value}{detail}")
    lines.append(f"- Overall: {'all gates passed' if gate_eval.all_passed else 'SOME GATES FAILED'}")

    if gate_eval.is_invalid:
        lines.extend([
            "",
            "**Score is INVALID.**",
            *[f"- {reason}" for reason in gate_eval.invalid_reasons],
        ])

    ceilings = compute_score_ceilings(gate_eval)
    has_ceilings = any(v is not None for v in (ceilings.result_ceiling, ceilings.process_ceiling, ceilings.verification_ceiling))
    if has_ceilings or ceilings.overall_invalid:
        lines.extend([
            "",
            "## Score Ceilings",
        ])
        if ceilings.result_ceiling is not None:
            lines.append(f"- Result score ceiling: {ceilings.result_ceiling}")
        if ceilings.process_ceiling is not None:
            lines.append(f"- Process score ceiling: {ceilings.process_ceiling}")
        if ceilings.verification_ceiling is not None:
            lines.append(f"- Verification score ceiling: {ceilings.verification_ceiling}")
        for reason in ceilings.reasons:
            lines.append(f"  - {reason}")

    path = run.root / "scoring_package.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


def _format_telemetry_scoring(telemetry: dict[str, object]) -> list[str]:
    lines = [
        "",
        "## Process Telemetry",
        f"- Changed file count: {telemetry.get('changed_file_count', 0)}",
        f"- Added lines: {telemetry.get('added_lines', 0)}",
        f"- Deleted lines: {telemetry.get('deleted_lines', 0)}",
        f"- Untracked file count: {telemetry.get('untracked_file_count', 0)}",
        f"- Commit count: {telemetry.get('commit_count', 0)}",
        f"- Rollback signal: {telemetry.get('rollback_signal', '') or '(none)'}",
    ]
    forbidden = telemetry.get("forbidden_path_touches", [])
    if isinstance(forbidden, list) and forbidden:
        lines.append(f"- Forbidden path touches: {', '.join(forbidden)}")
    else:
        lines.append("- Forbidden path touches: (none)")

    suspicious = telemetry.get("suspicious_path_touches", [])
    if isinstance(suspicious, list) and suspicious:
        lines.append(f"- Suspicious path touches: {', '.join(suspicious)}")
    else:
        lines.append("- Suspicious path touches: (none)")
    return lines


def _evidence_item_found(
    item: str,
    completion_path: Path,
    diff_path: Path,
    evidence_status: str,
    changed_files: list[str],
) -> bool:
    if item == "completion_report.md":
        return completion_path.exists()
    if item == "diff.patch":
        return diff_path.exists() and diff_path.stat().st_size > 0
    if item == "evidence.yaml":
        return evidence_status != "unknown"
    if item == "changed_files":
        return bool(changed_files)
    return False


def _render_provenance_section(provenance: CaseProvenance) -> list[str]:
    """Render the case provenance section for the scoring package."""
    lines = [
        "",
        "## Case Provenance & Source Risk",
        "",
        "This section describes how the case was constructed and the reliability of its evidence sources.",
        "It is scorer-facing metadata — not execution-agent oracle evidence.",
        "",
        f"- Primary source type: `{provenance.primary_source_type}`",
        f"- Contributing source types: {', '.join(f'`{t}`' for t in provenance.source_types) if provenance.source_types else '(none)'}",
        f"- Base commit source: `{provenance.base_commit_source or 'none'}`",
        f"- Base commit confidence: `{provenance.base_commit_confidence}`",
        f"- Merged confidence: `{provenance.confidence}`",
        f"- Candidate score: {provenance.candidate_score:.3f}",
        f"- Synthetic case: {'yes' if provenance.is_synthetic else 'no'}",
        f"- Overall risk level: `{provenance.risk_level}`",
    ]

    if provenance.risk_reasons:
        lines.append("")
        lines.append("### Risk Factors")
        for reason in provenance.risk_reasons:
            lines.append(f"- {reason}")

    if provenance.conflicts:
        lines.append("")
        lines.append("### Source Conflicts")
        for conflict in provenance.conflicts:
            severity_mark = _conflict_severity_mark(conflict.severity)
            lines.append(f"- [{severity_mark}] `{conflict.conflict_type}`: {conflict.description}")

    if provenance.is_low_confidence:
        lines.extend([
            "",
            "**Note:** This is a low-confidence case. Scores should be interpreted with caution.",
        ])

    return lines


def _conflict_severity_mark(severity: str) -> str:
    if severity == "high":
        return "HIGH"
    if severity == "medium":
        return "MED"
    return "LOW"


def _anonymous_runner_id(run: ReplayRun) -> str:
    run_yaml = run.root / "run.yaml"
    if run_yaml.exists():
        data = parse_simple_yaml(run_yaml)
        stored = str(data.get("anonymous_runner_id", "")).strip()
        if stored:
            return stored
    return _stable_anonymous_runner_id(run.id)


def _stable_anonymous_runner_id(run_id: str) -> str:
    return f"runner-{stable_hash(run_id, length=10)}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_all_cases(
    cases_root: str | Path,
    runs_root: str | Path,
    reports_root: str | Path,
) -> Path:
    """Generate a comprehensive cross-case aggregate report with scoring data.

    Reads evaluation recommendations, evidence telemetry, and gate results
    for every case/run pair, then produces a ranked aggregate overview with
    per-case detail sections and a conclusions block.
    """
    from .evaluation import read_recommendation, read_summary, read_attempts

    cases_dir = Path(cases_root)
    runs_dir = Path(runs_root)
    reports_dir = Path(reports_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Replay Checker — 综合评分报告",
        "",
        "跨 case 汇总，包含评分、证据门控、等级和排名。",
        "",
    ]

    # Collect data for every case that has runs with evidence
    case_rows: list[dict[str, object]] = []
    total_runs = 0
    total_cases_with_evidence = 0
    tiers: dict[str, int] = {}

    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir() and (d / "case.yaml").exists())

    for case_dir in case_dirs:
        case_data = parse_simple_yaml(case_dir / "case.yaml")
        case_id = str(case_data.get("id", case_dir.name))
        case_type = str(case_data.get("source_type", "unknown"))

        run_dirs = sorted(
            d for d in runs_dir.iterdir()
            if d.is_dir() and d.name.startswith(case_id)
        )
        if not run_dirs:
            continue

        runs_with_evidence: list[dict[str, object]] = []
        for run_dir in run_dirs:
            ev_path = run_dir / "evidence" / "evidence.yaml"
            if not ev_path.exists():
                continue
            ev = parse_simple_yaml(ev_path)
            telemetry = ev.get("telemetry", None)
            changed = len(list(ev.get("changed_files") or []))
            added = "-"
            deleted = "-"
            commits = "-"
            if isinstance(telemetry, dict):
                added = str(telemetry.get("added_lines", "-"))
                deleted = str(telemetry.get("deleted_lines", "-"))
                commits = str(telemetry.get("commit_count", "-"))
            run_yaml_path = run_dir / "run.yaml"
            anon = run_dir.name
            label = ""
            if run_yaml_path.exists():
                rd = parse_simple_yaml(run_yaml_path)
                anon = str(rd.get("anonymous_runner_id", run_dir.name))
                label = str(rd.get("runner_label", ""))
            runs_with_evidence.append(dict(
                run_id=run_dir.name, anon=anon, label=label, changed=changed,
                added=added, deleted=deleted, commits=commits,
                status=str(ev.get("status", "unknown")),
            ))

        if not runs_with_evidence:
            continue

        total_cases_with_evidence += 1
        total_runs += len(runs_with_evidence)

        # Read evaluation data
        score = 0.0
        sentence = ""
        tier = "unknown"
        validity = "unvalidated"
        confidence = "low"

        rec = read_recommendation(case_dir)
        if rec.score > 0 or rec.sentence:
            score = rec.score
            sentence = rec.sentence
            validity = rec.validity
            confidence = rec.confidence

        summary = read_summary(case_dir)
        if summary.total_score > 0:
            score = summary.total_score
            tier = summary.tier.value
            validity = summary.validity
            confidence = summary.confidence

        run_ids = [str(r["run_id"]) for r in runs_with_evidence]
        if _all_run_result_evidence_blocked(run_ids, runs_dir):
            score = 0.0
            sentence = (
                "Score is invalid — insufficient evidence: no run has a "
                "non-empty diff and changed-file evidence."
            )
            tier = "invalid"
            validity = "no valid runs with result evidence"
            confidence = "low"

        tiers[tier] = tiers.get(tier, 0) + 1

        case_rows.append(dict(
            case_id=case_id, case_type=case_type,
            score=score, tier=tier, sentence=sentence,
            validity=validity, confidence=confidence,
            runs=runs_with_evidence,
        ))

    # Sort by score descending
    case_rows.sort(key=lambda r: -float(str(r["score"])))

    # ---- Aggregate summary ----
    lines.extend([
        "## 汇总概览",
        "",
        f"- 评估 case 数: {total_cases_with_evidence}",
        f"- 评估 run 总数: {total_runs}",
    ])
    if tiers:
        lines.append(f"- 等级分布: {', '.join(f'{t}: {c}' for t, c in sorted(tiers.items()))}")
    if case_rows:
        avg = sum(float(str(r["score"])) for r in case_rows) / len(case_rows)
        lines.append(f"- 平均分: {avg:.1f} / 100")
        lines.append(f"- 最高分: {float(str(case_rows[0]['score'])):.0f} ({case_rows[0]['case_id']})")
        lines.append(f"- 最低分: {float(str(case_rows[-1]['score'])):.0f} ({case_rows[-1]['case_id']})")
    lines.append("")

    # ---- Ranking table ----
    lines.extend([
        "## 综合排名",
        "",
        "| 排名 | Case | 类型 | 得分 | 等级 | 结论 |",
        "|---|---|---|---|---|---|",
    ])
    for idx, row in enumerate(case_rows):
        tier = str(row["tier"])
        emoji = {"transformative": "++", "excellent": "+", "solved": "=", "partial": "~", "failed": "-", "invalid": "x"}.get(tier, "?")
        lines.append(
            f"| {idx + 1} | {row['case_id']} | {row['case_type']} | "
            f"{float(str(row['score'])):.0f} | {emoji} {tier} | {row['sentence']} |"
        )
    lines.append("")

    # ---- Per-case detail ----
    lines.extend([
        "## 各 Case 详情",
        "",
    ])
    for row in case_rows:
        lines.extend([
            f"### {row['case_id']} — {float(str(row['score'])):.0f}/100 ({row['tier']})",
            "",
            f"- **结论**: {row['sentence']}",
            f"- **有效性**: {row['validity']}",
            f"- **置信度**: {row['confidence']}",
            f"- **来源类型**: {row['case_type']}",
            "",
            "| Run | Agent | 匿名 ID | 状态 | 变更文件 | +行 | -行 | 提交 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for r in row["runs"]:
            label_display = r['label'] if r['label'] else '-'
            lines.append(
                f"| {r['run_id']} | {label_display} | {r['anon']} | {r['status']} | "
                f"{r['changed']} | {r['added']} | {r['deleted']} | {r['commits']} |"
            )
        lines.append("")

    # ---- Conclusions ----
    lines.append("## 评估结论")
    lines.append("")

    if not case_rows:
        lines.append("暂无含证据的 run 可供评估。")
    else:
        scored = [r for r in case_rows if float(str(r["score"])) > 0]
        passing = [r for r in scored if str(r["tier"]) not in ("failed", "invalid")]
        failing = [r for r in case_rows if str(r["tier"]) in ("failed", "invalid")]

        if failing:
            lines.append(f"### 需要重跑: {len(failing)} 个 case 未通过证据门控")
            for r in failing:
                lines.append(f"- `{r['case_id']}`: {r['sentence']} (得分 {float(str(r['score'])):.0f})")
            lines.append("")
            lines.append("这些 case 的 diff 为空或缺失关键证据。重新执行 agent 任务并运行 `collect-run` 后再评分。")
            lines.append("")

        if passing:
            lines.append(f"### 可用: {len(passing)} 个 case 通过评估")
            for r in passing:
                lines.append(f"- `{r['case_id']}`: {float(str(r['score'])):.0f}/100 — {r['sentence']}")
            lines.append("")

        if len(scored) >= 2:
            best = scored[0]
            worst = scored[-1]
            lines.append(f"### 最佳表现: `{best['case_id']}` ({float(str(best['score'])):.0f}/100)")
            lines.append(f"### 最弱表现: `{worst['case_id']}` ({float(str(worst['score'])):.0f}/100)")

    report = reports_dir / "AGGREGATE_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def compare_case(
    case: ReplayCase,
    *,
    runs_root: str | Path,
    reports_root: str | Path,
    detailed: bool = False,
) -> Path:
    root = Path(reports_root)
    root.mkdir(parents=True, exist_ok=True)

    # Collect run IDs for this case
    run_ids: list[str] = []
    for run_dir in sorted(Path(runs_root).glob(f"{case.id}-*")):
        evidence_path = run_dir / "evidence" / "evidence.yaml"
        if evidence_path.exists():
            run_ids.append(run_dir.name)

    # Produce output using evaluation system if available
    output = _build_compare_output(case, run_ids, runs_root, detailed=detailed)

    report = root / f"{case.id}.md"
    report.write_text(f"# Replay Comparison: {case.id}\n\n{output}\n", encoding="utf-8")

    return report


def _build_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_root: str | Path,
    *,
    detailed: bool = False,
) -> str:
    runs_dir = Path(runs_root)

    if not run_ids:
        return "No runs with evidence found for this case."

    if not detailed:
        return _build_default_compare_output(case, run_ids, runs_dir)

    return _build_detailed_compare_output(case, run_ids, runs_dir)


def _build_default_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_dir: Path,
) -> str:
    """Concise default output: score, recommendation sentence, validity, confidence."""
    if _all_run_result_evidence_blocked(run_ids, runs_dir):
        return "\n".join([
            "0 / 100",
            "Score is invalid — insufficient evidence: no run has a non-empty diff and changed-file evidence.",
            "Validity: no valid runs with result evidence; Confidence: low",
        ])

    recommendation = read_recommendation(case.root)

    if _has_recommendation(recommendation):
        return _format_default_recommendation(recommendation)

    return _format_missing_evaluation(run_ids)


def _all_run_result_evidence_blocked(run_ids: list[str], runs_dir: Path) -> bool:
    """Return true when every visible run lacks minimum result evidence."""
    if not run_ids:
        return False

    checked = 0
    for run_id in run_ids:
        run_root = runs_dir / run_id
        evidence_path = run_root / "evidence" / "evidence.yaml"
        if not evidence_path.exists():
            continue
        evidence = parse_simple_yaml(evidence_path)
        changed_files = list(evidence.get("changed_files") or [])
        gate_eval = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id=run_id,
            runner_label=None,
            changed_files=changed_files,
        )
        ceilings = compute_score_ceilings(gate_eval)
        checked += 1
        if ceilings.result_ceiling != 0.0 and changed_files:
            return False

    return checked > 0


def _build_detailed_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_dir: Path,
) -> str:
    """Detailed per-run output with telemetry."""
    lines: list[str] = []

    lines.extend([
        "## Per-Run Details",
        "",
    ])
    for run_id in run_ids:
        run_dir = runs_dir / run_id
        ev_path = run_dir / "evidence" / "evidence.yaml"
        run_yaml_path = run_dir / "run.yaml"
        anonymous = run_id
        if run_yaml_path.exists():
            run_data = parse_simple_yaml(run_yaml_path)
            anonymous = str(run_data.get("anonymous_runner_id", run_id))
        if ev_path.exists():
            ev = parse_simple_yaml(ev_path)
            lines.extend([
                f"### {anonymous}",
                f"- Status: {ev.get('status', 'unknown')}",
                f"- Changed files: {len(list(ev.get('changed_files') or []))}",
            ])
            telemetry = ev.get("telemetry", None)
            if isinstance(telemetry, dict):
                lines.extend([
                    f"- Added lines: {telemetry.get('added_lines', '-')}",
                    f"- Deleted lines: {telemetry.get('deleted_lines', '-')}",
                    f"- Untracked files: {telemetry.get('untracked_file_count', '-')}",
                    f"- Commits: {telemetry.get('commit_count', '-')}",
                ])
                forbidden = telemetry.get("forbidden_path_touches", [])
                if isinstance(forbidden, list) and forbidden:
                    lines.append(f"- WARNING: Forbidden path touches: {', '.join(forbidden)}")
            lines.append("")

    return "\n".join(lines)


def _has_recommendation(recommendation: Recommendation) -> bool:
    return bool(recommendation.sentence.strip()) or recommendation.score > 0.0


def _format_default_recommendation(recommendation: Recommendation) -> str:
    score = f"{recommendation.score:.0f}"
    sentence = recommendation.sentence.strip() or "Evaluation complete; inspect detailed output before making a decision."
    return "\n".join([
        f"{score} / 100",
        sentence,
        f"Validity: {recommendation.validity}; Confidence: {recommendation.confidence}",
    ])


def _format_missing_evaluation(run_ids: list[str]) -> str:
    return "\n".join([
        "No evaluation available for this case.",
        "Run score-run or refresh evaluation state, then compare again.",
        f"Runs with evidence: {len(run_ids)}",
    ])
