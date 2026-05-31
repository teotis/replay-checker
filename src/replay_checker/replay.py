from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .core import stable_hash
from .candidates import CandidateCase, build_case_candidates, select_case_candidate
from .packages import ExecutionPackageRenderer, compile_execution_package
from .scoring import compute_score_ceilings, evaluate_evidence_gates, parse_rubric
from .sources import EvidenceRecord, EvidenceSourceConfig, discover_evidence_sources

_GIT_TIMEOUT = 30


def _git_run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = _GIT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with a timeout."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _git_output(args: list[str], *, cwd: Path) -> str:
    """Run a git command and return stdout (empty string on failure/timeout)."""
    try:
        result = _git_run(args, cwd=cwd)
        return result.stdout if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""


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
class EvidenceGate:
    name: str
    passed: bool
    detail: str


def parse_simple_yaml(path: str | Path) -> dict[str, object]:
    """Parse the tiny YAML subset this project writes: scalar keys and string lists."""
    target = Path(path)
    data: dict[str, object] = {}
    current_list: str | None = None
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") and current_list:
            data.setdefault(current_list, [])
            assert isinstance(data[current_list], list)
            data[current_list].append(line[4:])
            continue
        current_list = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            data[key] = []
            current_list = key
        else:
            data[key] = value
    return data


def _write_simple_yaml(path: Path, data: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"{key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        for line in md.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "verification" in line.lower() and "command" in line.lower():
                in_verification = True
                continue
            if in_verification:
                stripped = line.strip()
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
    return _git_output(["rev-parse", "HEAD"], cwd=project_path)


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

    commits = []
    for line in output.strip().splitlines():
        if not line.strip():
            continue
        sha, message = line.split(" ", 1)
        msg_lower = message.lower()
        if any(skip in msg_lower for skip in ["format", "style", "lint", "bump", "dependabot", "lock"]):
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


def _write_case_yaml(case: ReplayCase) -> None:
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
    _write_simple_yaml(case.root / "case.yaml", data)


def _discover_case_evidence(
    project: Path,
    *,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None,
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


def validate_case(case: ReplayCase) -> list[str]:
    """Check that a ReplayCase has all required metadata fields.

    Returns a list of missing or invalid fields.
    """
    missing: list[str] = []
    if not case.id:
        missing.append("id")
    if not case.project_path:
        missing.append("project_path")
    if not case.source_type:
        missing.append("source_type")
    if not case.base_source:
        missing.append("base_source")
    if not (case.root / "evidence_sources.md").exists():
        missing.append("evidence_sources.md")
    return missing


def intake(
    *,
    cases_root: str | Path,
    project_path: str | Path,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> ReplayCase:
    """Auto-generate a replay case from a project directory.

    Scans for orchestration kits, handoff plans, or falls back to git history.
    Builds evidence candidates, selects the best one, and writes candidate_report.md.
    """
    project = Path(project_path).resolve()
    cases = Path(cases_root)

    source = _detect_source(project)

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

    is_synthetic = source["source_type"] == "git_history"

    # Discover evidence and build candidates
    evidence_records = _discover_case_evidence(
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
    )

    _write_case_yaml(case)
    _write_evidence_sources(case, evidence_records)

    if case.synthetic_case:
        _write_synthetic_task(case, project, evidence_records)
        _save_reference_evidence(case, project)
    else:
        _write_case_task(case, evidence_records)

    return case


def create_case(
    *,
    cases_root: str | Path,
    project_path: str | Path,
    plan_path: str | Path,
    base_commit: str,
    case_id: str,
    verification_commands: list[str] | tuple[str, ...] | None = None,
) -> ReplayCase:
    root = Path(cases_root) / case_id
    root.mkdir(parents=True, exist_ok=True)
    commands = tuple(verification_commands or ())
    case = ReplayCase(
        id=case_id,
        root=root,
        project_path=Path(project_path).resolve(),
        plan_path=Path(plan_path).resolve(),
        base_commit=base_commit,
        verification_commands=commands,
        base_source="manual",
        base_confidence="user_supplied",
        source_type="manual",
        source_path=str(Path(plan_path).resolve()),
        selection_reason="Manual case created from explicit plan and base commit",
        synthetic_case=False,
        evidence_sources=(),
    )
    _write_case_yaml(case)
    _write_evidence_sources(case, [])
    _write_case_task(case)
    return case


def load_case(cases_root: str | Path, case_id: str) -> ReplayCase:
    root = Path(cases_root) / case_id
    data = parse_simple_yaml(root / "case.yaml")
    commands = tuple(data.get("verification_commands", []))
    return ReplayCase(
        id=str(data["id"]),
        root=root,
        project_path=Path(str(data["project_path"])),
        plan_path=Path(str(data["plan_path"])),
        base_commit=str(data["base_commit"]),
        verification_commands=commands,
        base_source=str(data.get("base_source", "")),
        base_confidence=str(data.get("base_confidence", "unknown")),
        source_type=str(data.get("source_type", "manual")),
        source_path=str(data.get("source_path", "")),
        selection_reason=str(data.get("selection_reason", "")),
        synthetic_case=str(data.get("synthetic_case", "false")).lower() == "true",
        evidence_sources=tuple(data.get("evidence_sources", [])),
        selected_candidate_id=str(data.get("selected_candidate_id", "")),
        candidate_score=float(str(data.get("candidate_score", "0"))),
        candidate_source_type=str(data.get("candidate_source_type", "")),
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
        f"- Base commit: `{case.base_commit}`",
    ]

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
        ])
    else:
        lines.extend([
            f"- Plan package: `{case.plan_path}`",
            f"- Source type: {case.source_type}",
        ])

    if case.verification_commands:
        lines.extend([
            "",
            "## Verification Commands",
            *[f"- `{command}`" for command in case.verification_commands],
        ])

    _append_conversation_evidence(lines, evidence_records)

    (case.root / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _find_synthetic_target(project_path: Path, base_commit: str, base_source: str) -> str | None:
    """Find the target commit that a synthetic case should attempt to reproduce."""
    if base_source == "synthetic_target_parent":
        output = _git_output(
            ["log", "--no-merges", "--format=%H", f"{base_commit}..HEAD", "--"],
            cwd=project_path,
        )
        commits = [c for c in output.strip().splitlines() if c]
        return commits[-1] if commits else None

    if base_source == "synthetic_first_commit":
        return base_commit

    return None


def _save_reference_evidence(case: ReplayCase, project_path: Path) -> None:
    """Persist reference/oracle evidence for synthetic cases outside agent-visible paths."""
    target = _find_synthetic_target(project_path, case.base_commit, case.base_source)
    if not target:
        return

    log_output = _git_output(["log", "--format=%s%n%b", "-1", target], cwd=project_path)
    # Handle root commit (no parent) vs normal commit
    parent_output = _git_output(["rev-parse", f"{target}^"], cwd=project_path)
    if parent_output:
        diff_output = _git_output(["diff", f"{target}^..{target}", "--binary"], cwd=project_path)
    else:
        diff_output = _git_output(
            ["diff", "4b825dc642cb6eb9a060e54bf8994a3e3f7b9b6a", target, "--binary"],
            cwd=project_path,
        )

    changed_output = _git_output(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", target],
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
            "commit_message": log_output.strip(),
            "changed_files": changed_files,
        },
    )


def _write_synthetic_task(
    case: ReplayCase,
    project_path: Path,
    evidence_records: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
) -> None:
    """Write task context for synthetic cases with intent info but not the answer."""
    target = _find_synthetic_target(project_path, case.base_commit, case.base_source)
    commit_message = ""
    if target:
        commit_message = _git_output(["log", "--format=%s", "-1", target], cwd=project_path).strip()

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
        f"- Base commit: `{case.base_commit}`",
        f"- Source type: `synthetic` (generated from git history, not a human-written plan)",
        f"- Selection reason: {case.selection_reason}",
    ]

    if commit_message:
        lines.extend([
            "",
            "## Reconstructed Task",
            "",
            f"The historical commit message was: **{commit_message}**",
            "",
            f"Starting from base commit `{case.base_commit}`, implement the changes described by this commit message.",
            "The original implementation exists as a later commit — produce equivalent changes independently.",
        ])
    else:
        lines.extend([
            "",
            "## Note",
            "This is a synthetic case reconstructed from git history.",
            "Reference evidence (commit diff) is stored outside this task and not shown to the executing agent.",
            "Review the project state at the base commit and identify meaningful improvements.",
        ])

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
    existing = sorted(runs_root.glob(f"{case_id}-*"))
    return f"{case_id}-{len(existing) + 1:03d}"


def prepare_run(case: ReplayCase, *, runs_root: str | Path, runner_label: str) -> ReplayRun:
    missing = validate_case(case)
    if missing:
        raise ValueError(
            f"Case {case.id} is incomplete — missing: {', '.join(missing)}. "
            "Required fields: base_source, base_confidence, source_type, source_path, selection_reason, evidence_sources.md"
        )
    runs = Path(runs_root)
    run_id = _next_run_id(case.id, runs)
    run_root = runs / run_id
    workspace = run_root / "workspace"
    run_root.mkdir(parents=True, exist_ok=False)
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
    command = ["git", "-C", str(case.project_path), "worktree", "add", "--detach", str(workspace), "--", case.base_commit]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=_GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError("git worktree add timed out")
    if completed.returncode == 0:
        return
    if not (case.project_path / ".git").exists():
        shutil.copytree(case.project_path, workspace, ignore=shutil.ignore_patterns(".git"))
        return
    raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git worktree failed")


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
    data = parse_simple_yaml(root / "run.yaml")
    case = load_case(cases_root, str(data["case_id"]))
    return ReplayRun(
        id=str(data["id"]),
        root=root,
        case=case,
        runner_label=str(data.get("runner_label", "")),
        workspace=Path(str(data["workspace"])),
    )


def collect_run(run: ReplayRun) -> Evidence:
    evidence_dir = run.root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    diff = _git_capture(run.workspace, ["diff", "--binary"])
    status = _git_capture(run.workspace, ["status", "--short"])
    changed_files = tuple(_parse_changed_files(status))
    diff_path = evidence_dir / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8")
    (evidence_dir / "status.txt").write_text(status, encoding="utf-8")
    completion = run.root / "completion_report.md"
    run_status = _completion_status(completion)

    missing: list[str] = []
    if run_status == "missing-completion-report":
        missing.append("completion_report.md")
    if not diff.strip():
        missing.append("diff.patch (empty)")
    if not changed_files:
        missing.append("changed_files (none)")

    _write_simple_yaml(
        evidence_dir / "evidence.yaml",
        {
            "run_id": run.id,
            "status": run_status,
            "changed_files": list(changed_files),
            "diff_path": diff_path,
            "missing_fields": missing,
        },
    )
    return Evidence(run.id, run_status, changed_files, diff_path, tuple(missing))


def _git_capture(cwd: Path, args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False, timeout=_GIT_TIMEOUT)
        return completed.stdout if completed.returncode == 0 else completed.stderr
    except subprocess.TimeoutExpired:
        return ""


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


def _completion_status(path: Path) -> str:
    if not path.exists():
        return "missing-completion-report"
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^status:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "reported"


def validate_case(case: ReplayCase) -> list[str]:
    """Return a list of missing/incomplete fields for a case.

    An empty list means the case is complete.
    """
    missing: list[str] = []
    if not case.base_source:
        missing.append("base_source")
    if not case.base_confidence or case.base_confidence == "unknown":
        missing.append("base_confidence")
    if not case.source_type:
        missing.append("source_type")
    if not case.source_path and case.source_type not in ("git_history", "no_git", "empty_history"):
        missing.append("source_path")
    if not case.selection_reason:
        missing.append("selection_reason")
    if not case.evidence_sources and case.source_type != "manual":
        missing.append("evidence_sources")
    evidence_md = case.root / "evidence_sources.md"
    if not evidence_md.exists():
        missing.append("evidence_sources.md")
    return missing


def build_evidence_gates(
    run: ReplayRun,
    *,
    evidence: dict[str, object],
    completion_path: Path,
    diff_path: Path,
    runner_label: str,
) -> list[EvidenceGate]:
    """Build structured evidence validity gates for the scoring package."""
    changed_files = list(evidence.get("changed_files", []))
    evidence_status = str(evidence.get("status", "unknown"))
    missing_fields = list(evidence.get("missing_fields", []))

    diff_ok = diff_path.exists() and diff_path.stat().st_size > 0
    completion_ok = completion_path.exists()
    changed_ok = bool(changed_files)

    # Reference leakage: _reference/ content must not appear in scoring evidence
    ref_leaked = False
    for p in (diff_path, completion_path):
        if p.exists():
            text = p.read_text(encoding="utf-8", errors="ignore")
            if "_reference/" in text or "_reference\\" in text:
                ref_leaked = True
                break

    # Runner identity leakage: the actual runner label must not appear in scoring evidence
    identity_leaked = False
    if runner_label and len(runner_label) > 2:
        for p in (diff_path, completion_path):
            if p.exists():
                text = p.read_text(encoding="utf-8", errors="ignore")
                if runner_label in text:
                    identity_leaked = True
                    break

    return [
        EvidenceGate(
            name="diff_present",
            passed=diff_ok,
            detail="diff exists and is non-empty" if diff_ok else "diff missing or empty",
        ),
        EvidenceGate(
            name="completion_report",
            passed=completion_ok,
            detail="completion report found" if completion_ok else "completion report missing",
        ),
        EvidenceGate(
            name="changed_files",
            passed=changed_ok,
            detail=f"{len(changed_files)} file(s) changed" if changed_ok else "no changed files",
        ),
        EvidenceGate(
            name="reference_leakage",
            passed=not ref_leaked,
            detail="no reference content detected in scoring evidence" if not ref_leaked else "reference content leaked into scoring evidence",
        ),
        EvidenceGate(
            name="runner_identity_leakage",
            passed=not identity_leaked,
            detail="runner identity not exposed in scoring evidence" if not identity_leaked else "runner label found in scoring evidence",
        ),
        EvidenceGate(
            name="verification_present",
            passed=evidence_status not in ("unknown", "missing-completion-report"),
            detail=f"evidence status: {evidence_status}",
        ),
    ]


def _compute_score_ceilings(
    gates: list[EvidenceGate],
    rubric: dict[str, object],
) -> dict[str, int]:
    """Compute score ceilings based on gate results and rubric settings."""
    base_result = int(rubric.get("ceiling_result", 100))
    base_process = int(rubric.get("ceiling_process", 100))
    no_diff_ceiling = int(rubric.get("ceiling_no_diff", 20))
    no_completion_ceiling = int(rubric.get("ceiling_no_completion", 30))
    missing_penalty = int(rubric.get("ceiling_missing_evidence_penalty", 40))

    result_ceiling = base_result
    process_ceiling = base_process

    gate_map = {g.name: g.passed for g in gates}

    if not gate_map.get("diff_present", True):
        result_ceiling = min(result_ceiling, no_diff_ceiling)
    if not gate_map.get("completion_report", True):
        result_ceiling = min(result_ceiling, no_completion_ceiling)
        process_ceiling = min(process_ceiling, no_completion_ceiling)
    if not gate_map.get("changed_files", True):
        result_ceiling = min(result_ceiling, missing_penalty)

    # Any gate failure reduces both ceilings
    failed_count = sum(1 for g in gates if not g.passed)
    if failed_count:
        penalty = missing_penalty * failed_count
        result_ceiling = max(0, result_ceiling - penalty)
        process_ceiling = max(0, process_ceiling - penalty)

    return {"result_ceiling": result_ceiling, "process_ceiling": process_ceiling}


def score_run(run: ReplayRun, *, rubric_path: str | Path) -> Path:
    rubric = parse_rubric(Path(rubric_path))
    anonymous = _anonymous_runner_id(run)
    evidence = parse_simple_yaml(run.root / "evidence" / "evidence.yaml")
    missing = evidence.get("missing_fields", [])
    changed_files = evidence.get("changed_files", [])
    diff_path = run.root / "evidence" / "diff.patch"
    completion_path = run.root / "completion_report.md"
    evidence_status = evidence.get("status", "unknown")

    gates = build_evidence_gates(
        run,
        evidence=evidence,
        completion_path=completion_path,
        diff_path=diff_path,
        runner_label=run.runner_label,
    )
    ceilings = _compute_score_ceilings(gates, rubric)

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
        "## Evidence Validity Gates",
        *[f"- [{('PASS' if g.passed else 'FAIL')}] {g.name}: {g.detail}" for g in gates],
        "",
        "## Score Ceilings",
        f"- Result ceiling: {ceilings['result_ceiling']}",
        f"- Process ceiling: {ceilings['process_ceiling']}",
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

    from .scoring import EvidenceGate
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


def compare_case(case: ReplayCase, *, runs_root: str | Path, reports_root: str | Path) -> Path:
    root = Path(reports_root)
    root.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Replay Comparison: {case.id}",
        "",
        "| Run | Status | Changed Files |",
        "|---|---|---|",
    ]
    for run_dir in sorted(Path(runs_root).glob(f"{case.id}-*")):
        evidence_path = run_dir / "evidence" / "evidence.yaml"
        if not evidence_path.exists():
            continue
        evidence = parse_simple_yaml(evidence_path)
        changed = ", ".join(evidence.get("changed_files", []))
        lines.append(f"| {run_dir.name} | {evidence.get('status', 'unknown')} | {changed} |")
    report = root / f"{case.id}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
