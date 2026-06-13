from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from .candidates import CandidateCase, build_case_candidates, select_case_candidate
from .case_depth import (
    SituationProfile,
    build_situation_profile,
    situation_profile_to_case_yaml,
)
from .case_dedupe import build_case_fingerprint, find_duplicates, scan_all_fingerprints
from .case_paths import case_inventory_root, resolve_case_dir
from .case_reconstruction import (
    _build_reconstruction_context,
    _append_conversation_evidence,
    _extract_goal_section,
    _save_reference_evidence,
    _write_case_task,
    _write_synthetic_task,
)
from .case_validation import case_quality_gate_issues, validate_case, validate_case_task
from .core import stable_hash, sanitize_slug
from .git_utils import git_output
from .outcome_feedback import build_candidate_selection_feedback
from .replay_types import IntakeConfig, PlanPackage, ReplayCase
from .sources import EvidenceRecord, EvidenceSourceConfig, discover_evidence_sources
from .yaml_lite import parse_simple_yaml, write_simple_yaml

_GIT_TIMEOUT = 30
_MAX_EVIDENCE_SOURCES = 10
_MAX_CANDIDATE_REPORT_ENTRIES = 50


def _git_output(args: list[str], *, cwd: Path) -> str:
    return git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


def _git_head(project_path: Path) -> str:
    return _git_output(["rev-parse", "HEAD"], cwd=project_path).strip()


def _first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.parent.name


def _write_simple_yaml(path: Path, data: dict[str, object], *, indent: int = 0) -> None:
    write_simple_yaml(path, data, indent=indent)


_COMMAND_PREFIXES = (
    "rtk ", "python ", "python3 ", "pytest", "make", "bash ", "sh ",
    "git ", "npm ", "pnpm ", "yarn ", "uv ", "cargo ", "go ", "gradle ",
    "./gradlew", "ruby ", "bundle ", "ls ", "cat ", "echo ",
)


def _looks_like_command(line: str) -> bool:
    return any(line.startswith(p) for p in _COMMAND_PREFIXES)


def _extract_verification(project_path: Path, plan_dir: Path) -> list[str]:
    """Extract verification commands from plan markdown files."""
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
                    if cmd.startswith("`"):
                        end = cmd.rfind("`", 1)
                        cmd = cmd[1:end] if end > 0 else cmd[1:]
                    if cmd and _looks_like_command(cmd):
                        commands.append(cmd)
                elif stripped and not stripped.startswith("-"):
                    break
    return commands


def _resolve_commit(project_path: Path, sha: str) -> str:
    if not sha:
        return ""
    try:
        output = _git_output(["rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=project_path)
    except subprocess.TimeoutExpired:
        return ""
    return output.strip()


def _commit_exists(project_path: Path, sha: str) -> bool:
    return bool(_resolve_commit(project_path, sha))


def _git_recent_activity(project_path: Path, plan_dir: Path) -> bool:
    """Check if plan files have been modified in the last 30 days."""
    output = _git_output(
        ["log", "--oneline", "--since=30.days", "--", str(plan_dir)],
        cwd=project_path,
    )
    return bool(output.strip())


def _score_plan_candidate(candidate_dir: Path, project_path: Path) -> tuple[int, str]:
    """Score a plan candidate directory."""
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


def _detect_all_sources(project_path: Path) -> list[tuple[int, str, Path, str]]:
    """Return all plan candidates sorted by score descending."""
    plans_dir = project_path / "docs" / "plans"
    if not plans_dir.exists():
        return []

    candidates: list[tuple[int, str, Path, str]] = []

    for index in sorted(plans_dir.glob("**/INDEX.md")):
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
    """Detect the best task source in a project."""
    candidates = _detect_all_sources(project_path)

    if candidates:
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

    return {
        "source_type": "git_history",
        "source_path": "",
        "plan_path": "",
        "title": "",
        "selection_reason": "No plan package found, will generate synthetic case from git history",
        "verification_commands": [],
    }


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


def _infer_base_commit(project_path: Path, source: dict[str, object]) -> tuple[str, str, str]:
    """Infer the base commit for a case."""
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

    status_output = _git_output(["status", "--porcelain"], cwd=project_path)
    if status_output.strip():
        head = _git_head(project_path)
        return (head, "head_dirty", "low")

    source_type = source["source_type"]

    log_output = _git_output(["log", "--oneline", "-1"], cwd=project_path)
    if not log_output.strip():
        return ("", "empty_history", "low")

    if source_type in ("orchestration_kit", "handoff_plan"):
        return _base_from_plan(project_path, source)

    if source_type == "git_history":
        return _base_from_synthetic(project_path)

    head = _git_head(project_path)
    return (head, "head_fallback", "low")


def _base_from_plan(project_path: Path, source: dict[str, object]) -> tuple[str, str, str]:
    """Find the base commit for a plan-based case."""
    plan_path = source.get("source_path", "")
    if not plan_path:
        head = _git_head(project_path)
        return (head, "head_fallback", "low")

    output = _git_output(
        ["log", "--diff-filter=A", "--format=%H", "--", plan_path],
        cwd=project_path,
    )
    commits = [c for c in output.strip().splitlines() if c]
    if not commits:
        head = _git_head(project_path)
        return (head, "head_fallback", "low")

    first_commit = commits[-1]

    parent_output = _git_output(["rev-parse", f"{first_commit}^"], cwd=project_path)
    if parent_output:
        return (parent_output.strip(), "plan_first_commit_parent", "high")

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

    target_sha, _ = commits[0]
    parent_output = _git_output(["rev-parse", f"{target_sha}^"], cwd=project_path)
    if parent_output:
        return (parent_output.strip(), "synthetic_target_parent", "high")

    return (target_sha, "synthetic_first_commit", "medium")


def _generate_case_id(project: Path, source: dict[str, object]) -> str:
    """Generate a deterministic case ID from project name and source."""
    project_name = sanitize_slug(project.name)
    source_hash = stable_hash(str(source.get("source_path", "")) + str(source.get("title", "")))
    return f"{project_name}-{source_hash}"


def _write_case_yaml(
    case: ReplayCase,
    *,
    reconstruction_risk: str = "",
    reconstruction_confidence: str = "",
    situation_profile: SituationProfile | None = None,
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
    if situation_profile is not None:
        data.update(situation_profile_to_case_yaml(situation_profile))
    elif case.depth_score > 0:
        data["depth_score"] = str(case.depth_score)
        data["depth_level"] = case.depth_level
        data["episode_label"] = case.episode_label
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
    situation_profile: SituationProfile | None = None,
) -> tuple[ReplayCase, str]:
    """Check for duplicates after case creation and rewrite case.yaml."""
    fp = build_case_fingerprint(case.root)
    existing = scan_all_fingerprints(cases_root)
    existing_filtered = {k: v for k, v in existing.items() if k != case.id}
    existing_dirs: dict[str, Path] = {
        cid: cases_root / cid for cid in existing if cid != case.id
    }

    if not existing_filtered:
        _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                         reconstruction_confidence=reconstruction_confidence,
                         situation_profile=situation_profile,
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
                         situation_profile=situation_profile,
                         case_fingerprint=fp.fingerprint, duplicate_status="exact_duplicate",
                         duplicate_of=best.case_id, duplicate_reasons=best.reasons)
        return case, "exact_duplicate"

    if likely_matches:
        best = likely_matches[0]
        _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                         reconstruction_confidence=reconstruction_confidence,
                         situation_profile=situation_profile,
                         case_fingerprint=fp.fingerprint, duplicate_status="likely_duplicate",
                         duplicate_of=best.case_id, duplicate_reasons=best.reasons)
        return case, "likely_duplicate"

    _write_case_yaml(case, reconstruction_risk=reconstruction_risk,
                     reconstruction_confidence=reconstruction_confidence,
                     situation_profile=situation_profile,
                     case_fingerprint=fp.fingerprint, duplicate_status="unique")
    return case, "unique"


def _discover_case_evidence(
    project: Path,
    *,
    codex_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    claude_history_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    scan_warnings: list[str] | None = None,
) -> list[EvidenceRecord]:
    if codex_history_roots is None and claude_history_roots is None:
        return discover_evidence_sources(project, warnings=scan_warnings)
    return discover_evidence_sources(
        project,
        EvidenceSourceConfig(
            codex_history_roots=tuple(Path(path) for path in (codex_history_roots or ())),
            claude_history_roots=tuple(Path(path) for path in (claude_history_roots or ())),
        ),
        warnings=scan_warnings,
    )


def _evidence_record_label(record: EvidenceRecord) -> str:
    return f"{record.source_type}: {record.summary}"


def _write_evidence_sources(case: ReplayCase, records: list[EvidenceRecord], *, scan_warnings: list[str] | None = None) -> None:
    lines = [
        f"# Evidence Sources: {case.id}",
        "",
        "These are bounded summaries. Raw history logs are not copied into the case.",
        "",
    ]
    if not records:
        lines.append("- none")
    for record in records[:_MAX_EVIDENCE_SOURCES]:
        lines.append(f"- `{record.source_type}` confidence={record.confidence}: {record.summary}")
        lines.append(f"  - path: `{record.path}`")
    remaining = len(records) - _MAX_EVIDENCE_SOURCES
    if remaining > 0:
        lines.append(f"- ... {remaining} additional source(s) omitted (cap: {_MAX_EVIDENCE_SOURCES})")
    if scan_warnings:
        lines.extend(["", "## Scan Warnings", ""])
        for w in scan_warnings:
            lines.append(f"- {w}")
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

    shown_candidates = candidates[:_MAX_CANDIDATE_REPORT_ENTRIES]
    if len(candidates) > len(shown_candidates):
        lines.extend([
            "",
            f"Showing top {len(shown_candidates)} candidates.",
        ])

    lines.extend(["", "## Evidence Summary", ""])

    for idx, candidate in enumerate(shown_candidates):
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

    omitted = len(candidates) - len(shown_candidates)
    if omitted > 0:
        lines.extend([
            f"{omitted} additional candidate(s) omitted from the verbose report.",
            "",
        ])

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


def discover_plan_packages(project_path: str | Path) -> list[PlanPackage]:
    root = Path(project_path)
    plans = []
    for index in sorted((root / "docs" / "plans").glob("**/INDEX.md")):
        package_dir = index.parent / "packages"
        package_count = len(list(package_dir.glob("*.md"))) if package_dir.exists() else 0
        plans.append(PlanPackage(index, _first_heading(index), package_count))
    return plans


def intake(
    *,
    config: IntakeConfig,
    _force_source: dict[str, object] | None = None,
    _cached_evidence: list[EvidenceRecord] | None = None,
    _scan_warnings: list[str] | None = None,
) -> ReplayCase:
    """Auto-generate a replay case from a project directory."""
    project = config.project_path.resolve()
    cases = Path(config.cases_root)

    scan_warnings: list[str] = list(_scan_warnings) if _scan_warnings is not None else []
    evidence_records = _cached_evidence if _cached_evidence is not None else _discover_case_evidence(
        project,
        codex_history_roots=config.codex_history_roots or None,
        claude_history_roots=config.claude_history_roots or None,
        scan_warnings=scan_warnings,
    )
    candidate_feedback = build_candidate_selection_feedback(
        _project_case_roots_for_feedback(cases, project)
    )
    candidates = build_case_candidates(
        evidence_records,
        project_path=project,
        candidate_feedback=candidate_feedback,
    )
    selected_candidate = select_case_candidate(candidates)

    if _force_source is not None:
        source = _force_source
    else:
        source = _source_from_candidate(project, selected_candidate) or _detect_source(project)

    is_git = (project / ".git").exists()
    if not is_git:
        source = {**source, "source_type": "no_git", "selection_reason": "No .git directory found"}

    base_commit, base_source, base_confidence = _infer_base_commit(project, source)
    if base_source == "empty_history" and source["source_type"] == "git_history":
        source = {**source, "source_type": "empty_history", "selection_reason": "Git repository has no commits"}

    case_id = _generate_case_id(project, source)
    case_dir = case_inventory_root(cases, project) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    _SYNTHETIC_BASE_SOURCES = {"synthetic_target_parent", "synthetic_first_commit"}
    is_synthetic = (
        source["source_type"] == "git_history"
        and base_source in _SYNTHETIC_BASE_SOURCES
    )

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
        evidence_sources=tuple(_evidence_record_label(record) for record in evidence_records[:_MAX_EVIDENCE_SOURCES]),
        selected_candidate_id=selected_candidate.candidate_id if selected_candidate else "",
        candidate_score=selected_candidate.relevance_score if selected_candidate else 0.0,
        candidate_source_type=selected_candidate.source_type if selected_candidate else "",
        reference_target=str(source.get("reference_target", "")),
    )

    reconstruction_risk = ""
    reconstruction_confidence = ""
    recon_ctx: dict[str, object] = {}
    if is_synthetic:
        recon_ctx = _build_reconstruction_context(case, project, list(evidence_records))
        risk_notes = recon_ctx.get("risk_notes", [])
        if isinstance(risk_notes, list) and risk_notes:
            reconstruction_risk = "; ".join(risk_notes)
            if any("Low confidence" in r or "No verification hints" in r or "Commit message only" in r for r in risk_notes):
                reconstruction_confidence = "low"
            else:
                reconstruction_confidence = "medium"
        else:
            reconstruction_confidence = "high"
    else:
        plan_goal = _extract_goal_section(case.plan_path) if case.plan_path else ""
        recon_ctx = {
            "plan_goal": plan_goal,
            "verification_commands": list(case.verification_commands),
            "risk_notes": list(selected_candidate.risks) if selected_candidate else [],
        }

    situation_profile = build_situation_profile(
        list(evidence_records),
        selected_candidate=selected_candidate,
        reconstruction_context=recon_ctx,
    )
    case = replace(
        case,
        depth_score=situation_profile.depth_score,
        depth_level=situation_profile.depth_level,
        episode_label=situation_profile.episode_label,
    )

    _write_case_yaml(
        case,
        reconstruction_risk=reconstruction_risk,
        reconstruction_confidence=reconstruction_confidence,
        situation_profile=situation_profile,
    )
    _write_evidence_sources(case, evidence_records, scan_warnings=scan_warnings)

    _save_reference_evidence(case, project)

    if case.synthetic_case:
        _write_synthetic_task(case, project, evidence_records, situation_profile)
    else:
        _write_case_task(case, evidence_records, situation_profile)

    case, _dup_status = _apply_duplicate_check(
        case,
        cases,
        allow_duplicate=config.allow_duplicate,
        reconstruction_risk=reconstruction_risk,
        reconstruction_confidence=reconstruction_confidence,
        situation_profile=situation_profile,
    )

    return case


def _project_case_roots_for_feedback(cases_root: Path, project: Path) -> tuple[Path, ...]:
    project_root = case_inventory_root(cases_root, project)
    if not project_root.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(project_root.iterdir())
        if path.is_dir() and (path / "case.yaml").is_file()
    )


def _source_from_candidate(
    project: Path,
    candidate: CandidateCase | None,
) -> dict[str, object] | None:
    if candidate is None:
        return None

    source_path = candidate.primary_source
    plan_path = ""
    verification_commands: list[str] = []
    title = candidate.candidate_id
    if candidate.source_type == "plan":
        resolved = _resolve_candidate_path(project, source_path)
        source_type, resolved, is_package_source = _resolve_plan_candidate_source(project, resolved)
        if is_package_source:
            source_path = str(resolved)
        plan_path = str(resolved)
        if resolved.exists():
            title = _first_heading(resolved)
            verification_commands = _extract_verification(project, resolved.parent)
    else:
        source_type = candidate.source_type

    return {
        "source_type": source_type,
        "source_path": source_path,
        "plan_path": plan_path,
        "title": title,
        "selection_reason": (
            f"Selected candidate {candidate.candidate_id} "
            f"score {candidate.relevance_score:.3f}"
        ),
        "verification_commands": verification_commands,
    }


def _resolve_candidate_path(project: Path, source_path: str) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    return project / path


def _resolve_plan_candidate_source(project: Path, path: Path) -> tuple[str, Path, bool]:
    for parent in (path.parent, *path.parents):
        if parent == project.parent:
            break
        index = parent / "INDEX.md"
        if index.is_file() and (
            (parent / "packages").is_dir()
            or (parent / "status").is_dir()
            or (parent / "launchers").is_dir()
        ):
            return ("orchestration_kit", index, True)
        if parent == project:
            break
    return ("plan", path, False)


def batch_intake(
    *,
    config: IntakeConfig,
    min_score: int = 3,
    max_cases: int = 50,
) -> list[ReplayCase]:
    """Extract cases from all orchestration kits in a project."""
    project = config.project_path.resolve()
    all_sources = _detect_all_sources(project)

    if not all_sources:
        return []

    batch_scan_warnings: list[str] = []
    cached_evidence = _discover_case_evidence(
        project,
        codex_history_roots=config.codex_history_roots or None,
        claude_history_roots=config.claude_history_roots or None,
        scan_warnings=batch_scan_warnings,
    )

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
                config=config,
                _force_source=src,
                _cached_evidence=cached_evidence,
                _scan_warnings=batch_scan_warnings,
            )
            if case.id in seen_case_ids:
                continue
            quality_issues = case_quality_gate_issues(case)
            if quality_issues:
                if case.id == expected_case_id and case.root.exists():
                    shutil.rmtree(case.root, ignore_errors=True)
                continue
            created.append(case)
            seen_case_ids.add(case.id)
        except Exception as exc:
            print(
                f"warning: batch intake skipped {src.get('source_path', '<unknown>')}: {exc}",
                file=sys.stderr,
            )
            continue

    return created


def create_case(
    *,
    config: IntakeConfig,
    plan_path: str | Path,
    base_commit: str,
    case_id: str,
    verification_commands: list[str] | tuple[str, ...] | None = None,
    _scan_warnings: list[str] | None = None,
) -> ReplayCase:
    project = config.project_path.resolve()
    root = case_inventory_root(config.cases_root, project) / case_id
    root.mkdir(parents=True, exist_ok=True)
    raw_plan = Path(plan_path)
    plan = raw_plan.resolve() if raw_plan.is_absolute() else (project / raw_plan).resolve()
    cases = Path(config.cases_root)

    commands = tuple(verification_commands or ())
    if not commands and plan.exists():
        commands = tuple(_extract_verification(project, plan.parent))

    scan_warnings: list[str] = list(_scan_warnings) if _scan_warnings is not None else []
    evidence_records = _discover_case_evidence(project, scan_warnings=scan_warnings)
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
    _write_evidence_sources(case, evidence_records, scan_warnings=scan_warnings)
    _write_case_task(case, evidence_records)
    _save_reference_evidence(case, project)

    case, _dup_status = _apply_duplicate_check(
        case,
        cases,
        allow_duplicate=config.allow_duplicate,
    )

    return case


def load_case(cases_root: str | Path, case_id: str) -> ReplayCase:
    root = resolve_case_dir(cases_root, case_id)
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
        depth_score=int(str(data.get("depth_score", "0") or "0")),
        depth_level=str(data.get("depth_level", "")),
        episode_label=str(data.get("episode_label", "")),
    )
