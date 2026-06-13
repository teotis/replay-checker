from __future__ import annotations

from pathlib import Path

from .core import Diagnostic
from .replay_types import ReplayCase
from .yaml_lite import parse_simple_yaml

_MAX_EVIDENCE_SOURCES = 10
_MAX_PLAN_LINES = 200


def validate_case(case: ReplayCase) -> list[Diagnostic]:
    """Return missing/incomplete field diagnostics for a case.

    An empty list means the case is complete.
    """
    diagnostics: list[Diagnostic] = []
    if not case.base_source:
        diagnostics.append(Diagnostic("error", "missing_field.base_source", "base_source is empty"))
    if not case.base_confidence or case.base_confidence == "unknown":
        diagnostics.append(Diagnostic("error", "missing_field.base_confidence", "base_confidence is unknown or empty"))
    if not case.base_commit and case.source_type not in ("no_git", "empty_history", "manual"):
        diagnostics.append(Diagnostic("error", "missing_field.base_commit", "base_commit is empty for git-based source"))
    if not case.source_type:
        diagnostics.append(Diagnostic("error", "missing_field.source_type", "source_type is empty"))
    if not case.source_path and case.source_type not in ("git_history", "no_git", "empty_history"):
        diagnostics.append(Diagnostic("error", "missing_field.source_path", "source_path is empty"))
    if not case.selection_reason:
        diagnostics.append(Diagnostic("error", "missing_field.selection_reason", "selection_reason is empty"))
    if not case.plan_path and case.source_type in ("orchestration_kit", "handoff_plan"):
        diagnostics.append(Diagnostic("error", "missing_field.plan_path", "plan_path is required for orchestration_kit/handoff_plan"))
    evidence_md = case.root / "evidence_sources.md"
    if not evidence_md.exists():
        diagnostics.append(Diagnostic("error", "missing_file.evidence_sources", "evidence_sources.md not found"))
    return diagnostics


def _warn_case_quality(case: ReplayCase) -> list[Diagnostic]:
    """Return quality warnings for a case (non-blocking)."""
    diagnostics: list[Diagnostic] = []

    evidence_md = case.root / "evidence_sources.md"
    if evidence_md.exists():
        content = evidence_md.read_text(encoding="utf-8")
        source_count = sum(1 for line in content.splitlines() if line.strip().startswith("-"))
        if source_count > _MAX_EVIDENCE_SOURCES:
            diagnostics.append(Diagnostic(
                "warning", "scope.evidence_sources_count",
                f"evidence_sources 有 {source_count} 项（上限 {_MAX_EVIDENCE_SOURCES}），任务范围可能过大",
            ))

    if case.plan_path and case.source_type == "orchestration_kit":
        plan = Path(case.plan_path)
        if plan.exists():
            line_count = len(plan.read_text(encoding="utf-8").splitlines())
            if line_count > _MAX_PLAN_LINES:
                diagnostics.append(Diagnostic(
                    "warning", "scope.plan_size",
                    f"plan 文件 {plan.name} 有 {line_count} 行（建议 ≤{_MAX_PLAN_LINES}），任务可能过于复杂",
                ))

    return diagnostics


def validate_case_task(case: ReplayCase) -> list[Diagnostic]:
    """Check case task.md content quality for agent executability."""
    task_md = case.root / "task.md"
    if not task_md.exists():
        return [Diagnostic("error", "missing_file.task_md", "missing case task.md")]

    content = task_md.read_text(encoding="utf-8")
    diagnostics: list[Diagnostic] = []

    if "## Goal" not in content:
        diagnostics.append(Diagnostic("error", "content.missing_goal", "case task.md 缺少 ## Goal 段落"))
    else:
        goal_text = content.split("## Goal")[1].split("##")[0].strip()
        if len(goal_text) < 20:
            diagnostics.append(Diagnostic(
                "warning", "content.goal_too_short",
                f"case task.md Goal 过于简略（{len(goal_text)} 字符，建议 ≥20）",
            ))

    if case.source_type == "git_history" and "## Change Summary" not in content:
        diagnostics.append(Diagnostic("error", "content.missing_change_summary", "git_history case 缺少 ## Change Summary"))

    return diagnostics


def case_quality_gate_issues(case: ReplayCase) -> list[Diagnostic]:
    """Return blocking quality diagnostics for batch-extracted cases."""
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(validate_case(case))
    diagnostics.extend(validate_case_task(case))

    if not case.verification_commands:
        diagnostics.append(Diagnostic("error", "missing.verification_commands", "missing_verification_commands"))

    ref_dir = case.root / "_reference"
    diff_path = ref_dir / "diff.patch"
    if not diff_path.exists() or diff_path.stat().st_size == 0:
        diagnostics.append(Diagnostic("error", "missing.reference_diff", "missing_reference_diff"))

    metadata_path = ref_dir / "reference_metadata.yaml"
    metadata = parse_simple_yaml(metadata_path) if metadata_path.exists() else {}
    changed_files = metadata.get("changed_files") or []
    if not isinstance(changed_files, list) or not changed_files:
        diagnostics.append(Diagnostic("error", "missing.reference_changed_files", "missing_reference_changed_files"))

    return diagnostics
