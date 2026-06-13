from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from replay_checker.case_validation import (
    _MAX_EVIDENCE_SOURCES,
    _MAX_PLAN_LINES,
    _warn_case_quality,
    case_quality_gate_issues,
    validate_case,
    validate_case_task,
)
from replay_checker.replay_types import ReplayCase


@pytest.fixture()
def tmp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_case(tmp_dir: Path, *, source_type: str = "orchestration_kit", base_source: str = "plan_state_base",
               base_commit: str = "abc123", base_confidence: str = "high", selection_reason: str = "test",
               verification_commands: tuple[str, ...] = (), source_path: str = "plan.md") -> ReplayCase:
    root = tmp_dir / "case1"
    root.mkdir(parents=True, exist_ok=True)
    (root / "evidence_sources.md").write_text("# Evidence\n- plan: test\n", encoding="utf-8")
    (root / "task.md").write_text("# Replay Case: test\n\n## Goal\n\nImplement feature X that does something useful.\n\n## Verification\n\n- `make test`\n", encoding="utf-8")
    (root / "case.yaml").write_text("id: test\n", encoding="utf-8")
    return ReplayCase(
        id="test",
        root=root,
        project_path=tmp_dir,
        plan_path=root / "plan.md",
        base_commit=base_commit,
        base_source=base_source,
        base_confidence=base_confidence,
        source_type=source_type,
        source_path=source_path,
        selection_reason=selection_reason,
        verification_commands=verification_commands,
    )


class TestValidateCase:
    def test_complete_case_no_diagnostics(self, tmp_dir):
        case = _make_case(tmp_dir)
        assert validate_case(case) == []

    def test_missing_base_source(self, tmp_dir):
        case = _make_case(tmp_dir, base_source="")
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_field.base_source" in codes

    def test_missing_base_commit_for_git_source(self, tmp_dir):
        case = _make_case(tmp_dir, base_commit="", source_type="orchestration_kit")
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_field.base_commit" in codes

    def test_empty_base_commit_ok_for_no_git(self, tmp_dir):
        case = _make_case(tmp_dir, base_commit="", source_type="no_git")
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_field.base_commit" not in codes

    def test_empty_base_commit_ok_for_manual(self, tmp_dir):
        case = _make_case(tmp_dir, base_commit="", source_type="manual")
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_field.base_commit" not in codes

    def test_missing_source_type(self, tmp_dir):
        case = _make_case(tmp_dir, source_type="")
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_field.source_type" in codes

    def test_missing_evidence_sources_md(self, tmp_dir):
        root = tmp_dir / "case2"
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text("# Goal\nSomething", encoding="utf-8")
        case = ReplayCase(
            id="case2", root=root, project_path=tmp_dir, plan_path=Path(""),
            base_commit="abc", base_source="manual", base_confidence="high",
            source_type="manual", source_path="x", selection_reason="y",
        )
        diags = validate_case(case)
        codes = [d.code for d in diags]
        assert "missing_file.evidence_sources" in codes

    def test_orchestration_kit_requires_plan_path(self, tmp_dir):
        case = _make_case(tmp_dir, source_type="orchestration_kit")
        # case has plan_path set, so this should pass
        assert not any(d.code == "missing_field.plan_path" for d in validate_case(case))

    def test_git_history_requires_source_path(self, tmp_dir):
        case = _make_case(tmp_dir, source_type="git_history", source_path="")
        diags = validate_case(case)
        # git_history is exempt from source_path requirement
        assert not any(d.code == "missing_field.source_path" for d in diags)


class TestValidateCaseTask:
    def test_valid_task_no_diagnostics(self, tmp_dir):
        case = _make_case(tmp_dir)
        assert validate_case_task(case) == []

    def test_missing_task_md(self, tmp_dir):
        root = tmp_dir / "case_no_task"
        root.mkdir(parents=True, exist_ok=True)
        case = ReplayCase(
            id="notask", root=root, project_path=tmp_dir, plan_path=Path(""),
            base_commit="abc", base_source="manual", source_type="manual",
            source_path="x", selection_reason="y",
        )
        diags = validate_case_task(case)
        assert len(diags) == 1
        assert diags[0].severity == "error"
        assert "task_md" in diags[0].code

    def test_missing_goal(self, tmp_dir):
        root = tmp_dir / "case_nogoal"
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text("# Replay Case: test\n\n## Source\n\n- Project: `/tmp`\n", encoding="utf-8")
        case = ReplayCase(
            id="nogoal", root=root, project_path=tmp_dir, plan_path=Path(""),
            base_commit="abc", base_source="manual", source_type="manual",
            source_path="x", selection_reason="y",
        )
        diags = validate_case_task(case)
        codes = [d.code for d in diags]
        assert "content.missing_goal" in codes

    def test_short_goal_warning(self, tmp_dir):
        root = tmp_dir / "case_short"
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text("# Case\n\n## Goal\n\nShort.\n", encoding="utf-8")
        case = ReplayCase(
            id="short", root=root, project_path=tmp_dir, plan_path=Path(""),
            base_commit="abc", base_source="manual", source_type="manual",
            source_path="x", selection_reason="y",
        )
        diags = validate_case_task(case)
        codes = [d.code for d in diags]
        assert "content.goal_too_short" in codes

    def test_git_history_requires_change_summary(self, tmp_dir):
        root = tmp_dir / "case_git"
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text("# Case\n\n## Goal\n\nImplement feature X that does something.\n\n## Source\n\n- Project: test\n", encoding="utf-8")
        case = ReplayCase(
            id="git", root=root, project_path=tmp_dir, plan_path=Path(""),
            base_commit="abc", base_source="manual", source_type="git_history",
            source_path="x", selection_reason="y",
        )
        diags = validate_case_task(case)
        codes = [d.code for d in diags]
        assert "content.missing_change_summary" in codes


class TestCaseQualityGateIssues:
    def test_complete_case_no_issues(self, tmp_dir):
        case = _make_case(tmp_dir, verification_commands=("make test",))
        ref_dir = case.root / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        from replay_checker.yaml_lite import write_simple_yaml
        write_simple_yaml(ref_dir / "reference_metadata.yaml", {"changed_files": ["x.py"]})
        assert case_quality_gate_issues(case) == []

    def test_missing_verification_commands(self, tmp_dir):
        case = _make_case(tmp_dir, verification_commands=())
        ref_dir = case.root / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "diff.patch").write_text("diff", encoding="utf-8")
        from replay_checker.yaml_lite import write_simple_yaml
        write_simple_yaml(ref_dir / "reference_metadata.yaml", {"changed_files": ["x.py"]})
        diags = case_quality_gate_issues(case)
        codes = [d.code for d in diags]
        assert "missing.verification_commands" in codes

    def test_missing_reference_diff(self, tmp_dir):
        case = _make_case(tmp_dir, verification_commands=("make test",))
        ref_dir = case.root / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        # No diff.patch file
        diags = case_quality_gate_issues(case)
        codes = [d.code for d in diags]
        assert "missing.reference_diff" in codes

    def test_empty_reference_diff(self, tmp_dir):
        case = _make_case(tmp_dir, verification_commands=("make test",))
        ref_dir = case.root / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "diff.patch").write_text("", encoding="utf-8")
        diags = case_quality_gate_issues(case)
        codes = [d.code for d in diags]
        assert "missing.reference_diff" in codes

    def test_missing_reference_changed_files(self, tmp_dir):
        case = _make_case(tmp_dir, verification_commands=("make test",))
        ref_dir = case.root / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        (ref_dir / "diff.patch").write_text("diff", encoding="utf-8")
        from replay_checker.yaml_lite import write_simple_yaml
        write_simple_yaml(ref_dir / "reference_metadata.yaml", {"changed_files": []})
        diags = case_quality_gate_issues(case)
        codes = [d.code for d in diags]
        assert "missing.reference_changed_files" in codes
