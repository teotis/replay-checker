from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from replay_checker.candidates import CandidateCase
from replay_checker.case_intake import (
    _detect_all_sources,
    _detect_source,
    _generate_case_id,
    _git_head,
    _looks_like_command,
    _read_plan_state_bounds,
    _resolve_commit,
    _write_candidate_report,
    _write_evidence_sources,
    _write_case_yaml,
    _infer_base_commit,
    discover_plan_packages,
    intake,
    load_case,
)
from replay_checker.case_paths import case_inventory_root
from replay_checker.evaluation import TaskOutcomeEntry, append_task_outcome
from replay_checker.replay_types import IntakeConfig, PlanPackage, ReplayCase
from replay_checker.sources import EvidenceRecord
from replay_checker.yaml_lite import parse_simple_yaml, write_simple_yaml


@pytest.fixture()
def git_project(tmp_path):
    """Create a minimal git project with a plan file."""
    import subprocess
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), capture_output=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(tmp_path), capture_output=True)
    return tmp_path


@pytest.fixture()
def project_with_plans(git_project):
    """Extend git_project with a plan directory (committed)."""
    import subprocess
    plans_dir = git_project / "docs" / "plans" / "my-plan"
    plans_dir.mkdir(parents=True)
    index = plans_dir / "INDEX.md"
    index.write_text("# My Plan\n\n## Goal\n\nFix the bug\n\n## Verification Commands\n\n- `make test`\n", encoding="utf-8")
    pkg_dir = plans_dir / "packages"
    pkg_dir.mkdir()
    (pkg_dir / "01-step.md").write_text("# Step 1\n\nDo stuff\n", encoding="utf-8")
    status_dir = plans_dir / "status"
    status_dir.mkdir()
    subprocess.run(["git", "add", "."], cwd=str(git_project), capture_output=True)
    subprocess.run(["git", "commit", "-m", "add orchestration kit"], cwd=str(git_project), capture_output=True)
    return git_project


class TestLooksLikeCommand:
    def test_python_command(self):
        assert _looks_like_command("python3 -m pytest") is True

    def test_make_command(self):
        assert _looks_like_command("make test") is True

    def test_prose(self):
        assert _looks_like_command("This is a description of something.") is False

    def test_rtk_prefix(self):
        assert _looks_like_command("rtk make test") is True

    def test_pytest(self):
        assert _looks_like_command("pytest tests/ -q") is True


class TestGenerateCaseId:
    def test_deterministic(self, tmp_path):
        source = {"source_path": "/some/path", "title": "My Plan"}
        id1 = _generate_case_id(tmp_path, source)
        id2 = _generate_case_id(tmp_path, source)
        assert id1 == id2

    def test_different_sources(self, tmp_path):
        id1 = _generate_case_id(tmp_path, {"source_path": "/a", "title": "A"})
        id2 = _generate_case_id(tmp_path, {"source_path": "/b", "title": "B"})
        assert id1 != id2


class TestDiscoverPlanPackages:
    def test_discovers_plans(self, project_with_plans):
        packages = discover_plan_packages(project_with_plans)
        assert len(packages) >= 1
        assert any(p.package_count > 0 for p in packages)

    def test_empty_project(self, tmp_path):
        (tmp_path / "src").mkdir()
        packages = discover_plan_packages(tmp_path)
        assert packages == []


class TestDetectSource:
    def test_git_history_fallback(self, git_project):
        source = _detect_source(git_project)
        # Either finds a plan or falls back to git_history
        assert source["source_type"] in ("git_history", "orchestration_kit", "handoff_plan")

    def test_plan_detection(self, project_with_plans):
        source = _detect_source(project_with_plans)
        # Should find the plan
        assert "source_type" in source


class TestDetectAllSources:
    def test_with_plans(self, project_with_plans):
        sources = _detect_all_sources(project_with_plans)
        assert len(sources) >= 1
        score, reasons, path, source_type = sources[0]
        assert score > 0

    def test_without_plans(self, git_project):
        sources = _detect_all_sources(git_project)
        assert sources == []


class TestResolveCommit:
    def test_valid_commit(self, git_project):
        head = _git_head(git_project)
        resolved = _resolve_commit(git_project, head[:8])
        assert len(resolved) == 40  # Full SHA

    def test_empty_sha(self, git_project):
        assert _resolve_commit(git_project, "") == ""

    def test_invalid_sha(self, git_project):
        assert _resolve_commit(git_project, "0000000000000000000000000000000000000000") == ""


class TestGitHead:
    def test_returns_sha(self, git_project):
        head = _git_head(git_project)
        assert len(head) == 40


class TestInferBaseCommit:
    def test_plan_source(self, project_with_plans):
        source = {"source_type": "orchestration_kit", "source_path": str(project_with_plans / "docs" / "plans" / "my-plan" / "INDEX.md")}
        commit, src, conf = _infer_base_commit(project_with_plans, source)
        assert commit  # Should find something
        assert src in ("plan_first_commit_parent", "plan_first_commit", "head_fallback")

    def test_git_history_source(self, git_project):
        source = {"source_type": "git_history"}
        commit, src, conf = _infer_base_commit(git_project, source)
        assert src in ("synthetic_target_parent", "synthetic_first_commit", "head_fallback")

    def test_no_git_project(self, tmp_path):
        (tmp_path / "src").mkdir()
        commit, src, conf = _infer_base_commit(tmp_path, {"source_type": "git_history"})
        assert src == "no_git"
        assert conf == "low"


class TestReadPlanStateBounds:
    def test_missing_state_file(self, tmp_path):
        assert _read_plan_state_bounds(tmp_path, tmp_path) == ("", "")

    def test_state_with_bounds(self, git_project, project_with_plans):
        base, target = _read_plan_state_bounds(git_project, project_with_plans / "docs" / "plans" / "my-plan")
        # State file doesn't exist yet, so should be empty
        assert base == ""


class TestWriteCandidateReport:
    def test_empty_candidates(self, tmp_path):
        case_dir = tmp_path / "case1"
        case_dir.mkdir()
        _write_candidate_report(case_dir, "case1", [], None)
        report = (case_dir / "candidate_report.md").read_text(encoding="utf-8")
        assert "Total candidates: 0" in report
        assert "No evidence candidates" in report

    def test_candidate_report_caps_verbose_entries(self, tmp_path):
        case_dir = tmp_path / "case1"
        case_dir.mkdir()
        candidates = [
            CandidateCase(
                candidate_id=f"plan-{idx:03d}",
                source_type="plan",
                primary_source=f"docs/plans/{idx:03d}.md",
                relevance_score=1.0 - idx / 1000,
            )
            for idx in range(75)
        ]

        _write_candidate_report(case_dir, "case1", candidates, candidates[0])

        report = (case_dir / "candidate_report.md").read_text(encoding="utf-8")
        assert "Total candidates: 75" in report
        assert "Showing top 50 candidates" in report
        assert "25 additional candidate(s) omitted" in report
        assert "`plan-049`" in report
        assert "`plan-050`" not in report


class TestIntakeCandidateFeedback:
    def test_selected_candidate_becomes_actual_case_source(self, git_project):
        cases_root = git_project.parent / "cases"
        weaker = EvidenceRecord(
            source_type="plan",
            path=git_project / "docs" / "plans" / "weaker.md",
            summary="Implement weaker task",
            metadata={"relative_path": "docs/plans/weaker.md"},
            relevance_score=0.50,
            relevance_reasons=("plan doc",),
            task_signal="plan_task",
        )
        selected = EvidenceRecord(
            source_type="plan",
            path=git_project / "docs" / "plans" / "selected.md",
            summary="Implement selected task",
            metadata={"relative_path": "docs/plans/selected.md"},
            relevance_score=0.70,
            relevance_reasons=("plan doc",),
            task_signal="plan_task",
        )

        case = intake(
            config=IntakeConfig(project_path=git_project, cases_root=cases_root),
            _cached_evidence=[weaker, selected],
        )

        assert case.selected_candidate_id == "plan-selected"
        assert case.source_type == "plan"
        assert case.source_path == "docs/plans/selected.md"
        assert case.plan_path == git_project / "docs" / "plans" / "selected.md"
        assert "Selected candidate plan-selected" in case.selection_reason
        data = (case.root / "case.yaml").read_text(encoding="utf-8")
        assert "source_path: docs/plans/selected.md" in data

    def test_intake_writes_case_depth_metadata(self, git_project):
        cases_root = git_project.parent / "cases"
        selected = EvidenceRecord(
            source_type="plan",
            path=git_project / "docs" / "plans" / "selected.md",
            summary="Implement selected task with observable acceptance",
            metadata={
                "relative_path": "docs/plans/selected.md",
                "mentioned_files": "src/main.py,tests/test_main.py",
            },
            relevance_score=0.70,
            relevance_reasons=("plan doc",),
            task_signal="plan_task",
        )
        history = EvidenceRecord(
            source_type="codex_history",
            path=git_project / "session.jsonl",
            summary="User reported blocked verification for src/main.py",
            metadata={"mentioned_files": "src/main.py"},
            relevance_score=0.52,
            relevance_reasons=("task action verbs in summary",),
            task_signal="conversation_task",
        )

        case = intake(
            config=IntakeConfig(project_path=git_project, cases_root=cases_root),
            _cached_evidence=[selected, history],
        )

        data = parse_simple_yaml(case.root / "case.yaml")
        assert case.depth_score >= 60
        assert case.depth_level in {"solid", "rich"}
        assert case.episode_label == "plan-history"
        assert int(str(data["depth_score"])) >= 60
        assert data["depth_level"] in {"solid", "rich"}
        assert "plan-history" in str(data["episode_label"])
        task_text = (case.root / "task.md").read_text(encoding="utf-8")
        assert "## Situation Context" in task_text
        assert "## Failure Boundaries" in task_text

    def test_selected_package_doc_resolves_to_orchestration_kit_index(self, project_with_plans):
        case = intake(
            config=IntakeConfig(
                project_path=project_with_plans,
                cases_root=project_with_plans.parent / "cases",
            )
        )

        assert case.source_type == "orchestration_kit"
        assert case.source_path.endswith("docs/plans/my-plan/INDEX.md")
        assert case.plan_path == project_with_plans / "docs" / "plans" / "my-plan" / "INDEX.md"
        assert case.base_source in ("plan_first_commit_parent", "plan_first_commit")
        assert case.depth_score >= 50
        assert case.depth_level in {"solid", "rich"}
        task_text = (case.root / "task.md").read_text(encoding="utf-8")
        assert "Fix the bug" in task_text
        assert "`make test`" in task_text

    def test_prior_task_outcomes_lower_matching_candidate(self, git_project):
        cases_root = git_project.parent / "cases"
        prior_case = case_inventory_root(cases_root, git_project) / "old-case"
        prior_case.mkdir(parents=True)
        write_simple_yaml(
            prior_case / "case.yaml",
            {
                "id": "old-case",
                "project_path": str(git_project),
                "source_type": "manual",
                "source_path": "docs/plans/risky.md",
                "selection_reason": "prior generated case",
                "selected_candidate_id": "plan-risky",
                "candidate_source_type": "plan",
                "candidate_score": "0.60",
            },
        )
        (prior_case / "task.md").write_text("## Goal\nRisky task.\n", encoding="utf-8")
        append_task_outcome(
            prior_case,
            TaskOutcomeEntry(
                package_id="old-package",
                task_contract_id="old-contract",
                outcome="blocked",
                blocked_reason="missing baseline",
                acceptance_gaps=("no observable acceptance",),
                false_positive=True,
                verification_status="weak",
            ),
        )
        risky = EvidenceRecord(
            source_type="plan",
            path=git_project / "docs" / "plans" / "risky.md",
            summary="Implement risky task",
            metadata={"relative_path": "docs/plans/risky.md"},
            relevance_score=0.60,
            relevance_reasons=("plan doc",),
            task_signal="plan_task",
        )
        safer = EvidenceRecord(
            source_type="plan",
            path=git_project / "docs" / "plans" / "safer.md",
            summary="Implement safer task",
            metadata={"relative_path": "docs/plans/safer.md"},
            relevance_score=0.55,
            relevance_reasons=("plan doc",),
            task_signal="plan_task",
        )

        case = intake(
            config=IntakeConfig(project_path=git_project, cases_root=cases_root),
            _force_source={
                "source_type": "manual",
                "source_path": "manual-source",
                "plan_path": "",
                "title": "Manual source",
                "selection_reason": "test forced source",
                "verification_commands": [],
            },
            _cached_evidence=[risky, safer],
        )

        report = (case.root / "candidate_report.md").read_text(encoding="utf-8")
        assert "- ID: `plan-safer`" in report
        assert "prior outcome feedback" in report
        assert "prior outcome feedback lowered this candidate" in report


class TestWriteEvidenceSources:
    def test_writes_evidence_sources(self, tmp_path):
        from replay_checker.sources import EvidenceRecord
        case = ReplayCase(
            id="test", root=tmp_path, project_path=tmp_path, plan_path=Path(""),
            base_commit="abc", source_type="manual", source_path="x",
            selection_reason="y",
        )
        records = [EvidenceRecord(source_type="plan", confidence="high", path=Path("/tmp"), summary="plan found")]
        _write_evidence_sources(case, records)
        content = (tmp_path / "evidence_sources.md").read_text(encoding="utf-8")
        assert "plan" in content
        assert "plan found" in content

    def test_empty_records(self, tmp_path):
        case = ReplayCase(
            id="test", root=tmp_path, project_path=tmp_path, plan_path=Path(""),
            base_commit="abc", source_type="manual", source_path="x",
            selection_reason="y",
        )
        _write_evidence_sources(case, [])
        content = (tmp_path / "evidence_sources.md").read_text(encoding="utf-8")
        assert "- none" in content


class TestWriteCaseYaml:
    def test_writes_basic_case(self, tmp_path):
        case = ReplayCase(
            id="test", root=tmp_path, project_path=tmp_path, plan_path=Path("plan.md"),
            base_commit="abc123", base_source="manual", base_confidence="high",
            source_type="manual", source_path="plan.md", selection_reason="test",
        )
        _write_case_yaml(case)
        from replay_checker.yaml_lite import parse_simple_yaml
        data = parse_simple_yaml(tmp_path / "case.yaml")
        assert data["id"] == "test"
        assert data["base_commit"] == "abc123"


class TestLoadCase:
    def test_roundtrip(self, tmp_path):
        from replay_checker.yaml_lite import write_simple_yaml
        # Create a case directory under a parent "cases" root
        cases_root = tmp_path / "cases"
        case_dir = cases_root / "test"
        case_dir.mkdir(parents=True)
        case = ReplayCase(
            id="test", root=case_dir, project_path=tmp_path, plan_path=Path("plan.md"),
            base_commit="abc123", verification_commands=("make test",),
            base_source="manual", base_confidence="high",
            source_type="manual", source_path="plan.md", selection_reason="test",
        )
        _write_case_yaml(case)
        loaded = load_case(cases_root, "test")
        assert loaded.id == "test"
        assert loaded.base_commit == "abc123"
        assert loaded.verification_commands == ("make test",)

    def test_missing_id_raises(self, tmp_path):
        from replay_checker.yaml_lite import write_simple_yaml
        cases_root = tmp_path / "cases"
        case_dir = cases_root / "noid"
        case_dir.mkdir(parents=True)
        write_simple_yaml(case_dir / "case.yaml", {"not_id": "value"})
        with pytest.raises(ValueError, match="missing required field"):
            load_case(cases_root, "noid")
