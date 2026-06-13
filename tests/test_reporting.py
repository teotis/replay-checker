"""Focused owner-module tests for reporting.py.

Verifies concise/detailed comparison, stale-score suppression, and cross-case
reporting without importing replay.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from replay_checker.evaluation import Recommendation, write_recommendation
from replay_checker.reporting import (
    all_run_result_evidence_blocked,
    build_compare_output,
    compare_case,
    format_default_recommendation,
    format_missing_evaluation,
    has_recommendation,
    report_all_cases,
)
from replay_checker.replay_types import ReplayCase


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path, case_id: str = "test-case") -> ReplayCase:
    root = tmp_path / case_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "case.yaml").write_text(
        f"id: {case_id}\nsource_type: manual\n",
        encoding="utf-8",
    )
    (root / "evidence_sources.md").write_text("- none\n", encoding="utf-8")
    return ReplayCase(
        id=case_id,
        root=root,
        project_path=tmp_path,
        plan_path=Path(""),
        base_commit="abc123",
        verification_commands=(),
        base_source="head_fallback",
        base_confidence="low",
        source_type="manual",
        source_path="",
        selection_reason="test",
        synthetic_case=False,
        evidence_sources=(),
    )


def _write_fake_evidence(run_dir: Path, changed_files: list[str] | None = None) -> None:
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    if changed_files is None:
        changed_files = ["x"]
    cf_yaml = "\n".join(f"  - {f}" for f in changed_files)
    (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (ev_dir / "evidence.yaml").write_text(
        f"run_id: eval-run\nstatus: completed\nchanged_files:\n{cf_yaml}\n",
        encoding="utf-8",
    )


def _write_fake_run_yaml(run_dir: Path, anonymous_runner_id: str = "anon-xyz") -> None:
    (run_dir / "run.yaml").write_text(
        f"anonymous_runner_id: {anonymous_runner_id}\nrunner_label: test-agent\n",
        encoding="utf-8",
    )


# =========================================================================
# build_compare_output — Concise vs Detailed
# =========================================================================


class TestBuildCompareOutput:
    """Test the public build_compare_output function."""

    def test_empty_run_ids_returns_no_runs(self, tmp_path):
        case = _make_case(tmp_path)
        output = build_compare_output(case, [], tmp_path / "runs")
        assert "No runs with evidence found" in output

    def test_default_output_shows_score(self, tmp_path):
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=["a.py"])

        rec = Recommendation(score=85.0, sentence="Good job.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        output = build_compare_output(case, ["test-case-001"], tmp_path / "runs", detailed=False)
        assert "85" in output
        assert "Good job" in output

    def test_detailed_output_shows_per_run_details(self, tmp_path):
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=["a.py", "b.py"])
        _write_fake_run_yaml(run_root)

        rec = Recommendation(score=85.0, sentence="Good job.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        output = build_compare_output(case, ["test-case-001"], tmp_path / "runs", detailed=True)
        assert "## Per-Run Details" in output
        assert "anon-xyz" in output
        assert "Status: completed" in output

    def test_detailed_output_matches_facade(self, tmp_path):
        """build_compare_output matches the facade's _build_compare_output."""
        from replay_checker.replay import _build_compare_output as facade_compare

        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=["x.py"])
        _write_fake_run_yaml(run_root)

        rec = Recommendation(score=70.0, sentence="OK.", validity="valid", confidence="low")
        write_recommendation(case.root, rec)

        for detailed in (False, True):
            new_out = build_compare_output(case, ["test-case-001"], tmp_path / "runs", detailed=detailed)
            old_out = facade_compare(case, ["test-case-001"], tmp_path / "runs", detailed=detailed)
            assert new_out == old_out, f"Mismatch for detailed={detailed}"


# =========================================================================
# Stale-score suppression
# =========================================================================


class TestStaleScoreSuppression:
    """Diff present but no changed files must suppress score to 0/100."""

    def test_score_suppressed_when_changed_files_empty(self, tmp_path):
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=[])

        rec = Recommendation(score=90.0, sentence="Stale positive.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        output = build_compare_output(case, ["test-case-001"], tmp_path / "runs")
        assert "0 / 100" in output
        assert "insufficient evidence" in output.lower()

    def test_score_unblocked_when_changed_files_present(self, tmp_path):
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=["a.py"])

        rec = Recommendation(score=90.0, sentence="Good.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        output = build_compare_output(case, ["test-case-001"], tmp_path / "runs")
        assert "90" in output
        assert "insufficient evidence" not in output.lower()


# =========================================================================
# all_run_result_evidence_blocked
# =========================================================================


class TestAllRunResultEvidenceBlocked:
    """Test the evidence-blocking gate function."""

    def test_empty_run_ids_returns_false(self, tmp_path):
        assert all_run_result_evidence_blocked([], tmp_path / "runs") is False

    def test_all_blocked_returns_true(self, tmp_path):
        run_root = tmp_path / "runs" / "case-001"
        _write_fake_evidence(run_root, changed_files=[])
        assert all_run_result_evidence_blocked(["case-001"], tmp_path / "runs") is True

    def test_one_unblocked_returns_false(self, tmp_path):
        run_root = tmp_path / "runs" / "case-001"
        _write_fake_evidence(run_root, changed_files=["a.py"])
        assert all_run_result_evidence_blocked(["case-001"], tmp_path / "runs") is False

    def test_mixed_blocked_returns_false(self, tmp_path):
        r1 = tmp_path / "runs" / "case-001"
        r2 = tmp_path / "runs" / "case-002"
        _write_fake_evidence(r1, changed_files=[])
        _write_fake_evidence(r2, changed_files=["a.py"])
        assert all_run_result_evidence_blocked(["case-001", "case-002"], tmp_path / "runs") is False

    def test_matches_facade_behavior(self, tmp_path):
        """all_run_result_evidence_blocked matches the facade function."""
        from replay_checker.replay import _all_run_result_evidence_blocked as facade_fn

        run_root = tmp_path / "runs" / "case-001"
        _write_fake_evidence(run_root, changed_files=["a.py"])
        run_ids = ["case-001"]
        runs_dir = tmp_path / "runs"
        assert all_run_result_evidence_blocked(run_ids, runs_dir) == facade_fn(run_ids, runs_dir)


# =========================================================================
# Recommendation formatting
# =========================================================================


class TestRecommendationFormatting:
    """Test has_recommendation and format_default_recommendation."""

    def test_has_recommendation_true_when_score_positive(self):
        rec = Recommendation(score=50.0, sentence="")
        assert has_recommendation(rec) is True

    def test_has_recommendation_true_when_sentence_present(self):
        rec = Recommendation(score=0.0, sentence="OK")
        assert has_recommendation(rec) is True

    def test_has_recommendation_false_when_empty(self):
        rec = Recommendation(score=0.0, sentence="")
        assert has_recommendation(rec) is False

    def test_format_default_recommendation(self):
        rec = Recommendation(score=82.0, sentence="Strong.", validity="valid", confidence="medium")
        out = format_default_recommendation(rec)
        assert "82 / 100" in out
        assert "Strong." in out
        assert "validity: valid" in out.lower()

    def test_format_default_recommendation_no_sentence(self):
        rec = Recommendation(score=45.0, sentence="", validity="valid", confidence="low")
        out = format_default_recommendation(rec)
        assert "45 / 100" in out
        assert "inspect detailed output" in out

    def test_format_missing_evaluation(self):
        out = format_missing_evaluation(["run-001", "run-002"])
        assert "No evaluation available" in out
        assert "2" in out


# =========================================================================
# Cross-case reporting
# =========================================================================


class TestCrossCaseReporting:
    """Test report_all_cases aggregate output."""

    def test_empty_cases_dir(self, tmp_path):
        cases_root = tmp_path / "cases"
        cases_root.mkdir()
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        reports_root = tmp_path / "reports"
        path = report_all_cases(cases_root, runs_root, reports_root)
        text = path.read_text(encoding="utf-8")
        assert "暂无含证据的 run 可供评估" in text

    def test_single_case_with_evidence(self, tmp_path):
        cases_root = tmp_path / "cases"
        case = _make_case(cases_root, "case-001")

        runs_root = tmp_path / "runs"
        run_dir = runs_root / "case-001-001"
        _write_fake_evidence(run_dir, changed_files=["a.py"])
        _write_fake_run_yaml(run_dir)

        rec = Recommendation(score=72.0, sentence="Good.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        reports_root = tmp_path / "reports"
        path = report_all_cases(cases_root, runs_root, reports_root)
        text = path.read_text(encoding="utf-8")
        assert "## 综合排名" in text
        assert "72" in text
        assert "case-001" in text

    def test_report_all_cases_matches_facade(self, tmp_path):
        """report_all_cases produces the same file as the facade."""
        from replay_checker.replay import report_all_cases as facade_report

        cases_root = tmp_path / "cases"
        case = _make_case(cases_root, "case-001")

        runs_root = tmp_path / "runs"
        run_dir = runs_root / "case-001-001"
        _write_fake_evidence(run_dir, changed_files=["a.py"])
        _write_fake_run_yaml(run_dir)

        rec = Recommendation(score=60.0, sentence="OK.", validity="valid", confidence="low")
        write_recommendation(case.root, rec)

        # Run from owner module
        new_reports = tmp_path / "reports_new"
        new_path = report_all_cases(cases_root, runs_root, new_reports)

        # Run from facade
        old_reports = tmp_path / "reports_old"
        old_path = facade_report(cases_root, runs_root, old_reports)

        assert new_path.read_text(encoding="utf-8") == old_path.read_text(encoding="utf-8")
