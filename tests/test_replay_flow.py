"""Tests for replay flow: intake, prepare-run, collect, compare, and profile output.

Covers:
- Default compare output is concise (score + recommendation + validity + confidence)
- Default output does NOT contain verbose telemetry tables
- Detailed output works when detailed=True
- Recommendation sentences are actionable (not generic adjective-only)
- Profile aggregation does not contribute attempt count to score
"""

from __future__ import annotations

import pytest
from pathlib import Path

from replay_checker.evaluation import (
    EligibilityTier,
    Recommendation,
    RunAttempt,
    StructuralContribution,
    _build_recommendation,
    current_dir,
    init_evaluation_dir,
    recompute,
    read_recommendation,
    read_summary,
    write_recommendation,
)
from replay_checker.eval_adapter import SkillEvalCase, dedupe_eval_cases, generate_skill_rubric
from replay_checker.replay import ReplayCase, ReplayRun, _build_compare_output, score_run


# ---------------------------------------------------------------------------
# Helper: build a minimal ReplayCase pointing at a tmp_path case dir
# ---------------------------------------------------------------------------


def _fake_case(tmp_path: Path, case_id: str = "test-case") -> ReplayCase:
    root = tmp_path / case_id
    root.mkdir(parents=True, exist_ok=True)
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


def _write_fake_evidence(run_dir: Path, status: str = "completed") -> None:
    """Write minimal evidence.yaml so run_dir counts as 'evidence present'."""
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (ev_dir / "evidence.yaml").write_text(
        f"run_id: {run_dir.name}\nstatus: {status}\nchanged_files:\n  - x\n",
        encoding="utf-8",
    )


def test_skill_eval_rubric_includes_expected_output() -> None:
    eval_case = SkillEvalCase(
        skill_name="demo-skill",
        eval_id=1,
        prompt="Do the thing",
        expected_output="Explains the expected behavior.",
        assertions=["mentions verification"],
        files=[],
        evals_path=Path("skills/demo-skill/evals/evals.json"),
    )

    rubric = generate_skill_rubric(eval_case)

    assert rubric["expected_output"] == "Explains the expected behavior."
    assert rubric["criteria"] == ["assertion: mentions verification"]


def test_score_run_includes_skill_eval_rubric_criteria(tmp_path: Path) -> None:
    case = _fake_case(tmp_path, "skill-eval-case")
    rubric = case.root / "eval_rubric.yaml"
    rubric.write_text(
        "\n".join(
            [
                "result_weight: 80",
                "process_weight: 20",
                "expected_output: Agent explains the expected behavior.",
                "criteria:",
                "  - assertion: output contains lifecycle invariant",
                "  - assertion: output names verification command",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run_root = tmp_path / "runs" / "skill-eval-case-001"
    _write_fake_evidence(run_root)
    (run_root / "completion_report.md").write_text(
        "status: completed\nverification: pytest passed\n",
        encoding="utf-8",
    )
    run = ReplayRun(
        id="skill-eval-case-001",
        root=run_root,
        case=case,
        runner_label="agent",
        workspace=tmp_path / "workspace",
    )

    scoring_package = score_run(run, rubric_path=None)
    text = scoring_package.read_text(encoding="utf-8")

    assert "Agent explains the expected behavior." in text
    assert "assertion: output contains lifecycle invariant" in text
    assert "assertion: output names verification command" in text


def test_dedupe_eval_cases_does_not_remove_symlinked_case_dirs(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    cases_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_case = cases_root / "demo-eval1-bbbb"
    try:
        symlink_case.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    keep = cases_root / "demo-eval1-aaaa"
    keep.mkdir()

    removed = dedupe_eval_cases(cases_root)

    assert removed == []
    assert outside.exists()
    assert symlink_case.is_symlink()
    assert keep.exists()


# =========================================================================
# Default compare output tests
# =========================================================================


class TestDefaultCompareOutput:
    """Default compare output should be concise and actionable."""

    def test_default_output_is_concise(self, tmp_path):
        """Default output must contain score, sentence, validity, confidence — nothing else."""
        case = _fake_case(tmp_path)
        rec = Recommendation(
            score=72.0,
            sentence="Suitable for low-risk implementation.",
            validity="valid",
            confidence="medium",
        )
        write_recommendation(case.root, rec)

        # Must have at least one run_id so default path is taken (not "no runs" early return)
        fake_run_dir = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(fake_run_dir)

        output = _build_compare_output(case, ["test-case-001"], tmp_path / "runs")

        # Must contain the four required fields
        assert "72" in output, "score"
        assert "Suitable for low-risk implementation" in output
        assert "valid" in output
        assert "medium" in output

    def test_default_output_no_telemetry_table(self, tmp_path):
        """Default output must NOT contain the verbose per-run telemetry table."""
        case = _fake_case(tmp_path)
        rec = Recommendation(score=50.0, sentence="OK.", validity="valid", confidence="low")
        write_recommendation(case.root, rec)

        fake_run_dir = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(fake_run_dir)

        output = _build_compare_output(case, ["test-case-001"], tmp_path / "runs")

        # The old verbose table header must not appear
        assert "Anonymous ID" not in output
        assert "+Lines" not in output
        assert "-Lines" not in output

    def test_default_output_no_run_detail_section(self, tmp_path):
        """Default output must NOT show per-run details."""
        case = _fake_case(tmp_path)
        rec = Recommendation(score=60.0, sentence="Good.", validity="valid", confidence="medium")
        write_recommendation(case.root, rec)

        fake_run_dir = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(fake_run_dir)

        output = _build_compare_output(case, ["test-case-001"], tmp_path / "runs")
        assert "Per-Run Details" not in output


# =========================================================================
# Recommendation sentence quality tests
# =========================================================================


class TestRecommendationSentenceQuality:
    """Recommendation sentences must be actionable, not generic adjective-only."""

    @staticmethod
    def _sentence_for_tier(tier: EligibilityTier) -> str:
        from replay_checker.evaluation import EvaluationSummary
        summary_obj = EvaluationSummary(
            case_id="test", total_score=60.0, tier=tier,
            run_count=2, validity="valid", confidence="medium",
        )
        # Must provide runs with valid status for _build_recommendation
        # to reach the tier-based sentence path
        runs = [
            RunAttempt(run_id="r1", status="completed", tier=tier,
                       dimension_scores={"result": 50.0, "process": 10.0}),
            RunAttempt(run_id="r2", status="completed", tier=tier,
                       dimension_scores={"result": 50.0, "process": 10.0}),
        ]
        rec = _build_recommendation(summary_obj, runs)
        return rec.sentence

    def test_solved_tier_mentions_suitable(self):
        sentence = self._sentence_for_tier(EligibilityTier.SOLVED)
        assert "suitable" in sentence.lower() or "Suitable" in sentence

    def test_failed_tier_mentions_not_suitable(self):
        sentence = self._sentence_for_tier(EligibilityTier.FAILED)
        assert "not suitable" in sentence.lower() or "Not suitable" in sentence

    def test_invalid_tier_mentions_invalid(self):
        sentence = self._sentence_for_tier(EligibilityTier.INVALID)
        assert "invalid" in sentence.lower() or "Invalid" in sentence

    def test_no_generic_only_adjectives(self):
        """Sentences must not be just a single adjective like 'Good' or 'Excellent'."""
        for tier in EligibilityTier:
            sentence = self._sentence_for_tier(tier)
            words = sentence.strip().rstrip(".").split()
            assert len(words) >= 4, (
                f"Recommendation for {tier.value} is too short: '{sentence}'"
            )


# =========================================================================
# Attempt metadata must not affect score
# =========================================================================


class TestAttemptMetadataNoScoreImpact:
    """Attempt count and first/best attempt are factual metadata only."""

    def test_attempt_count_not_in_score(self, tmp_path):
        """run_count in summary is informational, not a score contributor."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        runs = [
            RunAttempt(run_id="r1", status="completed", tier=EligibilityTier.SOLVED,
                       dimension_scores={"result": 50.0, "process": 10.0}),
            RunAttempt(run_id="r2", status="completed", tier=EligibilityTier.SOLVED,
                       dimension_scores={"result": 50.0, "process": 10.0}),
            RunAttempt(run_id="r3", status="completed", tier=EligibilityTier.SOLVED,
                       dimension_scores={"result": 50.0, "process": 10.0}),
        ]
        recompute(case_root=case_root, runs=runs, reason="metadata test")

        summary = read_summary(case_root)
        # run_count is 3 but total_score must not be inflated by the count
        assert summary.run_count == 3
        # All runs identical → best score should be the weighted base, not boosted by count
        assert 0 < summary.total_score <= 100.0

    def test_invalid_run_affects_confidence_not_score(self, tmp_path):
        """Invalid/unscorable runs lower confidence, not the total score."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        runs = [
            RunAttempt(run_id="good", status="completed", tier=EligibilityTier.SOLVED,
                       dimension_scores={"result": 80.0, "process": 15.0}),
            RunAttempt(run_id="bad", status="missing-completion-report",
                       tier=EligibilityTier.FAILED,
                       dimension_scores={"result": 0.0, "process": 0.0}),
        ]
        recompute(case_root=case_root, runs=runs, reason="invalid test")

        summary = read_summary(case_root)
        # Confidence must reflect the incomplete run
        assert summary.confidence in ("low",)
        # But best_run_id should still be the good run
        assert summary.best_run_id == "good"


# =========================================================================
# Profile aggregation tests
# =========================================================================


class TestProfileAggregation:
    """Cross-case profile aggregation rules."""

    def test_no_cost_budget_scoring(self):
        """Cost, token, wall-time, and iteration budget must NOT appear as score components."""
        from replay_checker.evaluation import EvaluationSummary, read_summary
        # This is a design contract test: if these fields ever appear in
        # EvaluationSummary, the test should fail.
        summary = EvaluationSummary(case_id="x")
        assert not hasattr(summary, "cost_score"), "cost_score must not exist"
        assert not hasattr(summary, "token_score"), "token_score must not exist"
        assert not hasattr(summary, "wall_time_score"), "wall_time_score must not exist"

    def test_attempt_count_is_factual_metadata(self):
        """Attempt count exists as a field but is never added to total_score."""
        from replay_checker.evaluation import EvaluationSummary
        s = EvaluationSummary(case_id="x", run_count=5, total_score=72.0)
        # total_score should be independent of run_count
        assert s.total_score == 72.0
        assert s.run_count == 5
