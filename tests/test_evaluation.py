"""Tests for evaluation storage contract: data types, YAML round-trips, and IO helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import replay_checker.evaluation as evaluation_module
from replay_checker.evaluation import (
    ComparisonCandidate,
    ComparisonLabel,
    DimensionDef,
    EligibilityTier,
    EvaluationSummary,
    IncomparableReason,
    PairwiseComparison,
    RankingEntry,
    Recommendation,
    RunAttempt,
    ScoreHistoryEntry,
    _comparison_filename,
    _compute_information_gain,
    _from_yaml,
    _group_ranking_by_band,
    _to_yaml,
    append_score_history,
    build_comparison_queue,
    current_dir,
    eval_dir,
    init_evaluation_dir,
    read_attempts,
    read_comparison,
    read_comparison_queue,
    read_dimensions,
    read_ranking,
    read_recommendation,
    read_score_history,
    read_summary,
    recompute,
    snapshot_current,
    snapshots_dir,
    write_attempts,
    write_comparison,
    write_comparison_queue,
    write_dimensions,
    write_ranking,
    write_recommendation,
    write_summary,
)


# ---------------------------------------------------------------------------
# YAML round-trip tests
# ---------------------------------------------------------------------------


def test_yaml_round_trip_flat_dict():
    data = {"key": "value", "count": "42", "flag": "true"}
    parsed = _from_yaml(_to_yaml(data))
    assert parsed == data


def test_yaml_round_trip_nested_dict():
    inner = {"x": "1", "y": "2"}
    data = {"section": "\n", "section2": "\n"}
    text = _to_yaml({"section": inner})
    parsed = _from_yaml(text)
    assert parsed["section"]["x"] == "1"
    assert parsed["section"]["y"] == "2"


def test_yaml_round_trip_list_of_dicts():
    items = [
        {"name": "a", "score": "10"},
        {"name": "b", "score": "20"},
    ]
    text = _to_yaml(items)
    parsed = _from_yaml(text)
    assert len(parsed) >= 1


def test_yaml_round_trip_list_of_scalars():
    data = {"items": ["a", "b", "c"]}
    text = _to_yaml(data)
    parsed = _from_yaml(text)
    assert parsed["items"] == ["a", "b", "c"]


def test_yaml_parses_empty_string():
    assert _from_yaml("") == {}


def test_yaml_handles_comments():
    text = "# comment\nkey: value\n# another comment\n"
    assert _from_yaml(text) == {"key": "value"}


def test_yaml_handles_booleans():
    text = "flag: true\n"
    assert _from_yaml(text) == {"flag": "true"}


def test_yaml_handles_floats():
    text = "score: 42.5\n"
    assert _from_yaml(text) == {"score": "42.5"}


# ---------------------------------------------------------------------------
# Directory layout tests
# ---------------------------------------------------------------------------


def test_eval_dir_returns_correct_path():
    result = eval_dir("/tmp/case-1")
    assert result == Path("/tmp/case-1/evaluation")


def test_current_dir_returns_correct_path():
    result = current_dir("/tmp/case-1")
    assert result == Path("/tmp/case-1/evaluation/current")


def test_snapshots_dir_returns_correct_path():
    result = snapshots_dir("/tmp/case-1")
    assert result == Path("/tmp/case-1/evaluation/snapshots")


def test_init_evaluation_dir_creates_structure(tmp_path):
    case = tmp_path / "case-1"
    cur = init_evaluation_dir(case)
    assert cur.exists()
    assert (cur / "rankings").exists()
    assert (cur / "comparisons").exists()


def test_init_evaluation_dir_is_idempotent(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    init_evaluation_dir(case)
    assert (case / "evaluation" / "current").exists()


# ---------------------------------------------------------------------------
# summary.yaml round-trip tests
# ---------------------------------------------------------------------------


def test_summary_round_trip(tmp_path):
    case = tmp_path / "case-1"
    summary = EvaluationSummary(
        case_id="case-1",
        tier=EligibilityTier.SOLVED,
        validity="valid",
        confidence="high",
        total_score=72.5,
        run_count=3,
        best_run_id="run-abc",
    )
    write_summary(case, summary)
    loaded = read_summary(case)
    assert loaded.case_id == "case-1"
    assert loaded.tier == EligibilityTier.SOLVED
    assert loaded.validity == "valid"
    assert loaded.confidence == "high"
    assert loaded.total_score == 72.5
    assert loaded.run_count == 3
    assert loaded.best_run_id == "run-abc"


def test_summary_defaults_when_missing(tmp_path):
    case = tmp_path / "case-1"
    loaded = read_summary(case)
    assert loaded.case_id == ""


def test_summary_file_in_current_dir(tmp_path):
    case = tmp_path / "case-1"
    write_summary(case, EvaluationSummary(case_id="test"))
    assert (case / "evaluation" / "current" / "summary.yaml").exists()


# ---------------------------------------------------------------------------
# dimensions.yaml round-trip tests
# ---------------------------------------------------------------------------


def test_dimensions_round_trip(tmp_path):
    case = tmp_path / "case-1"
    dims = [
        DimensionDef(name="correctness", weight=30.0, description="Core correctness"),
        DimensionDef(name="maintainability", weight=15.0, description=""),
    ]
    write_dimensions(case, dims)
    loaded = read_dimensions(case)
    assert len(loaded) == 2
    assert loaded[0].name == "correctness"
    assert loaded[0].weight == 30.0
    assert loaded[1].name == "maintainability"
    assert loaded[1].weight == 15.0


def test_dimensions_empty_when_missing(tmp_path):
    case = tmp_path / "case-1"
    assert read_dimensions(case) == []


# ---------------------------------------------------------------------------
# attempts.yaml round-trip tests
# ---------------------------------------------------------------------------


def test_attempts_round_trip(tmp_path):
    case = tmp_path / "case-1"
    attempts = [
        RunAttempt(
            run_id="run-1",
            anonymous_id="anon-1",
            status="completed",
            tier=EligibilityTier.SOLVED,
            total_score=72.5,
            dimension_scores={"correctness": 25.0, "maintainability": 10.0},
        ),
        RunAttempt(
            run_id="run-2",
            anonymous_id="anon-2",
            status="completed",
            tier=EligibilityTier.PARTIAL,
            total_score=50.0,
        ),
    ]
    write_attempts(case, attempts)
    loaded = read_attempts(case)
    assert len(loaded) == 2
    assert loaded[0].run_id == "run-1"
    assert loaded[0].anonymous_id == "anon-1"
    assert loaded[0].tier == EligibilityTier.SOLVED
    assert loaded[0].total_score == 72.5
    assert loaded[0].dimension_scores == {"correctness": 25.0, "maintainability": 10.0}
    assert loaded[1].run_id == "run-2"
    assert loaded[1].dimension_scores == {}


def test_attempts_empty_when_missing(tmp_path):
    case = tmp_path / "case-1"
    assert read_attempts(case) == []


# ---------------------------------------------------------------------------
# rankings round-trip tests
# ---------------------------------------------------------------------------


def test_ranking_round_trip(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    entries = [
        RankingEntry(run_id="run-a", rank=1, score=30.0, band=0),
        RankingEntry(run_id="run-b", rank=2, score=20.0, band=1),
    ]
    write_ranking(case, "correctness", entries)
    loaded = read_ranking(case, "correctness")
    assert len(loaded) == 2
    assert loaded[0].run_id == "run-a"
    assert loaded[0].rank == 1
    assert loaded[0].score == 30.0
    assert loaded[0].band == 0


def test_ranking_empty_when_missing(tmp_path):
    case = tmp_path / "case-1"
    assert read_ranking(case, "nonexistent") == []


# ---------------------------------------------------------------------------
# comparisons round-trip tests
# ---------------------------------------------------------------------------


def test_comparison_round_trip(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    comp = PairwiseComparison(
        dimension="correctness",
        run_a="run-1",
        run_b="run-2",
        label=ComparisonLabel.BETTER,
        evidence="diff shows fewer bugs in run-1",
    )
    write_comparison(case, comp)
    loaded = read_comparison(case, "run-1", "run-2")
    assert loaded.dimension == "correctness"
    assert loaded.label == ComparisonLabel.BETTER
    assert loaded.evidence == "diff shows fewer bugs in run-1"


def test_comparison_incomparable_with_reasons(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    comp = PairwiseComparison(
        dimension="maintainability",
        run_a="run-1",
        run_b="run-2",
        label=ComparisonLabel.INCOMPARABLE,
        evidence="Different approaches to error handling",
        incomparable_reasons=[
            IncomparableReason.DIFFERENT_STRATEGY,
            IncomparableReason.DIFFERENT_RISK_PROFILE,
        ],
    )
    write_comparison(case, comp)
    loaded = read_comparison(case, "run-1", "run-2")
    assert loaded.label == ComparisonLabel.INCOMPARABLE
    assert len(loaded.incomparable_reasons) == 2
    assert IncomparableReason.DIFFERENT_STRATEGY in loaded.incomparable_reasons
    assert IncomparableReason.DIFFERENT_RISK_PROFILE in loaded.incomparable_reasons


def test_comparison_defaults_when_missing(tmp_path):
    case = tmp_path / "case-1"
    loaded = read_comparison(case, "a", "b")
    assert loaded.run_a == "a"
    assert loaded.run_b == "b"
    assert loaded.label == ComparisonLabel.TIE


# ---------------------------------------------------------------------------
# comparison filename tests
# ---------------------------------------------------------------------------


def test_comparison_filename_is_order_independent():
    a = _comparison_filename("run-x", "run-y")
    b = _comparison_filename("run-y", "run-x")
    assert a == b


def test_comparison_filename_contains_both_ids():
    fn = _comparison_filename("aaa", "bbb")
    assert "aaa" in fn
    assert "bbb" in fn


# ---------------------------------------------------------------------------
# score_history.jsonl tests
# ---------------------------------------------------------------------------


def test_score_history_append_and_read(tmp_path):
    case = tmp_path / "case-1"
    entry = ScoreHistoryEntry(
        run_id="run-1",
        timestamp="2026-01-01T00:00:00Z",
        total_score=72.5,
        tier="solved",
        dimensions={"correctness": 25.0},
        recomputation_reason="new run added",
    )
    append_score_history(case, entry)
    loaded = read_score_history(case)
    assert len(loaded) == 1
    assert loaded[0].run_id == "run-1"
    assert loaded[0].total_score == 72.5
    assert loaded[0].recomputation_reason == "new run added"


def test_score_history_append_multiple(tmp_path):
    case = tmp_path / "case-1"
    append_score_history(case, ScoreHistoryEntry(run_id="r1", total_score=70.0))
    append_score_history(case, ScoreHistoryEntry(run_id="r2", total_score=85.0))
    loaded = read_score_history(case)
    assert len(loaded) == 2


def test_score_history_empty_when_missing(tmp_path):
    case = tmp_path / "case-1"
    assert read_score_history(case) == []


def test_score_history_at_eval_root(tmp_path):
    """score_history.jsonl lives at evaluation/ root, not under current/."""
    case = tmp_path / "case-1"
    append_score_history(case, ScoreHistoryEntry(run_id="r1"))
    assert (case / "evaluation" / "score_history.jsonl").exists()
    assert not (case / "evaluation" / "current" / "score_history.jsonl").exists()


def test_score_history_is_valid_jsonl(tmp_path):
    case = tmp_path / "case-1"
    append_score_history(case, ScoreHistoryEntry(run_id="r1", total_score=50.0))
    path = case / "evaluation" / "score_history.jsonl"
    lines = path.read_text().rstrip("\n").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["run_id"] == "r1"


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_current_copies_yaml_files(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    write_summary(case, EvaluationSummary(case_id="case-1", tier=EligibilityTier.SOLVED))
    write_dimensions(case, [DimensionDef(name="correctness", weight=30.0)])

    snap = snapshot_current(case, "2026-01-01T000000")
    assert snap == case / "evaluation" / "snapshots" / "2026-01-01T000000"
    assert (snap / "summary.yaml").exists()
    assert (snap / "dimensions.yaml").exists()


def test_snapshot_current_missing_current_raises(tmp_path):
    case = tmp_path / "case-1"
    with pytest.raises(FileNotFoundError):
        snapshot_current(case, "2026-01-01T000000")


def test_snapshot_current_overwrites_existing(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    write_summary(case, EvaluationSummary(case_id="case-1", total_score=50.0))
    snapshot_current(case, "ts1")
    # overwrite with new score
    write_summary(case, EvaluationSummary(case_id="case-1", total_score=75.0))
    snapshot_current(case, "ts1")  # should not fail
    snap_summary = read_summary_from_path(
        case / "evaluation" / "snapshots" / "ts1" / "summary.yaml"
    )
    assert snap_summary.total_score == 75.0


def test_recompute_snapshot_timestamp_includes_subsecond_precision(tmp_path, monkeypatch):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    write_summary(case, EvaluationSummary(case_id="case-1", total_score=50.0))
    timestamps: list[str] = []

    def capture_snapshot(case_root: str | Path, timestamp: str) -> Path:
        timestamps.append(timestamp)
        return snapshots_dir(case_root) / timestamp

    monkeypatch.setattr(evaluation_module, "snapshot_current", capture_snapshot)

    recompute(case_root=case, runs=[_make_run("run-1", result=10.0)], reason="timestamp test")

    assert timestamps
    assert "." in timestamps[0]


def read_summary_from_path(path: Path) -> EvaluationSummary:
    from replay_checker.evaluation import _from_yaml
    data = _from_yaml(path.read_text(encoding="utf-8"))
    return EvaluationSummary(
        case_id=str(data.get("case_id", "")),
        tier=EligibilityTier(str(data.get("tier", "invalid"))),
        validity=str(data.get("validity", "unvalidated")),
        confidence=str(data.get("confidence", "low")),
        total_score=float(str(data.get("total_score", "0"))),
        run_count=int(str(data.get("run_count", "0"))),
        best_run_id=str(data.get("best_run_id", "")),
    )


# ---------------------------------------------------------------------------
# IncomparableReason enum tests
# ---------------------------------------------------------------------------


def test_incomparable_reason_has_five_members():
    reasons = list(IncomparableReason)
    assert len(reasons) == 5
    values = {r.value for r in reasons}
    assert values == {
        "different_strategy",
        "different_risk_profile",
        "insufficient_evidence",
        "task_ambiguity",
        "depends_on_user_lens",
    }


# ---------------------------------------------------------------------------
# EligibilityTier and ComparisonLabel enum tests
# ---------------------------------------------------------------------------


def test_eligibility_tier_members():
    assert EligibilityTier.INVALID.value == "invalid"
    assert EligibilityTier.TRANSFORMATIVE.value == "transformative"


def test_comparison_label_has_incomparable():
    assert ComparisonLabel.INCOMPARABLE.value == "incomparable"


# ---------------------------------------------------------------------------
# Recommendation round-trip tests
# ---------------------------------------------------------------------------


def test_recommendation_round_trip(tmp_path):
    case = tmp_path / "case-1"
    rec = Recommendation(
        score=72.5,
        sentence="Suitable for low-risk implementation.",
        validity="valid",
        confidence="medium",
    )
    write_recommendation(case, rec)
    loaded = read_recommendation(case)
    assert loaded.score == 72.5
    assert loaded.sentence == "Suitable for low-risk implementation."
    assert loaded.validity == "valid"
    assert loaded.confidence == "medium"


def test_recommendation_defaults_when_missing(tmp_path):
    case = tmp_path / "case-1"
    loaded = read_recommendation(case)
    assert loaded.score == 0.0
    assert loaded.sentence == ""


def test_recommendation_file_in_current_dir(tmp_path):
    case = tmp_path / "case-1"
    write_recommendation(case, Recommendation(score=85.0))
    assert (case / "evaluation" / "current" / "recommendation.yaml").exists()


# ---------------------------------------------------------------------------
# Full recomputation tests
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str,
    tier: EligibilityTier = EligibilityTier.SOLVED,
    status: str = "completed",
    **dim_scores: float,
) -> RunAttempt:
    return RunAttempt(
        run_id=run_id,
        anonymous_id=f"anon-{run_id}",
        status=status,
        tier=tier,
        total_score=0.0,  # will be recomputed
        dimension_scores=dict(dim_scores),
    )


def test_recompute_empty_runs(tmp_path):
    case = tmp_path / "case-1"
    cur = recompute(case_root=case, runs=[], reason="initial")
    assert cur.exists()
    summary = read_summary(case)
    assert summary.run_count == 0
    assert summary.tier == EligibilityTier.INVALID
    rec = read_recommendation(case)
    assert "No runs" in rec.sentence


def test_recompute_single_run(tmp_path):
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-1", result=25.0, process=15.0),
    ]
    cur = recompute(case_root=case, runs=runs, reason="single run test")

    assert cur.exists()
    summary = read_summary(case)
    assert summary.run_count == 1
    assert summary.best_run_id == "run-1"

    # Rankings should exist for both default dimensions
    result_ranking = read_ranking(case, "result")
    assert len(result_ranking) == 1
    assert result_ranking[0].run_id == "run-1"

    process_ranking = read_ranking(case, "process")
    assert len(process_ranking) == 1

    # Recommendation should be set
    rec = read_recommendation(case)
    assert rec.score > 0.0
    assert len(rec.sentence) > 0

    # Score history should have one entry
    history = read_score_history(case)
    assert len(history) == 1
    assert history[0].run_id == "run-1"


def test_invalid_tier_run_cannot_be_best_when_valid_run_scores_zero(tmp_path):
    case = tmp_path / "case-1"
    runs = [
        _make_run(
            "invalid-first",
            tier=EligibilityTier.INVALID,
            status="invalid",
            result=90.0,
            process=20.0,
        ),
        _make_run(
            "valid-zero",
            tier=EligibilityTier.FAILED,
            status="completed",
            result=0.0,
            process=0.0,
        ),
    ]

    recompute(case_root=case, runs=runs, reason="invalid best regression")

    summary = read_summary(case)
    assert summary.best_run_id == "valid-zero"
    assert summary.total_score == 0.0


def test_recompute_multiple_runs_with_rankings(tmp_path):
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-a", result=30.0, process=10.0),
        _make_run("run-b", result=20.0, process=18.0),
        _make_run("run-c", result=10.0, process=15.0),
    ]
    recompute(case_root=case, runs=runs, reason="multi-run test")

    # Rankings should sort by score descending
    result_ranking = read_ranking(case, "result")
    assert result_ranking[0].run_id == "run-a"
    assert result_ranking[0].score == 30.0
    assert result_ranking[0].rank == 1
    assert result_ranking[2].run_id == "run-c"
    assert result_ranking[2].score == 10.0

    process_ranking = read_ranking(case, "process")
    assert process_ranking[0].run_id == "run-b"
    assert process_ranking[0].score == 18.0

    # Comparisons should exist for all pairs
    comp_ab = read_comparison(case, "run-a", "run-b")
    assert comp_ab.dimension == "result"
    assert comp_ab.label in (ComparisonLabel.BETTER, ComparisonLabel.MUCH_BETTER)

    # Summary should pick best run
    summary = read_summary(case)
    assert summary.run_count == 3

    # History: 3 entries (one per run)
    history = read_score_history(case)
    assert len(history) == 3


def test_recompute_preserves_snapshot_of_prior_state(tmp_path):
    case = tmp_path / "case-1"
    runs_v1 = [_make_run("run-a", result=20.0, process=10.0)]
    recompute(case_root=case, runs=runs_v1, reason="v1")

    # Verify current has v1 state
    summary_v1 = read_summary(case)
    assert summary_v1.run_count == 1

    # Recomputation with different runs should snapshot the old state
    runs_v2 = [
        _make_run("run-a", result=20.0, process=10.0),
        _make_run("run-b", result=30.0, process=15.0),
    ]
    recompute(case_root=case, runs=runs_v2, reason="v2")

    # Current should reflect v2
    summary_v2 = read_summary(case)
    assert summary_v2.run_count == 2

    # Snapshot should exist with v1 state
    snap_dir = snapshots_dir(case)
    snapshots = list(snap_dir.iterdir())
    assert len(snapshots) >= 1

    # The snapshot should contain v1's summary
    snap_summary_path = snapshots[0] / "summary.yaml"
    assert snap_summary_path.exists()


def test_recompute_deterministic_for_same_inputs(tmp_path):
    case_a = tmp_path / "case-a"
    case_b = tmp_path / "case-b"
    runs = [
        _make_run("run-1", result=25.0, process=15.0),
        _make_run("run-2", result=20.0, process=18.0),
    ]

    recompute(case_root=case_a, runs=runs, reason="test-a")
    recompute(case_root=case_b, runs=runs, reason="test-b")

    # Summaries should be identical
    summary_a = read_summary(case_a)
    summary_b = read_summary(case_b)
    assert summary_a.total_score == summary_b.total_score
    assert summary_a.best_run_id == summary_b.best_run_id
    assert summary_a.tier == summary_b.tier

    # Rankings should be identical
    rank_a = read_ranking(case_a, "result")
    rank_b = read_ranking(case_b, "result")
    assert len(rank_a) == len(rank_b)
    for ea, eb in zip(rank_a, rank_b):
        assert ea.run_id == eb.run_id
        assert ea.rank == eb.rank
        assert ea.score == eb.score
        assert ea.band == eb.band

    # Recommendations should match
    rec_a = read_recommendation(case_a)
    rec_b = read_recommendation(case_b)
    assert rec_a.score == rec_b.score
    assert rec_a.sentence == rec_b.sentence


def test_recompute_adding_run_changes_all_rankings(tmp_path):
    """Adding a new run must trigger full recomputation, not just patch the new run."""
    case = tmp_path / "case-1"

    # Initial state: 2 runs
    runs_v1 = [
        _make_run("run-a", result=30.0, process=10.0),
        _make_run("run-b", result=20.0, process=18.0),
    ]
    recompute(case_root=case, runs=runs_v1, reason="v1")

    rank_v1 = read_ranking(case, "result")
    assert len(rank_v1) == 2
    assert rank_v1[0].run_id == "run-a"
    assert rank_v1[0].rank == 1
    assert rank_v1[1].run_id == "run-b"
    assert rank_v1[1].rank == 2

    # Add a third run that outperforms on result
    runs_v2 = [
        _make_run("run-a", result=30.0, process=10.0),
        _make_run("run-b", result=20.0, process=18.0),
        _make_run("run-c", result=35.0, process=5.0),
    ]
    recompute(case_root=case, runs=runs_v2, reason="v2 — new run added")

    rank_v2 = read_ranking(case, "result")
    assert len(rank_v2) == 3
    # run-c should now be rank 1
    assert rank_v2[0].run_id == "run-c"
    assert rank_v2[0].rank == 1
    assert rank_v2[0].score == 35.0
    # run-a drops to rank 2
    assert rank_v2[1].run_id == "run-a"
    assert rank_v2[1].rank == 2
    # run-b drops to rank 3
    assert rank_v2[2].run_id == "run-b"
    assert rank_v2[2].rank == 3

    # Summary should reflect the new best run
    summary_v2 = read_summary(case)
    assert summary_v2.best_run_id == "run-c"

    # History should have 5 entries total (2 from v1 + 3 from v2)
    history = read_score_history(case)
    assert len(history) == 5


def test_recompute_removing_run_changes_current(tmp_path):
    """Removing a run and recomputing should produce different current/ state."""
    case = tmp_path / "case-1"
    runs_v1 = [
        _make_run("run-a", result=30.0, process=10.0),
        _make_run("run-b", result=20.0, process=18.0),
    ]
    recompute(case_root=case, runs=runs_v1, reason="v1")
    best_v1 = read_summary(case).best_run_id

    # Remove the best run
    runs_v2 = [_make_run("run-b", result=20.0, process=18.0)]
    recompute(case_root=case, runs=runs_v2, reason="v2 — run removed")

    best_v2 = read_summary(case).best_run_id
    assert best_v2 != best_v1
    assert best_v2 == "run-b"

    rank = read_ranking(case, "result")
    assert len(rank) == 1


def test_recompute_with_incomplete_runs(tmp_path):
    """Runs with missing evidence should be marked incomparable."""
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-a", result=25.0, process=15.0),
        _make_run("run-b", status="missing-completion-report"),
    ]
    recompute(case_root=case, runs=runs, reason="incomplete test")

    # Comparison between valid and incomplete should be INCOMPARABLE
    comp = read_comparison(case, "run-a", "run-b")
    assert comp.label == ComparisonLabel.INCOMPARABLE
    assert IncomparableReason.INSUFFICIENT_EVIDENCE in comp.incomparable_reasons

    # Summary should note the incomplete run
    summary = read_summary(case)
    assert summary.run_count == 2
    # run-b shouldn't be best since it's incomplete
    assert summary.best_run_id == "run-a"


def test_recompute_with_custom_dimensions(tmp_path):
    case = tmp_path / "case-1"
    dims = [
        DimensionDef(name="correctness", weight=60.0, description="Accuracy"),
        DimensionDef(name="maintainability", weight=20.0, description="Code quality"),
        DimensionDef(name="completeness", weight=20.0, description="Scope coverage"),
    ]
    runs = [
        _make_run("run-a", correctness=28.0, maintainability=10.0, completeness=15.0),
        _make_run("run-b", correctness=20.0, maintainability=18.0, completeness=10.0),
    ]
    recompute(case_root=case, runs=runs, dimensions=dims, reason="custom dims")

    # Rankings should exist for each custom dimension
    for dim_name in ("correctness", "maintainability", "completeness"):
        rank = read_ranking(case, dim_name)
        assert len(rank) == 2, f"ranking for {dim_name} should have 2 entries"

    # Written dimensions should match
    loaded_dims = read_dimensions(case)
    assert len(loaded_dims) == 3
    dim_names = {d.name for d in loaded_dims}
    assert dim_names == {"correctness", "maintainability", "completeness"}


def test_recompute_score_history_append_only(tmp_path):
    """score_history.jsonl must only grow, never shrink or overwrite."""
    case = tmp_path / "case-1"

    recompute(case_root=case, runs=[_make_run("run-1", result=20.0)], reason="first")
    h1 = read_score_history(case)
    assert len(h1) == 1

    recompute(case_root=case, runs=[
        _make_run("run-1", result=20.0),
        _make_run("run-2", result=25.0),
    ], reason="second")
    h2 = read_score_history(case)
    assert len(h2) == 3  # 1 from first + 2 from second

    recompute(case_root=case, runs=[
        _make_run("run-1", result=20.0),
        _make_run("run-2", result=25.0),
        _make_run("run-3", result=30.0),
    ], reason="third")
    h3 = read_score_history(case)
    assert len(h3) == 6  # 3 from before + 3 new

    # Old entries must be preserved
    assert h3[0].run_id == h1[0].run_id
    assert h3[0].total_score == h1[0].total_score


def test_recompute_upon_first_run_no_snapshot_created(tmp_path):
    """First recomputation on a new case should not create a snapshot."""
    case = tmp_path / "case-1"
    runs = [_make_run("run-1", result=20.0, process=10.0)]
    recompute(case_root=case, runs=runs, reason="initial")

    snap_dir = snapshots_dir(case)
    snap_dir_exists = snap_dir.exists()
    # snapshots/ dir should not exist (no prior state to snapshot)
    assert not snap_dir_exists or len(list(snap_dir.iterdir())) == 0


def test_recompute_band_grouping(tmp_path):
    """Runs within 5% of the top score should share the same band."""
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-a", result=100.0),
        _make_run("run-b", result=97.0),  # within 5% → same band
        _make_run("run-c", result=80.0),  # >5% below → different band
    ]
    recompute(case_root=case, runs=runs, reason="band test")

    rank = read_ranking(case, "result")
    assert rank[0].band == 0  # top band
    assert rank[1].band == 0  # within 5% → same band
    assert rank[2].band == 1  # >5% below → next band


# ---------------------------------------------------------------------------
# Information-gain comparison engine tests
# ---------------------------------------------------------------------------


def _make_ranking_entry(run_id: str, rank: int, score: float, band: int) -> RankingEntry:
    return RankingEntry(run_id=run_id, rank=rank, score=score, band=band)


# ---------------------------------------------------------------------------
# _group_ranking_by_band
# ---------------------------------------------------------------------------


def test_group_ranking_by_band_single_band():
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 95.0, 0),
    ]
    groups = _group_ranking_by_band(entries)
    assert len(groups) == 1
    assert 0 in groups
    assert len(groups[0]) == 2


def test_group_ranking_by_band_multiple_bands():
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
        _make_ranking_entry("c", 3, 50.0, 2),
    ]
    groups = _group_ranking_by_band(entries)
    assert len(groups) == 3
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1


def test_group_ranking_by_band_mixed():
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 95.0, 0),
        _make_ranking_entry("c", 3, 80.0, 1),
        _make_ranking_entry("d", 4, 50.0, 2),
        _make_ranking_entry("e", 5, 48.0, 2),
    ]
    groups = _group_ranking_by_band(entries)
    assert len(groups[0]) == 2  # band 0: a, b
    assert len(groups[1]) == 1  # band 1: c
    assert len(groups[2]) == 2  # band 2: d, e


# ---------------------------------------------------------------------------
# _compute_information_gain
# ---------------------------------------------------------------------------


def test_gain_intra_band_higher_than_cross_band():
    """Same-band pairs should have higher gain than cross-band."""
    a = _make_ranking_entry("a", 1, 100.0, 0)
    b = _make_ranking_entry("b", 2, 95.0, 0)
    c = _make_ranking_entry("c", 3, 50.0, 2)

    band_sizes = {0: 2, 1: 0, 2: 1}
    max_band_size = 2

    gain_intra = _compute_information_gain(
        entry_a=a, entry_b=b, band_sizes=band_sizes, max_band_size=max_band_size,
    )
    gain_cross = _compute_information_gain(
        entry_a=a, entry_b=c, band_sizes=band_sizes, max_band_size=max_band_size,
    )

    assert gain_intra > gain_cross, (
        f"Intra-band gain ({gain_intra}) should exceed cross-band gain ({gain_cross})"
    )


def test_gain_larger_band_higher_priority():
    """Pairs in larger bands should have higher gain than pairs in smaller bands."""
    # Band 0: 4 runs, Band 1: 2 runs
    a = _make_ranking_entry("a", 1, 100.0, 0)
    b = _make_ranking_entry("b", 2, 96.0, 0)
    c = _make_ranking_entry("c", 5, 50.0, 1)
    d = _make_ranking_entry("d", 6, 48.0, 1)

    band_sizes = {0: 4, 1: 2}
    max_band_size = 4

    gain_large = _compute_information_gain(
        entry_a=a, entry_b=b, band_sizes=band_sizes, max_band_size=max_band_size,
    )
    gain_small = _compute_information_gain(
        entry_a=c, entry_b=d, band_sizes=band_sizes, max_band_size=max_band_size,
    )

    assert gain_large > gain_small, (
        f"Large-band gain ({gain_large}) should exceed small-band gain ({gain_small})"
    )


def test_gain_closer_scores_higher_priority_within_band():
    """Within the same band, closer scores should have higher gain."""
    a = _make_ranking_entry("a", 1, 100.0, 0)
    b = _make_ranking_entry("b", 2, 99.0, 0)  # very close to a
    c = _make_ranking_entry("c", 3, 95.0, 0)  # farther from a

    band_sizes = {0: 3}
    max_band_size = 3

    gain_close = _compute_information_gain(
        entry_a=a, entry_b=b, band_sizes=band_sizes, max_band_size=max_band_size,
    )
    gain_far = _compute_information_gain(
        entry_a=a, entry_b=c, band_sizes=band_sizes, max_band_size=max_band_size,
    )

    assert gain_close > gain_far, (
        f"Close-score gain ({gain_close}) should exceed far-score gain ({gain_far})"
    )


def test_gain_adjacent_band_higher_than_distant():
    """Adjacent-band pairs should have higher gain than distant-band pairs."""
    a = _make_ranking_entry("a", 1, 100.0, 0)
    b = _make_ranking_entry("b", 2, 80.0, 1)  # adjacent to a
    c = _make_ranking_entry("c", 3, 50.0, 3)  # distant from a

    band_sizes = {0: 1, 1: 1, 3: 1}
    max_band_size = 1

    gain_adj = _compute_information_gain(
        entry_a=a, entry_b=b, band_sizes=band_sizes, max_band_size=max_band_size,
    )
    gain_dist = _compute_information_gain(
        entry_a=a, entry_b=c, band_sizes=band_sizes, max_band_size=max_band_size,
    )

    assert gain_adj > gain_dist, (
        f"Adjacent-band gain ({gain_adj}) should exceed distant-band gain ({gain_dist})"
    )


# ---------------------------------------------------------------------------
# build_comparison_queue
# ---------------------------------------------------------------------------


def test_queue_empty_for_single_entry():
    rankings = {"result": [_make_ranking_entry("a", 1, 100.0, 0)]}
    queue = build_comparison_queue(rankings=rankings)
    assert queue == []


def test_queue_single_pair():
    rankings = {"result": [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
    ]}
    queue = build_comparison_queue(rankings=rankings)
    assert len(queue) == 1
    assert queue[0].dimension == "result"
    assert {queue[0].run_a, queue[0].run_b} == {"a", "b"}


def test_queue_intra_band_before_cross_band():
    """Pairs within the same band must sort before cross-band pairs."""
    # Band 0: a, b (intra-band)  Band 1: c
    rankings = {"result": [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 95.0, 0),
        _make_ranking_entry("c", 3, 50.0, 1),
    ]}
    queue = build_comparison_queue(rankings=rankings)

    # (a,b) should be first (intra-band), then (a,c) and (b,c) (cross-band)
    intra_pair = {"a", "b"}
    first_pair = {queue[0].run_a, queue[0].run_b}
    assert first_pair == intra_pair, (
        f"First pair should be the intra-band pair (a,b), got {first_pair}"
    )


def test_queue_larger_band_first():
    """Larger intra-band groups should be prioritized."""
    # Band 0: a (size 1), Band 1: b, c (size 2)
    rankings = {"result": [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
        _make_ranking_entry("c", 3, 78.0, 1),
    ]}
    queue = build_comparison_queue(rankings=rankings)

    # (b,c) should be first (intra-band pair in the largest band), then cross-band
    first_pair = {queue[0].run_a, queue[0].run_b}
    assert first_pair == {"b", "c"}, (
        f"First pair should be the larger-band intra pair (b,c), got {first_pair}"
    )


def test_queue_tie_deterministic():
    """Equal uncertainty should produce deterministic ordering."""
    # All bands size 1 → no intra-band pairs, all cross-band
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
        _make_ranking_entry("c", 3, 60.0, 2),
    ]
    rankings = {"result": entries}

    q1 = build_comparison_queue(rankings=rankings)
    q2 = build_comparison_queue(rankings=rankings)

    # Same input → same output
    assert len(q1) == len(q2)
    for c1, c2 in zip(q1, q2):
        assert c1.run_a == c2.run_a
        assert c1.run_b == c2.run_b
        assert c1.gain == c2.gain


def test_queue_adjacent_bands_before_distant():
    """Adjacent-band pairs should have higher priority than distant-band pairs."""
    # 4 runs, all in separate bands
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
        _make_ranking_entry("c", 3, 60.0, 2),
        _make_ranking_entry("d", 4, 40.0, 3),
    ]
    rankings = {"result": entries}
    queue = build_comparison_queue(rankings=rankings)

    # Adjacent pairs (0-1, 1-2, 2-3) should precede distant pairs (0-2, 0-3, 1-3)
    adjacent_pairs = [{"a", "b"}, {"b", "c"}, {"c", "d"}]
    top_three = [
        {queue[0].run_a, queue[0].run_b},
        {queue[1].run_a, queue[1].run_b},
        {queue[2].run_a, queue[2].run_b},
    ]
    for pair in adjacent_pairs:
        assert pair in top_three, f"Adjacent pair {pair} should be in top three, got {top_three}"


def test_queue_median_fallback():
    """When all bands are size 1, median-relative pairs get priority."""
    # 5 runs, all separate bands
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
        _make_ranking_entry("c", 3, 60.0, 2),  # median (rank 3 of 5)
        _make_ranking_entry("d", 4, 40.0, 3),
        _make_ranking_entry("e", 5, 20.0, 4),
    ]
    rankings = {"result": entries}
    queue = build_comparison_queue(rankings=rankings)

    # The first comparison should involve the median (c)
    first_pair = {queue[0].run_a, queue[0].run_b}
    assert "c" in first_pair, (
        f"First pair should involve median 'c', got {first_pair}"
    )


def test_queue_excludes_existing_pairs():
    """Pairs in existing_pairs should be excluded from the queue."""
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 95.0, 0),
        _make_ranking_entry("c", 3, 50.0, 1),
    ]
    rankings = {"result": entries}
    existing = {("a", "b", "result")}
    queue = build_comparison_queue(rankings=rankings, existing_pairs=existing)

    # (a,b) should be excluded; only (a,c) and (b,c) remain
    for candidate in queue:
        pair = {candidate.run_a, candidate.run_b}
        assert pair != {"a", "b"}, "Existing pair (a,b) should be excluded"


def test_queue_multiple_dimensions():
    """Queue should interleave candidates from different dimensions."""
    rankings = {
        "result": [
            _make_ranking_entry("a", 1, 100.0, 0),
            _make_ranking_entry("b", 2, 80.0, 1),
        ],
        "process": [
            _make_ranking_entry("a", 1, 100.0, 0),
            _make_ranking_entry("b", 2, 95.0, 0),  # intra-band!
        ],
    }
    queue = build_comparison_queue(rankings=rankings)

    # The process intra-band pair should come first (higher gain)
    first = queue[0]
    assert first.dimension == "process", (
        f"First comparison should be process (intra-band), got {first.dimension}"
    )


def test_queue_sorted_by_dimension():
    """Queue should sort by dimension name as last tie-breaker."""
    entries = [
        _make_ranking_entry("a", 1, 100.0, 0),
        _make_ranking_entry("b", 2, 80.0, 1),
    ]
    # Two dimensions with exactly the same pair structure
    rankings = {
        "process": entries,
        "result": entries,
    }
    queue = build_comparison_queue(rankings=rankings)

    # "process" < "result" alphabetically, so process should come first
    assert queue[0].dimension == "process", (
        f"process should sort before result, got {queue[0].dimension}"
    )


# ---------------------------------------------------------------------------
# Comparison queue persistence round-trip
# ---------------------------------------------------------------------------


def test_queue_round_trip(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)

    queue = [
        ComparisonCandidate(
            dimension="result", run_a="a", run_b="b", gain=0.75,
            band_a=0, band_b=0, reason="Intra-band uncertainty in band 0 (3 runs tied)",
        ),
        ComparisonCandidate(
            dimension="result", run_a="a", run_b="c", gain=0.25,
            band_a=0, band_b=1, reason="Adjacent bands 0 and 1 — boundary may shift",
        ),
    ]
    write_comparison_queue(case, queue)
    loaded = read_comparison_queue(case)

    assert len(loaded) == 2
    assert loaded[0].dimension == "result"
    assert loaded[0].run_a == "a"
    assert loaded[0].run_b == "b"
    assert loaded[0].gain == 0.75
    assert loaded[0].band_a == 0
    assert loaded[0].band_b == 0
    assert loaded[1].gain == 0.25
    assert loaded[1].reason == "Adjacent bands 0 and 1 — boundary may shift"


def test_queue_read_empty_when_missing(tmp_path):
    case = tmp_path / "case-1"
    assert read_comparison_queue(case) == []


def test_queue_file_in_correct_location(tmp_path):
    case = tmp_path / "case-1"
    init_evaluation_dir(case)
    queue = [ComparisonCandidate(
        dimension="result", run_a="a", run_b="b", gain=1.0,
        band_a=0, band_b=0, reason="test",
    )]
    write_comparison_queue(case, queue)
    assert (case / "evaluation" / "current" / "comparisons" / "_queue.yaml").exists()


# ---------------------------------------------------------------------------
# recompute writes comparison queue
# ---------------------------------------------------------------------------


def test_recompute_writes_comparison_queue(tmp_path):
    """After recompute(), the comparison queue should be persisted."""
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-a", result=100.0, process=50.0),
        _make_run("run-b", result=97.0, process=45.0),
        _make_run("run-c", result=50.0, process=80.0),
    ]
    recompute(case_root=case, runs=runs, reason="queue test")

    queue = read_comparison_queue(case)
    assert len(queue) > 0, "Queue should not be empty after recompute"

    # Intra-band pair for result (run-a and run-b within 5%) should be first
    # since they share a band
    intra_found = False
    for c in queue:
        if {c.run_a, c.run_b} == {"run-a", "run-b"} and c.dimension == "result":
            intra_found = True
            break
    assert intra_found, "Intra-band pair (run-a, run-b) should be in the queue"


def test_recompute_queue_explains_unexpected_order(tmp_path):
    """High-gain comparisons should have reasons explaining their selection."""
    case = tmp_path / "case-1"
    runs = [
        _make_run("run-a", result=100.0, process=50.0),
        _make_run("run-b", result=97.0, process=48.0),
        _make_run("run-c", result=50.0, process=80.0),
    ]
    recompute(case_root=case, runs=runs, reason="explainability test")

    queue = read_comparison_queue(case)
    for c in queue:
        assert c.reason, f"Each candidate must have a reason; missing for {c.run_a} vs {c.run_b}"
        assert len(c.reason) > 5, f"Reason should be descriptive: {c.reason}"


def test_recompute_queue_deterministic_across_runs(tmp_path):
    """Same inputs in different case directories → same queue order."""
    runs = [
        _make_run("run-a", result=100.0, process=50.0),
        _make_run("run-b", result=97.0, process=48.0),
        _make_run("run-c", result=50.0, process=80.0),
    ]

    case1 = tmp_path / "case-1"
    case2 = tmp_path / "case-2"
    recompute(case_root=case1, runs=runs, reason="test 1")
    recompute(case_root=case2, runs=runs, reason="test 2")

    q1 = read_comparison_queue(case1)
    q2 = read_comparison_queue(case2)

    assert len(q1) == len(q2)
    for c1, c2 in zip(q1, q2):
        assert c1.dimension == c2.dimension
        assert {c1.run_a, c1.run_b} == {c2.run_a, c2.run_b}
        assert c1.gain == c2.gain


# ---------------------------------------------------------------------------
# Information-gain selection example
# ---------------------------------------------------------------------------


def test_queue_small_example():
    """Small example showing why one target was chosen over the median.

    Scenario: 3 runs, 2 share a band → intra-band comparison is chosen
    instead of comparing against the median because the band indicates
    uncertain ordering that needs resolution.
    """
    # run-a (100) and run-b (95) are in band 0 (within 5%).
    # run-c (50) is alone in band 1.
    entries = [
        _make_ranking_entry("run-a", 1, 100.0, 0),
        _make_ranking_entry("run-b", 2, 95.0, 0),
        _make_ranking_entry("run-c", 3, 50.0, 1),
    ]
    rankings = {"result": entries}
    queue = build_comparison_queue(rankings=rankings)

    first = queue[0]
    assert {first.run_a, first.run_b} == {"run-a", "run-b"}, (
        f"Chose {first.run_a} vs {first.run_b} over median comparison "
        f"because they share a band (uncertain ordering)"
    )
    # The median (run-b) comparison would be run-b vs run-c, but that's
    # a cross-band comparison with lower gain
    assert first.gain > 0.5, (
        f"Intra-band gain ({first.gain}) should be substantial"
    )


def test_queue_median_fallback_with_no_bands():
    """When all runs have separate bands, median-relative pairs are preferred."""
    entries = [
        _make_ranking_entry("run-a", 1, 100.0, 0),
        _make_ranking_entry("run-b", 2, 70.0, 1),
        _make_ranking_entry("run-c", 3, 50.0, 2),  # median
        _make_ranking_entry("run-d", 4, 30.0, 3),
        _make_ranking_entry("run-e", 5, 10.0, 4),
    ]
    rankings = {"result": entries}
    queue = build_comparison_queue(rankings=rankings)

    # With 5 runs all separate, median is run-c (band 2)
    # First comparison should involve run-c
    first_pair = {queue[0].run_a, queue[0].run_b}
    assert "run-c" in first_pair, (
        f"Median run-c should be in first comparison, got {first_pair}"
    )


# ============================================================================
# Structural contribution tests
# ============================================================================

from replay_checker.evaluation import (
    StructuralContribution,
    StructuralContributionLevel,
    STRUCTURAL_CONTRIBUTION_CAPS,
    apply_structural_cap,
    determine_eligibility,
    promote_tier_from_scores,
    _compute_incomparability_ratio,
    _confidence_from_runs_and_incomparability,
)


class TestStructuralContributionLevel:
    def test_three_sublevels_exist(self):
        assert StructuralContributionLevel.LOCAL_IMPROVEMENT.value == "local_improvement"
        assert StructuralContributionLevel.CLASS_DELETION.value == "class_deletion"
        assert StructuralContributionLevel.PROBLEM_REFRAMING.value == "problem_reframing"

    def test_all_levels_are_unique(self):
        values = [e.value for e in StructuralContributionLevel]
        assert len(values) == len(set(values))


class TestStructuralContribution:
    def test_default_zero(self):
        sc = StructuralContribution()
        assert sc.local_improvement == 0.0
        assert sc.class_deletion == 0.0
        assert sc.problem_reframing == 0.0
        assert sc.total == 0.0
        assert not sc.has_any

    def test_total_sums_sublevels(self):
        sc = StructuralContribution(
            local_improvement=5.0,
            class_deletion=3.0,
            problem_reframing=2.0,
        )
        assert sc.total == 10.0
        assert sc.has_any

    def test_has_any_false_when_all_zero(self):
        sc = StructuralContribution()
        assert not sc.has_any

    def test_has_any_true_when_any_nonzero(self):
        sc = StructuralContribution(local_improvement=0.1)
        assert sc.has_any

    def test_to_dict(self):
        sc = StructuralContribution(local_improvement=5.0, class_deletion=3.0)
        d = sc.to_dict()
        assert d == {
            "local_improvement": 5.0,
            "class_deletion": 3.0,
            "problem_reframing": 0.0,
        }

    def test_from_dict(self):
        sc = StructuralContribution.from_dict({
            "local_improvement": 5.0,
            "class_deletion": 3.0,
            "problem_reframing": 1.0,
        })
        assert sc.local_improvement == 5.0
        assert sc.class_deletion == 3.0
        assert sc.problem_reframing == 1.0

    def test_from_dict_defaults_missing_keys(self):
        sc = StructuralContribution.from_dict({})
        assert sc.local_improvement == 0.0
        assert sc.class_deletion == 0.0
        assert sc.problem_reframing == 0.0

    def test_round_trip(self):
        original = StructuralContribution(
            local_improvement=4.0,
            class_deletion=2.0,
            problem_reframing=3.0,
        )
        restored = StructuralContribution.from_dict(original.to_dict())
        assert restored == original


class TestEligibilityCaps:
    def test_invalid_capped_at_zero(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.INVALID] == 0.0

    def test_failed_capped_at_10(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.FAILED] == 10.0

    def test_partial_capped_at_30(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.PARTIAL] == 30.0

    def test_solved_has_no_cap(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.SOLVED] is None

    def test_excellent_has_no_cap(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.EXCELLENT] is None

    def test_transformative_has_no_cap(self):
        assert STRUCTURAL_CONTRIBUTION_CAPS[EligibilityTier.TRANSFORMATIVE] is None

    def test_all_tiers_have_cap_entries(self):
        for tier in EligibilityTier:
            assert tier in STRUCTURAL_CONTRIBUTION_CAPS, f"{tier} missing from caps"


class TestDetermineEligibility:
    def test_invalid_when_no_diff_no_completion(self):
        tier = determine_eligibility(
            run_status="missing-completion-report",
            has_diff=False,
        )
        assert tier == EligibilityTier.INVALID

    def test_invalid_when_explicit_invalid_status(self):
        tier = determine_eligibility(run_status="invalid")
        assert tier == EligibilityTier.INVALID

    def test_invalid_when_many_gate_failures(self):
        tier = determine_eligibility(
            run_status="completed",
            gate_failures=3,
            has_diff=True,
        )
        assert tier == EligibilityTier.INVALID

    def test_failed_when_no_diff(self):
        tier = determine_eligibility(
            run_status="completed",
            has_diff=False,
        )
        assert tier == EligibilityTier.FAILED

    def test_failed_when_explicit_failure(self):
        tier = determine_eligibility(
            run_status="failed",
            has_diff=True,
        )
        assert tier == EligibilityTier.FAILED

    def test_failed_when_missing_completion_but_has_diff(self):
        tier = determine_eligibility(
            run_status="missing-completion-report",
            has_diff=True,
        )
        assert tier == EligibilityTier.FAILED

    def test_partial_when_missing_fields(self):
        tier = determine_eligibility(
            run_status="completed",
            has_diff=True,
            missing_fields=["evidence.yaml"],
        )
        assert tier == EligibilityTier.PARTIAL

    def test_partial_when_reported_only(self):
        tier = determine_eligibility(
            run_status="reported",
            has_diff=True,
        )
        assert tier == EligibilityTier.PARTIAL

    def test_solved_when_all_evidence_present(self):
        tier = determine_eligibility(
            run_status="completed",
            has_diff=True,
        )
        assert tier == EligibilityTier.SOLVED

    def test_solved_when_solved_status(self):
        tier = determine_eligibility(
            run_status="solved",
            has_diff=True,
        )
        assert tier == EligibilityTier.SOLVED

    def test_partial_not_promoted_for_missing_fields_even_with_completion(self):
        tier = determine_eligibility(
            run_status="completed",
            has_diff=True,
            missing_fields=["verification"],
        )
        assert tier == EligibilityTier.PARTIAL


class TestApplyStructuralCap:
    def test_no_cap_for_solved_tier(self):
        sc = StructuralContribution(local_improvement=20.0, class_deletion=15.0)
        result = apply_structural_cap(sc, EligibilityTier.SOLVED)
        assert result.local_improvement == 20.0
        assert result.class_deletion == 15.0

    def test_no_cap_for_excellent_tier(self):
        sc = StructuralContribution(problem_reframing=50.0)
        result = apply_structural_cap(sc, EligibilityTier.EXCELLENT)
        assert result.problem_reframing == 50.0

    def test_cap_applied_to_failed_tier(self):
        sc = StructuralContribution(local_improvement=5.0, class_deletion=5.0, problem_reframing=10.0)
        result = apply_structural_cap(sc, EligibilityTier.FAILED)
        assert result.total <= 10.0

    def test_cap_applied_to_partial_tier(self):
        sc = StructuralContribution(local_improvement=20.0, class_deletion=20.0)
        result = apply_structural_cap(sc, EligibilityTier.PARTIAL)
        assert result.total <= 30.0

    def test_cap_zero_for_invalid(self):
        sc = StructuralContribution(local_improvement=5.0)
        result = apply_structural_cap(sc, EligibilityTier.INVALID)
        assert result.total == 0.0

    def test_below_cap_not_scaled(self):
        sc = StructuralContribution(local_improvement=3.0)
        result = apply_structural_cap(sc, EligibilityTier.FAILED)
        assert result.local_improvement == 3.0  # unchanged

    def test_exactly_at_cap_not_scaled(self):
        sc = StructuralContribution(local_improvement=10.0)
        result = apply_structural_cap(sc, EligibilityTier.FAILED)
        assert result.local_improvement == 10.0

    def test_proportional_scaling_preserves_ratios(self):
        sc = StructuralContribution(local_improvement=10.0, class_deletion=10.0)
        result = apply_structural_cap(sc, EligibilityTier.FAILED)  # cap = 10
        assert result.total == pytest.approx(10.0)
        assert result.local_improvement == pytest.approx(result.class_deletion)

    def test_zero_contribution_unchanged_by_cap(self):
        sc = StructuralContribution()
        result = apply_structural_cap(sc, EligibilityTier.INVALID)
        assert result.total == 0.0


class TestPromoteTierFromScores:
    def test_solved_promotes_to_excellent(self):
        tier = promote_tier_from_scores(
            EligibilityTier.SOLVED, total_score=80.0,
            structural_contribution=StructuralContribution(),
        )
        assert tier == EligibilityTier.EXCELLENT

    def test_solved_with_structural_promotes_to_transformative(self):
        tier = promote_tier_from_scores(
            EligibilityTier.SOLVED, total_score=92.0,
            structural_contribution=StructuralContribution(local_improvement=5.0),
        )
        assert tier == EligibilityTier.TRANSFORMATIVE

    def test_solved_stays_solved_with_low_score(self):
        tier = promote_tier_from_scores(
            EligibilityTier.SOLVED, total_score=60.0,
            structural_contribution=StructuralContribution(),
        )
        assert tier == EligibilityTier.SOLVED

    def test_solved_no_transformative_without_structural(self):
        tier = promote_tier_from_scores(
            EligibilityTier.SOLVED, total_score=95.0,
            structural_contribution=StructuralContribution(),
        )
        assert tier == EligibilityTier.EXCELLENT  # 95 >= 75 but < 90 for transformative

    def test_failed_cannot_be_promoted(self):
        """Eligibility cap is hard: FAILED runs cannot be promoted."""
        tier = promote_tier_from_scores(
            EligibilityTier.FAILED, total_score=95.0,
            structural_contribution=StructuralContribution(local_improvement=5.0),
        )
        assert tier == EligibilityTier.FAILED

    def test_partial_cannot_be_promoted(self):
        """Eligibility cap is hard: PARTIAL runs cannot be promoted."""
        tier = promote_tier_from_scores(
            EligibilityTier.PARTIAL, total_score=90.0,
            structural_contribution=StructuralContribution(problem_reframing=10.0),
        )
        assert tier == EligibilityTier.PARTIAL

    def test_invalid_cannot_be_promoted(self):
        tier = promote_tier_from_scores(
            EligibilityTier.INVALID, total_score=100.0,
            structural_contribution=StructuralContribution(local_improvement=50.0),
        )
        assert tier == EligibilityTier.INVALID


class TestIncomparabilityRatio:
    def test_empty_comparisons_zero(self):
        assert _compute_incomparability_ratio([]) == 0.0

    def test_no_incomparable_zero(self):
        comps = [
            PairwiseComparison(dimension="result", run_a="a", run_b="b",
                               label=ComparisonLabel.BETTER),
            PairwiseComparison(dimension="result", run_a="a", run_b="c",
                               label=ComparisonLabel.TIE),
        ]
        assert _compute_incomparability_ratio(comps) == 0.0

    def test_all_incomparable_one(self):
        comps = [
            PairwiseComparison(dimension="result", run_a="a", run_b="b",
                               label=ComparisonLabel.INCOMPARABLE),
            PairwiseComparison(dimension="result", run_a="c", run_b="d",
                               label=ComparisonLabel.INCOMPARABLE),
        ]
        assert _compute_incomparability_ratio(comps) == 1.0

    def test_mixed_ratio(self):
        comps = [
            PairwiseComparison(dimension="result", run_a="a", run_b="b",
                               label=ComparisonLabel.BETTER),
            PairwiseComparison(dimension="result", run_a="a", run_b="c",
                               label=ComparisonLabel.INCOMPARABLE),
            PairwiseComparison(dimension="result", run_a="b", run_b="c",
                               label=ComparisonLabel.TIE),
            PairwiseComparison(dimension="result", run_a="a", run_b="d",
                               label=ComparisonLabel.INCOMPARABLE),
        ]
        assert _compute_incomparability_ratio(comps) == 0.5


class TestConfidenceFromRunsAndIncomparability:
    def test_high_incomparability_low_confidence(self):
        assert _confidence_from_runs_and_incomparability(3, 0.6) == "low"

    def test_medium_incomparability_with_enough_runs(self):
        assert _confidence_from_runs_and_incomparability(3, 0.3) == "medium"

    def test_medium_incomparability_few_runs_low(self):
        assert _confidence_from_runs_and_incomparability(2, 0.3) == "low"

    def test_low_incomparability_enough_runs_medium(self):
        assert _confidence_from_runs_and_incomparability(3, 0.0) == "medium"

    def test_low_incomparability_few_runs_low(self):
        assert _confidence_from_runs_and_incomparability(2, 0.0) == "low"


class TestIncomparablePreserved:
    """Incomparable must never be silently converted to a tie."""

    def test_incomparable_label_exists(self):
        assert ComparisonLabel.INCOMPARABLE.value == "incomparable"

    def test_incomparable_is_not_tie(self):
        assert ComparisonLabel.INCOMPARABLE != ComparisonLabel.TIE

    def test_all_incomparable_reasons_exist(self):
        expected = {
            "different_strategy",
            "different_risk_profile",
            "insufficient_evidence",
            "task_ambiguity",
            "depends_on_user_lens",
        }
        actual = {r.value for r in IncomparableReason}
        assert actual == expected

    def test_pairwise_comparison_stores_reasons(self):
        comp = PairwiseComparison(
            dimension="result",
            run_a="a",
            run_b="b",
            label=ComparisonLabel.INCOMPARABLE,
            incomparable_reasons=[
                IncomparableReason.DIFFERENT_STRATEGY,
                IncomparableReason.DEPENDS_ON_USER_LENS,
            ],
        )
        assert len(comp.incomparable_reasons) == 2
        assert IncomparableReason.DIFFERENT_STRATEGY in comp.incomparable_reasons
        assert IncomparableReason.DEPENDS_ON_USER_LENS in comp.incomparable_reasons


class TestRecomputeWithStructuralContribution:
    """Integration: recompute with structural contribution caps."""

    def test_failed_run_structural_capped(self, tmp_path):
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        failed_run = RunAttempt(
            run_id="run-failed",
            status="failed",
            tier=EligibilityTier.FAILED,
            dimension_scores={"result": 50.0, "process": 10.0},
            structural_contribution=StructuralContribution(
                local_improvement=5.0,
                class_deletion=10.0,
                problem_reframing=5.0,  # total 20, capped at 10
            ),
        )

        solved_run = RunAttempt(
            run_id="run-solved",
            status="completed",
            tier=EligibilityTier.SOLVED,
            dimension_scores={"result": 80.0, "process": 15.0},
            structural_contribution=StructuralContribution(
                local_improvement=5.0,
            ),
        )

        recompute(case_root=case_root, runs=[failed_run, solved_run], reason="test")

        summary = read_summary(case_root)
        assert summary.run_count == 2
        # The solved run should be best because its structural contribution
        # isn't capped and its base scores are higher
        assert summary.best_run_id == "run-solved"

        # Verify the failed run's structural contribution was capped
        attempts = read_attempts(case_root)
        failed_attempt = next(a for a in attempts if a.run_id == "run-failed")
        assert failed_attempt.tier == EligibilityTier.FAILED  # not promoted
        solved_attempt = next(a for a in attempts if a.run_id == "run-solved")
        assert solved_attempt.tier in (EligibilityTier.SOLVED, EligibilityTier.EXCELLENT)

    def test_partial_run_cannot_outrank_solved(self, tmp_path):
        """Structural contribution cannot lift a partial run above a solved one.

        Both runs have the same base dimension scores and same structural
        contribution, but the partial run's contribution is capped at 30
        while the solved run's is uncapped.
        """
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        structural = StructuralContribution(
            local_improvement=15.0,
            class_deletion=15.0,
            problem_reframing=15.0,  # total 45
        )

        partial_run = RunAttempt(
            run_id="run-partial",
            status="completed",
            tier=EligibilityTier.PARTIAL,
            dimension_scores={"result": 70.0, "process": 15.0},
            structural_contribution=structural,
        )

        solved_run = RunAttempt(
            run_id="run-solved",
            status="completed",
            tier=EligibilityTier.SOLVED,
            dimension_scores={"result": 70.0, "process": 15.0},
            structural_contribution=structural,  # same, but uncapped for SOLVED
        )

        recompute(case_root=case_root, runs=[partial_run, solved_run], reason="test")

        summary = read_summary(case_root)
        # Solved should win: same base score, same structural contribution,
        # but partial is capped at 30 while solved is uncapped (45 bonus)
        assert summary.best_run_id == "run-solved"

    def test_invalid_run_gets_no_score(self, tmp_path):
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        invalid_run = RunAttempt(
            run_id="run-invalid",
            status="invalid",
            tier=EligibilityTier.INVALID,
            dimension_scores={"result": 90.0, "process": 20.0},
            structural_contribution=StructuralContribution(problem_reframing=50.0),
        )

        valid_run = RunAttempt(
            run_id="run-valid",
            status="completed",
            tier=EligibilityTier.SOLVED,
            dimension_scores={"result": 50.0, "process": 10.0},
        )

        recompute(case_root=case_root, runs=[invalid_run, valid_run], reason="test")

        summary = read_summary(case_root)
        assert summary.best_run_id == "run-valid"

    def test_structural_sublevels_in_attempts_output(self, tmp_path):
        """Detailed artifacts show structural sublevels."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        run = RunAttempt(
            run_id="run-a",
            status="completed",
            tier=EligibilityTier.SOLVED,
            dimension_scores={"result": 75.0, "process": 15.0},
            structural_contribution=StructuralContribution(
                local_improvement=3.0,
                class_deletion=2.0,
                problem_reframing=1.0,
            ),
        )

        recompute(case_root=case_root, runs=[run], reason="test")

        # Read attempts and verify structural sublevels are present
        attempts = read_attempts(case_root)
        assert len(attempts) == 1
        sc = attempts[0].structural_contribution
        assert sc.local_improvement == 3.0
        assert sc.class_deletion == 2.0
        assert sc.problem_reframing == 1.0

        # Also verify the YAML file contains structural fields
        attempts_yaml = (current_dir(case_root) / "attempts.yaml").read_text("utf-8")
        assert "structural_local_improvement" in attempts_yaml
        assert "structural_class_deletion" in attempts_yaml
        assert "structural_problem_reframing" in attempts_yaml

    def test_incomparable_in_comparison_output(self, tmp_path):
        """Comparisons include incomparable reasons."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        complete_run = RunAttempt(
            run_id="run-complete",
            status="completed",
            tier=EligibilityTier.SOLVED,
            dimension_scores={"result": 80.0, "process": 15.0},
        )
        incomplete_run = RunAttempt(
            run_id="run-incomplete",
            status="missing-completion-report",
            tier=EligibilityTier.FAILED,
            dimension_scores={"result": 60.0, "process": 5.0},
        )

        recompute(case_root=case_root, runs=[complete_run, incomplete_run], reason="test")

        comp = read_comparison(case_root, "run-complete", "run-incomplete")
        assert comp.label == ComparisonLabel.INCOMPARABLE
        assert IncomparableReason.INSUFFICIENT_EVIDENCE in comp.incomparable_reasons

    def test_incomparability_reduces_confidence(self, tmp_path):
        """High incomparability ratio should reduce confidence to low."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        # All runs incomplete → all comparisons incomparable
        runs = []
        for i in range(3):
            runs.append(RunAttempt(
                run_id=f"run-{i}",
                status="missing-completion-report",
                tier=EligibilityTier.FAILED,
                dimension_scores={"result": 30.0, "process": 5.0},
            ))

        recompute(case_root=case_root, runs=runs, reason="test")
        summary = read_summary(case_root)
        # With 3 runs but all incomplete, confidence should be low
        assert summary.confidence == "low"

    def test_low_incomparability_with_multiple_runs_medium_confidence(self, tmp_path):
        """Multiple complete runs with low incomparability → medium confidence."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        init_evaluation_dir(case_root)

        runs = []
        for i in range(3):
            runs.append(RunAttempt(
                run_id=f"run-{i}",
                status="completed",
                tier=EligibilityTier.SOLVED,
                dimension_scores={
                    "result": 70.0 + i * 10.0,
                    "process": 15.0,
                },
            ))

        recompute(case_root=case_root, runs=runs, reason="test")
        summary = read_summary(case_root)
        # All runs complete → no incomparable comparisons → medium confidence
        assert summary.confidence == "medium"

    def test_comparison_with_different_strategies_preserved(self, tmp_path):
        """Incomparable with different_strategy reason must be preserved in
        the comparison file."""
        case_root = tmp_path / "test-case"
        case_root.mkdir()
        cur = current_dir(case_root)
        comps_dir = cur / "comparisons"
        comps_dir.mkdir(parents=True, exist_ok=True)

        comp = PairwiseComparison(
            dimension="result",
            run_a="run-a",
            run_b="run-b",
            label=ComparisonLabel.INCOMPARABLE,
            evidence="Strategies differ fundamentally",
            incomparable_reasons=[IncomparableReason.DIFFERENT_STRATEGY],
        )
        write_comparison(case_root, comp)
        restored = read_comparison(case_root, "run-a", "run-b")
        assert restored.label == ComparisonLabel.INCOMPARABLE
        assert IncomparableReason.DIFFERENT_STRATEGY in restored.incomparable_reasons
        assert "Strategies differ fundamentally" in restored.evidence


# ============================================================================
# Regression: Anonymity
# ============================================================================

from replay_checker.scoring import (
    EvidenceGate,
    evaluate_evidence_gates,
    compute_score_ceilings,
)


class TestAnonymityRegression:
    """Runner identity must never leak into scoring packages."""

    def test_runner_label_in_evidence_marks_invalid(self, tmp_path):
        """When the runner label appears in evidence.yaml, score must be invalid."""
        run_root = tmp_path / "run-exposed"
        run_root.mkdir()
        ev_dir = run_root / "evidence"
        ev_dir.mkdir()
        (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (ev_dir / "evidence.yaml").write_text(
            "run_id: run-exposed\nrunner: gpt-5-codex\nstatus: completed\n",
            encoding="utf-8",
        )
        (run_root / "completion_report.md").write_text(
            "Status: completed\nChanges: x\n", encoding="utf-8"
        )

        result = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id="run-exposed",
            runner_label="gpt-5-codex",
            changed_files=["x"],
        )

        identity_gate = [r for r in result.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN]
        assert len(identity_gate) == 1
        assert not identity_gate[0].passed, "identity gate should fail when label is exposed"

        ceilings = compute_score_ceilings(result)
        assert ceilings.overall_invalid, "score must be invalid when runner identity is exposed"

    def test_runner_label_absent_passes_gate(self, tmp_path):
        """When runner label is not in any evidence file, gate should pass."""
        run_root = tmp_path / "run-hidden"
        run_root.mkdir()
        ev_dir = run_root / "evidence"
        ev_dir.mkdir()
        (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (ev_dir / "evidence.yaml").write_text(
            "run_id: run-hidden\nstatus: completed\nchanged_files:\n  - x\n",
            encoding="utf-8",
        )
        (run_root / "completion_report.md").write_text(
            "Status: completed\nChanges: x\n", encoding="utf-8"
        )

        result = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id="run-hidden",
            runner_label="gpt-5-codex",
            changed_files=["x"],
        )

        identity_gate = [r for r in result.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN]
        assert len(identity_gate) == 1
        assert identity_gate[0].passed

    def test_anonymous_id_preserved_through_round_trip(self, tmp_path):
        """RunAttempt.anonymous_id must survive YAML serialization."""
        case = tmp_path / "case-1"
        attempts = [
            RunAttempt(
                run_id="run-abc",
                anonymous_id="anon-001",
                status="completed",
                tier=EligibilityTier.SOLVED,
                total_score=72.0,
            ),
        ]
        write_attempts(case, attempts)
        loaded = read_attempts(case)
        assert loaded[0].anonymous_id == "anon-001"
        assert loaded[0].run_id == "run-abc"


# ============================================================================
# Regression: Reference isolation
# ============================================================================


class TestReferenceIsolationRegression:
    """_reference/ directory presence must invalidate scores."""

    def test_reference_dir_marks_invalid(self, tmp_path):
        """If _reference/ exists in run root, score must be marked invalid."""
        run_root = tmp_path / "run-leaked"
        run_root.mkdir()
        ev_dir = run_root / "evidence"
        ev_dir.mkdir()
        (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (ev_dir / "evidence.yaml").write_text(
            "run_id: run-leaked\nstatus: completed\nchanged_files:\n  - x\n",
            encoding="utf-8",
        )
        (run_root / "completion_report.md").write_text(
            "Status: completed\nChanges: x\n", encoding="utf-8"
        )
        # Create _reference/ directory (oracle material)
        (run_root / "_reference").mkdir()
        (run_root / "_reference" / "target_diff.patch").write_text(
            "oracle diff\n", encoding="utf-8"
        )

        result = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id="run-leaked",
            runner_label="anonymous",
            changed_files=["x"],
        )

        ref_gate = [r for r in result.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT]
        assert len(ref_gate) == 1
        assert not ref_gate[0].passed
        assert "_reference/" in ref_gate[0].detail

        assert result.is_invalid
        assert any("_reference" in reason for reason in result.invalid_reasons)

        ceilings = compute_score_ceilings(result)
        assert ceilings.overall_invalid

    def test_no_reference_dir_passes_gate(self, tmp_path):
        """Without _reference/ and no oracle mentions, gate should pass."""
        run_root = tmp_path / "run-clean"
        run_root.mkdir()
        ev_dir = run_root / "evidence"
        ev_dir.mkdir()
        (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (ev_dir / "evidence.yaml").write_text(
            "run_id: run-clean\nstatus: completed\nchanged_files:\n  - x\n",
            encoding="utf-8",
        )
        (run_root / "completion_report.md").write_text(
            "Status: completed\nChanges: x\n", encoding="utf-8"
        )

        result = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id="run-clean",
            runner_label="anonymous",
            changed_files=["x"],
        )

        ref_gate = [r for r in result.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT]
        assert len(ref_gate) == 1
        assert ref_gate[0].passed
        assert not result.is_invalid

    def test_oracle_mention_in_completion_report_marks_invalid(self, tmp_path):
        """Mentioning 'oracle' in completion report should invalidate the score."""
        run_root = tmp_path / "run-oracle-mention"
        run_root.mkdir()
        ev_dir = run_root / "evidence"
        ev_dir.mkdir()
        (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
        (ev_dir / "evidence.yaml").write_text(
            "run_id: run-oracle-mention\nstatus: completed\nchanged_files:\n  - x\n",
            encoding="utf-8",
        )
        (run_root / "completion_report.md").write_text(
            "Status: completed\nI compared against the oracle solution and found...\n",
            encoding="utf-8",
        )

        result = evaluate_evidence_gates(
            evidence_root=run_root,
            run_id="run-oracle-mention",
            runner_label="anonymous",
            changed_files=["x"],
        )

        assert result.is_invalid
        assert any("oracle" in reason.lower() for reason in result.invalid_reasons)


# ============================================================================
# Regression: Deterministic recomputation (strengthened)
# ============================================================================


class TestDeterministicRecomputationRegression:
    """Recomputation must be deterministic and full, not incremental."""

    def test_same_inputs_produce_identical_total_scores(self, tmp_path):
        """Two separate recomputations with identical inputs must produce identical total scores."""
        runs = [
            _make_run("r1", result=75.0, process=20.0),
            _make_run("r2", result=60.0, process=18.0),
            _make_run("r3", result=45.0, process=12.0),
        ]

        case_a = tmp_path / "case-a"
        case_b = tmp_path / "case-b"
        recompute(case_root=case_a, runs=runs, reason="a")
        recompute(case_root=case_b, runs=runs, reason="b")

        s_a = read_summary(case_a)
        s_b = read_summary(case_b)
        assert s_a.total_score == s_b.total_score
        assert s_a.best_run_id == s_b.best_run_id
        assert s_a.tier == s_b.tier

    def test_run_order_does_not_affect_result(self, tmp_path):
        """Runs provided in different order must produce the same evaluation (sorted internally)."""
        runs_forward = [
            _make_run("r1", result=80.0, process=15.0),
            _make_run("r2", result=60.0, process=20.0),
        ]
        runs_reverse = list(reversed(runs_forward))

        case_f = tmp_path / "case-fwd"
        case_r = tmp_path / "case-rev"
        recompute(case_root=case_f, runs=runs_forward, reason="fwd")
        recompute(case_root=case_r, runs=runs_reverse, reason="rev")

        s_f = read_summary(case_f)
        s_r = read_summary(case_r)
        assert s_f.total_score == s_r.total_score
        assert s_f.best_run_id == s_r.best_run_id

    def test_recompute_snapshots_before_overwriting(self, tmp_path):
        """Each recomputation after the first must snapshot the previous current/.

        Note: snapshots use a second-precision UTC timestamp, so multiple
        recomputations within the same second may share a snapshot path.
        We verify at least one snapshot exists after multiple recomputations.
        """
        case = tmp_path / "case-1"
        recompute(case_root=case, runs=[_make_run("r1", result=30.0)], reason="v1")
        recompute(case_root=case, runs=[
            _make_run("r1", result=30.0),
            _make_run("r2", result=50.0),
        ], reason="v2")
        recompute(case_root=case, runs=[
            _make_run("r1", result=30.0),
            _make_run("r2", result=50.0),
            _make_run("r3", result=70.0),
        ], reason="v3")

        snap_dir = snapshots_dir(case)
        assert snap_dir.exists()
        snapshots = list(snap_dir.iterdir())
        assert len(snapshots) >= 1, "at least one snapshot should exist after recomputation"

        # Each snapshot should have a summary.yaml
        for snap in snapshots:
            assert (snap / "summary.yaml").exists()


# ============================================================================
# Regression: No iteration/cost score contribution (strengthened)
# ============================================================================


class TestNoCostScoreContributionRegression:
    """Cost, token, time, and iteration must never contribute to score."""

    def test_evaluation_summary_has_no_cost_fields(self):
        """EvaluationSummary must not have any cost/token/time/iteration fields."""
        from replay_checker.evaluation import EvaluationSummary
        s = EvaluationSummary(case_id="x")
        forbidden_attrs = [
            "cost_score", "token_score", "time_score", "iteration_score",
            "wall_time_score", "retry_score", "budget_score",
            "first_pass_score", "attempt_score",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(s, attr), f" EvaluationSummary must not have '{attr}'"

    def test_run_attempt_has_no_cost_fields(self):
        """RunAttempt must not have cost/token/iteration score fields."""
        a = RunAttempt(run_id="x")
        forbidden_attrs = [
            "cost_score", "token_score", "iteration_score", "time_score",
            "wall_time_score", "retry_count", "first_pass",
        ]
        for attr in forbidden_attrs:
            assert not hasattr(a, attr), f"RunAttempt must not have '{attr}'"

    def test_recompute_score_independent_of_attempt_count(self, tmp_path):
        """Adding more identical runs must not inflate the total score."""
        case_a = tmp_path / "case-1run"
        case_b = tmp_path / "case-5runs"

        one_run = [_make_run("r1", result=60.0, process=15.0)]
        five_runs = [
            _make_run(f"r{i}", result=60.0, process=15.0) for i in range(5)
        ]

        recompute(case_root=case_a, runs=one_run, reason="1 run")
        recompute(case_root=case_b, runs=five_runs, reason="5 runs")

        s1 = read_summary(case_a)
        s5 = read_summary(case_b)
        # Best score must be the same regardless of how many runs exist
        assert s1.total_score == s5.total_score
