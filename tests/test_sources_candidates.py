from __future__ import annotations

from pathlib import Path

import replay_checker.candidates as candidates_module
import replay_checker.sources as sources_module
from replay_checker.candidates import build_case_candidates
from replay_checker.outcome_feedback import CandidateSelectionFeedback
from replay_checker.sources import (
    EvidenceRecord,
    _iter_history_files,
    score_evidence_records,
)


def test_build_case_candidates_preserves_scored_zero_records(monkeypatch, tmp_path: Path) -> None:
    record = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "plan.md",
        summary="no actionable task signal",
        metadata={"relative_path": "docs/plans/plan.md"},
        relevance_score=0.0,
        relevance_reasons=("already scored as zero",),
        task_signal="plan_status_only",
    )

    def fail_if_rescored(*args, **kwargs):
        raise AssertionError("pre-scored records should not be scored again")

    monkeypatch.setattr(candidates_module, "score_evidence_records", fail_if_rescored)

    result = build_case_candidates([record], project_path=tmp_path)

    assert result[0].relevance_score == 0.0


def test_evidence_graph_boosts_candidates_with_cross_source_file_support(tmp_path: Path) -> None:
    weak_plan = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "weak.md",
        summary="Implement unrelated cleanup",
        metadata={"relative_path": "docs/plans/weak.md"},
        relevance_score=0.56,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    connected_plan = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "connected.md",
        summary="Implement app flow",
        metadata={
            "relative_path": "docs/plans/connected.md",
            "mentioned_files": "src/app.py",
        },
        relevance_score=0.53,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    supporting_history = EvidenceRecord(
        source_type="codex_history",
        path=tmp_path / "session.jsonl",
        summary="please implement the same app flow in src/app.py",
        metadata={
            "relative_path": "sessions/session.jsonl",
            "mentioned_files": "src/app.py",
        },
        relevance_score=0.20,
        relevance_reasons=("task action verbs in summary",),
        task_signal="conversation_task",
    )

    result = build_case_candidates(
        [weak_plan, connected_plan, supporting_history],
        project_path=tmp_path,
    )

    assert result[0].primary_source == "docs/plans/connected.md"
    assert result[0].relevance_score > weak_plan.relevance_score
    assert any("evidence graph" in reason for reason in result[0].selection_reasons)


def test_candidate_feedback_penalizes_prior_false_positive_source_path(tmp_path: Path) -> None:
    risky = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "risky.md",
        summary="Implement risky task",
        metadata={"relative_path": "docs/plans/risky.md"},
        relevance_score=0.60,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    safer = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "safer.md",
        summary="Implement safer task",
        metadata={"relative_path": "docs/plans/safer.md"},
        relevance_score=0.55,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    feedback = CandidateSelectionFeedback(
        source_path_penalties={"docs/plans/risky.md": 0.12},
        source_path_reasons={
            "docs/plans/risky.md": (
                "prior outcome feedback: false-positive completion",
            ),
        },
    )

    result = build_case_candidates(
        [risky, safer],
        project_path=tmp_path,
        candidate_feedback=feedback,
    )

    assert result[0].primary_source == "docs/plans/safer.md"
    risky_candidate = next(c for c in result if c.primary_source == "docs/plans/risky.md")
    assert risky_candidate.relevance_score == 0.48
    assert "prior outcome feedback: false-positive completion" in risky_candidate.selection_reasons
    assert "prior outcome feedback lowered this candidate" in risky_candidate.risks


def test_iter_history_files_honors_max_files(tmp_path: Path) -> None:
    root = tmp_path / "history"
    root.mkdir()
    for idx in range(3):
        (root / f"session-{idx}.jsonl").write_text("{}", encoding="utf-8")

    result = list(_iter_history_files([root], max_files=2))

    assert len(result) == 2


def test_score_evidence_records_reuses_project_file_inventory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    calls = 0

    def collect_once(project_path: Path) -> set[str]:
        nonlocal calls
        calls += 1
        return {"src/app.py"}

    monkeypatch.setattr(sources_module, "_collect_project_files", collect_once)
    records = [
        EvidenceRecord(
            source_type="codex_history",
            path=tmp_path / f"session-{idx}.jsonl",
            summary="please fix src/app.py",
            metadata={
                "matched_alias_type": "resolved_path",
                "mentioned_files": "src/app.py",
            },
        )
        for idx in range(2)
    ]

    scored = score_evidence_records(records, project_path=project)

    assert len(scored) == 2
    assert calls == 1
