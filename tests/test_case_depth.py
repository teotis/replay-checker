from __future__ import annotations

from pathlib import Path

from replay_checker.case_depth import (
    build_episode_profile,
    build_situation_profile,
)
from replay_checker.candidates import build_case_candidates
from replay_checker.sources import EvidenceRecord


def test_situation_profile_extracts_context_constraints_and_failure_boundaries(tmp_path: Path) -> None:
    records = [
        EvidenceRecord(
            source_type="plan",
            path=tmp_path / "plan.md",
            summary="Fix payment retry bug with observable acceptance",
            metadata={"mentioned_files": "src/pay.py"},
            relevance_score=0.62,
            task_signal="plan_task",
        ),
        EvidenceRecord(
            source_type="codex_history",
            path=tmp_path / "session.jsonl",
            summary="User reports weak verification and blocked retry test in tests/test_pay.py",
            metadata={"mentioned_files": "tests/test_pay.py"},
            relevance_score=0.54,
            task_signal="conversation_task",
        ),
    ]

    profile = build_situation_profile(
        records,
        selected_candidate=None,
        reconstruction_context={
            "commit_subject": "fix: retry payment failure",
            "verification_hints": ["Test directory `tests/` exists in project"],
            "risk_notes": ["Commit message only — no detailed body available"],
            "scope_summary": "1 source file(s), 1 test file(s)",
        },
    )

    assert "fix: retry payment failure" in profile.problem_context
    assert "1 source file(s), 1 test file(s)" in profile.constraints
    assert "Commit message only" in profile.failure_boundaries
    assert "Test directory `tests/` exists" in profile.observable_acceptance
    assert profile.depth_level == "rich"
    assert profile.depth_score >= 70


def test_episode_profile_identifies_cross_source_file_support(tmp_path: Path) -> None:
    records = [
        EvidenceRecord(
            source_type="plan",
            path=tmp_path / "docs" / "plans" / "payment.md",
            summary="Fix payment retry bug",
            metadata={"mentioned_files": "src/pay.py"},
            relevance_score=0.61,
            task_signal="plan_task",
        ),
        EvidenceRecord(
            source_type="codex_history",
            path=tmp_path / "session.jsonl",
            summary="retry bug touches src/pay.py and tests/test_pay.py",
            metadata={"mentioned_files": "src/pay.py,tests/test_pay.py"},
            relevance_score=0.58,
            task_signal="conversation_task",
        ),
    ]

    episode = build_episode_profile(records)

    assert episode.label == "plan-history"
    assert episode.anchor_paths == ("src/pay.py",)
    assert "episode support: plan + codex_history on src/pay.py" in episode.ranking_reason


def test_episode_profile_does_not_claim_episode_without_shared_anchor(tmp_path: Path) -> None:
    records = [
        EvidenceRecord(
            source_type="plan",
            path=tmp_path / "plan.md",
            summary="Fix payment retry bug",
            metadata={"mentioned_files": "src/pay.py"},
        ),
        EvidenceRecord(
            source_type="codex_history",
            path=tmp_path / "session.jsonl",
            summary="Investigate unrelated scoring issue",
            metadata={"mentioned_files": "src/scoring.py"},
        ),
    ]

    episode = build_episode_profile(records)

    assert episode.label == "unlinked-multi-source"
    assert episode.anchor_paths == ()
    assert episode.ranking_bonus == 0.0
    assert episode.ranking_reason == ""


def test_situation_profile_uses_plan_goal_and_verification_commands(tmp_path: Path) -> None:
    records = [
        EvidenceRecord(
            source_type="plan",
            path=tmp_path / "plan.md",
            summary="Package 01 discovery",
            metadata={"mentioned_files": "src/intake.py"},
        ),
        EvidenceRecord(
            source_type="claude_history",
            path=tmp_path / "session.jsonl",
            summary="Review src/intake.py extraction behavior",
            metadata={"mentioned_files": "src/intake.py"},
        ),
    ]

    profile = build_situation_profile(
        records,
        selected_candidate=None,
        reconstruction_context={
            "plan_goal": "Extract replayable situations from a real project with bounded evidence.",
            "verification_commands": [
                "rtk python3 tools/replay.py intake --project /project",
                "rtk make test",
            ],
            "risk_notes": ["Candidate evidence may be too broad without source agreement."],
        },
    )

    assert "Extract replayable situations" in profile.problem_context
    assert "rtk python3 tools/replay.py intake" in profile.observable_acceptance
    assert "Candidate evidence may be too broad" in profile.failure_boundaries
    assert profile.depth_level == "rich"
    assert profile.depth_score >= 70


def test_situation_profile_ignores_version_numbers_as_paths(tmp_path: Path) -> None:
    records = [
        EvidenceRecord(
            source_type="plan",
            path=tmp_path / "plan.md",
            summary="Replay Checker 3.1 validation using FINAL_REPORT.md",
            metadata={"mentioned_files": "3.1,FINAL_REPORT.md,src/intake.py"},
        ),
    ]

    profile = build_situation_profile(
        records,
        selected_candidate=None,
        reconstruction_context={},
    )

    assert "FINAL_REPORT.md" in profile.constraints
    assert "src/intake.py" in profile.constraints
    assert "3.1" not in profile.constraints


def test_episode_support_can_lift_connected_candidate(tmp_path: Path) -> None:
    weak_plan = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "weak.md",
        summary="Implement unrelated cleanup",
        metadata={"relative_path": "docs/plans/weak.md"},
        relevance_score=0.59,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    episode_plan = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "docs" / "plans" / "episode.md",
        summary="Fix payment retry bug",
        metadata={
            "relative_path": "docs/plans/episode.md",
            "mentioned_files": "src/pay.py",
        },
        relevance_score=0.52,
        relevance_reasons=("plan doc",),
        task_signal="plan_task",
    )
    supporting_history = EvidenceRecord(
        source_type="codex_history",
        path=tmp_path / "session.jsonl",
        summary="retry failure and weak verification in src/pay.py",
        metadata={
            "relative_path": "sessions/session.jsonl",
            "mentioned_files": "src/pay.py",
        },
        relevance_score=0.42,
        relevance_reasons=("task action verbs in summary",),
        task_signal="conversation_task",
    )

    result = build_case_candidates(
        [weak_plan, episode_plan, supporting_history],
        project_path=tmp_path,
    )

    assert result[0].primary_source == "docs/plans/episode.md"
    assert any("episode support" in reason for reason in result[0].selection_reasons)
