from replay_checker.candidates import (
    CandidateCase,
    build_case_candidates,
    select_case_candidate,
)
from replay_checker.sources import EvidenceRecord

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _plan_record(tmp_path, name, summary, base_commit=None):
    meta = {"relative_path": f"docs/plans/{name}"}
    if base_commit:
        meta["base_commit"] = base_commit
    return EvidenceRecord(
        source_type="plan",
        path=tmp_path / name,
        summary=summary,
        confidence="high",
        metadata=meta,
    )


def _history_record(source_type, summary):
    return EvidenceRecord(
        source_type=source_type,
        path=Path("/fake/session.jsonl"),
        summary=summary,
        confidence="high",
        metadata={},
    )


def test_build_candidates_ranks_plan_above_history(tmp_path):
    records = [
        _plan_record(tmp_path, "task-plan.md", "Implement scoring package"),
        _history_record("claude_history", "/Users/x/replay_checker"),
    ]
    candidates = build_case_candidates(records)

    assert len(candidates) == 2
    assert candidates[0].source_type == "plan"
    assert candidates[0].relevance_score >= candidates[1].relevance_score


def test_select_case_candidate_returns_highest_scored(tmp_path):
    records = [
        _plan_record(tmp_path, "low.md", "Just a state doc"),
        _plan_record(tmp_path, "high.md", "Implement candidate selection"),
    ]
    candidates = build_case_candidates(records)
    best = select_case_candidate(candidates)

    assert best is not None
    assert best.candidate_id == "plan-high"


def test_select_case_candidate_empty_list():
    assert select_case_candidate([]) is None


def test_build_candidates_groups_history_by_type(tmp_path):
    records = [
        _history_record("codex_history", "fix intake bug"),
        _history_record("codex_history", "run tests"),
        _history_record("claude_history", "build scoring package"),
    ]
    candidates = build_case_candidates(records)

    source_types = [c.source_type for c in candidates]
    assert source_types.count("codex_history") == 1
    assert source_types.count("claude_history") == 1


def test_build_candidates_adds_risks_for_path_only_history():
    rec = _history_record("codex_history", "/Users/x/projects/myproject")
    candidates = build_case_candidates([rec])

    assert len(candidates) == 1
    assert len(candidates[0].risks) > 0
    assert "path" in candidates[0].risks[0].lower()


def test_candidate_fields_populated():
    rec = EvidenceRecord(
        source_type="plan",
        path=Path("/tmp/test.md"),
        summary="Implement feature X",
        confidence="high",
        metadata={"relative_path": "docs/plans/test.md", "base_commit": "abc123def456"},
    )
    candidates = build_case_candidates([rec])
    c = candidates[0]

    assert c.candidate_id == "plan-test"
    assert c.source_type == "plan"
    assert c.primary_source == "docs/plans/test.md"
    assert c.base_commit == "abc123def456"
    assert c.base_confidence == "high"
    assert c.relevance_score > 0


def test_inspect_candidates_cli(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Demo Plan\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "inspect-candidates",
            "--project",
            str(project),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Candidates:" in result.stdout
    assert "plan-INDEX" in result.stdout
    assert "score:" in result.stdout
