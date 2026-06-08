from __future__ import annotations

from pathlib import Path

import replay_checker.candidates as candidates_module
import replay_checker.sources as sources_module
from replay_checker.candidates import build_case_candidates
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
