from replay_checker.sources import (
    EvidenceRecord,
    EvidenceSourceConfig,
    discover_evidence_sources,
    scan_claude_history,
    scan_codex_history,
    score_evidence_records,
)

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_discovers_configured_codex_history_for_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    history = tmp_path / "codex" / "sessions"
    history.mkdir(parents=True)
    item = history / "rollout.jsonl"
    item.write_text(
        '{"type":"turn_context","cwd":"' + str(project) + '"}\n'
        '{"type":"response_item","payload":{"role":"user","content":"fix intake"}}\n',
        encoding="utf-8",
    )

    records = scan_codex_history(project, [history])

    assert records[0].source_type == "codex_history"
    assert records[0].confidence == "high"
    assert "fix intake" in records[0].summary


def test_discovers_configured_claude_history_for_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    history = tmp_path / "claude" / "projects"
    history.mkdir(parents=True)
    item = history / "session.jsonl"
    item.write_text(
        '{"cwd":"' + str(project) + '","message":{"role":"user","content":"build scoring package"}}\n',
        encoding="utf-8",
    )

    records = scan_claude_history(project, [history])

    assert records[0].source_type == "claude_history"
    assert "build scoring package" in records[0].summary


def test_discover_sources_combines_history_and_plan_records(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Demo\n", encoding="utf-8")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":"user asked for demo"}\n',
        encoding="utf-8",
    )

    records = discover_evidence_sources(
        project,
        EvidenceSourceConfig(codex_history_roots=(codex,), claude_history_roots=()),
    )

    assert {record.source_type for record in records} >= {"plan", "codex_history"}


def test_discovery_ignores_macos_metadata_files(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Real Plan\n", encoding="utf-8")
    (plan_dir / "._INDEX.md").write_text("# Metadata Plan\n", encoding="utf-8")
    history = tmp_path / "history"
    history.mkdir()
    (history / "session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":"real conversation"}\n',
        encoding="utf-8",
    )
    (history / "._session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":"metadata conversation"}\n',
        encoding="utf-8",
    )

    records = discover_evidence_sources(
        project,
        EvidenceSourceConfig(codex_history_roots=(history,), claude_history_roots=()),
    )

    paths = {record.path.name for record in records}
    summaries = " ".join(record.summary for record in records)
    assert paths == {"INDEX.md", "session.jsonl"}
    assert "Metadata" not in summaries
    assert "metadata conversation" not in summaries


def test_inspect_sources_cli_prints_detected_records(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Demo Plan\n", encoding="utf-8")
    codex = tmp_path / "codex"
    codex.mkdir()
    (codex / "session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":"user asked for source inspection"}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "inspect-sources",
            "--project",
            str(project),
            "--codex-history-root",
            str(codex),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "plan" in result.stdout
    assert "codex_history" in result.stdout
    assert "source inspection" in result.stdout


def test_default_discovery_skips_home_history_for_temporary_projects(tmp_path):
    project = tmp_path / "project"
    project.mkdir()

    records = discover_evidence_sources(project)

    assert records == []


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

def test_score_plan_record_with_task_keyword(tmp_path):
    rec = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "task-plan.md",
        summary="Implement task scoring module",
        confidence="high",
        metadata={"relative_path": "docs/plans/task-plan.md"},
    )
    scored = score_evidence_records([rec])

    assert len(scored) == 1
    s = scored[0]
    assert s.relevance_score > 0.5
    assert s.task_signal == "plan_task"
    assert any("task" in r.lower() for r in s.relevance_reasons)


def test_score_plan_status_only_record(tmp_path):
    rec = EvidenceRecord(
        source_type="plan",
        path=tmp_path / "state.md",
        summary="Current project state",
        confidence="high",
        metadata={"relative_path": "control/state.md"},
    )
    scored = score_evidence_records([rec])

    assert scored[0].task_signal == "plan_status_only"
    assert scored[0].relevance_score < 0.5


def test_score_history_task_record():
    rec = EvidenceRecord(
        source_type="claude_history",
        path=Path("/fake/session.jsonl"),
        summary="fix intake scoring and build candidate module",
        confidence="high",
        metadata={},
    )
    scored = score_evidence_records([rec])

    assert scored[0].task_signal == "conversation_task"
    assert scored[0].relevance_score > 0.4


def test_score_history_path_only_mention():
    rec = EvidenceRecord(
        source_type="codex_history",
        path=Path("/fake/session.jsonl"),
        summary="/Users/x/projects/replay_checker",
        confidence="high",
        metadata={},
    )
    scored = score_evidence_records([rec])

    assert scored[0].task_signal == "conversation_mention"
    assert scored[0].relevance_score < 0.2


def test_score_evidence_records_is_deterministic(tmp_path):
    recs = [
        EvidenceRecord(source_type="plan", path=tmp_path / f"f{i}.md", summary=f"Task {i}")
        for i in range(5)
    ]
    run1 = score_evidence_records(recs)
    run2 = score_evidence_records(recs)

    assert [r.relevance_score for r in run1] == [r.relevance_score for r in run2]


def test_inspect_sources_cli_shows_score(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Demo Plan\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "inspect-sources",
            "--project",
            str(project),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "score=" in result.stdout
    assert "task=" in result.stdout
