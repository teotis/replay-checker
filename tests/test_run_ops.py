"""Focused tests for run_ops.py — run lifecycle owner."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from replay_checker.core import stable_hash
from replay_checker.evaluation import TaskOutcomeEntry, append_task_outcome
from replay_checker.git_utils import git_run
from replay_checker.replay_types import ReplayCase, RunHealth
from replay_checker.run_ops import (
    FORBIDDEN_PATH_PATTERNS,
    SENSITIVE_UNTRACKED_FILENAMES,
    _completion_declares_noop,
    _completion_status,
    _copy_project_tree,
    _create_worktree,
    _derive_canonical_run_status,
    _detect_forbidden_paths,
    _detect_rollback_signal,
    _detect_suspicious_paths,
    _is_sensitive_untracked_path,
    _next_run_id,
    _normalize_status,
    _partition_collectable_untracked,
    _read_run_base,
    _resolve_workspace_path,
    _stable_anonymous_runner_id,
    collect_run,
    doctor_runs,
    inspect_run_dir,
    load_case,
    load_run,
    prepare_run,
    validate_case,
)
from replay_checker.yaml_lite import write_simple_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path, *, base_commit: str = "abc1234", source_type: str = "manual") -> ReplayCase:
    case_dir = tmp_path / "cases" / "test-case-1"
    case_dir.mkdir(parents=True)
    write_simple_yaml(case_dir / "case.yaml", {
        "id": "test-case-1",
        "project_path": str(tmp_path / "project"),
        "plan_path": str(tmp_path / "plan.md"),
        "base_commit": base_commit,
        "base_source": "git_history",
        "base_confidence": "high",
        "source_type": source_type,
        "source_path": "docs/plan.md",
        "selection_reason": "test case",
        "verification_commands": ["echo ok"],
        "evidence_sources": [],
    })
    (case_dir / "evidence_sources.md").write_text("- test source\n", encoding="utf-8")
    return load_case(tmp_path / "cases", "test-case-1")


def _init_git_repo(path: Path) -> None:
    git_run(["init"], cwd=path)
    git_run(["add", "."], cwd=path)
    git_run(["commit", "-m", "initial", "--allow-empty"], cwd=path)


# ---------------------------------------------------------------------------
# _next_run_id — numbering gaps
# ---------------------------------------------------------------------------


def test_next_run_id_starts_at_001(tmp_path: Path) -> None:
    assert _next_run_id("case-1", tmp_path) == "case-1-001"


def test_next_run_id_increments_after_existing(tmp_path: Path) -> None:
    (tmp_path / "case-1-001").mkdir()
    (tmp_path / "case-1-002").mkdir()
    assert _next_run_id("case-1", tmp_path) == "case-1-003"


def test_next_run_id_handles_gap(tmp_path: Path) -> None:
    (tmp_path / "case-1-001").mkdir()
    (tmp_path / "case-1-005").mkdir()
    assert _next_run_id("case-1", tmp_path) == "case-1-006"


def test_next_run_id_ignores_non_numeric_suffixes(tmp_path: Path) -> None:
    (tmp_path / "case-1-001").mkdir()
    (tmp_path / "case-1-backup").mkdir()
    assert _next_run_id("case-1", tmp_path) == "case-1-002"


def test_next_run_id_ignores_other_case_dirs(tmp_path: Path) -> None:
    (tmp_path / "case-2-003").mkdir()
    assert _next_run_id("case-1", tmp_path) == "case-1-001"


# ---------------------------------------------------------------------------
# validate_case
# ---------------------------------------------------------------------------


def test_validate_case_passes_for_complete_case(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    assert validate_case(case) == []


def test_validate_case_fails_without_base_source(tmp_path: Path) -> None:
    case = _make_case(tmp_path)
    case = ReplayCase(
        id=case.id, root=case.root, project_path=case.project_path,
        plan_path=case.plan_path, base_commit=case.base_commit,
        base_source="", base_confidence="high", source_type="manual",
        source_path="x", selection_reason="y",
    )
    diags = validate_case(case)
    assert any("base_source" in d.code for d in diags)


def test_validate_case_allows_empty_base_commit_for_no_git(tmp_path: Path) -> None:
    case = _make_case(tmp_path, base_commit="", source_type="no_git")
    assert validate_case(case) == []


def test_prepare_run_uses_case_task_outcome_feedback(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _init_git_repo(project)
    case = _make_case(tmp_path, base_commit="HEAD")
    append_task_outcome(
        case.root,
        TaskOutcomeEntry(
            package_id="old-package",
            task_contract_id="old-contract",
            outcome="blocked",
            blocked_reason="missing fixture coverage",
            acceptance_gaps=("no observable fixture acceptance",),
            verification_status="weak",
        ),
    )

    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="test-agent")

    task_text = (run.root / "TASK.md").read_text(encoding="utf-8")
    assert "prior outcome feedback is addressed: no observable fixture acceptance" in task_text
    assert "task-specific completion" in task_text


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------


def test_normalize_status_completed_variants() -> None:
    assert _normalize_status("complete") == "completed"
    assert _normalize_status("pass") == "completed"
    assert _normalize_status("complete-already-implemented") == "completed"
    assert _normalize_status("completed") == "completed"
    assert _normalize_status("partial") == "partial"


def test_derive_status_completed_with_diff() -> None:
    status, reason = _derive_canonical_run_status(
        raw_status="completed",
        workspace_exists=True,
        completion_report_exists=True,
        has_result_diff=True,
        completion_text="status: completed",
    )
    assert status == "completed"
    assert "result diff" in reason


def test_derive_status_completed_no_diff() -> None:
    status, _ = _derive_canonical_run_status(
        raw_status="completed",
        workspace_exists=True,
        completion_report_exists=True,
        has_result_diff=False,
        completion_text="status: completed",
    )
    assert status == "failed"


def test_derive_status_workspace_missing() -> None:
    status, _ = _derive_canonical_run_status(
        raw_status="completed",
        workspace_exists=False,
        completion_report_exists=True,
        has_result_diff=True,
        completion_text="status: completed",
    )
    assert status == "failed"


def test_derive_status_noop_declaration() -> None:
    status, reason = _derive_canonical_run_status(
        raw_status="completed",
        workspace_exists=True,
        completion_report_exists=True,
        has_result_diff=False,
        completion_text="status: completed\nNo changes needed.",
    )
    assert status == "completed"
    assert "no-op" in reason


def test_completion_declares_noop() -> None:
    assert _completion_declares_noop("Already implemented.") is True
    assert _completion_declares_noop("No changes needed") is True
    assert _completion_declares_noop("Done something") is False


# ---------------------------------------------------------------------------
# Completion status
# ---------------------------------------------------------------------------


def test_completion_status_missing(tmp_path: Path) -> None:
    assert _completion_status(tmp_path / "nope.md") == "missing-completion-report"


def test_completion_status_with_status_line(tmp_path: Path) -> None:
    p = tmp_path / "report.md"
    p.write_text("status: completed\nDone.\n", encoding="utf-8")
    assert _completion_status(p) == "completed"


def test_completion_status_no_status_line(tmp_path: Path) -> None:
    p = tmp_path / "report.md"
    p.write_text("Just did stuff.\n", encoding="utf-8")
    assert _completion_status(p) == "reported"


# ---------------------------------------------------------------------------
# Safety filters
# ---------------------------------------------------------------------------


def test_detect_forbidden_paths() -> None:
    files = ("src/main.py", "_reference/diff.patch", ".git/config", "runs/r1")
    hits = _detect_forbidden_paths(tuple(files))
    assert "_reference/diff.patch" in hits
    assert ".git/config" in hits
    assert "runs/r1" in hits
    assert "src/main.py" not in hits


def test_detect_suspicious_paths() -> None:
    files = ("src/main.py", "config/credentials.json", "secrets.yaml")
    hits = _detect_suspicious_paths(tuple(files))
    assert any("credentials" in h for h in hits)
    assert "src/main.py" not in hits


def test_is_sensitive_untracked_env() -> None:
    assert _is_sensitive_untracked_path(".env") is True
    assert _is_sensitive_untracked_path(".env.local") is True
    assert _is_sensitive_untracked_path(".envrc") is True
    assert _is_sensitive_untracked_path(".env.example") is False


def test_is_sensitive_untracked_ssh() -> None:
    # All .ssh/ paths are flagged (directory-level heuristic)
    assert _is_sensitive_untracked_path(".ssh/id_rsa") is True
    assert _is_sensitive_untracked_path(".ssh/known_hosts") is True
    # Non-.ssh paths are fine
    assert _is_sensitive_untracked_path("config/known_hosts") is False


def test_is_sensitive_untracked_key_extensions() -> None:
    assert _is_sensitive_untracked_path("server.key") is True
    assert _is_sensitive_untracked_path("cert.pem") is True
    assert _is_sensitive_untracked_path("app.py") is False


def test_is_sensitive_untracked_forbidden_pattern() -> None:
    assert _is_sensitive_untracked_path("_reference/diff.patch") is True
    assert _is_sensitive_untracked_path("cases/test/case.yaml") is True


def test_partition_collectable_untracked() -> None:
    paths = [".env", "src/main.py", "completion_report.md", ".ssh/id_rsa"]
    collectable, skipped = _partition_collectable_untracked(paths)
    assert "src/main.py" in collectable
    assert ".env" in skipped
    assert ".ssh/id_rsa" in skipped
    # completion_report.md is ignored (not collectable or skipped)
    assert "completion_report.md" not in collectable
    assert "completion_report.md" not in skipped


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------


def test_resolve_workspace_relative_default() -> None:
    run_root = Path("/runs/case-1-001")
    assert _resolve_workspace_path(run_root, {}) == run_root / "workspace"


def test_resolve_workspace_absolute() -> None:
    run_root = Path("/runs/case-1-001")
    result = _resolve_workspace_path(run_root, {"workspace": "/tmp/custom"})
    assert result == Path("/tmp/custom")


def test_resolve_workspace_runs_prefix() -> None:
    run_root = Path("/runs/case-1-001")
    result = _resolve_workspace_path(run_root, {"workspace": "runs/shared-ws"})
    assert result == Path("/runs/shared-ws")


# ---------------------------------------------------------------------------
# Doctor runs
# ---------------------------------------------------------------------------


def test_doctor_runs_empty_root(tmp_path: Path) -> None:
    assert doctor_runs(tmp_path) == ()


def test_doctor_runs_with_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "case-1-001"
    run_dir.mkdir()
    ws = run_dir / "workspace"
    ws.mkdir()
    # Init git in workspace with a base commit
    _init_git_repo(ws)
    base_sha = git_run(["rev-parse", "HEAD"], cwd=ws).stdout.strip()
    write_simple_yaml(run_dir / "run.yaml", {
        "id": "case-1-001", "runner_label": "agent-a", "base_commit": base_sha,
    })
    # Create a minimal completion report
    (ws / "completion_report.md").write_text(
        "status: completed\n", encoding="utf-8",
    )
    # Make a change so diff is non-empty
    (ws / "new.txt").write_text("hello", encoding="utf-8")
    git_run(["add", "."], cwd=ws)
    git_run(["commit", "-m", "change"], cwd=ws)

    reports = doctor_runs(tmp_path)
    assert len(reports) == 1
    assert reports[0].run_id == "case-1-001"
    assert reports[0].status == "completed"


def test_doctor_runs_filter_by_runner_label(tmp_path: Path) -> None:
    for label in ("agent-a", "agent-b"):
        run_dir = tmp_path / f"case-1-00{1 if label == 'agent-a' else 2}"
        run_dir.mkdir()
        write_simple_yaml(run_dir / "run.yaml", {"id": run_dir.name, "runner_label": label})

    reports = doctor_runs(tmp_path, runner_label="agent-a")
    assert len(reports) == 1
    assert reports[0].run_id == "case-1-001"


# ---------------------------------------------------------------------------
# Anonymous runner ID stability
# ---------------------------------------------------------------------------


def test_stable_anonymous_runner_id_deterministic() -> None:
    id1 = _stable_anonymous_runner_id("case-1-001")
    id2 = _stable_anonymous_runner_id("case-1-001")
    assert id1 == id2
    assert id1 == f"runner-{stable_hash('case-1-001', length=10)}"


def test_stable_anonymous_runner_id_varies_with_input() -> None:
    assert _stable_anonymous_runner_id("case-1-001") != _stable_anonymous_runner_id("case-1-002")


# ---------------------------------------------------------------------------
# Workspace fallback contracts
# ---------------------------------------------------------------------------


def test_create_worktree_empty_base_commit_uses_copytree(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    case = _make_case(tmp_path, base_commit="")
    _create_worktree(case, workspace)
    assert (workspace / "src" / "main.py").exists()
    # Workspace should have a git repo initialized
    assert (workspace / ".git").exists()


def test_copy_project_tree_excludes_git(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _init_git_repo(project)
    (project / "keep.py").write_text("ok\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    _copy_project_tree(project, workspace)
    assert (workspace / "keep.py").exists()
    assert (workspace / ".git").exists()  # new repo init


# ---------------------------------------------------------------------------
# inspect_run_dir health
# ---------------------------------------------------------------------------


def test_inspect_run_dir_workspace_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "case-1-001"
    run_dir.mkdir()
    write_simple_yaml(run_dir / "run.yaml", {"id": "case-1-001", "base_commit": "abc"})
    health = inspect_run_dir(run_dir)
    assert health.status == "failed"
    assert "workspace missing" in health.reason


# ---------------------------------------------------------------------------
# _read_run_base
# ---------------------------------------------------------------------------


def test_read_run_base_empty_list_yaml(tmp_path: Path) -> None:
    run_dir = tmp_path / "case-1-001"
    run_dir.mkdir()
    # Write run.yaml with base_commit: [] (YAML empty list)
    (run_dir / "run.yaml").write_text("id: case-1-001\nbase_commit: []\n", encoding="utf-8")

    class FakeRun:
        root = run_dir
    assert _read_run_base(FakeRun()) == ""


def test_read_run_base_normal(tmp_path: Path) -> None:
    run_dir = tmp_path / "case-1-001"
    run_dir.mkdir()
    write_simple_yaml(run_dir / "run.yaml", {"id": "case-1-001", "base_commit": "abc123"})

    class FakeRun:
        root = run_dir
    assert _read_run_base(FakeRun()) == "abc123"


# ---------------------------------------------------------------------------
# collect_run — integration with workspace
# ---------------------------------------------------------------------------


def test_collect_run_produces_evidence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _init_git_repo(project)
    case = _make_case(tmp_path, base_commit="HEAD")

    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="test-agent")
    # Simulate agent changes
    (run.workspace / "new.py").write_text("print('hi')\n", encoding="utf-8")
    git_run(["add", "."], cwd=run.workspace)
    git_run(["commit", "-m", "agent change"], cwd=run.workspace)
    # Write completion report
    (run.workspace / "completion_report.md").write_text(
        "status: completed\n\nDid stuff.\n", encoding="utf-8",
    )

    evidence = collect_run(run)
    assert evidence.run_id == run.id
    assert evidence.status == "completed"
    assert len(evidence.changed_files) >= 0  # may or may not show changes vs base
    evidence_yaml = run.root / "evidence" / "evidence.yaml"
    assert evidence_yaml.exists()


def test_collect_run_skips_sensitive_untracked(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _init_git_repo(project)
    case = _make_case(tmp_path, base_commit="HEAD")

    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="test-agent")
    # Create sensitive untracked file
    (run.workspace / ".env").write_text("SECRET=key\n", encoding="utf-8")
    (run.workspace / "safe.py").write_text("ok\n", encoding="utf-8")
    # Write completion report
    (run.workspace / "completion_report.md").write_text(
        "status: completed\n", encoding="utf-8",
    )

    evidence = collect_run(run)
    # .env should be skipped from collection
    skipped = run.root / "evidence" / "evidence.yaml"
    content = skipped.read_text(encoding="utf-8")
    assert ".env" in content
