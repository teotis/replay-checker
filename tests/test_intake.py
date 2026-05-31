import subprocess
import sys
from pathlib import Path

from replay_checker.replay import (
    _extract_verification,
    intake,
    load_case,
    parse_simple_yaml,
)


ROOT = Path(__file__).resolve().parents[1]


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )


def test_intake_returns_case_with_metadata_fields(tmp_path):
    project = tmp_path / "sample_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial commit"], project)

    cases_root = tmp_path / "cases"
    case = intake(
        cases_root=cases_root,
        project_path=project,
    )

    case_yaml = parse_simple_yaml(case.root / "case.yaml")
    assert "base_confidence" in case_yaml
    assert "source_type" in case_yaml
    assert "selection_reason" in case_yaml


def test_intake_detects_orchestration_kit(tmp_path):
    project = tmp_path / "orch_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("pass\n", encoding="utf-8")
    # Create orchestration kit structure
    plan_dir = project / "docs" / "plans" / "demo"
    (plan_dir / "packages").mkdir(parents=True)
    (plan_dir / "launchers").mkdir(parents=True)
    (plan_dir / "status").mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text("# Demo Plan\n\n## Goal\nRefactor auth.\n", encoding="utf-8")
    (plan_dir / "packages" / "01-refactor.md").write_text("# Refactor\n\nSet X = 2.\n", encoding="utf-8")
    (plan_dir / "launchers" / "package-graph.tsv").write_text("pkg\tdeps\n", encoding="utf-8")
    (plan_dir / "status" / "state.tsv").write_text("id\tstatus\n", encoding="utf-8")
    (plan_dir / "FINAL_REPORT.md").write_text("# Done\n", encoding="utf-8")

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add orchestration kit"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "orchestration_kit"
    assert "INDEX.md" in case.source_path
    assert "orchestration" in case.selection_reason.lower()


def test_intake_detects_handoff_plan(tmp_path):
    project = tmp_path / "handoff_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "utils.py").write_text("def helper(): pass\n", encoding="utf-8")
    plan_dir = project / "docs" / "plans" / "feature-x"
    plan_dir.mkdir(parents=True)
    (plan_dir / "PLAN.md").write_text(
        "# Feature X\n\n"
        "## Goal\n"
        "Add logging.\n\n"
        "## Steps\n"
        "1. Import logging\n2. Add logger\n\n"
        "## Acceptance Criteria\n"
        "- Log level configurable\n\n"
        "## Verification Commands\n"
        "- rtk python3 -m pytest\n",
        encoding="utf-8",
    )

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add handoff plan"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "handoff_plan"
    assert "handoff" in case.selection_reason.lower()


def test_intake_generates_synthetic_case_from_git_history(tmp_path):
    project = tmp_path / "git_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("v1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial setup"], project)

    # Add a meaningful commit
    (project / "src" / "app.py").write_text("v2\n", encoding="utf-8")
    _git(["add", "."], project)
    _git(["commit", "-m", "feat: add user authentication module"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "git_history"
    assert case.synthetic_case is True
    assert case.base_confidence in ("high", "medium", "low")


def test_intake_low_confidence_on_uncommitted_files(tmp_path):
    project = tmp_path / "dirty_project"
    project.mkdir()
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    # One commit to have at least one
    (project / "README.md").write_text("hello\n", encoding="utf-8")
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)
    # Uncommitted change
    (project / "README.md").write_text("dirty\n", encoding="utf-8")

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.base_confidence == "low"


def test_intake_no_git_project_returns_failed_case(tmp_path):
    project = tmp_path / "nogit_project"
    project.mkdir()
    (project / "README.md").write_text("hello\n", encoding="utf-8")

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "no_git"
    assert case.base_confidence == "low"


def test_intake_empty_history_returns_failed_case(tmp_path):
    project = tmp_path / "empty_project"
    project.mkdir()
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "empty_history"
    assert case.base_confidence == "low"


def test_intake_cli_creates_case(tmp_path):
    project = tmp_path / "cli_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    cases_root = tmp_path / "cases"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "intake",
            "--cases-root",
            str(cases_root),
            "--project",
            str(project),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Created case:" in result.stdout
    assert "Next commands (copy-paste ready):" in result.stdout
    assert "rtk python3 tools/replay.py prepare-run" in result.stdout
    assert "rtk python3 tools/replay.py collect-run" in result.stdout
    assert "rtk python3 tools/replay.py score-run" in result.stdout
    assert "rtk python3 tools/replay.py compare" in result.stdout

    # Verify the generated case id appears in the next prepare-run command
    cases_list = sorted((cases_root).glob("*/case.yaml"))
    assert len(cases_list) == 1
    case_id = cases_list[0].parent.name
    assert f"--case {case_id}" in result.stdout


def test_intake_ranking_picks_highest_scoring_plan(tmp_path):
    project = tmp_path / "multi_plan"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("pass\n", encoding="utf-8")

    # Plan A: full orchestration kit (high score)
    plan_a = project / "docs" / "plans" / "plan-a"
    (plan_a / "packages").mkdir(parents=True)
    (plan_a / "launchers").mkdir(parents=True)
    (plan_a / "status").mkdir(parents=True)
    (plan_a / "INDEX.md").write_text("# Plan A\n", encoding="utf-8")
    (plan_a / "packages" / "01-do-x.md").write_text("# Do X\n", encoding="utf-8")
    (plan_a / "launchers" / "package-graph.tsv").write_text("pkg\tdeps\n", encoding="utf-8")
    (plan_a / "status" / "state.tsv").write_text("id\tstatus\n", encoding="utf-8")

    # Plan B: minimal plan (low score, only INDEX.md)
    plan_b = project / "docs" / "plans" / "plan-b"
    plan_b.mkdir(parents=True)
    (plan_b / "INDEX.md").write_text("# Plan B\n", encoding="utf-8")

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add two plans"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    # Plan A should win (higher score)
    assert case.source_type == "orchestration_kit"
    assert "plan-a" in case.source_path
    assert "Scored" in case.selection_reason


def test_intake_ranking_deterministic_on_equal_scores(tmp_path):
    project = tmp_path / "equal_plans"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("x\n", encoding="utf-8")

    # Two equal INDEX.md directories, same score
    plan_a = project / "docs" / "plans" / "aaa-plan"
    plan_a.mkdir(parents=True)
    (plan_a / "INDEX.md").write_text("# AAA\n", encoding="utf-8")

    plan_b = project / "docs" / "plans" / "bbb-plan"
    plan_b.mkdir(parents=True)
    (plan_b / "INDEX.md").write_text("# BBB\n", encoding="utf-8")

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add equal plans"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.source_type == "orchestration_kit"
    # Deterministic: sorts by path when scores tied
    assert "aaa-plan" in case.source_path


def test_synthetic_case_has_reference_evidence(tmp_path):
    project = tmp_path / "synth_ev"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("v1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    (project / "src" / "app.py").write_text("v2\n", encoding="utf-8")
    (project / "src" / "utils.py").write_text("def util(): pass\n", encoding="utf-8")
    _git(["add", "."], project)
    _git(["commit", "-m", "feat: add utility module"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert case.synthetic_case is True

    ref_dir = case.root / "_reference"
    assert ref_dir.is_dir()
    assert (ref_dir / "diff.patch").exists()
    assert (ref_dir / "reference_metadata.yaml").exists()

    meta = parse_simple_yaml(ref_dir / "reference_metadata.yaml")
    assert "target_commit" in meta
    assert "commit_message" in meta
    assert "changed_files" in meta


def test_synthetic_task_md_excludes_reference_diff(tmp_path):
    project = tmp_path / "synth_mask"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("old\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    (project / "src" / "main.py").write_text(
        "old\n\n# New feature added\ndef new_feature():\n    return 42\n",
        encoding="utf-8",
    )
    _git(["add", "."], project)
    _git(["commit", "-m", "feat: add new_feature function"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    task_md = (case.root / "task.md").read_text(encoding="utf-8")

    # Contains the commit message (intent)
    assert "add new_feature function" in task_md
    # Does NOT contain actual code from the reference implementation
    assert "def new_feature" not in task_md
    assert "return 42" not in task_md


def test_synthetic_task_md_contains_reconstructed_task(tmp_path):
    project = tmp_path / "synth_task"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "lib.py").write_text("# empty\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    (project / "src" / "lib.py").write_text("# updated\ndef greet():\n    return 'hi'\n", encoding="utf-8")
    _git(["add", "."], project)
    _git(["commit", "-m", "feat: implement greet function in lib"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    task_md = (case.root / "task.md").read_text(encoding="utf-8")

    assert "Reconstructed Task" in task_md
    assert "greet function" in task_md
    assert "Base commit:" in task_md


def test_intake_selection_reason_includes_score_details(tmp_path):
    project = tmp_path / "reason_proj"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("ok\n", encoding="utf-8")

    plan_dir = project / "docs" / "plans" / "demo"
    (plan_dir / "packages").mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text(
        "# Demo\n\n## Verification Commands\n- rtk make test\n",
        encoding="utf-8",
    )
    (plan_dir / "packages" / "01-task.md").write_text("# Task\n", encoding="utf-8")

    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add plan with verification"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    assert "Scored" in case.selection_reason
    assert "verification" in case.selection_reason
    assert "packages" in case.selection_reason


def test_intake_records_matching_conversation_evidence(tmp_path):
    project = tmp_path / "history_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("x\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    codex_root = tmp_path / "codex"
    codex_root.mkdir()
    (codex_root / "session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":"user asked to improve intake UX"}\n',
        encoding="utf-8",
    )

    case = intake(
        cases_root=tmp_path / "cases",
        project_path=project,
        codex_history_roots=(codex_root,),
        claude_history_roots=(),
    )

    case_yaml = parse_simple_yaml(case.root / "case.yaml")
    assert "codex_history" in " ".join(case_yaml.get("evidence_sources", []))

    evidence_md = (case.root / "evidence_sources.md").read_text(encoding="utf-8")
    assert "user asked to improve intake UX" in evidence_md

    task_md = (case.root / "task.md").read_text(encoding="utf-8")
    assert "Conversation Evidence" in task_md
    assert "user asked to improve intake UX" in task_md
    assert '{"cwd"' not in task_md


def test_intake_cli_accepts_history_roots(tmp_path):
    project = tmp_path / "cli_history_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("x\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    claude_root = tmp_path / "claude"
    claude_root.mkdir()
    (claude_root / "session.jsonl").write_text(
        '{"cwd":"' + str(project) + '","message":{"role":"user","content":"stabilize scoring package"}}\n',
        encoding="utf-8",
    )

    cases_root = tmp_path / "cases"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "intake",
            "--cases-root",
            str(cases_root),
            "--project",
            str(project),
            "--claude-history-root",
            str(claude_root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Evidence sources:" in result.stdout
    case_yaml = next(cases_root.glob("*/case.yaml"))
    data = parse_simple_yaml(case_yaml)
    assert "claude_history" in " ".join(data.get("evidence_sources", []))


def test_intake_creates_candidate_report(tmp_path):
    project = tmp_path / "candidate_proj"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    report_path = case.root / "candidate_report.md"
    assert report_path.exists(), "intake must generate candidate_report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert "Candidate Report" in report_text
    assert "Total candidates:" in report_text


def test_intake_case_yaml_includes_candidate_metadata(tmp_path):
    project = tmp_path / "meta_proj"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)

    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)

    case_yaml = parse_simple_yaml(case.root / "case.yaml")
    # source_type and base_source are always present
    assert "source_type" in case_yaml
    assert "base_source" in case_yaml


def test_intake_and_inspect_candidates_select_same_candidate(tmp_path):
    """intake and inspect-candidates should select the same primary candidate."""
    project = tmp_path / "consistent_proj"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("X = 1\n", encoding="utf-8")
    plan_dir = project / "docs" / "plans" / "demo"
    (plan_dir / "packages").mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text(
        "# Demo Plan\n\n## Verification Commands\n- rtk make test\n",
        encoding="utf-8",
    )
    (plan_dir / "packages" / "01-task.md").write_text("# Task\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "add plan"], project)

    # Run intake
    cases_root = tmp_path / "cases"
    case = intake(cases_root=cases_root, project_path=project)
    intake_candidate_id = case.selected_candidate_id

    # Run inspect-candidates CLI
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
    assert "Selected candidate:" in result.stdout

    # The candidate IDs should match
    if intake_candidate_id:
        assert intake_candidate_id in result.stdout


# ---------------------------------------------------------------------------
# _extract_verification command-shape filtering tests
# ---------------------------------------------------------------------------


def _write_plan(tmp_path, content):
    plan_dir = tmp_path / "plan"
    (plan_dir / "packages").mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text(content, encoding="utf-8")
    return plan_dir


def test_extract_verification_ignores_prose_bullets(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "# Plan\n\n"
        "## Verification Commands\n"
        "- Commit local package changes.\n"
        "- Write only their assigned coordinator status file.\n"
        "- Update the state ledger only through `bash orchestrate.sh mark-state`.\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == []


def test_extract_verification_keeps_valid_commands(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "- rtk python3 -m pytest tests/test_intake.py -q\n"
        "- make test\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == ["rtk python3 -m pytest tests/test_intake.py -q", "make test"]


def test_extract_verification_strips_surrounding_backticks(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "- `rtk make test`\n"
        "- `python3 -m pytest -q`\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == ["rtk make test", "python3 -m pytest -q"]


def test_extract_verification_no_nested_backtick_artifacts(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "- `bash orchestrate.sh mark-state`\n"
        "- Update the state ledger only through `bash orchestrate.sh mark-state`.\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == ["bash orchestrate.sh mark-state"]


def test_extract_verification_mixed_prose_and_commands(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "- `rtk make test`\n"
        "- Write only their assigned coordinator status file.\n"
        "- `git status`\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == ["rtk make test", "git status"]


def test_extract_verification_skips_unknown_prefix(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "- Verify merged-candidates.md exists and is non-empty.\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == []


def test_extract_verification_empty_lines_do_not_break_section(tmp_path):
    plan_dir = _write_plan(tmp_path, (
        "## Verification Commands\n"
        "\n"
        "- rtk make test\n"
    ))
    result = _extract_verification(tmp_path, plan_dir)
    assert result == ["rtk make test"]
