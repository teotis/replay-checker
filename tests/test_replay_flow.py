import subprocess
import sys
from pathlib import Path

from replay_checker.replay import (
    _completion_status,
    collect_run,
    compare_case,
    create_case,
    discover_plan_packages,
    load_case,
    parse_simple_yaml,
    prepare_run,
    score_run,
    validate_case,
)
from replay_checker.core import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def _git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)


def _make_history_project(tmp_path):
    project = tmp_path / "history_project"
    project.mkdir()
    (project / "docs" / "plans" / "demo" / "packages").mkdir(parents=True)
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    index = project / "docs" / "plans" / "demo" / "INDEX.md"
    index.write_text("# Demo Plan\n\n## Goal\nChange VALUE to 2.\n", encoding="utf-8")
    package = project / "docs" / "plans" / "demo" / "packages" / "01-change.md"
    package.write_text("# Change Package\n\nSet `VALUE = 2`.\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)
    base = _git(["rev-parse", "HEAD"], project).stdout.strip()
    return project, index, base


def test_discovers_orchestration_style_plan_packages(tmp_path):
    project, index, _base = _make_history_project(tmp_path)

    discovered = discover_plan_packages(project)

    assert discovered[0].plan_path == index
    assert discovered[0].package_count == 1
    assert discovered[0].title == "Demo Plan"


def test_create_case_writes_replay_contract(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"

    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    case_file = cases_root / "demo-case" / "case.yaml"
    task_file = cases_root / "demo-case" / "task.md"
    assert case.id == "demo-case"
    assert parse_simple_yaml(case_file)["base_commit"] == base
    assert "Do not launch or control an agent automatically" in task_file.read_text(encoding="utf-8")


def test_create_case_writes_complete_case_contract(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"

    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="complete-case",
    )

    case_yaml = parse_simple_yaml(case.root / "case.yaml")
    assert case_yaml["base_source"] == "manual"
    assert case_yaml["base_confidence"] == "user_supplied"
    assert case_yaml["source_type"] == "manual"
    assert case_yaml["source_path"] == str(index.resolve())
    assert case_yaml["selection_reason"] == "Manual case created from explicit plan and base commit"
    assert case_yaml["synthetic_case"] == "false"
    assert case_yaml["evidence_sources"] == []
    assert case_yaml["verification_commands"] == []
    assert (case.root / "evidence_sources.md").exists()
    assert validate_case(case) == []


def test_prepare_run_rejects_incomplete_case_yaml(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    case_dir = cases_root / "broken-case"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yaml").write_text(
        "\n".join([
            "id: broken-case",
            f"project_path: {project}",
            f"plan_path: {index}",
            f"base_commit: {base}",
            "verification_commands:",
            "",
        ]),
        encoding="utf-8",
    )
    (case_dir / "task.md").write_text("# Broken\n", encoding="utf-8")

    case = load_case(cases_root, "broken-case")

    try:
        prepare_run(case, runs_root=tmp_path / "runs", runner_label="Agent")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("prepare_run should reject incomplete case metadata")

    assert "Case broken-case is incomplete" in message
    assert "base_source" in message
    assert "evidence_sources.md" in message


def test_prepare_collect_score_and_compare_manual_agent_run(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    reports_root = tmp_path / "reports"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Claude Sonnet 4")

    run_file = runs_root / run.id / "run.yaml"
    task_file = runs_root / run.id / "TASK.md"
    assert parse_simple_yaml(run_file)["runner_label"] == "Claude Sonnet 4"
    assert parse_simple_yaml(run_file)["anonymous_runner_id"] == f"runner-{stable_hash(run.id, length=10)}"
    assert "Execution Package" in task_file.read_text(encoding="utf-8")
    assert (runs_root / run.id / "workspace" / "src" / "app.py").exists()

    (runs_root / run.id / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (runs_root / run.id / "completion_report.md").write_text(
        "status: completed\n\nChanged VALUE and checked manually.\n",
        encoding="utf-8",
    )

    evidence = collect_run(run)
    assert evidence.status == "completed"
    assert "src/app.py" in evidence.changed_files
    assert (runs_root / run.id / "evidence" / "diff.patch").exists()

    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")
    assert f"Anonymous runner: runner-{stable_hash(run.id, length=10)}" in scoring_text
    assert "Claude Sonnet 4" not in scoring_text
    assert "Result score weight: 80" in scoring_text

    report = compare_case(case, runs_root=runs_root, reports_root=reports_root)
    text = report.read_text(encoding="utf-8")
    assert "demo-case" in text
    assert run.id in text
    assert "completed" in text


def test_replay_cli_prepares_run_without_agent_command(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    create = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "create-case",
            "--cases-root",
            str(cases_root),
            "--project",
            str(project),
            "--plan",
            str(index),
            "--base",
            base,
            "--case-id",
            "cli-case",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert create.returncode == 0, create.stderr

    prepare = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "prepare-run",
            "--cases-root",
            str(cases_root),
            "--runs-root",
            str(runs_root),
            "--case",
            "cli-case",
            "--label",
            "manual-codex",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepare.returncode == 0, prepare.stderr
    assert "Copy TASK.md into your chosen agent" in prepare.stdout
    assert "rtk python3 tools/replay.py collect-run" in prepare.stdout


def test_prepare_run_generates_completion_report_template(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Claude Sonnet 4")

    template_path = runs_root / run.id / "completion_report_template.md"
    assert template_path.exists(), "prepare_run must generate completion_report_template.md"

    template_text = template_path.read_text(encoding="utf-8")
    assert "status:" in template_text, "template must include a status field"
    assert "# Completion Report" in template_text

    task_text = (runs_root / run.id / "TASK.md").read_text(encoding="utf-8")
    assert "completion_report_template.md" in task_text, "TASK.md must reference the template"
    assert "Protocol Version: replay-checker-task-v2" in task_text
    assert "## Expected Output" in task_text
    assert "## Forbidden Access" in task_text
    assert "## Verification Contract" in task_text


def test_collect_run_records_missing_evidence(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Agent X")

    # No changes made, no completion report written — all evidence should be missing
    evidence = collect_run(run)

    assert evidence.status == "missing-completion-report"
    assert "completion_report.md" in evidence.missing_fields
    assert "diff.patch (empty)" in evidence.missing_fields
    assert "changed_files (none)" in evidence.missing_fields

    evidence_yaml = parse_simple_yaml(runs_root / run.id / "evidence" / "evidence.yaml")
    assert "completion_report.md" in evidence_yaml.get("missing_fields", [])


def test_collect_run_with_full_evidence(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Agent Y")

    # Simulate a completed run with changes and completion report
    (run.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\nChanged VALUE to 2.\n",
        encoding="utf-8",
    )

    evidence = collect_run(run)
    assert evidence.status == "completed"
    assert len(evidence.missing_fields) == 0
    assert "src/app.py" in evidence.changed_files


def test_scoring_package_excludes_runner_label(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Claude Sonnet 4")

    (run.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\nChanged VALUE to 2.\n",
        encoding="utf-8",
    )

    collect_run(run)
    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")

    assert "Claude Sonnet 4" not in scoring_text, "runner_label must not appear in scoring package"
    assert f"Anonymous runner: runner-{stable_hash(run.id, length=10)}" in scoring_text
    assert "Bias Warnings" in scoring_text
    assert "runner label is intentionally excluded" in scoring_text


def test_anonymous_runner_id_is_stable_after_sibling_run_removed(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    first = prepare_run(case, runs_root=runs_root, runner_label="Agent A")
    second = prepare_run(case, runs_root=runs_root, runner_label="Agent B")

    (second.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (second.root / "completion_report.md").write_text("status: completed\n", encoding="utf-8")
    collect_run(second)
    before = score_run(second, rubric_path=ROOT / "rubrics" / "default.yaml").read_text(encoding="utf-8")

    for path in sorted(first.root.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    first.root.rmdir()

    after = score_run(second, rubric_path=ROOT / "rubrics" / "default.yaml").read_text(encoding="utf-8")

    assert before.split("Anonymous runner: ", 1)[1].splitlines()[0] == after.split("Anonymous runner: ", 1)[1].splitlines()[0]


def test_scoring_does_not_redact_common_runner_label_from_valid_content(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="completed")
    (run.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text("status: completed\n", encoding="utf-8")

    collect_run(run)
    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")

    assert "Status: completed" in scoring_text
    assert "[REDACTED]" not in scoring_text


def test_scoring_package_includes_evidence_inventory_and_missing(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="demo-case",
        verification_commands=["rtk python3 -m pytest"],
    )

    run = prepare_run(case, runs_root=runs_root, runner_label="Test Agent")

    # No changes, no completion report — produce missing evidence scenario
    collect_run(run)
    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")

    assert "Evidence Inventory" in scoring_text
    assert "Missing Evidence" in scoring_text
    assert "completion_report.md" in scoring_text or "missing-completion-report" in scoring_text
    assert "Minimum Evidence Check" in scoring_text
    assert "Bias Warnings" in scoring_text
    assert "Rubric Weights" in scoring_text
    assert "Result score weight:" in scoring_text
    assert "Process score weight:" in scoring_text
    assert "Protocol Version: replay-checker-score-v2" in scoring_text
    assert "## Required Inputs" in scoring_text
    assert "## Invalid Score Conditions" in scoring_text


def test_completion_status_helper(tmp_path):
    report = tmp_path / "report.md"
    assert _completion_status(report) == "missing-completion-report"

    report.write_text("status: completed\n\nDone.", encoding="utf-8")
    assert _completion_status(report) == "completed"

    report.write_text("no status line here\njust notes.", encoding="utf-8")
    assert _completion_status(report) == "reported"


def test_validate_case_returns_no_missing_for_complete_case(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="validate-case",
    )

    from replay_checker.replay import validate_case
    missing = validate_case(case)
    assert missing == []


def test_validate_case_returns_missing_fields(tmp_path):
    from replay_checker.replay import ReplayCase, validate_case

    case_dir = tmp_path / "incomplete"
    case_dir.mkdir()
    incomplete = ReplayCase(
        id="incomplete",
        root=case_dir,
        project_path=Path("/nonexistent"),
        plan_path=Path("/nonexistent"),
        base_commit="",
    )
    missing = validate_case(incomplete)
    assert "base_source" in missing


def test_scoring_package_includes_gate_assessment(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="gate-case",
        verification_commands=["rtk python3 -m pytest"],
    )
    run = prepare_run(case, runs_root=runs_root, runner_label="Test Agent")

    (run.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\nAll tests pass after verification.\n",
        encoding="utf-8",
    )
    collect_run(run)
    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")

    assert "## Evidence Gate Assessment" in scoring_text
    assert "[PASS]" in scoring_text or "[FAIL]" in scoring_text
    assert "Overall:" in scoring_text


def test_scoring_package_includes_score_ceilings_when_diff_missing(tmp_path):
    project, index, base = _make_history_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=index,
        base_commit=base,
        case_id="ceiling-case",
        verification_commands=["rtk python3 -m pytest"],
    )
    run = prepare_run(case, runs_root=runs_root, runner_label="Test Agent")

    # No changes made, no diff - should produce score ceilings
    (run.root / "completion_report.md").write_text(
        "status: completed\n\nVerified.\n",
        encoding="utf-8",
    )
    collect_run(run)
    scoring = score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    scoring_text = scoring.read_text(encoding="utf-8")

    assert "## Evidence Gate Assessment" in scoring_text
    # diff is missing, so result ceiling should be set
    assert "Result score ceiling: 0" in scoring_text or "Score Ceilings" in scoring_text
