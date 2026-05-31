from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from replay_checker.linters import lint_task
from replay_checker.replay import create_case, prepare_run

ROOT = Path(__file__).resolve().parents[1]


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True,
    )


def _make_project(tmp_path):
    """Create a minimal valid git project with a plan package."""
    project = tmp_path / "proj"
    project.mkdir()
    (project / "docs" / "plans" / "demo" / "packages").mkdir(parents=True)
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "docs" / "plans" / "demo" / "INDEX.md").write_text(
        "# Demo Plan\n\n## Goal\nChange VALUE to 2.\n", encoding="utf-8",
    )
    (project / "docs" / "plans" / "demo" / "packages" / "01-change.md").write_text(
        "# Change Package\n\nSet `VALUE = 2`.\n", encoding="utf-8",
    )
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)
    base = _git(["rev-parse", "HEAD"], project).stdout.strip()
    return project, base


def _make_valid_run(tmp_path):
    """Create a complete valid run directory with task package."""
    project, base = _make_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=project / "docs" / "plans" / "demo" / "INDEX.md",
        base_commit=base,
        case_id="test-case",
        verification_commands=["pytest"],
    )
    run = prepare_run(case, runs_root=runs_root, runner_label="Agent")
    return run


def test_valid_run_task_lints_clean(tmp_path):
    run = _make_valid_run(tmp_path)
    diagnostics = lint_task(run.root)
    assert diagnostics == [], f"expected clean lint, got: {diagnostics}"


def test_valid_case_task_lints_clean(tmp_path):
    project, base = _make_project(tmp_path)
    cases_root = tmp_path / "cases"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=project / "docs" / "plans" / "demo" / "INDEX.md",
        base_commit=base,
        case_id="case-only",
    )
    diagnostics = lint_task(case.root)
    assert diagnostics == [], f"expected clean lint, got: {diagnostics}"


def test_missing_task_file_detected(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    (empty / "case.yaml").write_text("id: x\n", encoding="utf-8")
    diagnostics = lint_task(empty)
    assert any("missing task file" in d for d in diagnostics)


def test_missing_goal_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    # Strip all headings so there is no goal or source statement
    lines = [
        line for line in content.splitlines()
        if not (line.startswith("# ") or line.startswith("## "))
    ]
    task_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert any("no goal or source statement" in d for d in diagnostics)


def test_missing_output_contract_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content.replace("## Agent Output Contract", "## Something Else"),
        encoding="utf-8",
    )
    diagnostics = lint_task(run.root)
    assert any("missing output contract" in d for d in diagnostics)


def test_missing_completion_report_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content.replace("completion_report", "agent_output").replace(
            "## Completion Report Schema", "## Report Schema"
        ),
        encoding="utf-8",
    )
    diagnostics = lint_task(run.root)
    assert any("missing completion report schema" in d for d in diagnostics)


def test_missing_forbidden_access_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content.replace("## Forbidden Access", "## Permissions"),
        encoding="utf-8",
    )
    diagnostics = lint_task(run.root)
    assert any("missing forbidden access" in d for d in diagnostics)


def test_reference_leak_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content + "\n## Secret\n\n_reference/diff.patch contains the answer.\n",
        encoding="utf-8",
    )
    diagnostics = lint_task(run.root)
    assert any("reference leak" in d for d in diagnostics)


def test_case_task_reference_leak_detected(tmp_path):
    project, base = _make_project(tmp_path)
    cases_root = tmp_path / "cases"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=project / "docs" / "plans" / "demo" / "INDEX.md",
        base_commit=base,
        case_id="leaky-case",
    )
    task_path = case.root / "task.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content + "\nSee _reference/diff.patch for details.\n",
        encoding="utf-8",
    )
    diagnostics = lint_task(case.root)
    assert any("reference leak" in d for d in diagnostics)


def test_invalid_base_commit_detected(tmp_path):
    project, _base = _make_project(tmp_path)
    cases_root = tmp_path / "cases"
    case_dir = cases_root / "bad-base"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yaml").write_text(
        "\n".join([
            "id: bad-base",
            f"project_path: {project}",
            f"plan_path: {project / 'docs' / 'plans' / 'demo' / 'INDEX.md'}",
            "base_commit: not-a-real-sha",
            "base_source: manual",
            "base_confidence: user_supplied",
            "source_type: manual",
            f"source_path: {project / 'docs' / 'plans' / 'demo' / 'INDEX.md'}",
            "selection_reason: test",
            "synthetic_case: false",
            "evidence_sources:",
            "verification_commands:",
        ]),
        encoding="utf-8",
    )
    (case_dir / "task.md").write_text(
        "# Test\n\n## Source\n\nBase commit: `not-a-real-sha`\n",
        encoding="utf-8",
    )
    (case_dir / "evidence_sources.md").write_text("# Evidence\n", encoding="utf-8")
    diagnostics = lint_task(case_dir)
    assert any("invalid" in d and "not-a-real-sha" in d for d in diagnostics)


def test_missing_protocol_version_detected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    task_path.write_text(
        content.replace("Protocol Version: replay-checker-task-v2", ""),
        encoding="utf-8",
    )
    diagnostics = lint_task(run.root)
    assert any("missing protocol version" in d for d in diagnostics)


def test_cli_lint_task_exits_nonzero_for_bad_package(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "case.yaml").write_text("id: bad\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "lint-task",
            "--run",
            str(bad_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "ERROR:" in result.stderr


def test_cli_lint_task_exits_zero_for_valid_package(tmp_path):
    run = _make_valid_run(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "lint-task",
            "--run",
            str(run.root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Lint clean" in result.stdout


def test_prose_verification_entry_rejected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- Run the tests to verify the changes work\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert any("non-command verification entry" in d for d in diagnostics)
    assert "Run the tests to verify the changes work" in diagnostics[0]


def test_prose_verification_entry_with_must_rejected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- Tests should all pass\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert any("non-command verification entry" in d for d in diagnostics)


def test_prose_verification_entry_with_to_verb_rejected(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- to confirm the fix is correct\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert any("non-command verification entry" in d for d in diagnostics)


def test_valid_command_verification_passes(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- `rtk python3 -m pytest -q`\n- make preflight\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert not any("non-command verification entry" in d for d in diagnostics)


def test_short_verification_entries_pass(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- pytest\n- make test\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert not any("non-command verification entry" in d for d in diagnostics)


def test_verification_entry_with_shell_syntax_passes(tmp_path):
    run = _make_valid_run(tmp_path)
    task_path = run.root / "TASK.md"
    content = task_path.read_text(encoding="utf-8")
    content += "\n## Verification Commands\n\n- cat file && echo ok\n- grep -r foo | wc -l\n"
    task_path.write_text(content, encoding="utf-8")
    diagnostics = lint_task(run.root)
    assert not any("non-command verification entry" in d for d in diagnostics)


# --- Execution package compiler tests (from 01-execution-package-compiler) ---

from replay_checker.packages import (
    ExecutionPackage,
    ExecutionPackageCompiler,
    ExecutionPackageRenderer,
    InputContract,
    OutputContract,
    compile_execution_package,
)


class TestExecutionPackageSchema:
    def test_has_all_required_sections(self):
        pkg = compile_execution_package(
            run_id="proj-0001",
            case_id="proj",
            case_task_path="/cases/proj/task.md",
            plan_path="/docs/INDEX.md",
            workspace="/runs/proj-0001/workspace",
            completion_template_path="/runs/proj-0001/completion_report_template.md",
            verification_commands=("rtk make test",),
        )
        assert pkg.goal
        assert pkg.starting_point
        assert pkg.allowed_scope
        assert pkg.expected_output
        assert pkg.verification_contract
        assert pkg.completion_report_schema
        assert pkg.forbidden_access
        assert pkg.ambiguity_handling
        assert pkg.source_material
        assert pkg.protocol_version == "replay-checker-task-v2"

    def test_protocol_version_is_v2(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert pkg.protocol_version == "replay-checker-task-v2"


class TestExecutionPackageCompilerSections:
    def test_forbidden_access_mentions_reference(self):
        compiler = ExecutionPackageCompiler(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        pkg = compiler.compile()
        assert "_reference/" in pkg.forbidden_access
        assert "secrets" in pkg.forbidden_access.lower()

    def test_goal_and_starting_point_are_non_empty(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert len(pkg.goal) > 10
        assert len(pkg.starting_point) > 10

    def test_verification_contract_includes_commands(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
            verification_commands=("rtk make test", "rtk make preflight"),
        )
        assert "rtk make test" in pkg.verification_contract
        assert "rtk make preflight" in pkg.verification_contract

    def test_verification_contract_empty_commands(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
            verification_commands=(),
        )
        assert "No verification commands" in pkg.verification_contract

    def test_completion_report_schema_includes_status(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert "status" in pkg.completion_report_schema.lower()

    def test_ambiguity_handling_mentions_interpretation(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert "interpretation" in pkg.ambiguity_handling.lower()

    def test_input_contract_captures_paths(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert pkg.input_contract.workspace == "/r1/ws"
        assert pkg.input_contract.case_task_path == "/c/task.md"
        assert pkg.input_contract.completion_template_path == "/r1/tpl.md"

    def test_output_contract_captures_workspace(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert pkg.output_contract.modify_files_inside_workspace == "/r1/ws"
        assert "completion_report.md" in pkg.output_contract.write_completion_report


class TestExecutionPackageRenderer:
    def test_render_includes_all_required_headers(self):
        pkg = compile_execution_package(
            run_id="proj-0001",
            case_id="proj",
            case_task_path="/cases/proj/task.md",
            plan_path="/docs/INDEX.md",
            workspace="/runs/proj-0001/workspace",
            completion_template_path="/runs/proj-0001/tpl.md",
            verification_commands=("rtk make test",),
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "Protocol Version: replay-checker-task-v2" in text
        assert "## Goal" in text
        assert "## Starting Point" in text
        assert "## Allowed Scope" in text
        assert "## Expected Output" in text
        assert "## Verification Contract" in text
        assert "## Completion Report Schema" in text
        assert "## Forbidden Access" in text
        assert "## Ambiguity Handling" in text
        assert "## Source" in text
        assert "## Inputs" in text
        assert "## Verification Commands" in text

    def test_render_includes_verification_commands(self):
        pkg = compile_execution_package(
            run_id="proj-0001",
            case_id="proj",
            case_task_path="/cases/proj/task.md",
            plan_path="/docs/INDEX.md",
            workspace="/runs/proj-0001/workspace",
            completion_template_path="/runs/proj-0001/tpl.md",
            verification_commands=("rtk make test",),
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "`rtk make test`" in text

    def test_render_no_conversation_evidence_section_when_empty(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "## Conversation Evidence" not in text

    def test_render_includes_conversation_evidence_when_present(self):
        pkg = ExecutionPackage(
            protocol_version="replay-checker-task-v2",
            run_id="r1",
            goal="test",
            starting_point="test",
            allowed_scope="test",
            expected_output="test",
            verification_contract="test",
            completion_report_schema="test",
            forbidden_access="test",
            ambiguity_handling="test",
            source_material="test",
            input_contract=InputContract(
                workspace="/r1/ws",
                case_task_path="/c/task.md",
                completion_template_path="/r1/tpl.md",
            ),
            output_contract=OutputContract(
                modify_files_inside_workspace="/r1/ws",
                write_completion_report="completion_report.md",
                include_status_summary="status",
            ),
            runner_instructions="test",
            conversation_evidence="codex_history: some summary",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "## Conversation Evidence" in text
        assert "some summary" in text

    def test_render_preserves_protocol_version(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "Protocol Version: replay-checker-task-v2" in text


class TestOracleIsolation:
    def test_execution_package_never_includes_reference_path(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        # _reference should only appear in forbidden_access as a prohibition
        assert "/_reference/diff.patch" not in text
        assert "/_reference/reference_metadata.yaml" not in text

    def test_forbidden_access_prohibits_reference(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert "Do not read `_reference/`" in pkg.forbidden_access

    def test_forbidden_access_prohibits_runner_identity(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert "runner identity" in pkg.forbidden_access.lower()

    def test_execution_package_no_scoring_details(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        # Should not expose scoring internals
        assert "result_weight" not in text
        assert "process_weight" not in text

    def test_execution_package_excludes_reference_diff_content(self):
        """Reference diffs must never be embedded in the execution package."""
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        renderer = ExecutionPackageRenderer()
        text = renderer.render(pkg)
        assert "diff.patch" not in text or text.count("diff.patch") == 0

    def test_input_contract_excludes_reference_paths(self):
        pkg = compile_execution_package(
            run_id="r1",
            case_id="c1",
            case_task_path="/c/task.md",
            plan_path="/p/INDEX.md",
            workspace="/r1/ws",
            completion_template_path="/r1/tpl.md",
        )
        assert "_reference" not in pkg.input_contract.workspace
        assert "_reference" not in pkg.input_contract.case_task_path
        assert "_reference" not in pkg.input_contract.completion_template_path
