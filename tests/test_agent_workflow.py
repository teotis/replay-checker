from __future__ import annotations

import os
import re
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from replay_checker.agent_workflow import (
    build_default_runner_label,
    create_extraction_flow,
    run_grading_flow,
)
from replay_checker.evaluation import Recommendation, write_recommendation
from replay_checker.replay import (
    batch_intake,
    collect_run,
    compare_case,
    create_case,
    doctor_runs,
    _extract_verification,
    inspect_run_dir,
    intake,
    parse_simple_yaml,
    prepare_run,
    report_all_cases,
)
from tools.replay import _print_preview


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)


def _init_project(path: Path) -> None:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "README.md").write_text("# Demo\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")


def test_default_runner_label_uses_date_and_agent_intro() -> None:
    label = build_default_runner_label(
        "Codex GPT-5, senior engineer; prefers tests",
        today=date(2026, 6, 1),
    )

    assert label == "2026-06-01-codex_gpt_5_senior_engineer_prefers_tests"


def test_create_extraction_flow_generates_kit_and_launch_options(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)

    flow = create_extraction_flow(
        project_path=project,
        scope="extract auth replay cases",
        output_root=tmp_path / "plans",
    )

    assert flow.kit_path.exists()
    assert flow.package_ids == (
        "01-plan-scan",
        "02-git-discovery",
        "03-case-merge",
        "99-finalize",
    )
    assert [option.key for option in flow.launch_options] == [
        "agents-view",
        "manual-prompts",
        "status-only",
    ]
    assert "Which launch mode" in flow.next_question


def test_external_extraction_kit_status_runs_outside_target_project(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)

    flow = create_extraction_flow(
        project_path=project,
        scope="external kit status",
        output_root=tmp_path / "external-plans",
    )

    result = subprocess.run(
        ["bash", str(flow.kit_path / "launchers" / "orchestrate.sh"), "status"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Coordinator consistency" in result.stdout


def test_generated_orchestrator_parses_colored_claude_session_id(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)

    flow = create_extraction_flow(
        project_path=project,
        scope="extract auth replay cases",
        output_root=tmp_path / "plans",
    )
    script = (flow.kit_path / "launchers" / "orchestrate.sh").read_text(encoding="utf-8")
    match = re.search(r"parse_session_id\(\) \{\n.*?\n\}", script, re.DOTALL)
    assert match is not None

    probe = tmp_path / "probe.sh"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"{match.group(0)}\n"
        "printf 'backgrounded · \\033[36m94390120\\033[39m\\n' | parse_session_id\n",
        encoding="utf-8",
    )

    result = subprocess.run(["bash", str(probe)], check=True, text=True, capture_output=True)

    assert result.stdout.strip() == "94390120"


def test_code_dependency_launch_uses_completed_upstream_commit(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    original_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    _git(project, "checkout", "-b", "agent/demo/01-base")
    (project / "upstream.txt").write_text("upstream work\n", encoding="utf-8")
    _git(project, "add", "upstream.txt")
    _git(project, "commit", "-m", "upstream work")
    upstream_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    _git(project, "checkout", original_branch)

    flow = create_extraction_flow(
        project_path=project,
        scope="serial code dependency",
        output_root=project / "docs" / "plans",
    )
    kit = flow.kit_path
    worktree_root = tmp_path / "worktrees"
    dependent_worktree = worktree_root / "02-dependent"

    (kit / "launchers" / "package-graph.tsv").write_text(
        "\t".join(
            [
                "package_id",
                "package_doc",
                "status_file",
                "dependencies",
                "dependency_type",
                "wave",
                "branch",
                "worktree",
                "manual",
                "finalize",
            ]
        )
        + "\n"
        + "\n".join(
            [
                "\t".join(
                    [
                        "01-base",
                        "packages/01-base/status.md",
                        "status/01-base.md",
                        "",
                        "code",
                        "1",
                        "agent/demo/01-base",
                        str(worktree_root / "01-base"),
                        "0",
                        "0",
                    ]
                ),
                "\t".join(
                    [
                        "02-dependent",
                        "packages/02-dependent/status.md",
                        "status/02-dependent.md",
                        "01-base",
                        "code",
                        "2",
                        "agent/demo/02-dependent",
                        str(dependent_worktree),
                        "0",
                        "0",
                    ]
                ),
                "\t".join(
                    [
                        "99-finalize",
                        "packages/99-finalize/status.md",
                        "status/99-finalize.md",
                        "01-base,02-dependent",
                        "status+code",
                        "final",
                        "agent/demo/99-finalize",
                        str(worktree_root / "99-finalize"),
                        "0",
                        "1",
                    ]
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (kit / "status" / "state.tsv").write_text(
        "\t".join(
            [
                "package_id",
                "state",
                "launched_at",
                "completed_at",
                "agent",
                "branch",
                "worktree",
                "base_commit",
                "commit_hash",
                "verification",
                "integration",
                "cleanup",
                "last_error",
                "failed_command",
                "conflict_files",
                "log_summary",
                "recovery_hint",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "01-base",
                "completed",
                "",
                "",
                "",
                "agent/demo/01-base",
                str(worktree_root / "01-base"),
                "",
                upstream_commit,
                "passed",
                "pending",
                "pending",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "02-dependent",
                "pending",
                "",
                "",
                "",
                "agent/demo/02-dependent",
                str(dependent_worktree),
                "",
                "",
                "pending",
                "pending",
                "pending",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        + "\n"
        + "\t".join(
            [
                "99-finalize",
                "pending",
                "",
                "",
                "",
                "agent/demo/99-finalize",
                str(worktree_root / "99-finalize"),
                "",
                "",
                "pending",
                "pending",
                "pending",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for package_id, state in {"01-base": "completed", "02-dependent": "pending", "99-finalize": "pending"}.items():
        status_file = kit / "status" / f"{package_id}.md"
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text(f"# {package_id}\n\n## State\n\n`{state}`\n", encoding="utf-8")
    (kit / "launchers" / "agent-prompts.md").write_text(
        "# Agent Prompts\n\n## Package: 02-dependent - Dependent\n\nDo nothing.\n\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_claude = fake_bin / "claude"
    fake_claude.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"--version\" ]; then echo fake claude; exit 0; fi\n"
        "if [ \"$1\" = \"logs\" ]; then exit 0; fi\n"
        "if [ \"$1\" = \"agents\" ]; then exit 0; fi\n"
        "if [ \"$1\" = \"--bg\" ]; then echo 'backgrounded · fake1234'; exit 0; fi\n"
        "echo \"unexpected claude args: $*\" >&2\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    subprocess.run(
        ["bash", str(kit / "launchers" / "orchestrate.sh"), "start"],
        cwd=project,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )

    dependent_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=dependent_worktree,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dependent_state = next(
        line for line in (kit / "status" / "state.tsv").read_text(encoding="utf-8").splitlines()
        if line.startswith("02-dependent\t")
    )

    assert dependent_head == upstream_commit
    assert dependent_state.split("\t")[7] == upstream_commit


def test_intake_from_git_history_writes_synthetic_case_task_and_reference(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    (project / "README.md").write_text("# Demo\n\nHistorical change\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(project, "commit", "-m", "add historical change")

    case = intake(cases_root=tmp_path / "cases", project_path=project)

    assert case.synthetic_case is True
    assert case.source_type == "git_history"
    assert (case.root / "task.md").exists()
    assert "The historical commit message was: **add historical change**" in (
        case.root / "task.md"
    ).read_text(encoding="utf-8")
    assert (case.root / "_reference" / "diff.patch").exists()

    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")

    assert run.id == f"{case.id}-001"
    assert (run.root / "TASK.md").exists()


def test_synthetic_execution_package_goal_uses_reconstructed_task(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    (project / "README.md").write_text("# Demo\n\nHistorical change\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(project, "commit", "-m", "add historical change")

    case = intake(cases_root=tmp_path / "cases", project_path=project)
    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")
    task_text = (run.root / "TASK.md").read_text(encoding="utf-8")

    assert "## Goal\nThe historical commit message was: **add historical change**" in task_text
    assert "Replay a historical project situation" not in task_text


def test_extract_verification_reads_bash_code_blocks(tmp_path: Path) -> None:
    project = tmp_path / "target"
    plan_dir = project / "docs" / "plans" / "demo"
    plan_dir.mkdir(parents=True)
    (plan_dir / "INDEX.md").write_text(
        "# Demo\n\n"
        "## Verification Commands\n\n"
        "```bash\n"
        "make test\n"
        "python3 tools/project.py check\n"
        "# prose comments are ignored\n"
        "```\n",
        encoding="utf-8",
    )

    assert _extract_verification(project, plan_dir) == [
        "make test",
        "python3 tools/project.py check",
    ]


def test_create_case_resolves_relative_plan_path_from_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "target"
    _init_project(project)
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        "# Demo Plan\n\n## Verification Commands\n\n```bash\nmake test\n```\n",
        encoding="utf-8",
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=Path("docs/plans/demo/INDEX.md"),
        base_commit=base,
        case_id="demo-case",
    )

    assert case.plan_path == plan.resolve()
    assert case.verification_commands == ("make test",)


def test_create_case_writes_goal_for_manual_plan(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        "# Demo Plan\n\n"
        "## Task\n\n"
        "Implement the focused demo improvement with regression coverage.\n",
        encoding="utf-8",
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )

    task = (case.root / "task.md").read_text(encoding="utf-8")
    assert "## Goal\nImplement the focused demo improvement with regression coverage." in task


def test_reference_evidence_uses_cumulative_base_to_target_diff(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")
    _git(project, "add", "docs/plans/demo/INDEX.md")
    _git(project, "commit", "-m", "add demo plan")
    (project / "app.py").write_text("print('done')\n", encoding="utf-8")
    _git(project, "add", "app.py")
    _git(project, "commit", "-m", "implement demo plan")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )

    diff = (case.root / "_reference" / "diff.patch").read_text(encoding="utf-8")
    assert "docs/plans/demo/INDEX.md" in diff
    assert "app.py" in diff


def test_reference_metadata_handles_multiline_commit_message(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")
    (project / "app.py").write_text("print('done')\n", encoding="utf-8")
    _git(project, "add", "docs/plans/demo/INDEX.md", "app.py")
    _git(project, "commit", "-m", "implement demo", "-m", "contains detail:\n- one\n- two")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    metadata = parse_simple_yaml(case.root / "_reference" / "reference_metadata.yaml")

    assert metadata["changed_files"] == ["app.py", "docs/plans/demo/INDEX.md"]
    assert "contains detail:" in metadata["commit_message"]


def test_batch_intake_uses_plan_state_base_commits(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base_one = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (project / "README.md").write_text("# Demo\n\nFirst change\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(project, "commit", "-m", "first change")
    base_two = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (project / "README.md").write_text("# Demo\n\nSecond change\n", encoding="utf-8")
    _git(project, "add", "README.md")
    _git(project, "commit", "-m", "second change")
    target_two = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    for name, base, target in (("kit-one", base_one, base_two), ("kit-two", base_two, target_two)):
        kit = project / "docs" / "plans" / name
        (kit / "packages").mkdir(parents=True)
        (kit / "status").mkdir()
        (kit / "INDEX.md").write_text(
            f"# {name}\n\n## Goal\nImplement the {name} replay target with focused regression coverage.\n\n"
            "## Verification Commands\n\n```bash\npython3 -m pytest\n```\n",
            encoding="utf-8",
        )
        (kit / "packages" / "01-work.md").write_text("# Work\n", encoding="utf-8")
        (kit / "status" / "state.tsv").write_text(
            "package_id\tstate\tlaunched_at\tcompleted_at\tagent\tbranch\tworktree\tbase_commit\tcommit_hash\tverification\tintegration\tcleanup\tlast_error\tfailed_command\tconflict_files\tlog_summary\trecovery_hint\n"
            f"01-work\tcompleted\t\t\t\t\t\t{base}\t{target}\t\t\t\t\t\t\t\t\n",
            encoding="utf-8",
        )
    _git(project, "add", "docs/plans")
    _git(project, "commit", "-m", "add replay kits")

    cases = batch_intake(
        cases_root=tmp_path / "cases",
        project_path=project,
        min_score=1,
        max_cases=10,
        allow_duplicate=True,
    )

    bases_by_plan = {Path(case.plan_path).parent.name: case.base_commit for case in cases}
    assert bases_by_plan["kit-one"] == base_one
    assert bases_by_plan["kit-two"] == base_two


def test_batch_intake_filters_low_quality_cases_before_returning(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    (project / "app.py").write_text("print('target')\n", encoding="utf-8")
    _git(project, "add", "app.py")
    _git(project, "commit", "-m", "target implementation")
    target = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()

    def write_kit(name: str, *, verification: bool, target_commit: str) -> None:
        kit = project / "docs" / "plans" / name
        (kit / "packages").mkdir(parents=True)
        (kit / "status").mkdir()
        verification_section = (
            "\n## Verification Commands\n\n```bash\npython3 -m pytest\n```\n"
            if verification else ""
        )
        (kit / "INDEX.md").write_text(
            f"# {name}\n\n## Goal\nImplement {name} safely.\n{verification_section}",
            encoding="utf-8",
        )
        (kit / "packages" / "01-work.md").write_text("# Work\n", encoding="utf-8")
        (kit / "status" / "state.tsv").write_text(
            "package_id\tstate\tlaunched_at\tcompleted_at\tagent\tbranch\tworktree\tbase_commit\tcommit_hash\tverification\tintegration\tcleanup\tlast_error\tfailed_command\tconflict_files\tlog_summary\trecovery_hint\n"
            f"01-work\tcompleted\t\t\t\t\t\t{base}\t{target_commit}\t\t\t\t\t\t\t\t\n",
            encoding="utf-8",
        )

    write_kit("good-kit", verification=True, target_commit=target)
    write_kit("no-verification-kit", verification=False, target_commit=target)
    write_kit("no-reference-kit", verification=True, target_commit=base)
    _git(project, "add", "docs/plans")
    _git(project, "commit", "-m", "add replay kits")

    cases = batch_intake(
        cases_root=tmp_path / "cases",
        project_path=project,
        min_score=1,
        max_cases=10,
    )

    assert [Path(case.plan_path).parent.name for case in cases] == ["good-kit"]
    assert sorted(path.name for path in (tmp_path / "cases").iterdir()) == [cases[0].id]


def test_grading_flow_collects_scores_and_writes_compare_report(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(
        case,
        runs_root=tmp_path / "runs",
        runner_label="2026-06-01-codex",
    )
    (run.workspace / "README.md").write_text("# Demo\n\nChanged\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\n## Verification\n\n- Not run\n",
        encoding="utf-8",
    )

    result = run_grading_flow(
        case_id=case.id,
        run_ids=(run.id,),
        cases_root=tmp_path / "cases",
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )

    assert result.scored_runs == (run.id,)
    assert (run.root / "evidence" / "evidence.yaml").exists()
    assert (run.root / "scoring_package.md").exists()
    assert result.compare_report == tmp_path / "reports" / "demo-case.md"
    assert result.compare_report.exists()


def test_prepare_run_allocates_next_run_id_after_numbering_gap(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    first = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")
    stale_manual = tmp_path / "runs" / "demo-case-003"
    stale_manual.mkdir(parents=True)
    (stale_manual / "run.yaml").write_text(
        "id: demo-case-003\ncase_id: demo-case\nrunner_label: manual\n",
        encoding="utf-8",
    )

    next_run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")

    assert first.id == "demo-case-001"
    assert next_run.id == "demo-case-004"


def test_collect_run_updates_run_yaml_with_canonical_completed_status(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")
    (run.workspace / "README.md").write_text("# Demo\n\nChanged\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text("status: completed\n", encoding="utf-8")

    collect_run(run)

    run_data = parse_simple_yaml(run.root / "run.yaml")
    assert run_data["status"] == "completed"
    assert run_data["raw_status"] == "completed"
    assert run_data["has_result_diff"] == "yes"
    assert run_data["completion_report_exists"] == "yes"


def test_collect_run_accepts_workspace_completion_report(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")
    (run.workspace / "README.md").write_text("# Demo\n\nChanged\n", encoding="utf-8")
    (run.workspace / "completion_report.md").write_text("status: complete\n", encoding="utf-8")

    collect_run(run)

    run_data = parse_simple_yaml(run.root / "run.yaml")
    assert run_data["status"] == "completed"
    assert run_data["raw_status"] == "complete"
    assert run_data["completion_report_exists"] == "yes"


def test_run_doctor_marks_partial_report_as_partial(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")
    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(case, runs_root=tmp_path / "runs", runner_label="agent")
    (run.root / "completion_report.md").write_text("status: partial\n", encoding="utf-8")

    health = inspect_run_dir(run.root)

    assert health.status == "partial"
    assert health.raw_status == "partial"
    assert health.reason == "completion report status is partial"


def test_run_doctor_marks_missing_workspace_as_failed(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_root = runs_root / "demo-case-001"
    run_root.mkdir(parents=True)
    (run_root / "run.yaml").write_text(
        "id: demo-case-001\n"
        "case_id: demo-case\n"
        "runner_label: agent\n"
        "workspace: runs/demo-case-001/workspace\n"
        "status: prepared\n",
        encoding="utf-8",
    )

    health = inspect_run_dir(run_root)

    assert health.status == "failed"
    assert health.reason == "workspace missing"


def test_doctor_runs_filters_by_runner_label_and_exposes_missing_reports(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    for run_id, label in (("demo-case-001", "wanted"), ("demo-case-002", "other")):
        run_root = runs_root / run_id
        workspace = run_root / "workspace"
        workspace.mkdir(parents=True)
        _git(workspace, "init")
        (run_root / "run.yaml").write_text(
            f"id: {run_id}\n"
            "case_id: demo-case\n"
            f"runner_label: {label}\n"
            "workspace: workspace\n"
            "status: prepared\n",
            encoding="utf-8",
        )

    reports = doctor_runs(runs_root, runner_label="wanted")

    assert [report.run_id for report in reports] == ["demo-case-001"]
    assert reports[0].status == "failed"
    assert reports[0].reason == "completion_report.md missing"


def test_doctor_runs_returns_empty_for_missing_runs_root(tmp_path: Path) -> None:
    reports = doctor_runs(tmp_path / "missing-runs")

    assert reports == ()


def test_compare_default_uses_concise_recommendation_when_available(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(
        case,
        runs_root=tmp_path / "runs",
        runner_label="2026-06-01-codex",
    )
    (run.workspace / "README.md").write_text("# Demo\n\nChanged\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\n## Verification\n\n- Not run\n",
        encoding="utf-8",
    )
    collect_run(run)
    write_recommendation(
        case.root,
        Recommendation(
            score=82.0,
            sentence=(
                "Suitable for low-risk implementation; not the first choice "
                "for open-ended architecture exploration."
            ),
            validity="valid",
            confidence="medium",
        ),
    )

    report = compare_case(case, runs_root=tmp_path / "runs", reports_root=tmp_path / "reports")
    text = report.read_text(encoding="utf-8")

    assert "82 / 100" in text
    assert "Suitable for low-risk implementation" in text
    assert "Validity: valid; Confidence: medium" in text


def test_compare_default_does_not_show_stale_positive_score_when_gates_fail(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(
        case,
        runs_root=tmp_path / "runs",
        runner_label="2026-06-01-codex",
    )
    (run.root / "completion_report.md").write_text(
        "status: completed\n\n## Verification\n\n- Not run\n",
        encoding="utf-8",
    )
    collect_run(run)
    write_recommendation(
        case.root,
        Recommendation(
            score=82.0,
            sentence="Stale positive recommendation.",
            validity="valid",
            confidence="medium",
        ),
    )

    report = compare_case(case, runs_root=tmp_path / "runs", reports_root=tmp_path / "reports")
    text = report.read_text(encoding="utf-8")

    assert "82 / 100" not in text
    assert "insufficient evidence" in text


def test_aggregate_report_does_not_rank_stale_positive_score_when_gates_fail(tmp_path: Path) -> None:
    project = tmp_path / "target"
    _init_project(project)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    plan = project / "docs" / "plans" / "demo" / "INDEX.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("# Demo Plan\n", encoding="utf-8")

    case = create_case(
        cases_root=tmp_path / "cases",
        project_path=project,
        plan_path=plan,
        base_commit=base,
        case_id="demo-case",
    )
    run = prepare_run(
        case,
        runs_root=tmp_path / "runs",
        runner_label="2026-06-01-codex",
    )
    (run.root / "completion_report.md").write_text(
        "status: completed\n\n## Verification\n\n- Not run\n",
        encoding="utf-8",
    )
    collect_run(run)
    write_recommendation(
        case.root,
        Recommendation(
            score=82.0,
            sentence="Stale positive recommendation.",
            validity="valid",
            confidence="medium",
        ),
    )

    report = report_all_cases(
        cases_root=tmp_path / "cases",
        runs_root=tmp_path / "runs",
        reports_root=tmp_path / "reports",
    )
    text = report.read_text(encoding="utf-8")

    assert "| 1 | demo-case | manual | 0 | x invalid |" in text
    assert "Stale positive recommendation" not in text
    assert "| Anonymous ID |" not in text


def test_wizard_preview_defaults_to_top_five_candidates(tmp_path: Path, capsys) -> None:
    project = tmp_path / "target"
    project.mkdir()
    candidates = [
        SimpleNamespace(
            candidate_id=f"candidate-{idx}",
            source_type="plan",
            relevance_score=1.0 - (idx / 100),
            primary_source=f"docs/plans/{idx}.md",
            risks=(),
        )
        for idx in range(8)
    ]
    preview = SimpleNamespace(
        project_path=project,
        scope="demo",
        has_git=True,
        git_clean=True,
        head_commit="abc123",
        signals=("8 case candidate(s) ranked",),
        risks=(),
        candidate_count=8,
        candidates=candidates,
        top_candidate=candidates[0],
        next_commands=("rtk python3 tools/replay.py intake --project /tmp/target",),
    )

    _print_preview(preview)
    output = capsys.readouterr().out

    assert "1. candidate-0" in output
    assert "5. candidate-4" in output
    assert "6. candidate-5" not in output
    assert "3 more candidate(s) hidden" in output
