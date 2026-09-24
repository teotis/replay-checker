from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from replay_checker.yaml_lite import write_simple_yaml
from tools import concurrent_tasks
from tools.replay import cmd_lint_task


def _write_case(root: Path, project: str, case_id: str) -> Path:
    case_dir = root / project / case_id
    case_dir.mkdir(parents=True)
    write_simple_yaml(
        case_dir / "case.yaml",
        {
            "id": case_id,
            "project_path": str(root.parent / "projects" / project),
            "base_source": "manual",
            "base_confidence": "high",
            "source_type": "manual",
            "source_path": "task.md",
            "selection_reason": "test",
        },
    )
    (case_dir / "task.md").write_text("# Task\n\n## Goal\n\nTest.\n", encoding="utf-8")
    return case_dir


def test_generate_manifest_only_lints_existing_cases(tmp_path: Path, monkeypatch) -> None:
    cases_root = tmp_path / "inventory"
    case_dir = _write_case(cases_root, "demo", "demo-1")
    monkeypatch.setattr(concurrent_tasks, "CASES_ROOT", cases_root)

    manifest = concurrent_tasks.generate_manifest(
        argparse.Namespace(project=None, max_concurrency=8)
    )

    assert len(manifest.tasks) == 1
    assert manifest.tasks[0].operation == "lint-task"
    assert manifest.tasks[0].args[-2:] == ["--run", str(case_dir)]


def test_run_manifest_rejects_arbitrary_argv(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "max_concurrency": 1,
                "tasks": [
                    {
                        "task_id": "T0001",
                        "case_id": "demo-1",
                        "project": "demo",
                        "operation": "lint-task",
                        "args": [
                            "/bin/sh",
                            "-c",
                            f"touch {marker}",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="allowlisted"):
        asyncio.run(
            concurrent_tasks.run_manifest(
                argparse.Namespace(
                    manifest=str(manifest_path),
                    dry_run=False,
                    max_concurrency=None,
                )
            )
        )

    assert not marker.exists()


def test_run_manifest_executes_generated_allowlisted_task(
    tmp_path: Path, monkeypatch
) -> None:
    cases_root = tmp_path / "inventory"
    _write_case(cases_root, "demo", "demo-1")
    monkeypatch.setattr(concurrent_tasks, "CASES_ROOT", cases_root)
    manifest = concurrent_tasks.generate_manifest(
        argparse.Namespace(project=None, max_concurrency=1)
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )

    asyncio.run(
        concurrent_tasks.run_manifest(
            argparse.Namespace(
                manifest=str(manifest_path),
                dry_run=False,
                max_concurrency=None,
            )
        )
    )

    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["status"] == "completed"


def test_phase_concurrency_budget_never_exceeds_global_limit() -> None:
    assert concurrent_tasks.phase_concurrency("intake-lint", 8) == 8
    assert concurrent_tasks.phase_concurrency("prepare-run", 8) == 1
    assert concurrent_tasks.package_parallelism("intake-lint", 8) == 1
    assert concurrent_tasks.package_parallelism("prepare-run", 8) == 8
    assert (
        concurrent_tasks.phase_concurrency("prepare-run", 8)
        * concurrent_tasks.package_parallelism("prepare-run", 8)
        <= 8
    )


def test_finalize_batch_rejects_failed_package_results(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    results = plan_root / "status" / "results"
    results.mkdir(parents=True)
    (results / "01.json").write_text(
        json.dumps({"package_id": "01", "project": "demo", "total": 2, "completed": 1, "failed": 1}),
        encoding="utf-8",
    )

    outcome = concurrent_tasks.finalize_batch(plan_root)

    assert outcome["ok"] is False
    assert outcome["failed"] == 1
    report = (plan_root / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "failed-no-merge" in report


def test_finalize_batch_requires_every_functional_package_result(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    (plan_root / "launchers").mkdir(parents=True)
    (plan_root / "launchers" / "package-graph.tsv").write_text(
        "package_id\tpackage_doc\tstatus_file\tdependencies\tdependency_type\twave\tbranch\tworktree\tmanual\tfinalize\n"
        "01\tpackages/01.md\tstatus/01.md\t\tstatus\t1\tb1\tnone\t0\t0\n"
        "02\tpackages/02.md\tstatus/02.md\t01\tstatus\t2\tb2\tnone\t0\t0\n"
        "99-finalize\tpackages/99.md\tstatus/99.md\t01,02\tstatus\tfinal\tb99\tnone\t0\t1\n",
        encoding="utf-8",
    )
    results = plan_root / "status" / "results"
    results.mkdir(parents=True)
    (results / "01.json").write_text(
        json.dumps({"package_id": "01", "project": "demo", "total": 1, "completed": 1, "failed": 0}),
        encoding="utf-8",
    )

    outcome = concurrent_tasks.finalize_batch(plan_root)

    assert outcome["ok"] is False
    assert outcome["missing_packages"] == ["02"]


def test_lint_task_cli_does_not_fail_on_warning_only(tmp_path: Path, capsys) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "task.md").write_text(
        "# Task\n\n## Goal\n\nA concrete historical task.\n",
        encoding="utf-8",
    )

    return_code = cmd_lint_task(argparse.Namespace(run=str(case_dir)))

    assert return_code == 0
    assert "WARNING:" in capsys.readouterr().err


def test_batch_launcher_propagates_dispatch_failure(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "docs" / "plans" / "batch-case-processing"
    plan_root = repo_root / "work" / "tmp" / f"pytest-batch-{tmp_path.name}"
    shutil.rmtree(plan_root, ignore_errors=True)
    shutil.copytree(source, plan_root)
    try:
        fake_dispatcher = plan_root / "fail_dispatcher.py"
        fake_dispatcher.write_text("raise SystemExit(7)\n", encoding="utf-8")
        launcher = plan_root / "launchers" / "orchestrate.sh"
        launcher.write_text(
            launcher.read_text(encoding="utf-8").replace(
                'DISPATCHER="$REPO_ROOT/tools/concurrent_tasks.py"',
                f'DISPATCHER="{fake_dispatcher}"',
            ),
            encoding="utf-8",
        )
        state_path = plan_root / "status" / "state.tsv"
        lines = state_path.read_text(encoding="utf-8").splitlines()
        rewritten = [lines[0]]
        for line in lines[1:]:
            columns = line.split("\t")
            columns[1] = "pending" if columns[0] == "01-intake-lint-image-factory" else "invalid"
            rewritten.append("\t".join(columns))
        state_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

        result = subprocess.run(
            ["bash", str(launcher), "start"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode != 0
        state = {
            line.split("\t")[0]: line.split("\t")[1]
            for line in state_path.read_text(encoding="utf-8").splitlines()[1:]
        }
        assert state["01-intake-lint-image-factory"] == "blocked"
    finally:
        shutil.rmtree(plan_root, ignore_errors=True)
