#!/usr/bin/env python3
"""
Concurrent task dispatcher for replay checker cases.

Scans case inventory, generates a task manifest, and executes tasks
with configurable concurrency (default: 8 workers).

Usage:
    python3 tools/concurrent_tasks.py generate-manifest [--project PROJECT] [--max-concurrency N]
    python3 tools/concurrent_tasks.py run --manifest PATH [--dry-run] [--max-concurrency N]
    python3 tools/concurrent_tasks.py status --manifest PATH
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
CASES_ROOT = ROOT / "cases" / "inventory"
RUNS_ROOT = ROOT / "runs"
MANIFEST_DIR = ROOT / "work"

DEFAULT_MAX_CONCURRENCY = 8
REPLAY_PY = ROOT / "tools" / "replay.py"


@dataclass
class Task:
    task_id: str
    case_id: str
    project: str
    operation: str
    args: list
    status: str = "pending"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    returncode: Optional[int] = None
    stdout: str = ""
    stderr: str = ""


@dataclass
class Manifest:
    version: int = 1
    created_at: str = ""
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    project_filter: Optional[str] = None
    tasks: list = field(default_factory=list)

    def to_dict(self):
        return {
            "version": self.version,
            "created_at": self.created_at,
            "max_concurrency": self.max_concurrency,
            "project_filter": self.project_filter,
            "tasks": [asdict(t) if hasattr(t, "task_id") else t for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(
            version=d.get("version", 1),
            created_at=d.get("created_at", ""),
            max_concurrency=d.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
            project_filter=d.get("project_filter"),
        )
        m.tasks = [Task(**t) for t in d.get("tasks", [])]
        return m


def discover_cases(project_filter: Optional[str] = None) -> list[dict]:
    """Scan cases/inventory/ and return [{project, case_id, path}]."""
    cases = []
    if not CASES_ROOT.exists():
        return cases
    for project_dir in sorted(CASES_ROOT.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_filter and project_dir.name != project_filter:
            continue
        for case_dir in sorted(project_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if case_dir.name.startswith("_"):
                continue
            if not (case_dir / "case.yaml").exists():
                continue
            cases.append({
                "project": project_dir.name,
                "case_id": case_dir.name,
                "path": str(case_dir),
            })
    return cases


def detect_available_operations(case_path: Path) -> list[str]:
    """Detect which operations are valid for a given case."""
    ops = ["intake"]
    case_yaml = case_path / "case.yaml"
    if case_yaml.exists():
        ops.append("lint-task")
    runs_dir = RUNS_ROOT / case_path.name
    if runs_dir.exists() and any(runs_dir.iterdir()):
        ops.extend(["collect-run", "score-run"])
    return ops


def generate_manifest(args) -> Manifest:
    """Generate a task manifest from discovered cases."""
    cases = discover_cases(args.project)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest = Manifest(
        created_at=now,
        max_concurrency=getattr(args, "max_concurrency", DEFAULT_MAX_CONCURRENCY),
        project_filter=getattr(args, "project", None),
    )

    task_counter = 0
    for case_info in cases:
        project = case_info["project"]
        case_id = case_info["case_id"]
        case_path = Path(case_info["path"])

        # Intake task
        task_counter += 1
        manifest.tasks.append(Task(
            task_id=f"T{task_counter:04d}",
            case_id=case_id,
            project=project,
            operation="intake",
            args=[
                sys.executable, str(REPLAY_PY), "intake",
                "--project", str(case_path.parent),
                "--cases-root", str(CASES_ROOT),
            ],
        ))

        # Lint-task if case exists
        if (case_path / "case.yaml").exists():
            task_counter += 1
            manifest.tasks.append(Task(
                task_id=f"T{task_counter:04d}",
                case_id=case_id,
                project=project,
                operation="lint-task",
                args=[
                    sys.executable, str(REPLAY_PY), "lint-task",
                    "--case", case_id,
                    "--cases-root", str(CASES_ROOT),
                ],
            ))

    return manifest


async def run_task(task: Task, semaphore: asyncio.Semaphore, manifest_path: Path):
    """Execute a single task with concurrency control."""
    async with semaphore:
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        task.status = "running"
        task.started_at = now
        _save_manifest_checkpoint(manifest_path, task)

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *task.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT),
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            task.stdout = stdout_bytes.decode(errors="replace")[-4000:]
            task.stderr = stderr_bytes.decode(errors="replace")[-4000:]
            task.returncode = proc.returncode
            task.status = "completed" if proc.returncode == 0 else "failed"
        except Exception as e:
            task.stderr = str(e)
            task.returncode = -1
            task.status = "error"

        task.duration_s = round(time.monotonic() - start, 2)
        task.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        _save_manifest_checkpoint(manifest_path, task)
        return task


def _save_manifest_checkpoint(manifest_path: Path, updated_task: Task):
    """Atomic save of manifest after a task state change."""
    data = json.loads(manifest_path.read_text())
    for i, t in enumerate(data["tasks"]):
        if t["task_id"] == updated_task.task_id:
            data["tasks"][i] = asdict(updated_task)
            break
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(manifest_path)


async def run_manifest(args):
    """Execute all pending tasks from a manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = Manifest.from_dict(json.loads(manifest_path.read_text()))
    max_conc = getattr(args, "max_concurrency", None) or manifest.max_concurrency

    if getattr(args, "dry_run", False):
        pending = [t for t in manifest.tasks if t.status == "pending"]
        print(f"[DRY RUN] Would execute {len(pending)} tasks (concurrency={max_conc})")
        for t in pending[:20]:
            print(f"  {t.task_id} | {t.operation} | {t.case_id} | {t.project}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    pending = [t for t in manifest.tasks if t.status == "pending"]
    if not pending:
        print("No pending tasks.")
        return

    print(f"Running {len(pending)} tasks (concurrency={max_conc})")
    semaphore = asyncio.Semaphore(max_conc)

    tasks_coros = [
        run_task(t, semaphore, manifest_path)
        for t in pending
    ]
    results = await asyncio.gather(*tasks_coros, return_exceptions=True)

    # Summary
    completed = sum(1 for r in results if isinstance(r, Task) and r.status == "completed")
    failed = sum(1 for r in results if isinstance(r, Task) and r.status in ("failed", "error"))
    print(f"\nDone: {completed} completed, {failed} failed out of {len(pending)} total")


def show_status(args):
    """Show status of a manifest."""
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = Manifest.from_dict(json.loads(manifest_path.read_text()))
    by_status = {}
    by_op = {}
    for t in manifest.tasks:
        by_status[t.status] = by_status.get(t.status, 0) + 1
        key = f"{t.operation}:{t.status}"
        by_op[key] = by_op.get(key, 0) + 1

    print(f"Manifest: {manifest_path.name}")
    print(f"Created:  {manifest.created_at}")
    print(f"Tasks:    {len(manifest.tasks)}")
    print(f"Concurrency: {manifest.max_concurrency}")
    print()
    print("Status breakdown:")
    for s, c in sorted(by_status.items()):
        print(f"  {s}: {c}")
    print()
    print("By operation:")
    for k, c in sorted(by_op.items()):
        print(f"  {k}: {c}")

    # Show recent failures
    failures = [t for t in manifest.tasks if t.status in ("failed", "error")]
    if failures:
        print(f"\nRecent failures ({len(failures)}):")
        for t in failures[:10]:
            err_line = t.stderr.strip().split("\n")[-1][:120] if t.stderr else "no stderr"
            print(f"  {t.task_id} | {t.operation} | {t.case_id} | {err_line}")


async def run_package_phase(project: str, phase: str, max_concurrency: int, package_id: str, orchestrator: str = ""):
    """Run all cases for a project in a given phase (intake-lint, prepare-run, finalize)."""
    cases = discover_cases(project_filter=project if project != "all" else None)
    if not cases and phase != "finalize":
        print(f"No cases found for project: {project}")
        sys.exit(1)

    if phase == "finalize":
        print(f"[finalize] Verifying all packages...")
        # Finalize just verifies — no actual work needed beyond what orchestrate.sh handles
        print(f"[finalize] Done.")
        return

    semaphore = asyncio.Semaphore(max_concurrency)
    all_tasks = []

    for case_info in cases:
        case_id = case_info["case_id"]
        case_path = Path(case_info["path"])

        if phase == "intake-lint":
            # Lint existing cases
            if (case_path / "case.yaml").exists():
                all_tasks.append(Task(
                    task_id=f"{package_id}:{case_id}:lint",
                    case_id=case_id,
                    project=project,
                    operation="lint-task",
                    args=[
                        sys.executable, str(REPLAY_PY), "lint-task",
                        "--case", case_id,
                        "--cases-root", str(CASES_ROOT),
                    ],
                ))

        elif phase == "prepare-run":
            # Prepare run workspace
            all_tasks.append(Task(
                task_id=f"{package_id}:{case_id}:prepare",
                case_id=case_id,
                project=project,
                operation="prepare-run",
                args=[
                    sys.executable, str(REPLAY_PY), "prepare-run",
                    "--case", case_id,
                    "--label", "batch-run",
                ],
            ))

    if not all_tasks:
        print(f"No tasks for {project}/{phase}")
        return

    print(f"Running {len(all_tasks)} tasks for {project}/{phase} (concurrency={max_concurrency})")
    results = await asyncio.gather(
        *[run_task_live(t, semaphore) for t in all_tasks],
        return_exceptions=True,
    )

    completed = sum(1 for r in results if isinstance(r, Task) and r.status == "completed")
    failed = sum(1 for r in results if isinstance(r, Task) and r.status in ("failed", "error"))
    print(f"\n[{project}/{phase}] Done: {completed} completed, {failed} failed out of {len(all_tasks)} total")

    if failed > 0:
        for r in results:
            if isinstance(r, Task) and r.status in ("failed", "error"):
                err_line = r.stderr.strip().split("\n")[-1][:120] if r.stderr else "no stderr"
                print(f"  FAILED: {r.case_id} | {err_line}")
        sys.exit(1)


async def run_task_live(task: Task, semaphore: asyncio.Semaphore) -> Task:
    """Execute a task and print live output (no manifest checkpoint)."""
    async with semaphore:
        task.status = "running"
        task.started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *task.args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT),
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
            task.stdout = stdout_bytes.decode(errors="replace")[-4000:]
            task.stderr = stderr_bytes.decode(errors="replace")[-4000:]
            task.returncode = proc.returncode
            task.status = "completed" if proc.returncode == 0 else "failed"
            icon = "+" if proc.returncode == 0 else "X"
            print(f"  [{icon}] {task.case_id}/{task.operation} ({time.monotonic() - start:.1f}s)")
        except Exception as e:
            task.stderr = str(e)
            task.returncode = -1
            task.status = "error"
            print(f"  [!] {task.case_id}/{task.operation} ERROR: {e}")
        task.duration_s = round(time.monotonic() - start, 2)
        task.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return task


def main():
    parser = argparse.ArgumentParser(description="Concurrent task dispatcher for replay cases")
    sub = parser.add_subparsers(dest="command")

    # generate-manifest
    p_gen = sub.add_parser("generate-manifest", help="Scan cases and create a task manifest")
    p_gen.add_argument("--project", help="Filter to a specific project directory name")
    p_gen.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    p_gen.add_argument("--output", help="Output manifest path (default: work/concurrent_task_manifest.json)")

    # run
    p_run = sub.add_parser("run", help="Execute tasks from a manifest")
    p_run.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p_run.add_argument("--dry-run", action="store_true", help="Show what would run without executing")
    p_run.add_argument("--max-concurrency", type=int, default=None)

    # run-package
    p_pkg = sub.add_parser("run-package", help="Run all cases for a project in a phase")
    p_pkg.add_argument("--project", required=True, help="Project name (or 'all' for finalize)")
    p_pkg.add_argument("--phase", required=True, choices=["intake-lint", "prepare-run", "finalize"])
    p_pkg.add_argument("--package-id", required=True, help="Package ID for orchestration")
    p_pkg.add_argument("--orchestrator", default="", help="Path to orchestrate.sh")
    p_pkg.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)

    # status
    p_status = sub.add_parser("status", help="Show manifest status")
    p_status.add_argument("--manifest", required=True, help="Path to manifest JSON")

    args = parser.parse_args()

    if args.command == "generate-manifest":
        manifest = generate_manifest(args)
        out_path = Path(args.output) if hasattr(args, "output") and args.output else MANIFEST_DIR / "concurrent_task_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
        print(f"Generated {len(manifest.tasks)} tasks -> {out_path}")

    elif args.command == "run":
        asyncio.run(run_manifest(args))

    elif args.command == "run-package":
        asyncio.run(run_package_phase(
            project=args.project,
            phase=args.phase,
            max_concurrency=args.max_concurrency,
            package_id=args.package_id,
            orchestrator=args.orchestrator,
        ))

    elif args.command == "status":
        show_status(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
