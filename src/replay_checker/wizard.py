"""Natural language wizard for case discovery planning.

Provides the 3.0 user-facing entry point that lets users specify a project
and natural-language case discovery scope. Generates a self-contained
orchestration kit that external agents can execute to discover replayable cases.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .core import sanitize_slug


@dataclass(frozen=True)
class WizardResult:
    kit_path: Path
    project_path: Path
    scope: str
    plan_name: str
    start_command: str


def validate_project_path(project: str | Path) -> Path:
    """Validate that project path exists and is a directory."""
    path = Path(project).resolve()
    if not path.exists():
        raise ValueError(f"Project path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Project path is not a directory: {path}")
    return path


def generate_plan_name(scope: str) -> str:
    """Generate a deterministic plan directory name from scope text and date."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = sanitize_slug(scope[:60], fallback="discovery")
    return f"{date_str}-{slug}"


def generate_discovery_kit(
    project_path: str | Path,
    scope: str,
    *,
    output_root: str | Path | None = None,
) -> WizardResult:
    """Generate a case discovery orchestration kit.

    Args:
        project_path: Path to the target project.
        scope: Natural-language discovery scope.
        output_root: Where to create the kit. Defaults to <project>/docs/plans/.

    Returns:
        WizardResult with kit path and metadata.
    """
    project = Path(project_path).resolve()
    scope_text = scope.strip()
    if not scope_text:
        raise ValueError("Scope text must not be empty")

    plan_name = generate_plan_name(scope_text)
    if output_root is not None:
        kit_root = Path(output_root).resolve() / plan_name
    else:
        kit_root = project / "docs" / "plans" / plan_name

    kit_root.mkdir(parents=True, exist_ok=True)

    # Write INDEX.md
    _write_index(kit_root, project, scope_text)

    # Write launchers
    launchers = kit_root / "launchers"
    launchers.mkdir(parents=True, exist_ok=True)
    _copy_orchestrate_sh(launchers)
    _write_package_graph(kit_root, project, scope_text)
    _write_agent_prompts(kit_root, project, scope_text)

    # Write status directory
    status = kit_root / "status"
    status.mkdir(parents=True, exist_ok=True)
    _write_initial_state(kit_root)
    _write_status_files(kit_root)

    # Write package definitions
    packages = kit_root / "packages"
    packages.mkdir(parents=True, exist_ok=True)
    _write_plan_scan_package(packages, project, scope_text)
    _write_git_discovery_package(packages, project, scope_text)
    _write_merge_package(packages, project, scope_text)
    _write_finalize_package(packages, project, scope_text)

    start_cmd = f"bash {kit_root / 'launchers' / 'orchestrate.sh'} start"

    return WizardResult(
        kit_path=kit_root,
        project_path=project,
        scope=scope_text,
        plan_name=plan_name,
        start_command=start_cmd,
    )


def wizard(
    project_path: str | Path,
    *,
    scope: str | None = None,
    interactive: bool = True,
) -> WizardResult:
    """Run the case discovery wizard.

    In interactive mode, prompts for scope if not provided.
    In non-interactive mode, raises ValueError if scope is missing.
    """
    project = validate_project_path(project_path)

    if not scope:
        if interactive:
            print("Enter a natural-language description of what you want to discover:")
            print("  Examples:")
            print('    "refactor the auth module"')
            print('    "add unit tests for the API layer"')
            print('    "migrate from REST to GraphQL"')
            print()
            scope = input("Scope: ").strip()
            if not scope:
                raise ValueError("Scope must not be empty")
        else:
            raise ValueError(
                "Scope is required in non-interactive mode. "
                "Use plan-case-discovery --scope <text>."
            )

    result = generate_discovery_kit(project, scope)

    print(f"Discovery kit generated: {result.kit_path}")
    print(f"Project: {result.project_path}")
    print(f"Scope: {result.scope}")
    print()
    print("Recommended next step:")
    print(f"  {result.start_command}")
    print()

    if interactive:
        print("Would you like to start the discovery now? [y/N] ", end="", flush=True)
        answer = input().strip().lower()
        if answer == "y":
            print(f"Run: {result.start_command}")
        else:
            print("Discovery kit is ready. Run the command above when you want to start.")
    else:
        print("Discovery kit is ready.")

    return result


def plan_case_discovery(
    project_path: str | Path,
    scope: str,
) -> WizardResult:
    """Scriptable entry point for case discovery planning.

    Non-interactive. Raises ValueError if inputs are invalid.
    """
    project = validate_project_path(project_path)
    if not scope or not scope.strip():
        raise ValueError("Scope must not be empty")

    result = generate_discovery_kit(project, scope)

    print(f"Discovery kit generated: {result.kit_path}")
    print(f"Project: {result.project_path}")
    print(f"Scope: {result.scope}")
    print()
    print("Recommended next step:")
    print(f"  {result.start_command}")

    return result


# ---------------------------------------------------------------------------
# Kit file writers
# ---------------------------------------------------------------------------

def _write_index(kit_root: Path, project: Path, scope: str) -> None:
    project_name = project.name
    text = f"""\
# {project_name} Case Discovery — Orchestration Index

## Goal

Discover replayable agent evaluation cases from the `{project}` project
using the following discovery scope:

> {scope}

The generated kit scans for plan-based task sources, git-history synthetic
cases, and (optionally) local agent conversation evidence. It merges
candidates, ranks them by relevance, and validates the final cases.

## User Entry Points

- **Manual**: copy prompts from `launchers/agent-prompts.md` into any agent platform.
- **Script**: run `bash launchers/orchestrate.sh start`.
- **Status**: run `bash launchers/orchestrate.sh status`.

## Repository And Branch Policy

- Main checkout: `{project}`
- Coordinator plan root: `{kit_root}`
- Mainline branch: `main`
- Integration branch: `agent/{project_name}-discovery/integration`
- Functional package branches: `agent/{project_name}-discovery/<package-id>`
- Implementation isolation: one worktree per functional package.
- Coordinator status/state files are not implementation artifacts and must not be committed.

Target project: `{project}` (read-only for discovery packages; never edit its files).

## Authorization

Package agents are authorized to:
- Create or reuse only their assigned worktree and branch.
- Edit only allowed paths.
- Run listed verification commands.
- Commit local package changes.
- Write only their assigned coordinator status file.
- Call `bash {kit_root}/launchers/orchestrate.sh advance --from <package-id>` after recording final status.

Forbidden without explicit user approval:
- Force-push or hard reset
- Delete branches/worktrees not recorded as created by this orchestration
- Edit files inside `{project}` outside the generated cases directory
- Add secrets or credentials

## Dependency Graph

| Package | Depends On | Dependency Type | Unlock Condition | Wave |
|---|---|---|---|---|
| 01-plan-scan | none | status | completed | 1 |
| 02-git-discovery | none | status | completed | 1 |
| 03-case-merge | 01-plan-scan, 02-git-discovery | status | completed | 2 |
| 99-finalize | all functional packages | status+code | all functional packages completed | final |

## Merge Strategy

- Functional merge order: 01-plan-scan, 02-git-discovery, 03-case-merge
- Conflict owner: `99-finalize`
- Mainline merge: local non-force merge after integration verification passes.

## Landing Strategy

- Primary: all discovery packages complete; cases generated and validated.
- Fallback: partial discovery if some sources unavailable (e.g., no git history).

## Stop Conditions

- Any functional package is `blocked`, `stale`, or `invalid`.
- Package evidence is incomplete.
- Merge conflict or verification failure occurs.
"""
    (kit_root / "INDEX.md").write_text(text, encoding="utf-8")


def _copy_orchestrate_sh(launchers: Path) -> None:
    """Copy the orchestrate.sh template into the kit.

    Looks for the template relative to this source file:
    src/replay_checker/wizard.py -> ../../docs/plans/.../orchestrate.sh
    Falls back to a stub if the template cannot be found.
    """
    import shutil

    # This file: .../src/replay_checker/wizard.py
    # Repo root: .../ (3 levels up from src/replay_checker/)
    src_replay_dir = Path(__file__).resolve().parent  # src/replay_checker/
    repo_root = src_replay_dir.parents[2]              # worktree root
    template_path = (
        repo_root
        / "docs"
        / "plans"
        / "replay-checker-3.0-natural-case-discovery"
        / "launchers"
        / "orchestrate.sh"
    )
    if not template_path.exists():
        # Walk up to find the main repo checkout (worktree is deeper)
        # Search for the repo marker (AGENTS.md or .git) up the tree
        candidate_path = src_replay_dir
        for _ in range(10):
            candidate_path = candidate_path.parent
            candidate = candidate_path / "docs" / "plans" / "replay-checker-3.0-natural-case-discovery" / "launchers" / "orchestrate.sh"
            if candidate.exists():
                template_path = candidate
                break
        else:
            (launchers / "orchestrate.sh").write_text(
                "#!/usr/bin/env bash\n# orchestrate.sh — generated by wizard\n"
                "echo 'orchestrate.sh stub: copy a real template here'\n",
                encoding="utf-8",
            )
            return
    shutil.copy2(template_path, launchers / "orchestrate.sh")
    (launchers / "orchestrate.sh").chmod(0o755)


def _write_package_graph(kit_root: Path, project: Path, scope: str) -> None:
    project_name = project.name
    branch_prefix = f"agent/{project_name}-discovery"
    lines = [
        "package_id\tpackage_doc\tstatus_file\tdependencies\tdependency_type\twave\tbranch\tworktree\tmanual\tfinalize",
        f"01-plan-scan\tpackages/01-plan-scan.md\tstatus/01-plan-scan.md\t\tstatus\t1\t{branch_prefix}/01-plan-scan\t.claude/worktrees/01-plan-scan\t0\t0",
        f"02-git-discovery\tpackages/02-git-discovery.md\tstatus/02-git-discovery.md\t\tstatus\t1\t{branch_prefix}/02-git-discovery\t.claude/worktrees/02-git-discovery\t0\t0",
        f"03-case-merge\tpackages/03-case-merge.md\tstatus/03-case-merge.md\t01-plan-scan,02-git-discovery\tstatus\t2\t{branch_prefix}/03-case-merge\t.claude/worktrees/03-case-merge\t0\t0",
        f"99-finalize\tpackages/99-finalize.md\tstatus/99-finalize.md\t01-plan-scan,02-git-discovery,03-case-merge\tstatus+code\tfinal\t{branch_prefix}/99-finalize\t.claude/worktrees/99-finalize\t0\t1",
    ]
    (kit_root / "launchers" / "package-graph.tsv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )


def _write_agent_prompts(kit_root: Path, project: Path, scope: str) -> None:
    project_name = project.name
    text = f"""\
# Agent Prompts — {project_name} Case Discovery

Copy the prompt for the package you want to execute into your agent platform.

## 01-plan-scan

You are executing package `01-plan-scan` for the {project_name} case discovery.

**Discovery scope**: {scope}

Scan `{project}` for plan-based task sources (orchestration kits, handoff plans,
planning docs with acceptance criteria). For each candidate, record:
- Path to the plan document
- Title and goal
- Package count (if an orchestration kit)
- Verification commands found
- Whether it matches the discovery scope above

Output a structured inventory in `inventory.md` at the kit root.

## 02-git-discovery

You are executing package `02-git-discovery` for the {project_name} case discovery.

**Discovery scope**: {scope}

Analyze git history in `{project}` to identify meaningful commits that could
serve as synthetic replay case targets. Focus on commits that match or relate
to the discovery scope above.

Output a structured inventory in `git-candidates.md` at the kit root.

## 03-case-merge

You are executing package `03-case-merge` for the {project_name} case discovery.

**Discovery scope**: {scope}

Read the inventories produced by 01-plan-scan and 02-git-discovery. Merge
overlapping candidates, rank by relevance to the scope, and output a final
case list in `merged-candidates.md`.

## 99-finalize

You are executing package `99-finalize`.

Read all completed package inventories, generate final cases using
`replay.py intake` or `replay.py create-case`, run validation, and
produce a completion report.
"""
    (kit_root / "launchers" / "agent-prompts.md").write_text(text, encoding="utf-8")


def _write_initial_state(kit_root: Path) -> None:
    header = (
        "package_id\tstate\tlaunched_at\tcompleted_at\tagent\tbranch\tworktree\t"
        "base_commit\tcommit_hash\tverification\tintegration\tcleanup\t"
        "last_error\tfailed_command\tconflict_files\tlog_summary\trecovery_hint"
    )
    packages = ["01-plan-scan", "02-git-discovery", "03-case-merge", "99-finalize"]
    rows = [header]
    for pkg in packages:
        rows.append(f"{pkg}\tpending\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t")
    (kit_root / "status" / "state.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_status_files(kit_root: Path) -> None:
    """Write status/<package-id>.md for every package in the graph."""
    package_ids = ["01-plan-scan", "02-git-discovery", "03-case-merge", "99-finalize"]
    for pkg_id in package_ids:
        md = (
            f"# {pkg_id} Status\n"
            "\n"
            "## State\n"
            "\n"
            "`pending`\n"
            "\n"
            "## Evidence\n"
            "\n"
            "- Worktree:\n"
            "- Branch:\n"
            "- Base commit:\n"
            "- Commit hash:\n"
            "- Changed files:\n"
            "- Verification:\n"
            "\n"
            "## Notes\n"
            "\n"
            "- Risks:\n"
            "- Blockers:\n"
            "- Recovery hint:\n"
        )
        (kit_root / "status" / f"{pkg_id}.md").write_text(md, encoding="utf-8")


def _write_plan_scan_package(packages: Path, project: Path, scope: str) -> None:
    text = f"""\
# 01-plan-scan

## Goal

Scan the target project for plan-based task sources that match the discovery scope.

**Discovery scope**: {scope}

## Allowed Paths

- `{project}/docs/` (read-only)
- `{project}/cases/` (write)
- Kit scratch directory

## Required Work

1. Scan `{project}/docs/plans/` for INDEX.md files (orchestration kits).
2. Scan for markdown files containing "acceptance criteria" and ("goal" or "steps").
3. Score each plan candidate by structural signals (packages, verification, git activity).
4. Record all candidates in an inventory file.
5. Highlight which candidates best match the discovery scope.

## Acceptance Criteria

- Inventory file lists all discovered plan sources with scores.
- Each candidate includes path, title, package count, and relevance assessment.

## Verification Commands

- `ls {project}/docs/plans/` (should list plan directories)
"""
    (packages / "01-plan-scan.md").write_text(text, encoding="utf-8")


def _write_git_discovery_package(packages: Path, project: Path, scope: str) -> None:
    text = f"""\
# 02-git-discovery

## Goal

Identify meaningful git commits in the target project that could serve as
synthetic replay case targets, aligned with the discovery scope.

**Discovery scope**: {scope}

## Allowed Paths

- `{project}/.git/` (read-only)
- Kit scratch directory

## Required Work

1. Analyze git log for the target project.
2. Filter out trivial commits (format, style, lint, bump, dependabot).
3. Identify commits whose messages or file changes relate to the scope.
4. For each candidate, record commit SHA, message, changed files, and relevance.
5. Output a structured candidate list.

## Acceptance Criteria

- Candidate list includes commit SHA, message summary, changed files count.
- Trivial commits are filtered out.
- Candidates ranked by relevance to the scope.

## Verification Commands

- `git -C {project} log --oneline -5` (should show recent commits)
"""
    (packages / "02-git-discovery.md").write_text(text, encoding="utf-8")


def _write_merge_package(packages: Path, project: Path, scope: str) -> None:
    text = f"""\
# 03-case-merge

## Goal

Merge plan-based and git-history case candidates into a ranked final list,
deduplicating and scoring by relevance to the discovery scope.

**Discovery scope**: {scope}

## Allowed Paths

- Kit root directory (read inventories, write merged output)
- `{project}/cases/` (write final cases via replay.py)

## Required Work

1. Read `inventory.md` from 01-plan-scan.
2. Read `git-candidates.md` from 02-git-discovery.
3. Deduplicate overlapping candidates (same file/commit referenced by both).
4. Rank all candidates by combined relevance score.
5. Output `merged-candidates.md` with the final ranked list.
6. Optionally generate cases using `replay.py intake` or `replay.py create-case`.

## Acceptance Criteria

- Merged list contains all unique candidates from both sources.
- Candidates are ranked by relevance to the scope.
- No duplicate entries for the same underlying source.

## Verification Commands

- Verify `merged-candidates.md` exists and is non-empty.
"""
    (packages / "03-case-merge.md").write_text(text, encoding="utf-8")


def _write_finalize_package(packages: Path, project: Path, scope: str) -> None:
    text = f"""\
# 99-finalize

## Goal

Validate all generated cases, produce a completion report, and clean up.

**Discovery scope**: {scope}

## Allowed Paths

- Kit root directory
- `{project}/cases/` (read validation)

## Required Work

1. Run `python3 tools/replay.py` validation commands on generated cases.
2. Produce `FINAL_REPORT.md` summarizing discovery results.
3. Record final state in state.tsv.
4. Verify no files in `{project}/` were modified.

## Acceptance Criteria

- All cases pass validation.
- FINAL_REPORT.md exists and lists case count and source breakdown.
- No modifications to the target project files.

## Verification Commands

- `python3 -m pytest tests/ -q` (replay_checker tests pass)
- Verify no unexpected changes in `{project}/`
"""
    (packages / "99-finalize.md").write_text(text, encoding="utf-8")
