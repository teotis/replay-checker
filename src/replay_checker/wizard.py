"""Natural language wizard for case discovery planning.

Provides the 3.0 user-facing entry point that lets users specify a project
and natural-language case discovery scope. Generates a self-contained
orchestration kit that external agents can execute to discover replayable cases.

3.1 adds fast local preview (--dry-run) and deterministic local execution
(--execute-local) that compile a case without generating a multi-agent kit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .candidates import CandidateCase, build_case_candidates, select_case_candidate
from .core import sanitize_slug
from .discovery.models import DiscoveryPackage
from .git_utils import git_output as shared_git_output
from .orchestration import (
    generate_agent_prompts,
    generate_graph_tsv,
    generate_package_doc,
    generate_state_tsv,
    generate_status_md,
)
from .sources import discover_evidence_sources

_GIT_TIMEOUT = 10


def _git_output(args: list[str], *, cwd: Path) -> str:
    return shared_git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


@dataclass(frozen=True)
class WizardResult:
    kit_path: Path
    project_path: Path
    scope: str
    plan_name: str
    start_command: str


@dataclass(frozen=True)
class WizardPreview:
    """Read-only preview: no kit, no case, no filesystem writes."""
    project_path: Path
    scope: str
    has_git: bool
    git_clean: bool
    head_commit: str
    plan_count: int
    history_count: int
    candidate_count: int
    top_candidate: CandidateCase | None
    candidates: list[CandidateCase] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_commands: list[str] = field(default_factory=list)


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

    Delegates file generation to orchestration.py as the single source of truth,
    then overlays wizard-specific customizations (INDEX.md, orchestrate.sh).
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

    # Generate standard kit files via orchestration.py (single truth source).
    packages = _wizard_packages(project)
    _write_kit_files(kit_root, project, packages)

    # Overlay wizard-specific customizations
    _write_index(kit_root, project, scope_text)

    launchers = kit_root / "launchers"
    launchers.mkdir(parents=True, exist_ok=True)
    _copy_orchestrate_sh(launchers, project)

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


def wizard_preview(
    project_path: str | Path,
    scope: str = "",
) -> WizardPreview:
    """Fast local-only preview: discover signals, rank candidates, no filesystem writes.

    Does not create worktrees, cases, runs, reports, or discovery kits.
    Returns a WizardPreview with extractability signals and next-step recommendations.
    """
    project = validate_project_path(project_path)
    scope_text = scope.strip()

    has_git = (project / ".git").exists()
    git_clean = True
    head_commit = ""
    if has_git:
        status = _git_output(["status", "--porcelain"], cwd=project)
        git_clean = not bool(status.strip())
        head_commit = _git_output(["rev-parse", "--short", "HEAD"], cwd=project).strip()

    records = discover_evidence_sources(project)
    candidates = build_case_candidates(records, project_path=project)
    top = select_case_candidate(candidates)

    plan_count = sum(1 for r in records if r.source_type == "plan")
    history_count = sum(1 for r in records if r.source_type in ("codex_history", "claude_history"))

    signals: list[str] = []
    risks: list[str] = []

    if has_git:
        signals.append("git repository detected")
        if git_clean:
            signals.append("working tree clean")
        else:
            risks.append("working tree is dirty — base commit inference may be less reliable")
    else:
        risks.append("no git repository — synthetic and base-commit features unavailable")

    if plan_count:
        signals.append(f"{plan_count} plan document(s) found")
    else:
        risks.append("no plan documents found under docs/plans/")

    if history_count:
        signals.append(f"{history_count} local history record(s) matched")
    else:
        signals.append("no local Codex/Claude history matches (may be normal for temp or new projects)")

    if candidates:
        signals.append(f"{len(candidates)} case candidate(s) ranked")
    else:
        risks.append("no case candidates could be built from available sources")

    if top:
        signals.append(f"top candidate: {top.candidate_id} (score={top.relevance_score:.3f}, {top.source_type})")
        if top.risks:
            risks.extend(top.risks)

    next_commands = [
        f"rtk python3 tools/replay.py wizard --project {project} --scope \"{scope_text or 'your scope here'}\"",
        f"rtk python3 tools/replay.py intake --project {project}",
        f"rtk python3 tools/replay.py wizard --project {project} --scope \"{scope_text or 'your scope here'}\" --execute-local --no-interactive",
    ]

    return WizardPreview(
        project_path=project,
        scope=scope_text,
        has_git=has_git,
        git_clean=git_clean,
        head_commit=head_commit,
        plan_count=plan_count,
        history_count=history_count,
        candidate_count=len(candidates),
        top_candidate=top,
        candidates=candidates,
        signals=signals,
        risks=risks,
        next_commands=next_commands,
    )


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


def _copy_orchestrate_sh(launchers: Path, project: Path) -> None:
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
            script = launchers / "orchestrate.sh"
            script.write_text(
                "#!/usr/bin/env bash\n# orchestrate.sh — generated by wizard\n"
                "echo 'orchestrate.sh stub: copy a real template here'\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            return
    target = launchers / "orchestrate.sh"
    shutil.copy2(template_path, target)
    _patch_orchestrate_repo_root(target, project)
    _patch_orchestrate_session_parser(target)
    target.chmod(0o755)


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _patch_orchestrate_repo_root(script: Path, project: Path) -> None:
    text = script.read_text(encoding="utf-8")
    old = 'REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"'
    new = f"REPO_ROOT={_shell_quote(project.as_posix())}"
    if old in text:
        script.write_text(text.replace(old, new), encoding="utf-8")


def _patch_orchestrate_session_parser(script: Path) -> None:
    text = script.read_text(encoding="utf-8")
    old = """parse_session_id() {
  awk '
    /backgrounded/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "backgrounded" || $i == "·" || $i == "-" || $i == "•") continue
        if ($i ~ /^[[:alnum:]_-]{6,}$/) {
          print $i
          exit
        }
      }
    }
  '
}
"""
    new = """parse_session_id() {
  awk '
    /backgrounded/ {
      for (i = 1; i <= NF; i++) {
        token = $i
        gsub(/\\033\\[[0-9;]*[[:alpha:]]/, "", token)
        if (token == "backgrounded" || token == "·" || token == "-" || token == "•") continue
        if (token ~ /^[[:alnum:]_-]{6,}$/) {
          print token
          exit
        }
      }
    }
  '
}
"""
    if old in text:
        script.write_text(text.replace(old, new), encoding="utf-8")


def _wizard_packages(project: Path) -> tuple[DiscoveryPackage, ...]:
    """Build wizard-specific discovery packages.

    Package IDs and descriptions are wizard-specific, but their structured
    definition is passed to orchestration.py for file generation.
    """
    project_name = project.name
    return (
        DiscoveryPackage(
            package_id="01-plan-scan",
            description="Scan the target project for plan-based task sources.",
            allowed_paths=(f"{project}/docs/", f"{project}/cases/"),
            forbidden_paths=("_reference/",),
            dependencies=(),
            dependency_type="status",
            wave=1,
        ),
        DiscoveryPackage(
            package_id="02-git-discovery",
            description="Identify meaningful git commits for synthetic replay case targets.",
            allowed_paths=(f"{project}/.git/",),
            forbidden_paths=("_reference/",),
            dependencies=(),
            dependency_type="status",
            wave=1,
        ),
        DiscoveryPackage(
            package_id="03-case-merge",
            description="Merge plan and git candidates into a ranked final list.",
            allowed_paths=(f"{project}/cases/",),
            forbidden_paths=("_reference/",),
            dependencies=("01-plan-scan", "02-git-discovery"),
            dependency_type="status",
            wave=2,
        ),
        DiscoveryPackage(
            package_id="99-finalize",
            description="Validate all generated cases, deduplicate, produce completion report.",
            allowed_paths=(f"{project}/cases/",),
            forbidden_paths=("_reference/",),
            dependencies=("01-plan-scan", "02-git-discovery", "03-case-merge"),
            dependency_type="status+code",
            wave=0,
            is_finalize=True,
        ),
    )


def _write_kit_files(
    kit_root: Path,
    project: Path,
    packages: tuple[DiscoveryPackage, ...],
) -> None:
    """Generate standard kit files via orchestration.py as the single truth source."""
    launchers = kit_root / "launchers"
    status_dir = kit_root / "status"
    packages_dir = kit_root / "packages"

    for d in (launchers, status_dir, packages_dir):
        d.mkdir(parents=True, exist_ok=True)

    (launchers / "package-graph.tsv").write_text(
        generate_graph_tsv(packages, kit_root, project),
        encoding="utf-8",
    )
    (launchers / "agent-prompts.md").write_text(
        generate_agent_prompts(packages, kit_root),
        encoding="utf-8",
    )
    (status_dir / "state.tsv").write_text(
        generate_state_tsv(packages),
        encoding="utf-8",
    )
    for pkg in packages:
        (status_dir / f"{pkg.package_id}.md").write_text(
            generate_status_md(pkg),
            encoding="utf-8",
        )
        (packages_dir / f"{pkg.package_id}.md").write_text(
            generate_package_doc(pkg),
            encoding="utf-8",
        )
