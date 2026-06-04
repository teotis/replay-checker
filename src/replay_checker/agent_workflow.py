"""Agent-platform workflow helpers for Replay Checker.

This module keeps the conversational workflow thin: it composes existing
discovery, run preparation, evidence collection, scoring package, and compare
operations without changing their underlying contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .core import sanitize_slug
from .replay import collect_run, compare_case, load_case, load_run, prepare_run, score_run
from .wizard import generate_discovery_kit


@dataclass(frozen=True)
class LaunchOption:
    key: str
    label: str
    command: str
    description: str


@dataclass(frozen=True)
class ExtractionFlow:
    kit_path: Path
    project_path: Path
    scope: str
    package_ids: tuple[str, ...]
    launch_options: tuple[LaunchOption, ...]
    next_question: str


@dataclass(frozen=True)
class PreparedRunFlow:
    run_id: str
    runner_label: str
    task_path: Path
    workspace: Path
    completion_template: Path


@dataclass(frozen=True)
class GradingFlowResult:
    case_id: str
    scored_runs: tuple[str, ...]
    scoring_packages: tuple[Path, ...]
    compare_report: Path


def build_default_runner_label(agent_intro: str, *, today: date | None = None) -> str:
    """Build the default execution identity from date plus agent self-intro."""
    day = today or date.today()
    intro = re.sub(r"[^a-z0-9]+", "_", agent_intro.lower()).strip("_") or "agent"
    intro = sanitize_slug(intro, fallback="agent")
    return f"{day.isoformat()}-{intro[:80].strip('_')}"


def create_extraction_flow(
    *,
    project_path: str | Path,
    scope: str,
    output_root: str | Path | None = None,
) -> ExtractionFlow:
    """Generate a discovery kit and return user-facing launch choices."""
    result = generate_discovery_kit(project_path, scope, output_root=output_root)
    package_ids = _read_package_ids(result.kit_path)
    launch_options = _build_launch_options(result.kit_path)
    return ExtractionFlow(
        kit_path=result.kit_path,
        project_path=result.project_path,
        scope=result.scope,
        package_ids=package_ids,
        launch_options=launch_options,
        next_question="Which launch mode should I use: agents-view, manual-prompts, or status-only?",
    )


def prepare_agent_run_flow(
    *,
    case_id: str,
    agent_intro: str | None = None,
    label: str | None = None,
    cases_root: str | Path = "cases",
    runs_root: str | Path = "runs",
) -> PreparedRunFlow:
    """Prepare a run with a product-friendly default runner identity."""
    runner_label = label or build_default_runner_label(agent_intro or "agent")
    case = load_case(cases_root, case_id)
    run = prepare_run(case, runs_root=runs_root, runner_label=runner_label)
    return PreparedRunFlow(
        run_id=run.id,
        runner_label=runner_label,
        task_path=run.root / "TASK.md",
        workspace=run.workspace,
        completion_template=run.root / "completion_report_template.md",
    )


def run_grading_flow(
    *,
    case_id: str,
    run_ids: tuple[str, ...] | list[str] | None = None,
    cases_root: str | Path = "cases",
    runs_root: str | Path = "runs",
    reports_root: str | Path = "reports",
    rubric_path: str | Path = "rubrics/default.yaml",
) -> GradingFlowResult:
    """Collect evidence, create scoring packages, and refresh the compare report."""
    case = load_case(cases_root, case_id)
    selected = tuple(run_ids) if run_ids else _discover_case_runs(case_id, runs_root)
    scoring_packages: list[Path] = []
    for run_id in selected:
        run = load_run(runs_root, run_id, cases_root=cases_root)
        collect_run(run)
        scoring_packages.append(score_run(run, rubric_path=rubric_path))
    report = compare_case(case, runs_root=runs_root, reports_root=reports_root)
    return GradingFlowResult(
        case_id=case_id,
        scored_runs=selected,
        scoring_packages=tuple(scoring_packages),
        compare_report=report,
    )


def _read_package_ids(kit_path: Path) -> tuple[str, ...]:
    graph = kit_path / "launchers" / "package-graph.tsv"
    if not graph.exists():
        return ()
    lines = graph.read_text(encoding="utf-8").splitlines()[1:]
    return tuple(line.split("\t", 1)[0] for line in lines if line.strip())


def _build_launch_options(kit_path: Path) -> tuple[LaunchOption, ...]:
    orchestrator = kit_path / "launchers" / "orchestrate.sh"
    prompts = kit_path / "launchers" / "agent-prompts.md"
    return (
        LaunchOption(
            key="agents-view",
            label="Agents View / script launch",
            command=f"bash {orchestrator} start",
            description="Launch ready discovery packages through the generated orchestrator, then monitor in the agent platform.",
        ),
        LaunchOption(
            key="manual-prompts",
            label="Manual prompt copy",
            command=f"open {prompts}",
            description="Copy package prompts into any agent platform without automatic launching.",
        ),
        LaunchOption(
            key="status-only",
            label="Status only",
            command=f"bash {orchestrator} status",
            description="Do not start workers yet; only inspect package readiness and coordinator consistency.",
        ),
    )


def _discover_case_runs(case_id: str, runs_root: str | Path) -> tuple[str, ...]:
    root = Path(runs_root)
    if not root.exists():
        return ()
    return tuple(path.name for path in sorted(root.glob(f"{case_id}-*")) if (path / "run.yaml").exists())
