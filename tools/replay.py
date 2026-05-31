#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_checker.linters import lint_task, lint_score  # noqa: E402
from replay_checker.replay import (  # noqa: E402
    collect_run,
    compare_case,
    create_case,
    discover_plan_packages,
    intake,
    load_case,
    load_run,
    prepare_run,
    score_run,
)
from replay_checker.candidates import build_case_candidates, select_case_candidate  # noqa: E402
from replay_checker.eval_adapter import load_skill_evals, write_skill_eval_case  # noqa: E402
from replay_checker.sources import (  # noqa: E402
    EvidenceSourceConfig,
    discover_evidence_sources,
    score_evidence_records,
)
from replay_checker.wizard import plan_case_discovery, wizard  # noqa: E402


def cmd_discover(args: argparse.Namespace) -> int:
    for package in discover_plan_packages(args.project):
        print(f"{package.plan_path}\t{package.package_count}\t{package.title}")
    return 0


def cmd_inspect_sources(args: argparse.Namespace) -> int:
    records = discover_evidence_sources(
        args.project,
        EvidenceSourceConfig(
            codex_history_roots=tuple(Path(path) for path in args.codex_history_root),
            claude_history_roots=tuple(Path(path) for path in args.claude_history_root),
        )
        if args.codex_history_root or args.claude_history_root
        else None,
    )
    scored = score_evidence_records(records)
    for record in scored:
        reasons = "; ".join(record.relevance_reasons) if record.relevance_reasons else "-"
        task = record.task_signal or "-"
        print(
            f"{record.source_type}\t{record.confidence}\t"
            f"score={record.relevance_score:.3f}\ttask={task}\t"
            f"{record.path}\t{record.summary}\t[{reasons}]"
        )
    return 0


def cmd_inspect_candidates(args: argparse.Namespace) -> int:
    records = discover_evidence_sources(
        args.project,
        EvidenceSourceConfig(
            codex_history_roots=tuple(Path(path) for path in args.codex_history_root),
            claude_history_roots=tuple(Path(path) for path in args.claude_history_root),
        )
        if args.codex_history_root or args.claude_history_root
        else None,
    )
    candidates = build_case_candidates(records, project_path=args.project)
    best = select_case_candidate(candidates)

    if not candidates:
        print("No candidates found.")
        return 0

    print(f"Candidates: {len(candidates)}")
    print()
    for idx, c in enumerate(candidates):
        tag = " [selected]" if best and c.candidate_id == best.candidate_id else ""
        print(f"  {idx + 1}. {c.candidate_id}{tag}")
        print(f"     source_type:  {c.source_type}")
        print(f"     primary:      {c.primary_source}")
        if c.supporting_sources:
            print(f"     supporting:   {len(c.supporting_sources)} other source(s)")
        if c.base_commit:
            print(f"     base_commit:  {c.base_commit[:12]} (confidence: {c.base_confidence})")
        else:
            print(f"     base_commit:  (none)")
        print(f"     score:        {c.relevance_score:.3f}")
        reasons = "; ".join(c.selection_reasons) if c.selection_reasons else "-"
        print(f"     reasons:      [{reasons}]")
        if c.risks:
            risks = "; ".join(c.risks)
            print(f"     risks:        [{risks}]")
        print()

    if best:
        print(f"Selected candidate: {best.candidate_id} (score {best.relevance_score:.3f})")
    return 0


def cmd_create_case(args: argparse.Namespace) -> int:
    case = create_case(
        cases_root=args.cases_root,
        project_path=args.project,
        plan_path=args.plan,
        base_commit=args.base,
        case_id=args.case_id,
        verification_commands=args.verify,
    )
    print(f"Created case: {case.root / 'case.yaml'}")
    return 0


def cmd_extract_skill_evals(args: argparse.Namespace) -> int:
    selected = set(args.skills or [])
    eval_cases = [
        eval_case for eval_case in load_skill_evals(args.skill_repo)
        if not selected or eval_case.skill_name in selected
    ]
    skills = {eval_case.skill_name for eval_case in eval_cases}
    for eval_case in eval_cases:
        case_root = write_skill_eval_case(eval_case, args.cases_root, args.skill_repo)
        print(f"Created case: {case_root / 'case.yaml'} ({eval_case.skill_name} eval {eval_case.eval_id})")
    print(f"Extracted {len(eval_cases)} cases from {len(skills)} skills.")
    return 0


def cmd_prepare_run(args: argparse.Namespace) -> int:
    case = load_case(args.cases_root, args.case)
    run = prepare_run(case, runs_root=args.runs_root, runner_label=args.label)
    print(f"Prepared run: {run.root}")
    print(f"Workspace: {run.workspace}")
    print("Copy TASK.md into your chosen agent, then return and run:")
    print(f"  rtk python3 tools/replay.py collect-run --run {run.id}")
    return 0


def cmd_collect_run(args: argparse.Namespace) -> int:
    run = load_run(args.runs_root, args.run, cases_root=args.cases_root)
    evidence = collect_run(run)
    print(f"Collected run: {evidence.run_id} status={evidence.status}")
    return 0


def cmd_score_run(args: argparse.Namespace) -> int:
    run = load_run(args.runs_root, args.run, cases_root=args.cases_root)
    path = score_run(run, rubric_path=args.rubric)
    print(f"Created scoring package: {path}")
    return 0


def cmd_intake(args: argparse.Namespace) -> int:
    case = intake(
        cases_root=args.cases_root,
        project_path=args.project,
        codex_history_roots=tuple(args.codex_history_root) if args.codex_history_root else None,
        claude_history_roots=tuple(args.claude_history_root) if args.claude_history_root else None,
    )
    print(f"Created case: {case.root / 'case.yaml'}")
    print(f"Source: {case.source_type} ({case.selection_reason})")
    if case.base_commit:
        print(f"Base: {case.base_commit[:12]} (confidence: {case.base_confidence})")
    else:
        print(f"Base: (none) (confidence: {case.base_confidence})")
    print(f"Evidence sources: {len(case.evidence_sources)}")
    print()
    print("Next commands (copy-paste ready):")
    print(f"  rtk python3 tools/replay.py prepare-run --case {case.id} --label \"<agent/model>\"")
    print()
    print("After agent execution:")
    print(f"  rtk python3 tools/replay.py collect-run --run <run-id>")
    print(f"  rtk python3 tools/replay.py score-run --run <run-id>")
    print(f"  rtk python3 tools/replay.py compare --case {case.id}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    case = load_case(args.cases_root, args.case)
    report = compare_case(case, runs_root=args.runs_root, reports_root=args.reports_root)
    print(f"Created comparison report: {report}")
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    try:
        result = wizard(
            args.project,
            scope=args.scope,
            interactive=args.interactive,
        )
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_plan_case_discovery(args: argparse.Namespace) -> int:
    try:
        result = plan_case_discovery(args.project, args.scope)
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_lint_task(args: argparse.Namespace) -> int:
    from pathlib import Path as _P

    run_root = _P(args.run)
    diagnostics = lint_task(run_root)
    if diagnostics:
        for msg in diagnostics:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    print("Lint clean: task package is well-formed.")
    return 0


def cmd_lint_score(args: argparse.Namespace) -> int:
    from pathlib import Path as _P

    run_root = _P(args.run)
    diagnostics = lint_score(run_root)
    if diagnostics:
        for msg in diagnostics:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1
    print("Lint clean: scoring package is well-formed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay historical project tasks for agent evaluation. "
        "Use intake to auto-generate cases, prepare-run to isolate a workspace, "
        "collect-run/score-run to gather evidence, and lint-task/lint-score to validate packages.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="find orchestration/handoff-style plan packages")
    discover.add_argument("--project", required=True)
    discover.set_defaults(func=cmd_discover)

    inspect = sub.add_parser("inspect-sources", help="inspect plan and local history evidence for a project")
    inspect.add_argument("--project", required=True)
    inspect.add_argument("--codex-history-root", action="append", default=[])
    inspect.add_argument("--claude-history-root", action="append", default=[])
    inspect.set_defaults(func=cmd_inspect_sources)

    inspect_c = sub.add_parser("inspect-candidates", help="preview ranked case candidates for a project")
    inspect_c.add_argument("--project", required=True)
    inspect_c.add_argument("--codex-history-root", action="append", default=[])
    inspect_c.add_argument("--claude-history-root", action="append", default=[])
    inspect_c.set_defaults(func=cmd_inspect_candidates)

    intake_parser = sub.add_parser("intake", help="auto-generate a case from a project directory")
    intake_parser.add_argument("--cases-root", default="cases")
    intake_parser.add_argument("--project", required=True)
    intake_parser.add_argument("--codex-history-root", action="append", default=[])
    intake_parser.add_argument("--claude-history-root", action="append", default=[])
    intake_parser.set_defaults(func=cmd_intake)

    create = sub.add_parser("create-case", help="create a replay case from a historical plan package")
    create.add_argument("--cases-root", default="cases")
    create.add_argument("--project", required=True)
    create.add_argument("--plan", required=True)
    create.add_argument("--base", required=True)
    create.add_argument("--case-id", required=True)
    create.add_argument("--verify", action="append", default=[])
    create.set_defaults(func=cmd_create_case)

    extract = sub.add_parser("extract-skill-evals", help="create replay cases from skill eval definitions")
    extract.add_argument("--cases-root", default="cases")
    extract.add_argument("--skill-repo", required=True)
    extract.add_argument("--skills", action="append", default=[])
    extract.set_defaults(func=cmd_extract_skill_evals)

    prepare = sub.add_parser("prepare-run", help="prepare an isolated user-driven run")
    prepare.add_argument("--cases-root", default="cases")
    prepare.add_argument("--runs-root", default="runs")
    prepare.add_argument("--case", required=True)
    prepare.add_argument("--label", required=True)
    prepare.set_defaults(func=cmd_prepare_run)

    collect = sub.add_parser("collect-run", help="collect evidence after the user-driven agent run")
    collect.add_argument("--cases-root", default="cases")
    collect.add_argument("--runs-root", default="runs")
    collect.add_argument("--run", required=True)
    collect.set_defaults(func=cmd_collect_run)

    score = sub.add_parser("score-run", help="create an anonymized scoring task package")
    score.add_argument("--cases-root", default="cases")
    score.add_argument("--runs-root", default="runs")
    score.add_argument("--run", required=True)
    score.add_argument("--rubric", default="rubrics/default.yaml")
    score.set_defaults(func=cmd_score_run)

    compare = sub.add_parser("compare", help="compare collected runs for one case")
    compare.add_argument("--cases-root", default="cases")
    compare.add_argument("--runs-root", default="runs")
    compare.add_argument("--reports-root", default="reports")
    compare.add_argument("--case", required=True)
    compare.set_defaults(func=cmd_compare)

    lint_task_parser = sub.add_parser(
        "lint-task",
        help="validate a task package for structural completeness and scope safety",
    )
    lint_task_parser.add_argument(
        "--run", required=True,
        help="path to the run or case directory containing TASK.md",
    )
    lint_task_parser.set_defaults(func=cmd_lint_task)

    lint_score_parser = sub.add_parser(
        "lint-score",
        help="validate a scoring package for evidence gates, score ceilings, and schema compliance",
    )
    lint_score_parser.add_argument(
        "--run", required=True,
        help="path to the run directory containing scoring_package.md",
    )
    lint_score_parser.set_defaults(func=cmd_lint_score)

    wizard_parser = sub.add_parser("wizard", help="interactive case discovery wizard")
    wizard_parser.add_argument("--project", required=True, help="path to the target project")
    wizard_parser.add_argument("--scope", default=None, help="natural-language discovery scope")
    wizard_parser.add_argument("--no-interactive", dest="interactive", action="store_false", default=True, help="disable interactive prompts")
    wizard_parser.set_defaults(func=cmd_wizard)

    plan_cd = sub.add_parser("plan-case-discovery", help="scriptable case discovery planning")
    plan_cd.add_argument("--project", required=True, help="path to the target project")
    plan_cd.add_argument("--scope", required=True, help="natural-language discovery scope")
    plan_cd.set_defaults(func=cmd_plan_case_discovery)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
