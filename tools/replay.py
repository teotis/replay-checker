#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from replay_checker.linters import lint_task, lint_score  # noqa: E402
from replay_checker.replay import (  # noqa: E402
    batch_intake,
    collect_run,
    compare_case,
    create_case,
    discover_plan_packages,
    doctor_runs,
    intake,
    inspect_run_dir,
    load_case,
    load_run,
    prepare_run,
    report_all_cases,
    score_run,
)
from replay_checker.candidates import build_case_candidates, select_case_candidate  # noqa: E402
from replay_checker.eval_adapter import load_skill_evals, write_skill_eval_case, dedupe_eval_cases  # noqa: E402
from replay_checker.case_dedupe import inspect_duplicates  # noqa: E402
from replay_checker.sources import (  # noqa: E402
    EvidenceSourceConfig,
    discover_evidence_sources,
    score_evidence_records,
)
from replay_checker.wizard import plan_case_discovery, wizard, wizard_preview  # noqa: E402
from replay_checker.agent_workflow import (  # noqa: E402
    create_extraction_flow,
    prepare_agent_run_flow,
    run_grading_flow,
)


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
    scored = score_evidence_records(records, project_path=args.project)
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
        allow_duplicate=getattr(args, 'allow_duplicate', False),
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
    removed = dedupe_eval_cases(args.cases_root)
    if removed:
        print(f"Dedup: removed {len(removed)} older duplicate case(s):")
        for kept, dropped in removed:
            print(f"  kept {kept}, removed {dropped}")
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
    health = inspect_run_dir(run.root)
    print(f"Collected run: {evidence.run_id} status={health.status} raw={evidence.status}")
    if health.status != "completed":
        print(f"Reason: {health.reason}")
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
        allow_duplicate=getattr(args, 'allow_duplicate', False),
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


def cmd_batch_intake(args: argparse.Namespace) -> int:
    min_score = getattr(args, "min_score", 3)
    max_cases = getattr(args, "max_cases", 50)
    cases = batch_intake(
        cases_root=args.cases_root,
        project_path=args.project,
        min_score=min_score,
        max_cases=max_cases,
        codex_history_roots=tuple(args.codex_history_root) if args.codex_history_root else None,
        claude_history_roots=tuple(args.claude_history_root) if args.claude_history_root else None,
        allow_duplicate=getattr(args, "allow_duplicate", False),
    )
    print(f"Batch intake complete: {len(cases)} cases created")
    print()
    for case in cases:
        score_label = f" (score: case.score)" if case.candidate_score > 0 else ""
        print(f"  {case.id} | {case.source_type} | base: {case.base_commit[:12] if case.base_commit else '(none)'}")
    print()
    print("Next steps:")
    print(f"  Review cases: ls {args.cases_root}/")
    print(f"  Prepare a run: rtk python3 tools/replay.py prepare-run --case <case-id> --label \"<agent/model>\"")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    case = load_case(args.cases_root, args.case)
    detailed = getattr(args, "detailed", False)
    report = compare_case(
        case,
        runs_root=args.runs_root,
        reports_root=args.reports_root,
        detailed=detailed,
    )
    print(f"Created comparison report: {report}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    report = report_all_cases(
        cases_root=args.cases_root,
        runs_root=args.runs_root,
        reports_root=args.reports_root,
    )
    print(f"Created cross-case report: {report}")
    return 0


def cmd_doctor_runs(args: argparse.Namespace) -> int:
    reports = doctor_runs(args.runs_root, runner_label=args.label)
    if args.format == "tsv":
        print("status\traw_status\tworkspace\tcompletion_report\tdiff\tchanged_files\treason\trun_id")
        for report in reports:
            print(
                f"{report.status}\t{report.raw_status}\t"
                f"{'yes' if report.workspace_exists else 'no'}\t"
                f"{'yes' if report.completion_report_exists else 'no'}\t"
                f"{'yes' if report.has_result_diff else 'no'}\t"
                f"{len(report.changed_files)}\t{report.reason}\t{report.run_id}"
            )
        return 0

    counts: dict[str, int] = {}
    for report in reports:
        counts[report.status] = counts.get(report.status, 0) + 1
    summary = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())) or "no runs"
    print(f"Run doctor: {len(reports)} run(s) inspected ({summary})")
    print()
    print("| Status | Raw | Workspace | Report | Diff | Files | Run | Reason |")
    print("|---|---|---|---|---|---:|---|---|")
    for report in reports:
        print(
            f"| {report.status} | {report.raw_status} | "
            f"{'yes' if report.workspace_exists else 'no'} | "
            f"{'yes' if report.completion_report_exists else 'no'} | "
            f"{'yes' if report.has_result_diff else 'no'} | "
            f"{len(report.changed_files)} | {report.run_id} | {report.reason} |"
        )
    return 0


def cmd_flow_extract(args: argparse.Namespace) -> int:
    flow = create_extraction_flow(
        project_path=args.project,
        scope=args.scope,
        output_root=args.output_root,
    )
    print(f"Extraction kit generated: {flow.kit_path}")
    print(f"Project: {flow.project_path}")
    print(f"Scope: {flow.scope}")
    print(f"Packages: {', '.join(flow.package_ids)}")
    print()
    print("Launch options:")
    for option in flow.launch_options:
        print(f"- {option.key}: {option.label}")
        print(f"  {option.description}")
        print(f"  Command: {option.command}")
    print()
    print(flow.next_question)
    return 0


def cmd_flow_prepare_run(args: argparse.Namespace) -> int:
    prepared = prepare_agent_run_flow(
        case_id=args.case,
        agent_intro=args.agent_intro,
        label=args.label,
        cases_root=args.cases_root,
        runs_root=args.runs_root,
    )
    print(f"Prepared run: {prepared.run_id}")
    print(f"Runner label: {prepared.runner_label}")
    print(f"Task: {prepared.task_path}")
    print(f"Workspace: {prepared.workspace}")
    print(f"Completion template: {prepared.completion_template}")
    print()
    print("Next: give TASK.md to the selected agent, then ask Replay Checker to grade the run.")
    return 0


def cmd_flow_grade(args: argparse.Namespace) -> int:
    result = run_grading_flow(
        case_id=args.case,
        run_ids=tuple(args.run),
        cases_root=args.cases_root,
        runs_root=args.runs_root,
        reports_root=args.reports_root,
        rubric_path=args.rubric,
    )
    print(f"Graded case: {result.case_id}")
    print(f"Runs: {', '.join(result.scored_runs) if result.scored_runs else '(none)'}")
    for path in result.scoring_packages:
        print(f"Scoring package: {path}")
    print(f"Comparison report: {result.compare_report}")
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            preview = wizard_preview(args.project, args.scope or "")
            _print_preview(preview)
            return 0

        if args.execute_local:
            preview = wizard_preview(args.project, args.scope or "")
            _print_preview(preview)

            if preview.top_candidate is None:
                print("Warning: preview found no top candidate. intake may still discover a case from git history or plan sources.")
                print()

            print("--- Compiling case via intake ---\n")
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
            print()
            print("Next commands (copy-paste ready):")
            print(f"  rtk python3 tools/replay.py prepare-run --case {case.id} --label \"<agent/model>\"")
            print()
            print("After agent execution:")
            print(f"  rtk python3 tools/replay.py collect-run --run <run-id>")
            print(f"  rtk python3 tools/replay.py score-run --run <run-id>")
            print(f"  rtk python3 tools/replay.py compare --case {case.id}")
            return 0

        result = wizard(
            args.project,
            scope=args.scope,
            interactive=args.interactive,
        )
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _print_preview(preview) -> None:
    """Print a human-readable dry-run preview to stdout."""
    print(f"=== Wizard Dry Run: {preview.project_path.name} ===")
    print()
    print(f"Project: {preview.project_path}")
    if preview.scope:
        print(f"Scope: {preview.scope}")
    print()

    print("--- Git Signals ---")
    print(f"Git repository: {'yes' if preview.has_git else 'no'}")
    if preview.has_git:
        print(f"Working tree: {'clean' if preview.git_clean else 'dirty'}")
        if preview.head_commit:
            print(f"HEAD: {preview.head_commit}")
    print()

    print("--- Signals ---")
    if preview.signals:
        for s in preview.signals:
            print(f"  + {s}")
    else:
        print("  (none)")
    print()

    if preview.risks:
        print("--- Risks ---")
        for r in preview.risks:
            print(f"  ! {r}")
        print()

    candidate_limit = 5
    print(f"--- Candidates ({preview.candidate_count}) ---")
    if preview.candidates:
        shown_candidates = preview.candidates[:candidate_limit]
        for idx, c in enumerate(shown_candidates):
            tag = " [top]" if preview.top_candidate and c.candidate_id == preview.top_candidate.candidate_id else ""
            print(f"  {idx + 1}. {c.candidate_id}{tag}")
            print(f"     type={c.source_type}  score={c.relevance_score:.3f}  primary={c.primary_source}")
            if c.risks:
                print(f"     risks: {'; '.join(c.risks)}")
        hidden_count = len(preview.candidates) - len(shown_candidates)
        if hidden_count > 0:
            print(f"  ... {hidden_count} more candidate(s) hidden; use inspect-candidates for the full audit list.")
    else:
        print("  No candidates found from available sources.")
    print()

    print("--- Next Step ---")
    for cmd in preview.next_commands:
        print(f"  {cmd}")


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


def cmd_dedupe_cases(args: argparse.Namespace) -> int:
    removed = dedupe_eval_cases(args.cases_root, dry_run=args.dry_run)
    if not removed:
        print("No duplicate cases found.")
        return 0
    label = "Would remove" if args.dry_run else "Removed"
    print(f"{label} {len(removed)} older duplicate case(s):")
    for kept, dropped in removed:
        print(f"  kept {kept}, removed {dropped}")
    return 0


def cmd_inspect_duplicates(args: argparse.Namespace) -> int:
    report = inspect_duplicates(Path(args.cases_root))
    total = report["total_cases"]
    exact = report["exact_clusters"]
    likely = report["likely_pairs"]

    print(f"Total cases scanned: {total}")
    print()

    if exact:
        print(f"Exact duplicate clusters: {len(exact)}")
        for rep, dup, reasons in exact:
            reason_str = ", ".join(reasons)
            print(f"  {rep} == {dup}  ({reason_str})")
        print()
    else:
        print("Exact duplicates: none")
        print()

    if likely:
        print(f"Likely duplicate pairs: {len(likely)}")
        for id_a, id_b, sim, reasons in likely:
            reason_str = ", ".join(reasons)
            print(f"  {id_a} ~ {id_b}  (sim={sim:.2f}, {reason_str})")
        print()
    else:
        print("Likely duplicates: none")
        print()

    if not exact and not likely:
        print("No duplicates found.")
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
    intake_parser.add_argument("--allow-duplicate", action="store_true", default=False,
                               help="create case even if exact duplicate exists")
    intake_parser.set_defaults(func=cmd_intake)

    batch_intake_parser = sub.add_parser(
        "batch-intake",
        help="extract cases from all orchestration kits in a project",
    )
    batch_intake_parser.add_argument("--cases-root", default="cases")
    batch_intake_parser.add_argument("--project", required=True)
    batch_intake_parser.add_argument("--min-score", type=int, default=3,
                                     help="minimum kit score to include (default: 3)")
    batch_intake_parser.add_argument("--max-cases", type=int, default=50,
                                     help="maximum number of cases to create (default: 50)")
    batch_intake_parser.add_argument("--codex-history-root", action="append", default=[])
    batch_intake_parser.add_argument("--claude-history-root", action="append", default=[])
    batch_intake_parser.add_argument("--allow-duplicate", action="store_true", default=False)
    batch_intake_parser.set_defaults(func=cmd_batch_intake)

    create = sub.add_parser("create-case", help="create a replay case from a historical plan package")
    create.add_argument("--cases-root", default="cases")
    create.add_argument("--project", required=True)
    create.add_argument("--plan", required=True)
    create.add_argument("--base", required=True)
    create.add_argument("--case-id", required=True)
    create.add_argument("--verify", action="append", default=[])
    create.add_argument("--allow-duplicate", action="store_true", default=False,
                        help="create case even if exact duplicate exists")
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
    compare.add_argument("--detailed", action="store_true", default=False,
                        help="show per-run details including telemetry")
    compare.set_defaults(func=cmd_compare)

    report_parser = sub.add_parser("report", help="generate cross-case report with anonymous runner IDs")
    report_parser.add_argument("--cases-root", default="cases")
    report_parser.add_argument("--runs-root", default="runs")
    report_parser.add_argument("--reports-root", default="reports")
    report_parser.set_defaults(func=cmd_report)

    doctor = sub.add_parser("doctor-runs", help="derive canonical run health from reports, workspaces, and diffs")
    doctor.add_argument("--runs-root", default="runs")
    doctor.add_argument("--label", default=None, help="only inspect runs with this runner_label")
    doctor.add_argument("--format", choices=("table", "tsv"), default="table")
    doctor.set_defaults(func=cmd_doctor_runs)

    flow = sub.add_parser(
        "flow",
        help="agent-platform workflow helpers for extract, prepare-run, and grade",
    )
    flow_sub = flow.add_subparsers(dest="flow_command", required=True)

    flow_extract = flow_sub.add_parser(
        "extract",
        help="generate a discovery kit and print launch choices",
    )
    flow_extract.add_argument("--project", required=True)
    flow_extract.add_argument("--scope", required=True)
    flow_extract.add_argument("--output-root", default=None)
    flow_extract.set_defaults(func=cmd_flow_extract)

    flow_prepare = flow_sub.add_parser(
        "prepare-run",
        help="prepare a run using date plus agent self-introduction as default identity",
    )
    flow_prepare.add_argument("--cases-root", default="cases")
    flow_prepare.add_argument("--runs-root", default="runs")
    flow_prepare.add_argument("--case", required=True)
    flow_prepare.add_argument("--agent-intro", default=None)
    flow_prepare.add_argument("--label", default=None)
    flow_prepare.set_defaults(func=cmd_flow_prepare_run)

    flow_grade = flow_sub.add_parser(
        "grade",
        help="collect evidence, create scoring packages, and refresh comparison report",
    )
    flow_grade.add_argument("--cases-root", default="cases")
    flow_grade.add_argument("--runs-root", default="runs")
    flow_grade.add_argument("--reports-root", default="reports")
    flow_grade.add_argument("--case", required=True)
    flow_grade.add_argument("--run", action="append", default=[])
    flow_grade.add_argument("--rubric", default="rubrics/default.yaml")
    flow_grade.set_defaults(func=cmd_flow_grade)

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

    dedupe_parser = sub.add_parser(
        "dedupe-cases",
        help="remove older duplicate eval cases, keeping the newest per skill-eval group",
    )
    dedupe_parser.add_argument("--cases-root", default="cases")
    dedupe_parser.add_argument("--dry-run", action="store_true", default=False,
                               help="show what would be removed without deleting")
    dedupe_parser.set_defaults(func=cmd_dedupe_cases)

    inspect_dup = sub.add_parser(
        "inspect-duplicates",
        help="audit all cases for content-aware exact and likely duplicates",
    )
    inspect_dup.add_argument("--cases-root", default="cases",
                             help="cases directory root to scan")
    inspect_dup.set_defaults(func=cmd_inspect_duplicates)

    wizard_parser = sub.add_parser("wizard", help="interactive case discovery wizard")
    wizard_parser.add_argument("--project", required=True, help="path to the target project")
    wizard_parser.add_argument("--scope", default=None, help="natural-language discovery scope")
    wizard_parser.add_argument("--no-interactive", dest="interactive", action="store_false", default=True, help="disable interactive prompts")
    wizard_parser.add_argument("--dry-run", action="store_true", default=False, help="fast local preview: show extractability signals without generating a kit or case")
    wizard_parser.add_argument("--execute-local", action="store_true", default=False, help="preview + auto-compile a case via intake (no kit, no agent launch)")
    wizard_parser.add_argument("--cases-root", default="cases", help="cases directory root (for --execute-local)")
    wizard_parser.add_argument("--codex-history-root", action="append", default=[], help="Codex history root(s)")
    wizard_parser.add_argument("--claude-history-root", action="append", default=[], help="Claude history root(s)")
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
