"""Per-case comparison and cross-case aggregate report projections.

This owner module reads evaluation data and evidence, then produces
concise/detailed per-case comparisons and cross-case aggregate reports.

Uses ``scoring``, ``evaluation``, ``replay_types``, ``yaml_lite``, and
``case_paths`` directly.  Does NOT import ``replay.py``.
"""

from __future__ import annotations

from pathlib import Path

from .case_paths import iter_case_dirs
from .evaluation import Recommendation, read_recommendation, read_summary
from .replay_types import ReplayCase
from .scoring import EvidenceGate, evaluate_run
from .yaml_lite import parse_simple_yaml


# ---------------------------------------------------------------------------
# Per-case comparison
# ---------------------------------------------------------------------------


def compare_case(
    case: ReplayCase,
    *,
    runs_root: str | Path,
    reports_root: str | Path,
    detailed: bool = False,
) -> Path:
    """Generate a per-case comparison report.

    Returns the path to ``<reports_root>/<case.id>.md``.
    """
    root = Path(reports_root)
    root.mkdir(parents=True, exist_ok=True)

    # Collect run IDs for this case
    run_ids: list[str] = []
    for run_dir in sorted(Path(runs_root).glob(f"{case.id}-*")):
        evidence_path = run_dir / "evidence" / "evidence.yaml"
        if evidence_path.exists():
            run_ids.append(run_dir.name)

    # Produce output using evaluation system if available
    output = build_compare_output(case, run_ids, runs_root, detailed=detailed)

    report = root / f"{case.id}.md"
    report.write_text(f"# Replay Comparison: {case.id}\n\n{output}\n", encoding="utf-8")

    return report


def build_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_root: str | Path,
    *,
    detailed: bool = False,
) -> str:
    """Build the comparison output body for *case*.

    Public alias for ``_build_compare_output``.
    """
    return _build_compare_output(case, run_ids, runs_root, detailed=detailed)


# ---------------------------------------------------------------------------
# Cross-case aggregate report
# ---------------------------------------------------------------------------


def report_all_cases(
    cases_root: str | Path,
    runs_root: str | Path,
    reports_root: str | Path,
) -> Path:
    """Generate a comprehensive cross-case aggregate report with scoring data.

    Returns the path to ``<reports_root>/AGGREGATE_REPORT.md``.
    """
    from .evaluation import read_attempts  # lazy to avoid circular

    cases_dir = Path(cases_root)
    runs_dir = Path(runs_root)
    reports_dir = Path(reports_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Replay Checker — 综合评分报告",
        "",
        "跨 case 汇总，包含评分、证据门控、等级和排名。",
        "",
    ]

    # Collect data for every case that has runs with evidence
    case_rows: list[dict[str, object]] = []
    total_runs = 0
    total_cases_with_evidence = 0
    tiers: dict[str, int] = {}

    case_dirs = iter_case_dirs(cases_dir)

    for case_dir in case_dirs:
        case_data = parse_simple_yaml(case_dir / "case.yaml")
        case_id = str(case_data.get("id", case_dir.name))
        case_type = str(case_data.get("source_type", "unknown"))

        run_dirs = sorted(
            d for d in runs_dir.iterdir()
            if d.is_dir() and d.name.startswith(case_id)
        )
        if not run_dirs:
            continue

        runs_with_evidence: list[dict[str, object]] = []
        for run_dir in run_dirs:
            ev_path = run_dir / "evidence" / "evidence.yaml"
            if not ev_path.exists():
                continue
            ev = parse_simple_yaml(ev_path)
            telemetry = ev.get("telemetry", None)
            changed = len(list(ev.get("changed_files") or []))
            added = "-"
            deleted = "-"
            commits = "-"
            if isinstance(telemetry, dict):
                added = str(telemetry.get("added_lines", "-"))
                deleted = str(telemetry.get("deleted_lines", "-"))
                commits = str(telemetry.get("commit_count", "-"))
            run_yaml_path = run_dir / "run.yaml"
            anon = run_dir.name
            label = ""
            if run_yaml_path.exists():
                rd = parse_simple_yaml(run_yaml_path)
                anon = str(rd.get("anonymous_runner_id", run_dir.name))
                label = str(rd.get("runner_label", ""))
            runs_with_evidence.append(dict(
                run_id=run_dir.name, anon=anon, label=label, changed=changed,
                added=added, deleted=deleted, commits=commits,
                status=str(ev.get("status", "unknown")),
            ))

        if not runs_with_evidence:
            continue

        total_cases_with_evidence += 1
        total_runs += len(runs_with_evidence)

        # Read evaluation data
        score = 0.0
        sentence = ""
        tier = "unknown"
        validity = "unvalidated"
        confidence = "low"

        rec = read_recommendation(case_dir)
        if rec.score > 0 or rec.sentence:
            score = rec.score
            sentence = rec.sentence
            validity = rec.validity
            confidence = rec.confidence

        summary = read_summary(case_dir)
        if summary.total_score > 0:
            score = summary.total_score
            tier = summary.tier.value
            validity = summary.validity
            confidence = summary.confidence

        run_ids = [str(r["run_id"]) for r in runs_with_evidence]
        if all_run_result_evidence_blocked(run_ids, runs_dir):
            score = 0.0
            sentence = (
                "Score is invalid — insufficient evidence: no run has a "
                "non-empty diff and changed-file evidence."
            )
            tier = "invalid"
            validity = "no valid runs with result evidence"
            confidence = "low"

        tiers[tier] = tiers.get(tier, 0) + 1

        case_rows.append(dict(
            case_id=case_id, case_type=case_type,
            score=score, tier=tier, sentence=sentence,
            validity=validity, confidence=confidence,
            runs=runs_with_evidence,
        ))

    # Sort by score descending
    case_rows.sort(key=lambda r: -float(str(r["score"])))

    # ---- Aggregate summary ----
    lines.extend([
        "## 汇总概览",
        "",
        f"- 评估 case 数: {total_cases_with_evidence}",
        f"- 评估 run 总数: {total_runs}",
    ])
    if tiers:
        lines.append(f"- 等级分布: {', '.join(f'{t}: {c}' for t, c in sorted(tiers.items()))}")
    if case_rows:
        avg = sum(float(str(r["score"])) for r in case_rows) / len(case_rows)
        lines.append(f"- 平均分: {avg:.1f} / 100")
        lines.append(f"- 最高分: {float(str(case_rows[0]['score'])):.0f} ({case_rows[0]['case_id']})")
        lines.append(f"- 最低分: {float(str(case_rows[-1]['score'])):.0f} ({case_rows[-1]['case_id']})")
    lines.append("")

    # ---- Ranking table ----
    lines.extend([
        "## 综合排名",
        "",
        "| 排名 | Case | 类型 | 得分 | 等级 | 结论 |",
        "|---|---|---|---|---|---|",
    ])
    for idx, row in enumerate(case_rows):
        tier = str(row["tier"])
        emoji = {"transformative": "++", "excellent": "+", "solved": "=", "partial": "~", "failed": "-", "invalid": "x"}.get(tier, "?")
        lines.append(
            f"| {idx + 1} | {row['case_id']} | {row['case_type']} | "
            f"{float(str(row['score'])):.0f} | {emoji} {tier} | {row['sentence']} |"
        )
    lines.append("")

    # ---- Per-case detail ----
    lines.extend([
        "## 各 Case 详情",
        "",
    ])
    for row in case_rows:
        lines.extend([
            f"### {row['case_id']} — {float(str(row['score'])):.0f}/100 ({row['tier']})",
            "",
            f"- **结论**: {row['sentence']}",
            f"- **有效性**: {row['validity']}",
            f"- **置信度**: {row['confidence']}",
            f"- **来源类型**: {row['case_type']}",
            "",
            "| Run | Agent | 匿名 ID | 状态 | 变更文件 | +行 | -行 | 提交 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for r in row["runs"]:
            label_display = r['label'] if r['label'] else '-'
            lines.append(
                f"| {r['run_id']} | {label_display} | {r['anon']} | {r['status']} | "
                f"{r['changed']} | {r['added']} | {r['deleted']} | {r['commits']} |"
            )
        lines.append("")

    # ---- Conclusions ----
    lines.append("## 评估结论")
    lines.append("")

    if not case_rows:
        lines.append("暂无含证据的 run 可供评估。")
    else:
        scored = [r for r in case_rows if float(str(r["score"])) > 0]
        passing = [r for r in scored if str(r["tier"]) not in ("failed", "invalid")]
        failing = [r for r in case_rows if str(r["tier"]) in ("failed", "invalid")]

        if failing:
            lines.append(f"### 需要重跑: {len(failing)} 个 case 未通过证据门控")
            for r in failing:
                lines.append(f"- `{r['case_id']}`: {r['sentence']} (得分 {float(str(r['score'])):.0f})")
            lines.append("")
            lines.append("这些 case 的 diff 为空或缺失关键证据。重新执行 agent 任务并运行 `collect-run` 后再评分。")
            lines.append("")

        if passing:
            lines.append(f"### 可用: {len(passing)} 个 case 通过评估")
            for r in passing:
                lines.append(f"- `{r['case_id']}`: {float(str(r['score'])):.0f}/100 — {r['sentence']}")
            lines.append("")

        if len(scored) >= 2:
            best = scored[0]
            worst = scored[-1]
            lines.append(f"### 最佳表现: `{best['case_id']}` ({float(str(best['score'])):.0f}/100)")
            lines.append(f"### 最弱表现: `{worst['case_id']}` ({float(str(worst['score'])):.0f}/100)")

    report = reports_dir / "AGGREGATE_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Evidence-blocking helpers
# ---------------------------------------------------------------------------


def all_run_result_evidence_blocked(run_ids: list[str], runs_dir: Path) -> bool:
    """Return True when every visible run lacks minimum result evidence.

    Public alias for ``_all_run_result_evidence_blocked``.
    """
    return _all_run_result_evidence_blocked(run_ids, runs_dir)


def has_recommendation(recommendation: Recommendation) -> bool:
    """Return True when a recommendation carries a meaningful score or sentence."""
    return _has_recommendation(recommendation)


def format_default_recommendation(recommendation: Recommendation) -> str:
    """Format a recommendation into concise comparison output."""
    return _format_default_recommendation(recommendation)


def format_missing_evaluation(run_ids: list[str]) -> str:
    """Format a message when no evaluation data is available."""
    return _format_missing_evaluation(run_ids)


# ---------------------------------------------------------------------------
# Private helpers (not part of public API)
# ---------------------------------------------------------------------------


def _build_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_root: str | Path,
    *,
    detailed: bool = False,
) -> str:
    runs_dir = Path(runs_root)

    if not run_ids:
        return "No runs with evidence found for this case."

    if not detailed:
        return _build_default_compare_output(case, run_ids, runs_dir)

    return _build_detailed_compare_output(case, run_ids, runs_dir)


def _build_default_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_dir: Path,
) -> str:
    """Concise default output: score, recommendation sentence, validity, confidence."""
    if _all_run_result_evidence_blocked(run_ids, runs_dir):
        return "\n".join([
            "0 / 100",
            "Score is invalid — insufficient evidence: no run has a non-empty diff and changed-file evidence.",
            "Validity: no valid runs with result evidence; Confidence: low",
        ])

    recommendation = read_recommendation(case.root)

    if _has_recommendation(recommendation):
        return _format_default_recommendation(recommendation)

    return _format_missing_evaluation(run_ids)


def _all_run_result_evidence_blocked(run_ids: list[str], runs_dir: Path) -> bool:
    """Return true when every visible run lacks minimum result evidence."""
    if not run_ids:
        return False

    checked = 0
    for run_id in run_ids:
        run_root = runs_dir / run_id
        evidence_path = run_root / "evidence" / "evidence.yaml"
        if not evidence_path.exists():
            continue
        evidence = parse_simple_yaml(evidence_path)
        changed_files = list(evidence.get("changed_files") or [])
        run_eval = evaluate_run(
            evidence_root=run_root,
            run_id=run_id,
            runner_label=None,
            changed_files=changed_files,
        )
        checked += 1
        if not run_eval.result_evidence_blocked:
            return False

    return checked > 0


def _build_detailed_compare_output(
    case: ReplayCase,
    run_ids: list[str],
    runs_dir: Path,
) -> str:
    """Detailed per-run output with telemetry."""
    lines: list[str] = []

    lines.extend([
        "## Per-Run Details",
        "",
    ])
    for run_id in run_ids:
        run_dir = runs_dir / run_id
        ev_path = run_dir / "evidence" / "evidence.yaml"
        run_yaml_path = run_dir / "run.yaml"
        anonymous = run_id
        if run_yaml_path.exists():
            run_data = parse_simple_yaml(run_yaml_path)
            anonymous = str(run_data.get("anonymous_runner_id", run_id))
        if ev_path.exists():
            ev = parse_simple_yaml(ev_path)
            lines.extend([
                f"### {anonymous}",
                f"- Status: {ev.get('status', 'unknown')}",
                f"- Changed files: {len(list(ev.get('changed_files') or []))}",
            ])
            telemetry = ev.get("telemetry", None)
            if isinstance(telemetry, dict):
                lines.extend([
                    f"- Added lines: {telemetry.get('added_lines', '-')}",
                    f"- Deleted lines: {telemetry.get('deleted_lines', '-')}",
                    f"- Untracked files: {telemetry.get('untracked_file_count', '-')}",
                    f"- Commits: {telemetry.get('commit_count', '-')}",
                ])
                forbidden = telemetry.get("forbidden_path_touches", [])
                if isinstance(forbidden, list) and forbidden:
                    lines.append(f"- WARNING: Forbidden path touches: {', '.join(forbidden)}")
            lines.append("")

    return "\n".join(lines)


def _has_recommendation(recommendation: Recommendation) -> bool:
    return bool(recommendation.sentence.strip()) or recommendation.score > 0.0


def _format_default_recommendation(recommendation: Recommendation) -> str:
    score = f"{recommendation.score:.0f}"
    sentence = recommendation.sentence.strip() or "Evaluation complete; inspect detailed output before making a decision."
    return "\n".join([
        f"{score} / 100",
        sentence,
        f"Validity: {recommendation.validity}; Confidence: {recommendation.confidence}",
    ])


def _format_missing_evaluation(run_ids: list[str]) -> str:
    return "\n".join([
        "No evaluation available for this case.",
        "Run score-run or refresh evaluation state, then compare again.",
        f"Runs with evidence: {len(run_ids)}",
    ])
