"""Scoring-package assembly, rendering, provenance, and anonymization helpers.

This owner module builds the scoring-package markdown file for a run,
including evidence inventory, telemetry, gate assessment, score ceilings,
provenance/risk metadata, and bias warnings.

Uses ``replay_types``, ``scoring``, ``core``, and ``yaml_lite`` directly.
Does NOT import ``replay.py``.
"""

from __future__ import annotations

from pathlib import Path

from .core import stable_hash
from .replay_types import ReplayRun
from .scoring import (
    CaseProvenance,
    EvidenceGate,
    RunEvaluation,
    build_case_provenance,
    evaluate_run,
    parse_rubric,
)
from .yaml_lite import parse_simple_yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_run(run: ReplayRun, *, rubric_path: str | Path | None = None) -> Path:
    """Build and write the scoring package markdown for *run*.

    Returns the path to ``<run_root>/scoring_package.md``.
    """
    evidence_yaml = run.root / "evidence" / "evidence.yaml"
    if not evidence_yaml.exists():
        raise FileNotFoundError(
            f"Evidence not found at {evidence_yaml}. "
            "Run collect-run before score-run."
        )
    resolved_rubric_path = _resolve_rubric_path(run, rubric_path)
    rubric = parse_rubric(resolved_rubric_path) if resolved_rubric_path is not None else {}
    anonymous = _anonymous_runner_id(run)
    evidence = parse_simple_yaml(run.root / "evidence" / "evidence.yaml")
    missing = list(evidence.get("missing_fields") or [])
    changed_files = list(evidence.get("changed_files") or [])
    diff_path = run.root / "evidence" / "diff.patch"
    completion_path = run.root / "completion_report.md"
    evidence_status = evidence.get("status", "unknown")

    lines = [
        f"# Scoring Package: {run.id}",
        "",
        "Protocol Version: replay-checker-score-v2",
        "",
        f"Anonymous runner: {anonymous}",
        f"Case: {run.case.id}",
        "",
        "## Required Inputs",
        f"- Evidence YAML: `{run.root / 'evidence' / 'evidence.yaml'}`",
        f"- Diff patch: `{diff_path}`",
        f"- Completion report: `{completion_path}`",
        f"- Case metadata: `{run.case.root / 'case.yaml'}`",
        "",
        "## Rubric Weights",
        f"- Result score weight: {rubric.get('result_weight', '80')}",
        f"- Process score weight: {rubric.get('process_weight', '20')}",
        "",
        "## Evidence Inventory",
        f"- Status: {evidence_status}",
        f"- Changed files ({len(changed_files)}): {', '.join(changed_files) if changed_files else '(none)'}",
        f"- Diff present: {diff_path.exists() and diff_path.stat().st_size > 0}",
        f"- Completion report: {completion_path.exists()}",
    ]

    expected_output = rubric.get("expected_output", "")
    criteria = rubric.get("criteria", [])
    if expected_output or criteria:
        lines.extend([
            "",
            "## Rubric Expectations",
        ])
        if expected_output:
            lines.append(f"- Expected output: {expected_output}")
        if isinstance(criteria, list) and criteria:
            lines.extend(f"- {item}" for item in criteria)

    # Include telemetry section
    telemetry_block = evidence.get("telemetry", None)
    if isinstance(telemetry_block, dict):
        lines.extend(_format_telemetry_scoring(telemetry_block))

    if missing:
        lines.extend([
            "",
            "## Missing Evidence",
            *[f"- {item}" for item in missing],
        ])
    else:
        lines.extend([
            "",
            "## Missing Evidence",
            "- (none)",
        ])

    minimum = rubric.get("minimum_evidence", [])
    if minimum:
        lines.extend([
            "",
            "## Minimum Evidence Check (from rubric)",
            *[f"- {item}: {'found' if _evidence_item_found(item, completion_path, diff_path, evidence_status, changed_files) else 'MISSING'}" for item in minimum],
        ])

    lines.extend([
        "",
        "## Bias Warnings",
        "- Do not infer quality from agent identity, model family, verbosity, or style similarity.",
        "- Score observable result evidence first; award process points only for cited process evidence.",
        "- The runner label is intentionally excluded from this package to prevent identity bias.",
        "- Mark the score invalid if evidence is missing or uncited.",
        "- Compare only evidence packs, not assumptions about the runner's capabilities.",
    ])

    # Case provenance — scorer-facing trust and risk metadata
    provenance = build_case_provenance(
        source_type=run.case.source_type,
        base_source=run.case.base_source,
        base_confidence=run.case.base_confidence,
        candidate_score=run.case.candidate_score,
        is_synthetic=run.case.synthetic_case,
        evidence_sources=run.case.evidence_sources,
        candidate_source_type=run.case.candidate_source_type,
    )
    lines.extend(_render_provenance_section(provenance))

    lines.extend([
        "",
        "## Process Scoring Guidance",
        "- Telemetry is a risk/scope signal, not an automatic quality score.",
        "- Do not penalize agents for low commit counts (commits are not required by default).",
        "- Do not penalize agents for having untracked files unless they contain secrets or forbidden paths.",
        "- Forbidden path touches (e.g. `_reference/`, `.git/`) should cap or warn on scope discipline.",
        "- Suspicious path touches (e.g. `.env`, `credentials`) warrant a note but are not automatic failures.",
        "- Assess rollback/reflog signals in context; iterative refinement is normal development practice.",
    ])

    lines.extend([
        "",
        "## Invalid Score Conditions",
        "- The score cites runner/model identity instead of evidence.",
        "- The score uses uncited assumptions about unavailable files or hidden context.",
        "- Required evidence is missing and the score does not mark the affected category as unverified.",
        "- The scoring agent reads `_reference/` oracle material while evaluating an executing agent's output.",
    ])

    lines.extend([
        "",
        "## Evidence References",
        f"- Diff: `{diff_path}`",
        f"- Evidence YAML: `{run.root / 'evidence' / 'evidence.yaml'}`",
        f"- Completion report: `{completion_path}`",
        f"- Case: `{run.case.root / 'case.yaml'}`",
    ])

    # Evidence gate and ceiling evaluation via the canonical snapshot.
    # Skip runner identity gate: run.yaml is internal metadata, not exposed to the scorer.
    run_eval = evaluate_run(
        evidence_root=run.root,
        run_id=run.id,
        runner_label=None,
        changed_files=changed_files if isinstance(changed_files, list) else [],
    )

    display_results = [r for r in run_eval.gate_results if r.gate != EvidenceGate.RUNNER_IDENTITY_HIDDEN]

    lines.extend([
        "",
        "## Evidence Gate Assessment",
    ])
    for result in display_results:
        status = "PASS" if result.passed else "FAIL"
        detail = f" — {result.detail}" if result.detail else ""
        lines.append(f"- [{status}] {result.gate.value}{detail}")
    lines.append(f"- Overall: {'all gates passed' if run_eval.all_gates_passed else 'SOME GATES FAILED'}")

    if run_eval.is_invalid:
        lines.extend([
            "",
            "**Score is INVALID.**",
            *[f"- {reason}" for reason in run_eval.invalid_reasons],
        ])

    has_ceilings = any(v is not None for v in (run_eval.result_ceiling, run_eval.process_ceiling, run_eval.verification_ceiling))
    if has_ceilings or run_eval.overall_invalid:
        lines.extend([
            "",
            "## Score Ceilings",
        ])
        if run_eval.result_ceiling is not None:
            lines.append(f"- Result score ceiling: {run_eval.result_ceiling}")
        if run_eval.process_ceiling is not None:
            lines.append(f"- Process score ceiling: {run_eval.process_ceiling}")
        if run_eval.verification_ceiling is not None:
            lines.append(f"- Verification score ceiling: {run_eval.verification_ceiling}")
        for reason in run_eval.ceiling_reasons:
            lines.append(f"  - {reason}")

    path = run.root / "scoring_package.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------------
# Provenance and anonymization helpers
# ---------------------------------------------------------------------------


def render_provenance_section(provenance: CaseProvenance) -> list[str]:
    """Render the case provenance section for the scoring package.

    Public alias for ``_render_provenance_section``.
    """
    return _render_provenance_section(provenance)


def anonymous_runner_id(run: ReplayRun) -> str:
    """Resolve the anonymous runner ID for *run*.

    Public alias for ``_anonymous_runner_id``.
    """
    return _anonymous_runner_id(run)


# ---------------------------------------------------------------------------
# Private helpers (not part of public API)
# ---------------------------------------------------------------------------


def _resolve_rubric_path(run: ReplayRun, rubric_path: str | Path | None) -> Path | None:
    if rubric_path is not None:
        return Path(rubric_path)
    case_rubric = run.case.root / "eval_rubric.yaml"
    if case_rubric.exists():
        return case_rubric
    default_rubric = Path("rubrics/default.yaml")
    return default_rubric if default_rubric.exists() else None


def _format_telemetry_scoring(telemetry: dict[str, object]) -> list[str]:
    lines = [
        "",
        "## Process Telemetry",
        f"- Changed file count: {telemetry.get('changed_file_count', 0)}",
        f"- Added lines: {telemetry.get('added_lines', 0)}",
        f"- Deleted lines: {telemetry.get('deleted_lines', 0)}",
        f"- Untracked file count: {telemetry.get('untracked_file_count', 0)}",
        f"- Commit count: {telemetry.get('commit_count', 0)}",
        f"- Rollback signal: {telemetry.get('rollback_signal', '') or '(none)'}",
    ]
    forbidden = telemetry.get("forbidden_path_touches", [])
    if isinstance(forbidden, list) and forbidden:
        lines.append(f"- Forbidden path touches: {', '.join(forbidden)}")
    else:
        lines.append("- Forbidden path touches: (none)")

    suspicious = telemetry.get("suspicious_path_touches", [])
    if isinstance(suspicious, list) and suspicious:
        lines.append(f"- Suspicious path touches: {', '.join(suspicious)}")
    else:
        lines.append("- Suspicious path touches: (none)")
    return lines


def _evidence_item_found(
    item: str,
    completion_path: Path,
    diff_path: Path,
    evidence_status: str,
    changed_files: list[str],
) -> bool:
    if item == "completion_report.md":
        return completion_path.exists()
    if item == "diff.patch":
        return diff_path.exists() and diff_path.stat().st_size > 0
    if item == "evidence.yaml":
        return evidence_status != "unknown"
    if item == "changed_files":
        return bool(changed_files)
    return False


def _render_provenance_section(provenance: CaseProvenance) -> list[str]:
    """Render the case provenance section for the scoring package."""
    lines = [
        "",
        "## Case Provenance & Source Risk",
        "",
        "This section describes how the case was constructed and the reliability of its evidence sources.",
        "It is scorer-facing metadata — not execution-agent oracle evidence.",
        "",
        f"- Primary source type: `{provenance.primary_source_type}`",
        f"- Contributing source types: {', '.join(f'`{t}`' for t in provenance.source_types) if provenance.source_types else '(none)'}",
        f"- Base commit source: `{provenance.base_commit_source or 'none'}`",
        f"- Base commit confidence: `{provenance.base_commit_confidence}`",
        f"- Merged confidence: `{provenance.confidence}`",
        f"- Candidate score: {provenance.candidate_score:.3f}",
        f"- Synthetic case: {'yes' if provenance.is_synthetic else 'no'}",
        f"- Overall risk level: `{provenance.risk_level}`",
    ]

    if provenance.risk_reasons:
        lines.append("")
        lines.append("### Risk Factors")
        for reason in provenance.risk_reasons:
            lines.append(f"- {reason}")

    if provenance.conflicts:
        lines.append("")
        lines.append("### Source Conflicts")
        for conflict in provenance.conflicts:
            severity_mark = _conflict_severity_mark(conflict.severity)
            lines.append(f"- [{severity_mark}] `{conflict.conflict_type}`: {conflict.description}")

    if provenance.is_low_confidence:
        lines.extend([
            "",
            "**Note:** This is a low-confidence case. Scores should be interpreted with caution.",
        ])

    return lines


def _conflict_severity_mark(severity: str) -> str:
    if severity == "high":
        return "HIGH"
    if severity == "medium":
        return "MED"
    return "LOW"


def _anonymous_runner_id(run: ReplayRun) -> str:
    run_yaml = run.root / "run.yaml"
    if run_yaml.exists():
        data = parse_simple_yaml(run_yaml)
        stored = str(data.get("anonymous_runner_id", "")).strip()
        if stored:
            return stored
    return _stable_anonymous_runner_id(run.id)


def _stable_anonymous_runner_id(run_id: str) -> str:
    return f"runner-{stable_hash(run_id, length=10)}"
