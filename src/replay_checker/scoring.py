"""Evidence validity gates and score ceiling rules for Replay Checker scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .yaml_lite import coerce_bool_int, parse_yaml_file


class EvidenceGate(str, Enum):
    """Evidence validity gate identifiers."""

    DIFF_PRESENT = "diff_present"
    COMPLETION_REPORT_PRESENT = "completion_report_present"
    CHANGED_FILES_PRESENT = "changed_files_present"
    VERIFICATION_CITED = "verification_cited"
    REFERENCE_ACCESS_ABSENT = "reference_access_absent"
    RUNNER_IDENTITY_HIDDEN = "runner_identity_hidden"


@dataclass
class GateResult:
    gate: EvidenceGate
    passed: bool
    detail: str = ""


@dataclass
class GateEvaluation:
    results: list[GateResult] = field(default_factory=list)
    all_passed: bool = True
    is_invalid: bool = False
    invalid_reasons: list[str] = field(default_factory=list)

    def add(self, result: GateResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.all_passed = False

    def add_invalid(self, reason: str) -> None:
        self.is_invalid = True
        self.invalid_reasons.append(reason)


@dataclass
class ScoreCeilings:
    result_ceiling: float | None = None
    process_ceiling: float | None = None
    verification_ceiling: float | None = None
    overall_invalid: bool = False
    reasons: list[str] = field(default_factory=list)


def evaluate_evidence_gates(
    *,
    evidence_root: Path,
    run_id: str | None = None,
    runner_label: str | None = None,
    changed_files: list[str] | None = None,
) -> GateEvaluation:
    """Evaluate evidence validity gates against a run's evidence directory.

    Args:
        evidence_root: Path to the run root (contains evidence/ and completion_report.md).
        run_id: Optional run ID used to check anonymous ID presence.
        runner_label: If provided, will be checked against evidence content.
        changed_files: Optional list of changed files from evidence.yaml.

    Returns:
        GateEvaluation with per-gate results and overall validity.
    """
    if changed_files is None:
        changed_files = []

    evaluation = GateEvaluation()
    evidence_dir = evidence_root / "evidence"
    diff_path = evidence_dir / "diff.patch"
    completion_path = evidence_root / "completion_report.md"
    evidence_yaml_path = evidence_dir / "evidence.yaml"

    evaluation.add(_check_diff_present(diff_path))
    evaluation.add(_check_completion_report_present(completion_path))
    evaluation.add(_check_changed_files_present(changed_files))
    evaluation.add(_check_verification_cited(completion_path, evidence_yaml_path))
    evaluation.add(_check_reference_access_absent(evidence_root, completion_path))
    evaluation.add(
        _check_runner_identity_hidden(
            evidence_root,
            evidence_yaml_path,
            run_id=run_id,
            runner_label=runner_label,
        )
    )

    _check_oracle_leakage(evaluation, evidence_root, completion_path)

    return evaluation


def compute_score_ceilings(
    gate_evaluation: GateEvaluation,
) -> ScoreCeilings:
    """Compute score ceilings based on evidence gate results.

    Rules:
    - No diff → result ceiling 0
    - No completion report → process ceiling 0
    - No verification cited → verification ceiling 0
    - Reference/oracle leakage → overall invalid (no valid score possible)
    - Runner identity exposed → overall invalid
    """
    ceilings = ScoreCeilings()

    if gate_evaluation.is_invalid:
        ceilings.overall_invalid = True
        ceilings.reasons.extend(gate_evaluation.invalid_reasons)
        return ceilings

    for result in gate_evaluation.results:
        if result.gate == EvidenceGate.DIFF_PRESENT and not result.passed:
            ceilings.result_ceiling = 0.0
            ceilings.reasons.append("No diff present → result score capped at 0")
        elif (
            result.gate == EvidenceGate.COMPLETION_REPORT_PRESENT and not result.passed
        ):
            ceilings.process_ceiling = 0.0
            ceilings.reasons.append(
                "No completion report → process score capped at 0"
            )
        elif (
            result.gate == EvidenceGate.VERIFICATION_CITED and not result.passed
        ):
            ceilings.verification_ceiling = 0.0
            ceilings.reasons.append(
                "No verification cited → verification subscore capped at 0"
            )
        elif (
            result.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN and not result.passed
        ):
            ceilings.overall_invalid = True
            ceilings.reasons.append(
                f"Runner identity exposed → score invalid ({result.detail})"
            )

    return ceilings


def apply_ceiling(score: float, ceiling: float | None) -> float:
    """Clamp a score to a ceiling. Returns score if ceiling is None."""
    if ceiling is None:
        return score
    return min(score, ceiling)


def parse_rubric(path: Path) -> dict[str, Any]:
    """Parse the YAML rubric file into a dict.

    Reads result_weight, process_weight, minimum_evidence,
    gates (optional), and ceilings (optional) sections.
    Handles scalars, string lists, and one level of nested dicts.
    """
    data = parse_yaml_file(
        path,
        scalar_parser=coerce_bool_int,
        parse_list_item_dicts=False,
        parse_inline_lists=True,
    )
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Private gate check helpers
# ---------------------------------------------------------------------------


def _check_diff_present(diff_path: Path) -> GateResult:
    exists = diff_path.exists() and diff_path.stat().st_size > 0
    return GateResult(
        gate=EvidenceGate.DIFF_PRESENT,
        passed=exists,
        detail="" if exists else "diff.patch is missing or empty",
    )


def _check_completion_report_present(completion_path: Path) -> GateResult:
    exists = completion_path.exists()
    return GateResult(
        gate=EvidenceGate.COMPLETION_REPORT_PRESENT,
        passed=exists,
        detail="" if exists else "completion_report.md is missing",
    )


def _check_changed_files_present(changed_files: list[str]) -> GateResult:
    has_files = bool(changed_files)
    return GateResult(
        gate=EvidenceGate.CHANGED_FILES_PRESENT,
        passed=has_files,
        detail="" if has_files else "no changed files in evidence",
    )


def _check_verification_cited(
    completion_path: Path, evidence_yaml_path: Path
) -> GateResult:
    """Check that verification steps are mentioned in completion report or evidence."""
    if completion_path.exists():
        text = completion_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(
            r"(verify|verification|test|pytest|make\s+test|check|pass|assert)",
            text,
            re.IGNORECASE,
        ):
            return GateResult(
                gate=EvidenceGate.VERIFICATION_CITED,
                passed=True,
            )

    if evidence_yaml_path.exists():
        text = evidence_yaml_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(
            r"(verify|verification|test|pytest|pass)",
            text,
            re.IGNORECASE,
        ):
            return GateResult(
                gate=EvidenceGate.VERIFICATION_CITED,
                passed=True,
            )

    return GateResult(
        gate=EvidenceGate.VERIFICATION_CITED,
        passed=False,
        detail="no verification steps cited in completion report or evidence",
    )


def _check_reference_access_absent(
    evidence_root: Path, completion_path: Path
) -> GateResult:
    """Detect if the run accessed _reference/ or oracle material."""
    ref_dir = evidence_root / "_reference"
    if ref_dir.exists():
        return GateResult(
            gate=EvidenceGate.REFERENCE_ACCESS_ABSENT,
            passed=False,
            detail="_reference/ directory exists in run",
        )

    if completion_path.exists():
        text = completion_path.read_text(encoding="utf-8", errors="ignore")
        if "_reference" in text or "oracle" in text.lower():
            return GateResult(
                gate=EvidenceGate.REFERENCE_ACCESS_ABSENT,
                passed=False,
                detail="reference to _reference/ or oracle found in completion report",
            )

    return GateResult(
        gate=EvidenceGate.REFERENCE_ACCESS_ABSENT,
        passed=True,
    )


def _check_runner_identity_hidden(
    evidence_root: Path,
    evidence_yaml_path: Path,
    *,
    run_id: str | None = None,
    runner_label: str | None = None,
) -> GateResult:
    """Verify the scoring evidence does not expose runner identity."""
    if runner_label and runner_label.lower() in ("", "unknown", "anonymous"):
        return GateResult(
            gate=EvidenceGate.RUNNER_IDENTITY_HIDDEN,
            passed=True,
        )

    evidence_dir = evidence_root / "evidence"
    for path in (evidence_dir / "evidence.yaml", evidence_root / "run.yaml"):
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if runner_label and runner_label in text:
                return GateResult(
                    gate=EvidenceGate.RUNNER_IDENTITY_HIDDEN,
                    passed=False,
                    detail=f"runner label '{runner_label}' found in {path.name}",
                )

    if completion_path := evidence_root / "completion_report.md":
        if completion_path.exists():
            text = completion_path.read_text(encoding="utf-8", errors="ignore")
            if runner_label and runner_label in text:
                return GateResult(
                    gate=EvidenceGate.RUNNER_IDENTITY_HIDDEN,
                    passed=False,
                    detail=f"runner label '{runner_label}' found in completion_report.md",
                )

    return GateResult(
        gate=EvidenceGate.RUNNER_IDENTITY_HIDDEN,
        passed=True,
    )


def _check_oracle_leakage(
    evaluation: GateEvaluation,
    evidence_root: Path,
    completion_path: Path,
) -> None:
    """Mark score invalid if reference/oracle material was accessed."""
    reasons: list[str] = []

    ref_dir = evidence_root / "_reference"
    if ref_dir.exists():
        reasons.append("_reference/ directory found in run evidence")

    if completion_path.exists():
        text = completion_path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\b(oracle|_reference)\b", text):
            reasons.append(
                "reference to oracle or _reference found in completion report"
            )

    for reason in reasons:
        evaluation.add_invalid(reason)


# ---------------------------------------------------------------------------
# Case provenance and source conflict detection
# ---------------------------------------------------------------------------


@dataclass
class SourceConflict:
    """A single detected conflict between evidence sources."""

    conflict_type: str
    description: str
    severity: str = "low"


@dataclass
class CaseProvenance:
    """Provenance metadata for a case — scorer-facing trust and risk info.

    This is downstream contract data, NOT execution-agent oracle evidence.
    It tells scorers what sources built the case, how reliable they are,
    and what conflicts were found during multi-source merge.
    """

    source_types: tuple[str, ...] = ()
    primary_source_type: str = ""
    base_commit_source: str = ""
    base_commit_confidence: str = "low"
    confidence: str = "medium"
    candidate_score: float = 0.0
    is_synthetic: bool = False
    conflicts: tuple[SourceConflict, ...] = ()
    risk_level: str = "low"
    risk_reasons: tuple[str, ...] = ()

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence == "low"


def build_case_provenance(
    *,
    source_type: str,
    base_source: str,
    base_confidence: str,
    candidate_score: float = 0.0,
    is_synthetic: bool = False,
    evidence_sources: tuple[str, ...] = (),
    candidate_source_type: str = "",
    merged_confidence: str = "medium",
    merged_risk_level: str = "low",
    merged_risk_reasons: tuple[str, ...] = (),
    merged_source_agents: tuple[str, ...] = (),
) -> CaseProvenance:
    """Build a CaseProvenance from case metadata and merged candidate data.

    Derives source_types from evidence_sources labels and candidate metadata.
    Detects conflicts between plan/git/history sources.
    """
    source_types = _derive_source_types(
        evidence_sources, source_type, candidate_source_type
    )
    conflicts = detect_source_conflicts(
        source_type=source_type,
        base_source=base_source,
        base_confidence=base_confidence,
        candidate_source_type=candidate_source_type,
        evidence_sources=evidence_sources,
        merged_confidence=merged_confidence,
        merged_risk_reasons=merged_risk_reasons,
    )
    risk_level, risk_reasons = _compute_provenance_risk(
        merged_risk_level=merged_risk_level,
        merged_risk_reasons=merged_risk_reasons,
        base_confidence=base_confidence,
        conflicts=conflicts,
        is_synthetic=is_synthetic,
    )

    return CaseProvenance(
        source_types=source_types,
        primary_source_type=candidate_source_type or source_type,
        base_commit_source=base_source,
        base_commit_confidence=base_confidence,
        confidence=merged_confidence,
        candidate_score=candidate_score,
        is_synthetic=is_synthetic,
        conflicts=tuple(conflicts),
        risk_level=risk_level,
        risk_reasons=tuple(risk_reasons),
    )


def detect_source_conflicts(
    *,
    source_type: str = "",
    base_source: str = "",
    base_confidence: str = "",
    candidate_source_type: str = "",
    evidence_sources: tuple[str, ...] = (),
    merged_confidence: str = "medium",
    merged_risk_reasons: tuple[str, ...] = (),
) -> list[SourceConflict]:
    """Detect conflicts between evidence sources used to build a case.

    Checks for:
    - Plan/git disagreement: plan doc exists but git history points elsewhere
    - Low-confidence history evidence
    - Missing/weak base commit source
    - Single-source cases (no corroboration)
    - Low merged confidence
    """
    conflicts: list[SourceConflict] = []

    # Check for weak base commit
    if base_confidence == "low" and base_source:
        conflicts.append(SourceConflict(
            conflict_type="weak_base_commit",
            description=f"Base commit source '{base_source}' has low confidence",
            severity="medium",
        ))
    if not base_source and source_type != "manual":
        conflicts.append(SourceConflict(
            conflict_type="missing_base_source",
            description="No base commit source recorded",
            severity="high",
        ))

    # Check for status-only / dependency-only history signals
    for reason in merged_risk_reasons:
        if "status-only" in reason:
            conflicts.append(SourceConflict(
                conflict_type="status_only_plan",
                description="Plan doc is status/tracking only, may lack actionable tasks",
                severity="medium",
            ))
        if "path-only" in reason:
            conflicts.append(SourceConflict(
                conflict_type="weak_history_signal",
                description="History evidence is path-only mention with no task detail",
                severity="medium",
            ))
        if "low-value" in reason:
            conflicts.append(SourceConflict(
                conflict_type="low_value_conversation",
                description="History conversation is about dependency/lockfile maintenance",
                severity="low",
            ))

    # Check if plan and history disagree on task signal
    source_labels = " ".join(evidence_sources).lower()
    has_plan = source_type in ("orchestration_kit", "handoff_plan") or "plan" in source_labels
    has_history = any(t in source_labels for t in ("codex_history", "claude_history", "git_history"))

    if has_plan and has_history and merged_confidence == "low":
        conflicts.append(SourceConflict(
            conflict_type="plan_history_disagreement",
            description="Plan and history sources disagree on task signal (low merged confidence)",
            severity="high",
        ))

    # Single-source cases are risky
    if not has_plan and has_history and source_type == "git_history":
        conflicts.append(SourceConflict(
            conflict_type="history_only",
            description="Case built from git history only — no plan or manual guidance",
            severity="medium",
        ))

    # Low merged confidence is always a conflict
    if merged_confidence == "low":
        conflicts.append(SourceConflict(
            conflict_type="low_confidence",
            description="Low merged confidence across all evidence sources",
            severity="high",
        ))

    return conflicts


def _derive_source_types(
    evidence_sources: tuple[str, ...],
    source_type: str,
    candidate_source_type: str,
) -> tuple[str, ...]:
    """Derive the set of source types that contributed to this case."""
    types: set[str] = set()

    # From evidence source labels
    for label in evidence_sources:
        label_lower = label.lower()
        if "plan:" in label_lower and source_type not in ("git_history", "no_git", "empty_history"):
            types.add("plan")
        if "codex_history:" in label_lower:
            types.add("codex_history")
        if "claude_history:" in label_lower:
            types.add("claude_history")

    # From case metadata
    if source_type in ("orchestration_kit", "handoff_plan"):
        types.add("plan")
    if source_type == "git_history":
        types.add("git_history")
    if source_type == "manual":
        types.add("manual")

    if candidate_source_type:
        types.add(candidate_source_type)

    if not types:
        types.add(source_type or "unknown")

    return tuple(sorted(types))


def _compute_provenance_risk(
    *,
    merged_risk_level: str,
    merged_risk_reasons: tuple[str, ...],
    base_confidence: str,
    conflicts: list[SourceConflict],
    is_synthetic: bool,
) -> tuple[str, tuple[str, ...]]:
    """Compute overall risk for the provenance section."""
    reasons: list[str] = list(merged_risk_reasons)

    if is_synthetic:
        reasons.append("synthetic case reconstructed from git history")
    if base_confidence == "low":
        reasons.append("low base commit confidence")

    high_severity = [c for c in conflicts if c.severity == "high"]
    if high_severity:
        return "high", tuple(reasons)
    if conflicts or merged_risk_level == "high":
        return "medium", tuple(reasons)
    if merged_risk_level == "medium" or reasons:
        return "medium", tuple(reasons)
    return "low", ()
