"""Summarize prior task outcomes for future package compilation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .evaluation import TaskOutcomeEntry, read_task_outcomes
from .yaml_lite import parse_simple_yaml


@dataclass
class CandidateSelectionFeedback:
    """Feedback that can adjust future candidate ranking."""

    candidate_id_penalties: dict[str, float] = field(default_factory=dict)
    candidate_id_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_path_penalties: dict[str, float] = field(default_factory=dict)
    source_path_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    source_type_penalties: dict[str, float] = field(default_factory=dict)
    source_type_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def penalty_for(
        self,
        *,
        candidate_id: str,
        source_type: str,
        source_path: str,
    ) -> tuple[float, tuple[str, ...]]:
        penalty = 0.0
        reasons: list[str] = []
        for key, penalties, reason_map in (
            (candidate_id, self.candidate_id_penalties, self.candidate_id_reasons),
            (source_path, self.source_path_penalties, self.source_path_reasons),
            (source_type, self.source_type_penalties, self.source_type_reasons),
        ):
            value = penalties.get(key, 0.0)
            if value:
                penalty += value
                reasons.extend(reason_map.get(key, ()))
        return min(penalty, 0.4), tuple(dict.fromkeys(reasons))

    @property
    def has_signal(self) -> bool:
        return any(
            (
                self.candidate_id_penalties,
                self.source_path_penalties,
                self.source_type_penalties,
            )
        )


def build_candidate_selection_feedback(
    case_roots: tuple[Path, ...] | list[Path],
) -> CandidateSelectionFeedback:
    """Build candidate ranking feedback from prior case outcome records."""

    feedback = CandidateSelectionFeedback()
    for case_root in case_roots:
        case_yaml = case_root / "case.yaml"
        if not case_yaml.is_file():
            continue
        metadata = parse_simple_yaml(case_yaml)
        outcomes = read_task_outcomes(case_root)
        penalty, reasons = _candidate_penalty_from_outcomes(outcomes)
        if not penalty:
            continue

        candidate_id = str(metadata.get("selected_candidate_id", "")).strip()
        source_path = str(metadata.get("source_path", "")).strip()
        source_type = str(
            metadata.get("candidate_source_type", "")
            or metadata.get("source_type", "")
        ).strip()

        if candidate_id:
            _add_feedback(
                feedback.candidate_id_penalties,
                feedback.candidate_id_reasons,
                candidate_id,
                penalty,
                reasons,
            )
        if source_path:
            _add_feedback(
                feedback.source_path_penalties,
                feedback.source_path_reasons,
                source_path,
                penalty,
                reasons,
            )
        if not candidate_id and not source_path and source_type:
            _add_feedback(
                feedback.source_type_penalties,
                feedback.source_type_reasons,
                source_type,
                penalty,
                reasons,
            )
    return feedback


def _candidate_penalty_from_outcomes(
    outcomes: tuple[TaskOutcomeEntry, ...] | list[TaskOutcomeEntry],
) -> tuple[float, tuple[str, ...]]:
    penalty = 0.0
    reasons: list[str] = []
    for outcome in outcomes:
        if outcome.false_positive:
            penalty += 0.12
            reasons.append("prior outcome feedback: false-positive completion")
        if outcome.outcome == "blocked" or outcome.blocked_reason:
            penalty += 0.06
            reasons.append("prior outcome feedback: blocked execution")
        if outcome.verification_status.lower() in {"weak", "missing", "failed"}:
            penalty += 0.03
            reasons.append("prior outcome feedback: weak verification")
        if outcome.acceptance_gaps:
            penalty += 0.02
            reasons.append("prior outcome feedback: acceptance gaps")
    return min(penalty, 0.30), tuple(dict.fromkeys(reasons))


def _add_feedback(
    penalties: dict[str, float],
    reasons_by_key: dict[str, tuple[str, ...]],
    key: str,
    penalty: float,
    reasons: tuple[str, ...],
) -> None:
    penalties[key] = min(penalties.get(key, 0.0) + penalty, 0.40)
    reasons_by_key[key] = tuple(dict.fromkeys(reasons_by_key.get(key, ()) + reasons))


@dataclass(frozen=True)
class TaskCompilerFeedback:
    """Compact signal derived from historical task package outcomes."""

    total_outcomes: int = 0
    blocked_count: int = 0
    false_positive_count: int = 0
    weak_verification_count: int = 0
    top_acceptance_gaps: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()

    @property
    def has_signal(self) -> bool:
        return any(
            (
                self.blocked_count,
                self.false_positive_count,
                self.weak_verification_count,
                self.top_acceptance_gaps,
            )
        )


def summarize_task_outcomes(
    outcomes: tuple[TaskOutcomeEntry, ...] | list[TaskOutcomeEntry],
) -> TaskCompilerFeedback:
    """Reduce task outcome history into package compiler guidance."""

    gap_counts: Counter[str] = Counter()
    blocked_reasons: list[str] = []
    blocked_count = 0
    false_positive_count = 0
    weak_verification_count = 0

    for outcome in outcomes:
        if outcome.outcome == "blocked" or outcome.blocked_reason:
            blocked_count += 1
        if outcome.blocked_reason and outcome.blocked_reason not in blocked_reasons:
            blocked_reasons.append(outcome.blocked_reason)
        if outcome.false_positive:
            false_positive_count += 1
        if outcome.verification_status.lower() in {"weak", "missing", "failed"}:
            weak_verification_count += 1
        for gap in outcome.acceptance_gaps:
            cleaned = " ".join(str(gap).split())
            if cleaned:
                gap_counts[cleaned] += 1

    return TaskCompilerFeedback(
        total_outcomes=len(outcomes),
        blocked_count=blocked_count,
        false_positive_count=false_positive_count,
        weak_verification_count=weak_verification_count,
        top_acceptance_gaps=tuple(gap for gap, _ in gap_counts.most_common(3)),
        blocked_reasons=tuple(blocked_reasons[:3]),
    )


def feedback_acceptance_criteria(feedback: TaskCompilerFeedback) -> tuple[str, ...]:
    """Translate feedback into observable acceptance hardening criteria."""

    if not feedback.has_signal:
        return ()

    criteria: list[str] = []
    for gap in feedback.top_acceptance_gaps:
        criteria.append(f"prior outcome feedback is addressed: {gap}.")
    if feedback.weak_verification_count:
        criteria.append(
            "Verification evidence cites command output that proves task-specific completion, not only generic test success."
        )
    if feedback.blocked_reasons:
        joined = "; ".join(feedback.blocked_reasons)
        criteria.append(f"Known blocker patterns are resolved or explicitly ruled out: {joined}.")
    if feedback.false_positive_count:
        criteria.append(
            "Completion evidence rules out prior false-positive patterns before claiming success."
        )
    return tuple(criteria)
