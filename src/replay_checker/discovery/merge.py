"""Merge, confidence, and risk logic for discovery outputs."""

from __future__ import annotations

from pathlib import Path

from ..candidates import CandidateCase, build_case_candidates, select_case_candidate
from ..sources import (
    EvidenceRecord,
    EvidenceSourceConfig,
    discover_evidence_sources,
    score_evidence_records,
)
from .models import DiscoveryOutput, MergedCandidate


def merge_discovery_outputs(
    outputs: list[DiscoveryOutput],
    *,
    project_path: str | Path | None = None,
) -> list[MergedCandidate]:
    """Merge multiple discovery outputs into a single ranked candidate list.

    Combines evidence records from all sources, deduplicates candidates,
    and assigns confidence/risk metadata to each.
    """
    all_records: list[EvidenceRecord] = []
    # Map evidence record path → contributing agents
    record_agent_map: dict[str, str] = {}

    for output in outputs:
        for record in output.records:
            key = record.path.as_posix()
            if key not in record_agent_map:
                record_agent_map[key] = output.source_agent
            all_records.append(record)

    scored = score_evidence_records(all_records, project_path=project_path) if all_records else []
    candidates = build_case_candidates(scored)
    deduped = dedupe_candidates(candidates)

    # Build a map from primary_source → contributing agents for candidates
    source_agents: dict[str, tuple[str, ...]] = {}
    for c in candidates:
        key = c.primary_source
        source_agents.setdefault(key, ())

    # Enrich source_agents from the record_agent_map
    agent_set_by_source: dict[str, set[str]] = {}
    for rec in all_records:
        source_key = _record_primary_source(rec)
        agent_set_by_source.setdefault(source_key, set()).add(
            record_agent_map.get(rec.path.as_posix(), rec.source_type)
        )
    source_agents = {k: tuple(v) for k, v in agent_set_by_source.items()}

    enriched = assign_confidence_and_risk(deduped, source_agents)
    enriched.sort(key=lambda c: c.candidate.relevance_score, reverse=True)
    return enriched


def _record_primary_source(record: EvidenceRecord) -> str:
    return record.metadata.get("relative_path", str(record.path))


def dedupe_candidates(candidates: list[CandidateCase]) -> list[CandidateCase]:
    """Remove duplicate candidates, keeping the highest-scored version.

    Two plan candidates are duplicates if they reference the same plan file.
    History candidates are deduplicated by source_type (already grouped by
    build_case_candidates).
    """
    seen_plan: dict[str, CandidateCase] = {}
    history_types: dict[str, CandidateCase] = {}
    result: list[CandidateCase] = []

    for c in candidates:
        if c.source_type == "plan":
            key = c.primary_source
            if key not in seen_plan or c.relevance_score > seen_plan[key].relevance_score:
                seen_plan[key] = c
        else:
            # History candidates are already grouped by source_type
            if c.source_type not in history_types:
                history_types[c.source_type] = c

    result.extend(seen_plan.values())
    result.extend(history_types.values())
    return result


def assign_confidence_and_risk(
    candidates: list[CandidateCase],
    agent_map: dict[str, list[str]] | None = None,
) -> list[MergedCandidate]:
    """Compute confidence and risk level for each candidate.

    Confidence is based on:
    - Source type (plan > code_analysis > history)
    - Base commit availability
    - Number of contributing discovery agents
    - Task signal quality

    Risk is based on:
    - Confidence level
    - Presence of risk markers from the candidate
    - Supporting source count
    - Cross-source conflicts with other candidates
    """
    result: list[MergedCandidate] = []
    for c in candidates:
        confidence = _compute_confidence(c, agent_map)
        risk_level, risk_reasons = _compute_risk(c, confidence, agent_map)
        contributing = tuple(
            _agents_for_candidate(c, agent_map)
        )
        result.append(MergedCandidate(
            candidate=c,
            contributing_agents=contributing,
            confidence=confidence,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
        ))

    # Detect cross-source conflicts across all enriched candidates
    _detect_cross_source_conflicts(result)

    return result


def compile_cases_from_discovery(
    merged_candidates: list[MergedCandidate],
    *,
    project_path: str | Path,
    selected_id: str = "",
) -> tuple[CandidateCase | None, list[CandidateCase]]:
    """Select the best candidate and prepare case compilation data.

    Returns (selected, all_candidates) where selected is the highest-confidence
    candidate to build a case from, and all_candidates is the full deduplicated
    list.

    Low-confidence candidates are not excluded — they still produce cases
    with explicit risk metadata.
    """
    if not merged_candidates:
        return None, []

    sorted_by_confidence = sorted(
        merged_candidates,
        key=lambda mc: (
            _confidence_rank(mc.confidence),
            mc.candidate.relevance_score,
        ),
        reverse=True,
    )

    selected_mc = sorted_by_confidence[0]
    all_candidates = [mc.candidate for mc in merged_candidates]
    return selected_mc.candidate, all_candidates


def compile_low_confidence_case(
    candidate: MergedCandidate,
    *,
    project_path: str | Path,
) -> dict[str, object]:
    """Build case metadata for a low-confidence candidate.

    Returns a dict with explicit risk markers that the case writer
    should include. Does not create files — the caller decides
    whether to persist the case.
    """
    return {
        "candidate_id": candidate.candidate.candidate_id,
        "source_type": candidate.candidate.source_type,
        "confidence": candidate.confidence,
        "risk_level": candidate.risk_level,
        "risk_reasons": list(candidate.risk_reasons),
        "relevance_score": candidate.candidate.relevance_score,
        "primary_source": candidate.candidate.primary_source,
        "selection_reasons": list(candidate.candidate.selection_reasons),
        "low_confidence": True,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _confidence_rank(confidence: str) -> int:
    return _CONFIDENCE_RANK.get(confidence, 0)


def _compute_confidence(
    candidate: CandidateCase,
    agent_map: dict[str, list[str]] | None,
) -> str:
    """Score-based confidence assignment."""
    score = candidate.relevance_score

    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _compute_risk(
    candidate: CandidateCase,
    confidence: str,
    agent_map: dict[str, list[str]] | None,
) -> tuple[str, tuple[str, ...]]:
    """Compute risk level and reasons for a candidate."""
    reasons: list[str] = []

    if confidence == "low":
        reasons.append("low confidence candidate")
    if candidate.risks:
        reasons.extend(candidate.risks)
    if not candidate.supporting_sources and candidate.source_type != "plan":
        reasons.append("no supporting sources")
    if candidate.base_confidence == "low":
        reasons.append("low base commit confidence")

    if not reasons:
        return "low", ()
    if confidence == "low":
        return "high", tuple(reasons)
    return "medium", tuple(reasons)


def _agents_for_candidate(
    candidate: CandidateCase,
    agent_map: dict[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    """Find which discovery agents contributed to this candidate."""
    if not agent_map:
        return (candidate.source_type,)
    key = candidate.primary_source
    agents = agent_map.get(key)
    return agents if agents else (candidate.source_type,)


def _detect_cross_source_conflicts(
    enriched: list[MergedCandidate],
) -> None:
    """Detect conflicts between candidates from different source types.

    Modifies MergedCandidate risk metadata in-place when cross-source
    disagreement is found (e.g., plan says one thing, git history says another).
    """
    if len(enriched) < 2:
        return

    plan_candidates = [m for m in enriched if m.candidate.source_type == "plan"]
    history_candidates = [m for m in enriched if m.candidate.source_type not in ("plan",)]

    # If we have both plan and history candidates but they disagree on task signal
    if plan_candidates and history_candidates:
        best_plan = max(plan_candidates, key=lambda m: m.candidate.relevance_score)
        best_history = max(history_candidates, key=lambda m: m.candidate.relevance_score)

        # Check if plan confidence is high but selected was history (disagreement)
        if best_plan.confidence == "high" and best_history.confidence != "high":
            for mc in enriched:
                new_reasons = list(mc.risk_reasons)
                new_reasons.append("plan and history sources disagree on primary task signal")
                object.__setattr__(mc, "risk_reasons", tuple(new_reasons))
                if mc.risk_level == "low":
                    object.__setattr__(mc, "risk_level", "medium")

    # Check for history-only cases with no plan corroboration
    if not plan_candidates and history_candidates:
        for mc in history_candidates:
            if (
                mc.candidate.base_confidence == "low"
                and "no plan corroboration" not in mc.risk_reasons
            ):
                new_reasons = list(mc.risk_reasons)
                new_reasons.append("no plan corroboration for history-only candidate")
                object.__setattr__(mc, "risk_reasons", tuple(new_reasons))
                if mc.risk_level == "low":
                    object.__setattr__(mc, "risk_level", "medium")
