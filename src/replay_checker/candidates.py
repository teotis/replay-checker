"""Case candidate ranking built from scored evidence records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .case_depth import build_episode_profile, episode_bonus_for_record
from .evidence_graph import build_evidence_graph
from .outcome_feedback import CandidateSelectionFeedback
from .sources import EvidenceRecord, score_evidence_records


@dataclass(frozen=True)
class CandidateCase:
    candidate_id: str
    source_type: str
    primary_source: str
    supporting_sources: tuple[str, ...] = ()
    base_commit: str | None = None
    base_source: str = ""
    base_confidence: str = "low"
    relevance_score: float = 0.0
    selection_reasons: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()


def build_case_candidates(
    records: list[EvidenceRecord],
    *,
    project_path: str | Path | None = None,
    candidate_feedback: CandidateSelectionFeedback | None = None,
) -> list[CandidateCase]:
    """Turn scored evidence records into ranked case candidates.

    Returns candidates ordered by relevance_score descending.
    """
    scored = records if _records_are_scored(records) else score_evidence_records(records, project_path=project_path)
    graph = build_evidence_graph(scored)
    episode = build_episode_profile(scored)
    plan_records = [r for r in scored if r.source_type == "plan"]
    history_records = [r for r in scored if r.source_type in ("codex_history", "claude_history")]

    candidates: list[CandidateCase] = []

    # Each plan doc becomes its own candidate
    for idx, rec in enumerate(plan_records):
        cid = _candidate_id("plan", rec.path)
        reasons = list(rec.relevance_reasons)
        graph_bonus, graph_reasons = graph.bonus_for(rec)
        reasons.extend(graph_reasons)
        episode_bonus, episode_reasons = episode_bonus_for_record(rec, episode)
        reasons.extend(episode_reasons)
        risks: list[str] = []
        primary_source = rec.metadata.get("relative_path", str(rec.path))
        feedback_penalty, feedback_reasons = _feedback_adjustment(
            candidate_feedback,
            candidate_id=cid,
            source_type=rec.source_type,
            source_path=primary_source,
        )
        reasons.extend(feedback_reasons)
        if feedback_penalty:
            risks.append("prior outcome feedback lowered this candidate")
        if rec.task_signal == "plan_status_only":
            risks.append("status-only plan doc may lack actionable tasks")
        if not rec.path.exists():
            risks.append("source file missing on disk")
        base_commit = rec.metadata.get("base_commit")
        base_source = "plan_metadata" if base_commit else ""
        candidates.append(
            CandidateCase(
                candidate_id=cid,
                source_type="plan",
                primary_source=primary_source,
                base_commit=base_commit,
                base_source=base_source,
                base_confidence="high" if base_commit else "low",
                relevance_score=_adjusted_score(
                    rec.relevance_score,
                    graph_bonus + episode_bonus,
                    feedback_penalty,
                ),
                selection_reasons=tuple(reasons),
                risks=tuple(risks),
            )
        )

    # Group history records by source_type into one candidate each
    by_type: dict[str, list[EvidenceRecord]] = {}
    for rec in history_records:
        by_type.setdefault(rec.source_type, []).append(rec)

    for source_type, recs in by_type.items():
        best = max(recs, key=lambda r: r.relevance_score)
        supporting = tuple(
            r.metadata.get("relative_path", str(r.path))
            for r in recs
            if r is not best
        )
        cid = _candidate_id(source_type, best.path)
        reasons = list(best.relevance_reasons)
        graph_bonus, graph_reasons = graph.bonus_for(best)
        reasons.extend(graph_reasons)
        episode_bonus, episode_reasons = episode_bonus_for_record(best, episode)
        reasons.extend(episode_reasons)
        risks: list[str] = []
        primary_source = best.metadata.get("relative_path", str(best.path))
        feedback_penalty, feedback_reasons = _feedback_adjustment(
            candidate_feedback,
            candidate_id=cid,
            source_type=best.source_type,
            source_path=primary_source,
        )
        reasons.extend(feedback_reasons)
        if feedback_penalty:
            risks.append("prior outcome feedback lowered this candidate")
        if best.task_signal == "conversation_mention":
            risks.append("conversation only mentions project path, no task detail")
        if best.task_signal == "conversation_low_value":
            risks.append("conversation is about dependency/lockfile changes")
        alias_type = best.metadata.get("matched_alias_type", "")
        if alias_type in ("basename", "normalized_name"):
            risks.append(f"matched project only by {alias_type}, may be a different project")
        candidates.append(
            CandidateCase(
                candidate_id=cid,
                source_type=source_type,
                primary_source=primary_source,
                supporting_sources=supporting,
                base_confidence="low",
                relevance_score=_adjusted_score(
                    best.relevance_score,
                    graph_bonus + episode_bonus,
                    feedback_penalty,
                ),
                selection_reasons=tuple(reasons),
                risks=tuple(risks),
            )
        )

    candidates.sort(key=lambda c: c.relevance_score, reverse=True)
    return candidates


def _records_are_scored(records: list[EvidenceRecord]) -> bool:
    if not records:
        return False
    return all(
        record.relevance_score != 0.0
        or bool(record.relevance_reasons)
        or bool(record.task_signal)
        for record in records
    )


def select_case_candidate(
    candidates: list[CandidateCase],
) -> CandidateCase | None:
    """Return the highest-scored candidate, or None if empty."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.relevance_score)


def _feedback_adjustment(
    feedback: CandidateSelectionFeedback | None,
    *,
    candidate_id: str,
    source_type: str,
    source_path: str,
) -> tuple[float, tuple[str, ...]]:
    if feedback is None or not feedback.has_signal:
        return 0.0, ()
    return feedback.penalty_for(
        candidate_id=candidate_id,
        source_type=source_type,
        source_path=source_path,
    )


def _adjusted_score(base: float, bonus: float, penalty: float) -> float:
    return round(max(min(base + bonus, 1.0) - penalty, 0.0), 3)


def _candidate_id(source_type: str, path: Path) -> str:
    stem = path.stem.replace(" ", "-")
    return f"{source_type}-{stem}"
