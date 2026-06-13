"""Situation-depth projection for replay case extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .sources import EvidenceRecord

if TYPE_CHECKING:
    from .candidates import CandidateCase


_HISTORY_TYPES = {"codex_history", "claude_history"}
_SOURCE_ORDER = ("plan", "codex_history", "claude_history", "git_history")
_FAILURE_MARKERS = (
    "blocked", "weak verification", "false positive", "false-positive",
    "risk", "missing", "failed", "gap",
)


@dataclass(frozen=True)
class EpisodeProfile:
    """A compact historical episode mined from cross-source evidence."""

    label: str = "single-source"
    anchor_paths: tuple[str, ...] = ()
    source_types: tuple[str, ...] = ()
    ranking_bonus: float = 0.0
    ranking_reason: str = ""

    @property
    def has_cross_source_support(self) -> bool:
        return bool(self.anchor_paths) and len(self.source_types) >= 2


@dataclass(frozen=True)
class SituationProfile:
    """A bounded profile describing why a replay case is deep or shallow."""

    problem_context: str = ""
    constraints: str = ""
    failure_boundaries: str = ""
    observable_acceptance: str = ""
    episode_label: str = "single-source"
    anchor_paths: tuple[str, ...] = ()
    depth_score: int = 0
    depth_level: str = "shallow"
    depth_reasons: tuple[str, ...] = ()


def build_episode_profile(records: list[EvidenceRecord] | tuple[EvidenceRecord, ...]) -> EpisodeProfile:
    """Mine a small episode from shared path mentions across evidence records."""

    path_sources: dict[str, set[str]] = {}
    for record in records:
        for path in _record_paths(record):
            path_sources.setdefault(path, set()).add(record.source_type)

    anchors = tuple(
        path
        for path, sources in sorted(path_sources.items())
        if len(sources) >= 2
    )
    if not anchors:
        source_types = _ordered_source_types({record.source_type for record in records})
        label = "unlinked-multi-source" if len(source_types) >= 2 else "single-source"
        return EpisodeProfile(label=label, source_types=source_types)

    first_sources = path_sources[anchors[0]]
    source_types = _ordered_source_types(first_sources)
    label = _episode_label(source_types)
    shown_sources = " + ".join(source_types)
    shown_paths = ", ".join(anchors[:3])
    return EpisodeProfile(
        label=label,
        anchor_paths=anchors,
        source_types=source_types,
        ranking_bonus=0.08,
        ranking_reason=f"episode support: {shown_sources} on {shown_paths}",
    )


def episode_bonus_for_record(record: EvidenceRecord, episode: EpisodeProfile) -> tuple[float, tuple[str, ...]]:
    """Return an explainable ranking bonus when a record belongs to an episode."""

    if not episode.has_cross_source_support:
        return 0.0, ()
    record_paths = set(_record_paths(record))
    if not record_paths.intersection(episode.anchor_paths):
        return 0.0, ()
    return episode.ranking_bonus, (episode.ranking_reason,)


def build_situation_profile(
    records: list[EvidenceRecord] | tuple[EvidenceRecord, ...],
    *,
    selected_candidate: "CandidateCase | None",
    reconstruction_context: dict[str, object] | None = None,
) -> SituationProfile:
    """Compile bounded evidence into a situation profile for a case."""

    ctx = reconstruction_context or {}
    episode = build_episode_profile(records)
    problem_parts: list[str] = []
    constraints: list[str] = []
    failures: list[str] = []
    acceptance: list[str] = []
    reasons: list[str] = []
    score = 0

    commit_subject = str(ctx.get("commit_subject", "") or "").strip()
    if commit_subject:
        problem_parts.append(commit_subject)
        score += 20
        reasons.append("commit subject")

    plan_goal = str(ctx.get("plan_goal", "") or "").strip()
    if plan_goal:
        problem_parts.append(plan_goal)
        score += 20
        reasons.append("plan goal")

    if selected_candidate is not None:
        problem_parts.append(f"Selected candidate `{selected_candidate.candidate_id}` from {selected_candidate.source_type}.")
        if selected_candidate.selection_reasons:
            constraints.append("Selection reasons: " + "; ".join(selected_candidate.selection_reasons[:3]))
        score += 10
        reasons.append("selected candidate")

    for record in records[:3]:
        if record.summary:
            problem_parts.append(record.summary)

    source_types = {record.source_type for record in records}
    if len(source_types) >= 2:
        score += 15
        reasons.append("multiple evidence source types")
        constraints.append("Evidence source types: " + ", ".join(_ordered_source_types(source_types)))

    mentioned_paths = _mentioned_paths(records)
    if mentioned_paths:
        score += 10
        reasons.append("mentioned project paths")
        constraints.append("Mentioned paths: " + ", ".join(mentioned_paths[:5]))

    scope_summary = str(ctx.get("scope_summary", "") or "").strip()
    if scope_summary and scope_summary != "unknown scope":
        score += 10
        reasons.append("scope summary")
        constraints.append(scope_summary)

    verification_hints = _list_of_str(ctx.get("verification_hints", []))
    if verification_hints:
        score += 20
        reasons.append("verification hints")
        acceptance.extend(verification_hints[:4])

    verification_commands = _list_of_str(ctx.get("verification_commands", []))
    if verification_commands:
        score += 20
        reasons.append("verification commands")
        acceptance.extend(f"Run `{command}`" for command in verification_commands[:4])

    test_paths = [path for path in mentioned_paths if _looks_like_test_path(path)]
    if test_paths:
        score += 5
        reasons.append("test path mentions")
        acceptance.append("Test-related paths mentioned: " + ", ".join(test_paths[:3]))

    for record in records[:5]:
        lowered = record.summary.lower()
        if any(marker in lowered for marker in ("observable acceptance", "verification", "test")):
            acceptance.append(record.summary)

    if acceptance and "acceptance text signals" not in reasons:
        score += 10
        reasons.append("acceptance text signals")

    risk_notes = _list_of_str(ctx.get("risk_notes", []))
    if risk_notes:
        score += 10
        reasons.append("reconstruction risk boundaries")
        failures.extend(risk_notes[:4])

    for record in records[:5]:
        lowered = record.summary.lower()
        if any(marker in lowered for marker in _FAILURE_MARKERS):
            failures.append(record.summary)

    if failures:
        score += 5
        reasons.append("failure boundary signals")

    if episode.has_cross_source_support:
        score += 10
        reasons.append("episode cross-source support")

    depth_score = max(0, min(score, 100))
    return SituationProfile(
        problem_context=_join_unique(problem_parts, limit=600),
        constraints=_join_unique(constraints, limit=500),
        failure_boundaries=_join_unique(failures, limit=500),
        observable_acceptance=_join_unique(acceptance, limit=500),
        episode_label=episode.label,
        anchor_paths=episode.anchor_paths,
        depth_score=depth_score,
        depth_level=_depth_level(depth_score),
        depth_reasons=tuple(reasons),
    )


def situation_profile_to_case_yaml(profile: SituationProfile) -> dict[str, object]:
    """Project profile metadata into case.yaml-safe scalar/list values."""

    return {
        "depth_score": str(profile.depth_score),
        "depth_level": profile.depth_level,
        "episode_label": profile.episode_label,
        "episode_anchor_paths": list(profile.anchor_paths),
        "depth_reasons": list(profile.depth_reasons),
    }


def _episode_label(source_types: tuple[str, ...]) -> str:
    if "plan" in source_types and any(src in _HISTORY_TYPES for src in source_types):
        return "plan-history"
    if source_types and all(src in _HISTORY_TYPES for src in source_types):
        return "history-only"
    if "git_history" in source_types:
        return "git-reconstruction"
    return "single-source"


def _ordered_source_types(source_types: set[str]) -> tuple[str, ...]:
    ordered = [src for src in _SOURCE_ORDER if src in source_types]
    ordered.extend(sorted(source_types.difference(ordered)))
    return tuple(ordered)


def _record_paths(record: EvidenceRecord) -> tuple[str, ...]:
    paths: list[str] = []
    raw = record.metadata.get("mentioned_files", "")
    paths.extend(part.strip() for part in raw.split(",") if part.strip())
    paths.extend(_extract_path_mentions(record.summary))
    normalized = (_normalize_path(path) for path in paths if path)
    return tuple(dict.fromkeys(path for path in normalized if _looks_like_project_path(path)))


def _mentioned_paths(records: list[EvidenceRecord] | tuple[EvidenceRecord, ...]) -> list[str]:
    paths: list[str] = []
    for record in records:
        paths.extend(_record_paths(record))
    return list(dict.fromkeys(paths))


def _extract_path_mentions(text: str) -> list[str]:
    matches = re.findall(r"(?:^|\s|`)([\w./-]+\.\w{1,10})(?:\s|$|`|[.,:;])", text)
    return [match.strip("./") for match in matches]


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def _looks_like_test_path(path: str) -> bool:
    p = Path(path)
    lowered = path.lower()
    return (
        "test" in p.parts
        or "tests" in p.parts
        or p.name.startswith("test_")
        or "_test." in lowered
        or ".test." in lowered
        or ".spec." in lowered
    )


def _looks_like_project_path(path: str) -> bool:
    if "/" in path:
        return True
    suffix = Path(path).suffix.lstrip(".")
    return bool(suffix) and any(char.isalpha() for char in suffix)


def _list_of_str(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _join_unique(parts: list[str], *, limit: int) -> str:
    cleaned = [" ".join(part.split()) for part in parts if " ".join(part.split())]
    text = " | ".join(dict.fromkeys(cleaned))
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _depth_level(score: int) -> str:
    if score >= 70:
        return "rich"
    if score >= 45:
        return "solid"
    return "shallow"
