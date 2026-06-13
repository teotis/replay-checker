"""Evaluation storage contract: data types, enums, and case-local IO helpers.

Storage layout::

    cases/<case-id>/evaluation/
      current/
        summary.yaml
        dimensions.yaml
        rankings/
        comparisons/
        recommendation.yaml
      snapshots/
        <timestamp>/
          summary.yaml
          rankings/
          comparisons/
          recommendation.yaml
      score_history.jsonl
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .yaml_lite import emit_yaml, parse_list_of_dicts as yaml_parse_list_of_dicts, parse_yaml_text


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EligibilityTier(str, Enum):
    INVALID = "invalid"
    FAILED = "failed"
    PARTIAL = "partial"
    SOLVED = "solved"
    EXCELLENT = "excellent"
    TRANSFORMATIVE = "transformative"


class ComparisonLabel(str, Enum):
    MUCH_BETTER = "much_better"
    BETTER = "better"
    TIE = "tie"
    WORSE = "worse"
    MUCH_WORSE = "much_worse"
    INCOMPARABLE = "incomparable"


class IncomparableReason(str, Enum):
    DIFFERENT_STRATEGY = "different_strategy"
    DIFFERENT_RISK_PROFILE = "different_risk_profile"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    TASK_AMBIGUITY = "task_ambiguity"
    DEPENDS_ON_USER_LENS = "depends_on_user_lens"


class StructuralContributionLevel(str, Enum):
    LOCAL_IMPROVEMENT = "local_improvement"
    CLASS_DELETION = "class_deletion"
    PROBLEM_REFRAMING = "problem_reframing"


# Maximum structural contribution points by eligibility tier.
# None means no cap — the full structural contribution applies.
STRUCTURAL_CONTRIBUTION_CAPS: dict[EligibilityTier, float | None] = {
    EligibilityTier.INVALID: 0.0,
    EligibilityTier.FAILED: 10.0,
    EligibilityTier.PARTIAL: 30.0,
    EligibilityTier.SOLVED: None,
    EligibilityTier.EXCELLENT: None,
    EligibilityTier.TRANSFORMATIVE: None,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EvaluationSummary:
    """Case-level evaluation summary stored in current/summary.yaml."""

    case_id: str
    tier: EligibilityTier = EligibilityTier.INVALID
    validity: str = "unvalidated"
    confidence: str = "low"
    total_score: float = 0.0
    run_count: int = 0
    best_run_id: str = ""


@dataclass
class DimensionDef:
    """Dimension definition with weight and description."""

    name: str
    weight: float = 0.0
    description: str = ""


@dataclass
class StructuralContribution:
    """Three sublevels of structural contribution, capped by eligibility tier."""

    local_improvement: float = 0.0
    class_deletion: float = 0.0
    problem_reframing: float = 0.0

    @property
    def total(self) -> float:
        return self.local_improvement + self.class_deletion + self.problem_reframing

    @property
    def has_any(self) -> bool:
        return self.total > 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "local_improvement": self.local_improvement,
            "class_deletion": self.class_deletion,
            "problem_reframing": self.problem_reframing,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "StructuralContribution":
        return cls(
            local_improvement=data.get("local_improvement", 0.0),
            class_deletion=data.get("class_deletion", 0.0),
            problem_reframing=data.get("problem_reframing", 0.0),
        )


@dataclass
class RunAttempt:
    """Metadata for a single run attempt stored in attempts.yaml."""

    run_id: str
    anonymous_id: str = ""
    status: str = "unknown"
    tier: EligibilityTier = EligibilityTier.INVALID
    total_score: float = 0.0
    dimension_scores: dict[str, float] = field(default_factory=dict)
    structural_contribution: StructuralContribution = field(
        default_factory=StructuralContribution
    )


@dataclass
class RankingEntry:
    """A single ranked position in a dimension ranking."""

    run_id: str
    rank: int = 0
    score: float = 0.0
    band: int = 0


@dataclass
class PairwiseComparison:
    """Pairwise comparison between two runs for a single dimension."""

    dimension: str
    run_a: str
    run_b: str
    label: ComparisonLabel = ComparisonLabel.TIE
    evidence: str = ""
    incomparable_reasons: list[IncomparableReason] = field(default_factory=list)


@dataclass
class ScoreHistoryEntry:
    """A single entry in the append-only score history log."""

    run_id: str
    timestamp: str = ""
    total_score: float = 0.0
    tier: str = ""
    dimensions: dict[str, float] = field(default_factory=dict)
    recomputation_reason: str = ""


@dataclass
class TaskOutcomeEntry:
    """Outcome replay record for a generated task package."""

    package_id: str
    task_contract_id: str = ""
    timestamp: str = ""
    outcome: str = "unknown"
    blocked_reason: str = ""
    acceptance_gaps: tuple[str, ...] = ()
    false_positive: bool = False
    landed: bool = False
    verification_status: str = "unknown"


@dataclass
class Recommendation:
    """Default user-facing recommendation stored in current/recommendation.yaml."""

    score: float = 0.0
    sentence: str = ""
    validity: str = "unvalidated"
    confidence: str = "low"


@dataclass
class ComparisonCandidate:
    """A pending pairwise comparison with its information gain.

    Candidates are sorted by gain to prioritize comparisons that most reduce
    ranking uncertainty.  Higher gain → compare this pair first.
    """

    dimension: str
    run_a: str
    run_b: str
    gain: float
    band_a: int
    band_b: int
    reason: str


# ---------------------------------------------------------------------------
# YAML emitter / parser
#
# Handles exactly the subset this module produces:
#   - flat key: value
#   - section header with nested key: value
#   - list of dict items
#   - section with list of dict items
# All values are scalars (str, int, float, bool) -- no nested containers
# beyond one level of dict-inside-list.
# ---------------------------------------------------------------------------


def _to_yaml(data: Any, indent: int = 0) -> str:
    """Serialize a Python value to a simple YAML string."""
    return emit_yaml(data, indent=indent)


def _from_yaml(text: str) -> Any:
    """Parse a simple YAML string into Python objects.

    Handles flat dicts, section headers with nested dicts, lists of dicts,
    and lists of scalars.  All leaf values are kept as strings; callers
    coerce to the expected types.
    """
    return parse_yaml_text(text)


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

EVALUATION_DIR = "evaluation"
CURRENT_DIR = "current"
SNAPSHOTS_DIR = "snapshots"


def eval_dir(case_root: str | Path) -> Path:
    """Return the evaluation directory path for a case."""
    return Path(case_root) / EVALUATION_DIR


def current_dir(case_root: str | Path) -> Path:
    """Return the current/ directory path for a case."""
    return eval_dir(case_root) / CURRENT_DIR


def snapshots_dir(case_root: str | Path) -> Path:
    """Return the snapshots/ directory path for a case."""
    return eval_dir(case_root) / SNAPSHOTS_DIR


def init_evaluation_dir(case_root: str | Path) -> Path:
    """Create the evaluation directory and subdirectories for a case.

    Creates current/ with rankings/ and comparisons/ subdirectories.
    Idempotent -- does not overwrite existing files.
    """
    cur = current_dir(case_root)
    cur.mkdir(parents=True, exist_ok=True)
    (cur / "rankings").mkdir(exist_ok=True)
    (cur / "comparisons").mkdir(exist_ok=True)
    return cur


# ---------------------------------------------------------------------------
# current/summary.yaml
# ---------------------------------------------------------------------------


def write_summary(case_root: str | Path, summary: EvaluationSummary) -> Path:
    cur = current_dir(case_root)
    cur.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "case_id": summary.case_id,
        "tier": summary.tier.value,
        "validity": summary.validity,
        "confidence": summary.confidence,
        "total_score": summary.total_score,
        "run_count": summary.run_count,
        "best_run_id": summary.best_run_id,
    }
    path = cur / "summary.yaml"
    path.write_text(_to_yaml(data) + "\n", encoding="utf-8")
    return path


def read_summary(case_root: str | Path) -> EvaluationSummary:
    cur = current_dir(case_root)
    path = cur / "summary.yaml"
    if not path.exists():
        return EvaluationSummary(case_id="")
    data = _from_yaml(path.read_text(encoding="utf-8"))
    return EvaluationSummary(
        case_id=str(data.get("case_id", "")),
        tier=EligibilityTier(str(data.get("tier", "invalid"))),
        validity=str(data.get("validity", "unvalidated")),
        confidence=str(data.get("confidence", "low")),
        total_score=float(str(data.get("total_score", "0"))),
        run_count=int(str(data.get("run_count", "0"))),
        best_run_id=str(data.get("best_run_id", "")),
    )


# ---------------------------------------------------------------------------
# current/dimensions.yaml
# ---------------------------------------------------------------------------


def write_dimensions(
    case_root: str | Path,
    dimensions: list[DimensionDef],
) -> Path:
    cur = current_dir(case_root)
    cur.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict[str, Any]] = {}
    for dim in dimensions:
        data[dim.name] = {
            "weight": dim.weight,
            "description": dim.description,
        }
    path = cur / "dimensions.yaml"
    path.write_text(_to_yaml(data) + "\n", encoding="utf-8")
    return path


def read_dimensions(case_root: str | Path) -> list[DimensionDef]:
    cur = current_dir(case_root)
    path = cur / "dimensions.yaml"
    if not path.exists():
        return []
    data = _from_yaml(path.read_text(encoding="utf-8"))
    result: list[DimensionDef] = []
    for name, fields in data.items():
        if isinstance(fields, dict):
            result.append(
                DimensionDef(
                    name=name,
                    weight=float(str(fields.get("weight", "0"))),
                    description=str(fields.get("description", "")),
                )
            )
    return result


# ---------------------------------------------------------------------------
# current/attempts.yaml
# ---------------------------------------------------------------------------


def write_attempts(
    case_root: str | Path,
    attempts: list[RunAttempt],
) -> Path:
    cur = current_dir(case_root)
    cur.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for a in attempts:
        item: dict[str, Any] = {
            "run_id": a.run_id,
            "anonymous_id": a.anonymous_id,
            "status": a.status,
            "tier": a.tier.value,
            "total_score": a.total_score,
        }
        for dim, score in sorted(a.dimension_scores.items()):
            item[f"dim_{dim}"] = score
        sc = a.structural_contribution
        item["structural_local_improvement"] = sc.local_improvement
        item["structural_class_deletion"] = sc.class_deletion
        item["structural_problem_reframing"] = sc.problem_reframing
        items.append(item)
    path = cur / "attempts.yaml"
    path.write_text(_to_yaml(items) + "\n", encoding="utf-8")
    return path


def _parse_list_of_dicts(text: str) -> list[dict[str, str]]:
    """Parse a top-level YAML list of dicts."""
    return yaml_parse_list_of_dicts(text)


def read_attempts(case_root: str | Path) -> list[RunAttempt]:
    cur = current_dir(case_root)
    path = cur / "attempts.yaml"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    items = _parse_list_of_dicts(text)
    result: list[RunAttempt] = []
    for item in items:
        dim_scores: dict[str, float] = {}
        extra: dict[str, str] = {}
        for k, v in item.items():
            if k.startswith("dim_"):
                dim_scores[k[4:]] = float(str(v))
            else:
                extra[k] = str(v)

        structural = StructuralContribution(
            local_improvement=float(extra.get("structural_local_improvement", "0")),
            class_deletion=float(extra.get("structural_class_deletion", "0")),
            problem_reframing=float(extra.get("structural_problem_reframing", "0")),
        )

        result.append(
            RunAttempt(
                run_id=extra.get("run_id", ""),
                anonymous_id=extra.get("anonymous_id", ""),
                status=extra.get("status", "unknown"),
                tier=EligibilityTier(extra.get("tier", "invalid")),
                total_score=float(extra.get("total_score", "0")),
                dimension_scores=dim_scores,
                structural_contribution=structural,
            )
        )
    return result


# ---------------------------------------------------------------------------
# current/rankings/<dimension>.yaml
# ---------------------------------------------------------------------------


def write_ranking(
    case_root: str | Path,
    dimension: str,
    entries: list[RankingEntry],
) -> Path:
    cur = current_dir(case_root)
    rankings_dir = cur / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"dimension": dimension}
    entry_dicts: list[dict[str, Any]] = []
    for e in entries:
        entry_dicts.append({
            "run_id": e.run_id,
            "rank": e.rank,
            "score": e.score,
            "band": e.band,
        })
    data["entries"] = entry_dicts
    path = rankings_dir / f"{dimension}.yaml"
    path.write_text(_to_yaml(data) + "\n", encoding="utf-8")
    return path


def read_ranking(
    case_root: str | Path,
    dimension: str,
) -> list[RankingEntry]:
    cur = current_dir(case_root)
    path = cur / "rankings" / f"{dimension}.yaml"
    if not path.exists():
        return []
    data = _from_yaml(path.read_text(encoding="utf-8"))
    entries_data = data.get("entries", [])
    if not isinstance(entries_data, list):
        return []
    result: list[RankingEntry] = []
    for item in entries_data:
        if isinstance(item, dict):
            result.append(
                RankingEntry(
                    run_id=str(item.get("run_id", "")),
                    rank=int(str(item.get("rank", "0"))),
                    score=float(str(item.get("score", "0"))),
                    band=int(str(item.get("band", "0"))),
                )
            )
    return result


# ---------------------------------------------------------------------------
# current/recommendation.yaml
# ---------------------------------------------------------------------------


def write_recommendation(
    case_root: str | Path,
    recommendation: Recommendation,
) -> Path:
    cur = current_dir(case_root)
    cur.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "score": recommendation.score,
        "sentence": recommendation.sentence,
        "validity": recommendation.validity,
        "confidence": recommendation.confidence,
    }
    path = cur / "recommendation.yaml"
    path.write_text(_to_yaml(data) + "\n", encoding="utf-8")
    return path


def read_recommendation(case_root: str | Path) -> Recommendation:
    cur = current_dir(case_root)
    path = cur / "recommendation.yaml"
    if not path.exists():
        return Recommendation()
    data = _from_yaml(path.read_text(encoding="utf-8"))
    return Recommendation(
        score=float(str(data.get("score", "0"))),
        sentence=str(data.get("sentence", "")),
        validity=str(data.get("validity", "unvalidated")),
        confidence=str(data.get("confidence", "low")),
    )


# ---------------------------------------------------------------------------
# current/comparisons/<run-a>__vs__<run-b>.yaml
# ---------------------------------------------------------------------------


def _comparison_filename(run_a: str, run_b: str) -> str:
    """Stable, order-independent filename for a pairwise comparison."""
    ids = sorted([run_a, run_b])
    return f"{ids[0]}__vs__{ids[1]}.yaml"


def write_comparison(
    case_root: str | Path,
    comparison: PairwiseComparison,
) -> Path:
    cur = current_dir(case_root)
    comps_dir = cur / "comparisons"
    comps_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "dimension": comparison.dimension,
        "run_a": comparison.run_a,
        "run_b": comparison.run_b,
        "label": comparison.label.value,
        "evidence": comparison.evidence,
    }
    if comparison.incomparable_reasons:
        data["incomparable_reasons"] = [r.value for r in comparison.incomparable_reasons]
    path = comps_dir / _comparison_filename(comparison.run_a, comparison.run_b)
    path.write_text(_to_yaml(data) + "\n", encoding="utf-8")
    return path


def read_comparison(
    case_root: str | Path,
    run_a: str,
    run_b: str,
) -> PairwiseComparison:
    cur = current_dir(case_root)
    path = cur / "comparisons" / _comparison_filename(run_a, run_b)
    if not path.exists():
        return PairwiseComparison(dimension="", run_a=run_a, run_b=run_b)
    data = _from_yaml(path.read_text(encoding="utf-8"))
    reasons: list[IncomparableReason] = []
    raw_reasons = data.get("incomparable_reasons", [])
    if isinstance(raw_reasons, list):
        for r in raw_reasons:
            try:
                reasons.append(IncomparableReason(str(r)))
            except ValueError:
                pass
    return PairwiseComparison(
        dimension=str(data.get("dimension", "")),
        run_a=str(data.get("run_a", "")),
        run_b=str(data.get("run_b", "")),
        label=ComparisonLabel(str(data.get("label", "tie"))),
        evidence=str(data.get("evidence", "")),
        incomparable_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# score_history.jsonl (at evaluation/ root, not under current/)
# ---------------------------------------------------------------------------


def append_score_history(
    case_root: str | Path,
    entry: ScoreHistoryEntry,
) -> Path:
    root = eval_dir(case_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "score_history.jsonl"
    record = {
        "run_id": entry.run_id,
        "timestamp": entry.timestamp,
        "total_score": entry.total_score,
        "tier": entry.tier,
        "dimensions": entry.dimensions,
        "recomputation_reason": entry.recomputation_reason,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read_score_history(case_root: str | Path) -> list[ScoreHistoryEntry]:
    root = eval_dir(case_root)
    path = root / "score_history.jsonl"
    if not path.exists():
        return []
    result: list[ScoreHistoryEntry] = []
    for line in path.read_text(encoding="utf-8").rstrip("\n").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result.append(
            ScoreHistoryEntry(
                run_id=str(record.get("run_id", "")),
                timestamp=str(record.get("timestamp", "")),
                total_score=float(record.get("total_score", 0)),
                tier=str(record.get("tier", "")),
                dimensions={
                    str(k): float(v)
                    for k, v in record.get("dimensions", {}).items()
                },
                recomputation_reason=str(record.get("recomputation_reason", "")),
            )
        )
    return result


# ---------------------------------------------------------------------------
# task_outcomes.jsonl (at evaluation/ root, not under current/)
# ---------------------------------------------------------------------------


def append_task_outcome(
    case_root: str | Path,
    entry: TaskOutcomeEntry,
) -> Path:
    root = eval_dir(case_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "task_outcomes.jsonl"
    record = {
        "package_id": entry.package_id,
        "task_contract_id": entry.task_contract_id,
        "timestamp": entry.timestamp,
        "outcome": entry.outcome,
        "blocked_reason": entry.blocked_reason,
        "acceptance_gaps": list(entry.acceptance_gaps),
        "false_positive": entry.false_positive,
        "landed": entry.landed,
        "verification_status": entry.verification_status,
    }
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return path


def read_task_outcomes(case_root: str | Path) -> list[TaskOutcomeEntry]:
    root = eval_dir(case_root)
    path = root / "task_outcomes.jsonl"
    if not path.exists():
        return []
    result: list[TaskOutcomeEntry] = []
    for line in path.read_text(encoding="utf-8").rstrip("\n").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result.append(
            TaskOutcomeEntry(
                package_id=str(record.get("package_id", "")),
                task_contract_id=str(record.get("task_contract_id", "")),
                timestamp=str(record.get("timestamp", "")),
                outcome=str(record.get("outcome", "unknown")),
                blocked_reason=str(record.get("blocked_reason", "")),
                acceptance_gaps=tuple(
                    str(item) for item in record.get("acceptance_gaps", [])
                ),
                false_positive=bool(record.get("false_positive", False)),
                landed=bool(record.get("landed", False)),
                verification_status=str(record.get("verification_status", "unknown")),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def snapshot_current(case_root: str | Path, timestamp: str) -> Path:
    """Copy the current/ evaluation state into snapshots/<timestamp>/.

    Returns the snapshot directory path.
    """
    import shutil

    cur = current_dir(case_root)
    snap = snapshots_dir(case_root) / timestamp
    if not cur.exists():
        raise FileNotFoundError(f"No current evaluation state at {cur}")

    if snap.exists():
        shutil.rmtree(snap)

    # Only copy yaml files and jsonl, skip large/transient artifacts
    def _copy_filter(src: str, names: list[str]) -> set[str]:
        ignore: set[str] = set()
        for name in names:
            full = Path(src) / name
            if full.is_dir():
                continue
            if not (name.endswith(".yaml") or name.endswith(".jsonl")):
                ignore.add(name)
        return ignore

    shutil.copytree(cur, snap, ignore=_copy_filter)
    return snap


# ---------------------------------------------------------------------------
# Eligibility determination and structural contribution caps
# ---------------------------------------------------------------------------


def determine_eligibility(
    *,
    run_status: str,
    gate_failures: int = 0,
    has_diff: bool = False,
    missing_fields: list[str] | None = None,
) -> EligibilityTier:
    """Determine eligibility tier from run evidence quality.

    Rules:
    - INVALID: gate failures invalidate the score (reference leakage, identity
      exposure) or no evidence whatsoever.
    - FAILED: no diff, missing completion report, or explicit failure status.
      Structural contribution capped at 10.
    - PARTIAL: some evidence missing but core evidence (diff, completion)
      is present. Structural contribution capped at 30.
    - SOLVED: all evidence present, run completed successfully.
    - EXCELLENT/TRANSFORMATIVE: assigned later by score thresholds, not by
      evidence alone.

    This function only determines the evidence-based floor. The caller may
    promote a SOLVED tier to EXCELLENT or TRANSFORMATIVE based on scores.
    """
    if missing_fields is None:
        missing_fields = []

    # Catastrophic failures → INVALID
    if run_status == "invalid":
        return EligibilityTier.INVALID

    # No completion report and no diff → INVALID
    if run_status == "missing-completion-report" and not has_diff:
        return EligibilityTier.INVALID

    # Gate failures: reference leakage, identity exposure, etc.
    if gate_failures >= 3:
        return EligibilityTier.INVALID

    # No diff at all → FAILED
    if not has_diff:
        return EligibilityTier.FAILED

    # Explicit failure status
    if run_status == "failed":
        return EligibilityTier.FAILED

    # Missing completion report but has diff → FAILED (can't evaluate without
    # knowing outcome)
    if run_status == "missing-completion-report":
        return EligibilityTier.FAILED

    # Has core evidence but some pieces missing → PARTIAL
    if missing_fields:
        return EligibilityTier.PARTIAL

    # Reporting without explicit completion → PARTIAL
    if run_status == "reported":
        return EligibilityTier.PARTIAL

    # All evidence present, run completed → SOLVED
    if run_status in ("completed", "solved"):
        return EligibilityTier.SOLVED

    return EligibilityTier.SOLVED


def apply_structural_cap(
    contribution: StructuralContribution,
    tier: EligibilityTier,
) -> StructuralContribution:
    """Apply eligibility cap to structural contribution.

    Returns a new StructuralContribution with each sublevel proportionally
    scaled so that the total does not exceed the cap for the given tier.

    When there is no cap (None), returns the contribution unchanged.
    """
    cap = STRUCTURAL_CONTRIBUTION_CAPS.get(tier)
    if cap is None:
        return contribution

    if not contribution.has_any:
        return contribution

    if contribution.total <= cap:
        return contribution

    # Scale proportionally
    scale = cap / contribution.total
    return StructuralContribution(
        local_improvement=contribution.local_improvement * scale,
        class_deletion=contribution.class_deletion * scale,
        problem_reframing=contribution.problem_reframing * scale,
    )


def promote_tier_from_scores(
    tier: EligibilityTier,
    total_score: float,
    structural_contribution: StructuralContribution,
) -> EligibilityTier:
    """Promote eligibility tier based on score thresholds.

    SOLVED runs with high scores and structural contribution can be promoted
    to EXCELLENT or TRANSFORMATIVE.  FAILED and PARTIAL runs CANNOT be
    promoted — the eligibility cap is hard.
    """
    if tier in (EligibilityTier.INVALID, EligibilityTier.FAILED, EligibilityTier.PARTIAL):
        return tier

    if total_score >= 90.0 and structural_contribution.has_any:
        return EligibilityTier.TRANSFORMATIVE
    if total_score >= 75.0:
        return EligibilityTier.EXCELLENT

    return tier


def _compute_incomparability_ratio(
    comparisons: list[PairwiseComparison],
) -> float:
    """Compute the fraction of pairwise comparisons that are incomparable.

    Returns a float in [0.0, 1.0]. A high ratio reduces confidence.
    """
    if not comparisons:
        return 0.0
    incomparable_count = sum(
        1 for c in comparisons if c.label == ComparisonLabel.INCOMPARABLE
    )
    return incomparable_count / len(comparisons)


# ---------------------------------------------------------------------------
# Information-gain comparison queue engine
# ---------------------------------------------------------------------------


def _group_ranking_by_band(
    entries: list[RankingEntry],
) -> dict[int, list[RankingEntry]]:
    """Group ranking entries by their band assignment."""
    bands: dict[int, list[RankingEntry]] = {}
    for e in entries:
        bands.setdefault(e.band, []).append(e)
    return bands


def _compute_information_gain(
    *,
    entry_a: RankingEntry,
    entry_b: RankingEntry,
    band_sizes: dict[int, int],
    max_band_size: int,
) -> float:
    """Compute how much ranking uncertainty a comparison would reduce.

    Within-band pairs (same band): high gain — ordering is uncertain.
    Adjacent-band pairs: medium gain — band boundaries may shift.
    Distant-band pairs: low gain — ordering is already clear from scores.
    """
    if entry_a.band == entry_b.band:
        band_size = band_sizes.get(entry_a.band, 1)
        max_score = max(entry_a.score, entry_b.score)
        score_diff = abs(entry_a.score - entry_b.score)
        closeness = 1.0 - (score_diff / max_score if max_score > 0 else 1.0)
        return (band_size / max_band_size) * closeness if max_band_size > 0 else 0.0
    else:
        band_dist = abs(entry_a.band - entry_b.band)
        return 1.0 / (band_dist + 1) / max_band_size if max_band_size > 0 else 0.0


def _comparison_reason(
    a: RankingEntry, b: RankingEntry, band_sizes: dict[int, int]
) -> str:
    """Human-readable reason for the information-gain assignment."""
    if a.band == b.band:
        size = band_sizes.get(a.band, 1)
        return f"Intra-band uncertainty in band {a.band} ({size} runs tied)"
    else:
        dist = abs(a.band - b.band)
        if dist == 1:
            return f"Adjacent bands {a.band} and {b.band} — boundary may shift"
        else:
            return f"Distant bands {a.band} and {b.band} — ordering clear"


def build_comparison_queue(
    *,
    rankings: dict[str, list[RankingEntry]],
    existing_pairs: set[tuple[str, str, str]] | None = None,
) -> list[ComparisonCandidate]:
    """Build a comparison queue ordered by information gain.

    Prioritizes comparisons that most reduce ranking uncertainty:
    1. Intra-band pairs (largest bands first, closest scores within band)
    2. Adjacent-band pairs
    3. Cross-band pairs (lowest priority)

    When no intra-band uncertainty exists (all bands size 1), the median
    entry is used as a fallback comparison anchor: pairs involving the
    median get a small gain boost so they sort above non-median pairs
    at the same band distance.

    Ties in gain are broken deterministically: by run_a, then run_b,
    then dimension.
    """
    existing = existing_pairs or set()
    candidates: list[ComparisonCandidate] = []

    for dimension in sorted(rankings.keys()):
        entries = rankings[dimension]
        if len(entries) < 2:
            continue

        bands = _group_ranking_by_band(entries)
        band_sizes = {b: len(v) for b, v in bands.items()}
        max_band_size = max(band_sizes.values()) if band_sizes else 1
        has_uncertain_bands = any(size >= 2 for size in band_sizes.values())

        # When no uncertain bands exist, use median-proximity as tiebreaker
        median_run_id: str | None = None
        if not has_uncertain_bands and len(entries) >= 3:
            median_run_id = entries[len(entries) // 2].run_id

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                pair_key = tuple(sorted([a.run_id, b.run_id]) + [dimension])
                if pair_key in existing:
                    continue

                gain = _compute_information_gain(
                    entry_a=a, entry_b=b,
                    band_sizes=band_sizes,
                    max_band_size=max_band_size,
                )

                # Median fallback: when no intra-band uncertainty, prefer
                # comparisons involving the median-ranked run.
                if median_run_id is not None and (
                    a.run_id == median_run_id or b.run_id == median_run_id
                ):
                    gain += 0.1

                reason = _comparison_reason(a, b, band_sizes)
                candidates.append(ComparisonCandidate(
                    dimension=dimension,
                    run_a=a.run_id,
                    run_b=b.run_id,
                    gain=gain,
                    band_a=a.band,
                    band_b=b.band,
                    reason=reason,
                ))

    # Sort: highest gain first, tie-break deterministically
    candidates.sort(key=lambda c: (-c.gain, c.run_a, c.run_b, c.dimension))
    return candidates


def write_comparison_queue(
    case_root: str | Path,
    queue: list[ComparisonCandidate],
) -> Path:
    """Persist the comparison queue to evaluation/current/comparisons/_queue.yaml."""
    cur = current_dir(case_root)
    comps_dir = cur / "comparisons"
    comps_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for c in queue:
        items.append({
            "dimension": c.dimension,
            "run_a": c.run_a,
            "run_b": c.run_b,
            "gain": c.gain,
            "band_a": c.band_a,
            "band_b": c.band_b,
            "reason": c.reason,
        })

    path = comps_dir / "_queue.yaml"
    path.write_text(_to_yaml(items) + "\n", encoding="utf-8")
    return path


def read_comparison_queue(
    case_root: str | Path,
) -> list[ComparisonCandidate]:
    """Read the persisted comparison queue."""
    cur = current_dir(case_root)
    path = cur / "comparisons" / "_queue.yaml"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    items = _parse_list_of_dicts(text)
    result: list[ComparisonCandidate] = []
    for item in items:
        result.append(ComparisonCandidate(
            dimension=str(item.get("dimension", "")),
            run_a=str(item.get("run_a", "")),
            run_b=str(item.get("run_b", "")),
            gain=float(str(item.get("gain", "0"))),
            band_a=int(str(item.get("band_a", "0"))),
            band_b=int(str(item.get("band_b", "0"))),
            reason=str(item.get("reason", "")),
        ))
    return result


# ---------------------------------------------------------------------------
# Full recomputation engine
# ---------------------------------------------------------------------------


# Default evaluation dimensions
_DEFAULT_DIMENSIONS: tuple[DimensionDef, ...] = (
    DimensionDef(name="result", weight=80.0, description="Observable result quality"),
    DimensionDef(name="process", weight=20.0, description="Process discipline and clarity"),
)

# Score thresholds for tie bands (as fraction of max observed score in dimension)
_TIE_BAND_RATIO: float = 0.05  # scores within 5% of each other are ties


def recompute(
    *,
    case_root: str | Path,
    runs: list[RunAttempt],
    dimensions: list[DimensionDef] | None = None,
    reason: str = "",
) -> Path:
    """Fully recompute the evaluation state for a case from scratch.

    If current/ already has a summary.yaml, snapshot it first.
    Then recompute all rankings, comparisons, summary, and recommendation
    from the provided runs and dimensions.

    The result is deterministic for the same ordered inputs.  Runs are
    processed in stable sort order (by run_id).

    Returns the path to the new current/ directory.
    """
    import datetime as _dt

    case = Path(case_root)
    cur = current_dir(case)

    # Snapshot existing current/ if it has real content
    if (cur / "summary.yaml").exists():
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
        snapshot_current(case, ts)

    # Use provided dimensions or defaults
    dims: list[DimensionDef] = list(dimensions) if dimensions else list(_DEFAULT_DIMENSIONS)

    # Sort runs stably for determinism
    sorted_runs = sorted(runs, key=lambda r: r.run_id)

    # Compute per-dimension rankings
    dim_names = sorted(d.name for d in dims)
    for dim_name in dim_names:
        _recompute_dimension_ranking(case, dim_name, sorted_runs)

    # Build the information-gain comparison queue from the just-computed rankings
    all_rankings: dict[str, list[RankingEntry]] = {}
    for dim_name in dim_names:
        entries = read_ranking(case, dim_name)
        if entries:
            all_rankings[dim_name] = entries
    queue = build_comparison_queue(rankings=all_rankings)
    write_comparison_queue(case, queue)

    # Compute pairwise comparisons in information-gain order
    for dim_name in dim_names:
        dim_queue = [c for c in queue if c.dimension == dim_name]
        _recompute_dimension_comparisons(case, dim_name, sorted_runs, queue=dim_queue)

    # Compute case-level summary
    summary = _build_summary(case, sorted_runs, dims)
    write_summary(case, summary)

    # Compute recommendation
    recommendation = _build_recommendation(summary, sorted_runs)
    write_recommendation(case, recommendation)

    # Write dimensions (may differ from defaults if passed in)
    write_dimensions(case, dims)

    # Write attempts
    write_attempts(case, sorted_runs)

    # Append score history entry for the best run (or each run if reason is provided)
    _record_recomputation(case, summary, sorted_runs, reason)

    return cur


def _recompute_dimension_ranking(
    case_root: Path,
    dimension: str,
    runs: list[RunAttempt],
) -> None:
    """Compute ranking entries for a single dimension.

    Runs are sorted by dimension score descending, with run_id as tie-breaker.
    Bands group runs whose scores fall within _TIE_BAND_RATIO of each other.
    """
    # Filter runs that have a score for this dimension
    scored = [
        (r, r.dimension_scores.get(dimension, 0.0))
        for r in runs
        if _is_scorable_run(r)
    ]
    if not scored:
        return

    # Sort by score descending, then run_id for determinism
    scored.sort(key=lambda x: (-x[1], x[0].run_id))

    max_score = max(s for _, s in scored) if scored else 1.0
    threshold = max_score * _TIE_BAND_RATIO if max_score > 0 else 0.0

    entries: list[RankingEntry] = []
    current_band = 0
    band_leader_score: float | None = None

    for rank_idx, (run, score) in enumerate(scored):
        if band_leader_score is None:
            band_leader_score = score
        elif (band_leader_score - score) > threshold:
            current_band += 1
            band_leader_score = score

        entries.append(RankingEntry(
            run_id=run.run_id,
            rank=rank_idx + 1,
            score=score,
            band=current_band,
        ))

    write_ranking(case_root, dimension, entries)


def _recompute_dimension_comparisons(
    case_root: Path,
    dimension: str,
    runs: list[RunAttempt],
    *,
    queue: list[ComparisonCandidate] | None = None,
) -> None:
    """Compute pairwise comparisons for all run pairs on a single dimension.

    When *queue* is provided, valid-run comparisons are computed in
    information-gain order (highest gain first).  Incomplete-run
    comparisons are always appended as INCOMPARABLE regardless of queue.

    Only compares runs that both have evidence (status is not missing-completion-report).
    Runs with missing evidence are marked incomparable with insufficient_evidence.
    """
    valid_runs = [r for r in runs if _is_scorable_run(r)]
    incomplete_runs = [r for r in runs if r.status == "missing-completion-report"]

    # Compute valid-run comparisons in queue order when available
    if queue:
        for candidate in queue:
            # Find the two RunAttempt objects
            run_lookup = {r.run_id: r for r in valid_runs}
            a = run_lookup.get(candidate.run_a)
            b = run_lookup.get(candidate.run_b)
            if a is None or b is None:
                continue
            score_a = a.dimension_scores.get(dimension, 0.0)
            score_b = b.dimension_scores.get(dimension, 0.0)
            label, evidence, reasons = _score_comparison(
                dimension, a.run_id, b.run_id, score_a, score_b
            )
            write_comparison(case_root, PairwiseComparison(
                dimension=dimension,
                run_a=a.run_id,
                run_b=b.run_id,
                label=label,
                evidence=evidence,
                incomparable_reasons=reasons,
            ))
    else:
        # Fallback: compare all pairs of valid runs
        for i in range(len(valid_runs)):
            for j in range(i + 1, len(valid_runs)):
                a, b = valid_runs[i], valid_runs[j]
                score_a = a.dimension_scores.get(dimension, 0.0)
                score_b = b.dimension_scores.get(dimension, 0.0)
                label, evidence, reasons = _score_comparison(
                    dimension, a.run_id, b.run_id, score_a, score_b
                )
                write_comparison(case_root, PairwiseComparison(
                    dimension=dimension,
                    run_a=a.run_id,
                    run_b=b.run_id,
                    label=label,
                    evidence=evidence,
                    incomparable_reasons=reasons,
                ))

    # Mark comparisons with incomplete runs as incomparable
    for inc in incomplete_runs:
        for valid in valid_runs:
            write_comparison(case_root, PairwiseComparison(
                dimension=dimension,
                run_a=inc.run_id,
                run_b=valid.run_id,
                label=ComparisonLabel.INCOMPARABLE,
                evidence=f"{inc.run_id} has incomplete evidence",
                incomparable_reasons=[IncomparableReason.INSUFFICIENT_EVIDENCE],
            ))

    # Pairs of incomplete runs
    for i in range(len(incomplete_runs)):
        for j in range(i + 1, len(incomplete_runs)):
            a, b = incomplete_runs[i], incomplete_runs[j]
            write_comparison(case_root, PairwiseComparison(
                dimension=dimension,
                run_a=a.run_id,
                run_b=b.run_id,
                label=ComparisonLabel.INCOMPARABLE,
                evidence="Both runs have insufficient evidence",
                incomparable_reasons=[IncomparableReason.INSUFFICIENT_EVIDENCE],
            ))


def _score_comparison(
    dimension: str,
    run_a: str,
    run_b: str,
    score_a: float,
    score_b: float,
) -> tuple[ComparisonLabel, str, list[IncomparableReason]]:
    """Determine comparison label from two dimension scores.

    Uses a relative threshold: the larger the max score, the wider the
    absolute gap needed for a clear win/loss.
    """
    if score_a == 0.0 and score_b == 0.0:
        return (
            ComparisonLabel.INCOMPARABLE,
            f"Both runs have zero {dimension} score",
            [IncomparableReason.INSUFFICIENT_EVIDENCE],
        )

    max_s = max(score_a, score_b)
    if max_s == 0:
        return (ComparisonLabel.TIE, f"Equal {dimension} scores (0)", [])

    relative_diff = abs(score_a - score_b) / max_s

    if relative_diff <= _TIE_BAND_RATIO:
        return (ComparisonLabel.TIE, f"Similar {dimension} scores ({score_a:.1f} vs {score_b:.1f})", [])
    elif relative_diff <= 0.20:
        if score_a > score_b:
            return (ComparisonLabel.BETTER, f"{run_a} leads {run_b} on {dimension} ({score_a:.1f} vs {score_b:.1f})", [])
        else:
            return (ComparisonLabel.WORSE, f"{run_b} leads {run_a} on {dimension} ({score_b:.1f} vs {score_a:.1f})", [])
    else:
        if score_a > score_b:
            return (ComparisonLabel.MUCH_BETTER, f"{run_a} significantly ahead of {run_b} on {dimension} ({score_a:.1f} vs {score_b:.1f})", [])
        else:
            return (ComparisonLabel.MUCH_WORSE, f"{run_b} significantly ahead of {run_a} on {dimension} ({score_b:.1f} vs {score_a:.1f})", [])


def _build_summary(
    case_root: Path,
    runs: list[RunAttempt],
    dims: list[DimensionDef],
) -> EvaluationSummary:
    """Build the case-level evaluation summary from recomputed rankings.

    Integrates structural contribution with eligibility caps and accounts
    for incomparable comparisons in confidence estimation.
    """
    case_id = case_root.name if case_root.name else "unknown"

    if not runs:
        return EvaluationSummary(
            case_id=case_id,
            tier=EligibilityTier.INVALID,
            validity="no runs available",
            confidence="low",
            total_score=0.0,
            run_count=0,
            best_run_id="",
        )

    valid_runs = [r for r in runs if _is_scorable_run(r)]

    # Compute weighted total scores with capped structural contribution
    dim_weights = {d.name: d.weight for d in dims}
    total_weight = sum(dim_weights.values()) or 1.0

    best_run_id = ""
    best_score: float | None = None
    best_tier = EligibilityTier.INVALID

    for run in runs:
        # Invalid runs get no score at all
        if run.tier == EligibilityTier.INVALID:
            run.total_score = 0.0
            continue

        # Base score from weighted dimension scores
        weighted = 0.0
        for dim in dims:
            weighted += run.dimension_scores.get(dim.name, 0.0) * dim_weights.get(dim.name, 0.0)
        base_score = weighted / total_weight if total_weight > 0 else 0.0

        # Apply eligibility cap to structural contribution
        capped_sc = apply_structural_cap(run.structural_contribution, run.tier)
        sc_bonus = capped_sc.total

        run.total_score = min(base_score + sc_bonus, 100.0)

        # Promote tier based on capped total score
        run.tier = promote_tier_from_scores(run.tier, run.total_score, capped_sc)

        if best_score is None or run.total_score > best_score:
            best_score = run.total_score
            best_run_id = run.run_id
            best_tier = run.tier

    # Determine validity and confidence including incomparability
    incomparability_ratio = 0.0
    if valid_runs and len(valid_runs) >= 2:
        # Collect comparisons from the comparisons directory
        comps = _collect_all_comparisons(case_root)
        incomparability_ratio = _compute_incomparability_ratio(comps)

    if not valid_runs:
        validity = "no valid runs with evidence"
        confidence = "low"
    elif len(valid_runs) == 1:
        validity = "single run — no comparison possible"
        confidence = "low"
    else:
        validity = "valid"
        confidence = _confidence_from_runs_and_incomparability(
            len(valid_runs), incomparability_ratio
        )

    # Overall tier is the best tier among valid runs
    tier_order = list(EligibilityTier)
    overall_tier = EligibilityTier.INVALID
    for run in valid_runs:
        if tier_order.index(run.tier) > tier_order.index(overall_tier):
            overall_tier = run.tier

    return EvaluationSummary(
        case_id=case_id,
        tier=overall_tier,
        validity=validity,
        confidence=confidence,
        total_score=best_score if best_score is not None else 0.0,
        run_count=len(runs),
        best_run_id=best_run_id,
    )


def _is_scorable_run(run: RunAttempt) -> bool:
    """Return True when a run has evidence and is not invalidated."""
    return run.status != "missing-completion-report" and run.tier != EligibilityTier.INVALID


def _collect_all_comparisons(case_root: Path) -> list[PairwiseComparison]:
    """Read all pairwise comparison files from the comparisons directory."""
    cur = current_dir(case_root)
    comps_dir = cur / "comparisons"
    if not comps_dir.exists():
        return []
    result: list[PairwiseComparison] = []
    for path in sorted(comps_dir.glob("*.yaml")):
        if path.name == "_queue.yaml":
            continue
        data = _from_yaml(path.read_text(encoding="utf-8"))
        run_a = str(data.get("run_a", ""))
        run_b = str(data.get("run_b", ""))
        if not run_a or not run_b:
            continue
        label = ComparisonLabel(str(data.get("label", "tie")))
        reasons: list[IncomparableReason] = []
        raw_reasons = data.get("incomparable_reasons", [])
        if isinstance(raw_reasons, list):
            for r in raw_reasons:
                try:
                    reasons.append(IncomparableReason(str(r)))
                except ValueError:
                    pass
        result.append(PairwiseComparison(
            dimension=str(data.get("dimension", "")),
            run_a=run_a,
            run_b=run_b,
            label=label,
            incomparable_reasons=reasons,
        ))
    return result


def _confidence_from_runs_and_incomparability(
    num_runs: int,
    incomparability_ratio: float,
) -> str:
    """Determine confidence level based on run count and incomparability.

    High incomparability ratios reduce confidence because much of the
    comparison space cannot be resolved into a total order.
    """
    if incomparability_ratio > 0.5:
        return "low"
    if incomparability_ratio > 0.25:
        return "medium" if num_runs >= 3 else "low"
    if num_runs >= 3:
        return "medium"
    return "low"


def _build_recommendation(
    summary: EvaluationSummary,
    runs: list[RunAttempt],
) -> Recommendation:
    """Build a user-facing recommendation from the evaluation summary."""
    score = summary.total_score

    if summary.run_count == 0:
        return Recommendation(
            score=0.0,
            sentence="No runs to evaluate.",
            validity=summary.validity,
            confidence=summary.confidence,
        )

    valid_runs = [r for r in runs if _is_scorable_run(r)]

    if not valid_runs:
        if any(r.tier == EligibilityTier.INVALID for r in runs):
            return Recommendation(
                score=0.0,
                sentence="Score is invalid — evidence gates not met.",
                validity=summary.validity,
                confidence=summary.confidence,
            )
        return Recommendation(
            score=0.0,
            sentence="All runs have insufficient evidence for evaluation.",
            validity=summary.validity,
            confidence=summary.confidence,
        )

    # Build recommendation sentence based on tier and score
    tier = summary.tier
    if tier == EligibilityTier.INVALID:
        sentence = "Score is invalid — evidence gates not met."
    elif tier == EligibilityTier.FAILED:
        sentence = "Not suitable — failed to meet minimum evidence requirements."
    elif tier == EligibilityTier.PARTIAL:
        sentence = "Partially suitable for low-risk tasks; not recommended for critical changes."
    elif tier == EligibilityTier.SOLVED:
        sentence = "Suitable for low-risk implementation and existing-plan execution; not the first choice for open-ended architecture exploration."
    elif tier == EligibilityTier.EXCELLENT:
        sentence = "Strong performer across evaluation dimensions; suitable for moderate-complexity tasks."
    elif tier == EligibilityTier.TRANSFORMATIVE:
        sentence = "Exceptional performance; suitable for high-complexity and open-ended tasks."
    else:
        sentence = "Evaluation complete."

    return Recommendation(
        score=score,
        sentence=sentence,
        validity=summary.validity,
        confidence=summary.confidence,
    )


def _record_recomputation(
    case_root: Path,
    summary: EvaluationSummary,
    runs: list[RunAttempt],
    reason: str,
) -> None:
    """Record a score history entry for the recomputation."""
    import datetime as _dt

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Record one entry per valid run that contributed to the recomputation
    for run in runs:
        entry = ScoreHistoryEntry(
            run_id=run.run_id,
            timestamp=timestamp,
            total_score=run.total_score,
            tier=summary.tier.value,
            dimensions=dict(run.dimension_scores),
            recomputation_reason=reason or "full recomputation",
        )
        append_score_history(case_root, entry)
