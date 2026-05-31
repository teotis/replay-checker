"""Evidence validity gates and score ceiling rules for Replay Checker scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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
    data: dict[str, Any] = {}
    current_list: str | None = None
    current_section: str | None = None
    current_subsection: str | None = None
    sub_data: dict[str, Any] | None = None

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        # "  - value" inside a list (top-level or inside a section)
        if line.startswith("  - "):
            if current_list and current_section is None:
                # Top-level list (e.g. minimum_evidence in flat format)
                data.setdefault(current_list, [])
                assert isinstance(data[current_list], list)
                data[current_list].append(line[4:])
                continue
            if current_section and current_subsection is None:
                # List items under a top-level section (e.g. minimum_evidence:)
                if data.get(current_section) is None or data[current_section] == {}:
                    data[current_section] = []
                if isinstance(data[current_section], list):
                    data[current_section].append(line[4:])
                    continue

        # list item inside a nested section (e.g. "    - diff_present")
        if line.strip().startswith("- ") and current_subsection is not None:
            data.setdefault(current_section, {})
            assert isinstance(data[current_section], dict)
            if data[current_section].get(current_subsection) is None:
                data[current_section][current_subsection] = []
            if not isinstance(data[current_section][current_subsection], list):
                # Subsection was already initialized as dict (nested keys), skip
                continue
            data[current_section][current_subsection].append(line.strip()[2:])
            continue

        # top-level "key:" or "key: value"
        indent = len(raw_line) - len(raw_line.lstrip())
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 0:
            # Top-level key
            current_list = None
            current_section = None
            current_subsection = None
            sub_data = None
            if value == "":
                # Could be a section with nested dicts or a list;
                # initialize as None, first child line determines type.
                current_section = key
                data[key] = None
            elif value.startswith("[") and value.endswith("]"):
                data[key] = [v.strip().strip('"') for v in value[1:-1].split(",") if v.strip()]
            else:
                try:
                    data[key] = int(value)
                except ValueError:
                    data[key] = value
        elif indent == 2 and current_section:
            # Second-level key (e.g. "required:" or "no_diff:")
            # Initialize section as dict if not already (was None from indent==0)
            if data.get(current_section) is None:
                data[current_section] = {}
            if value == "":
                current_subsection = key
                # Placeholder; will be set to list if list items follow,
                # or to dict if nested keys follow.
                sub_data = None
                data[current_section][key] = None
            else:
                current_subsection = None
                sub_data = None
                try:
                    data[current_section][key] = int(value)
                except ValueError:
                    if value == "true":
                        data[current_section][key] = True
                    elif value == "false":
                        data[current_section][key] = False
                    else:
                        data[current_section][key] = value
        elif indent == 4 and current_subsection is not None:
            # Third-level key (e.g. "result_ceiling: 0")
            section_data = data.get(current_section, {})
            current_val = section_data.get(current_subsection)
            if current_val is None:
                # Placeholder from indent==2: initialize as dict
                sub_data = {}
                data[current_section][current_subsection] = sub_data
            elif isinstance(current_val, dict):
                sub_data = current_val
            else:
                continue  # Already a list from list items; skip key-value parse
            try:
                sub_data[key] = int(value)
            except ValueError:
                if value == "true":
                    sub_data[key] = True
                elif value == "false":
                    sub_data[key] = False
                else:
                    sub_data[key] = value

    return data


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
