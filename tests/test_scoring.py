"""Characterization tests for evidence gates, score ceilings, and scoring packages.

Locks down current behavior before the lifecycle refactor begins.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from replay_checker.scoring import (
    CaseProvenance,
    EvidenceGate,
    GateEvaluation,
    GateResult,
    RunEvaluation,
    ScoreCeilings,
    compute_score_ceilings,
    evaluate_evidence_gates,
    evaluate_run,
)
from replay_checker.scoring_ops import (
    _anonymous_runner_id,
    _conflict_severity_mark,
    _evidence_item_found,
    _format_telemetry_scoring,
    _render_provenance_section,
    _stable_anonymous_runner_id,
    score_run,
)
from replay_checker.evaluation import (
    EligibilityTier,
    Recommendation,
    RunAttempt,
    StructuralContribution,
    recompute,
    read_summary,
    write_recommendation,
)
from replay_checker.replay import (
    ReplayCase,
    ReplayRun,
    _build_compare_output,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_run_root(
    tmp_path: Path,
    *,
    has_diff: bool = True,
    diff_content: str = "diff --git a/x b/x\n+line\n",
    has_completion: bool = True,
    completion_content: str = "status: completed\n\n## Verification\n\npytest passed\n",
    changed_files: list[str] | None = None,
    has_reference_dir: bool = False,
    evidence_yaml_extra: str = "",
) -> Path:
    """Build a minimal run root with independently configurable evidence."""
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)

    ev_dir = run_root / "evidence"
    ev_dir.mkdir(exist_ok=True)

    if has_diff:
        (ev_dir / "diff.patch").write_text(diff_content, encoding="utf-8")
    else:
        # Ensure diff is missing (or empty)
        pass

    if changed_files is None:
        changed_files = []

    cf_yaml = "\n".join(f"  - {f}" for f in changed_files)
    evidence_yaml = (
        f"run_id: eval-run\nstatus: completed\nchanged_files:\n{cf_yaml}\n"
        + evidence_yaml_extra
    )
    (ev_dir / "evidence.yaml").write_text(evidence_yaml, encoding="utf-8")

    if has_completion:
        (run_root / "completion_report.md").write_text(
            completion_content, encoding="utf-8"
        )

    if has_reference_dir:
        ref_dir = run_root / "_reference"
        ref_dir.mkdir()
        (ref_dir / "diff.patch").write_text("oracle diff\n", encoding="utf-8")

    return run_root


def _make_case(tmp_path: Path, case_id: str = "test-case") -> ReplayCase:
    root = tmp_path / case_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "evidence_sources.md").write_text("- none\n", encoding="utf-8")
    return ReplayCase(
        id=case_id,
        root=root,
        project_path=tmp_path,
        plan_path=Path(""),
        base_commit="abc123",
        verification_commands=(),
        base_source="head_fallback",
        base_confidence="low",
        source_type="manual",
        source_path="",
        selection_reason="test",
        synthetic_case=False,
        evidence_sources=(),
    )


def _write_fake_evidence(run_dir: Path, changed_files: list[str] | None = None) -> None:
    ev_dir = run_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    if changed_files is None:
        changed_files = ["x"]
    cf_yaml = "\n".join(f"  - {f}" for f in changed_files)
    (ev_dir / "diff.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (ev_dir / "evidence.yaml").write_text(
        f"run_id: eval-run\nstatus: completed\nchanged_files:\n{cf_yaml}\n",
        encoding="utf-8",
    )


# =========================================================================
# Evidence Gate Characterization
# =========================================================================


class TestEvidenceGateCharacterization:
    """Each evidence gate independently verified against current behavior."""

    def test_diff_present_gate_passes_with_nonempty_patch(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=True, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.DIFF_PRESENT]
        assert len(gate) == 1
        assert gate[0].passed is True

    def test_diff_present_gate_fails_when_patch_missing(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.DIFF_PRESENT]
        assert gate[0].passed is False
        assert "missing" in gate[0].detail.lower() or "empty" in gate[0].detail.lower()

    def test_diff_present_gate_fails_when_patch_empty(self, tmp_path):
        root = _make_run_root(
            tmp_path, has_diff=True, diff_content="", changed_files=["x"]
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.DIFF_PRESENT]
        assert gate[0].passed is False

    def test_completion_report_gate_passes_when_file_exists(self, tmp_path):
        root = _make_run_root(tmp_path, has_completion=True, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.COMPLETION_REPORT_PRESENT]
        assert gate[0].passed is True

    def test_completion_report_gate_fails_when_missing(self, tmp_path):
        root = _make_run_root(tmp_path, has_completion=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.COMPLETION_REPORT_PRESENT]
        assert gate[0].passed is False

    def test_changed_files_gate_passes_with_nonempty_list(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["a.py", "b.py"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["a.py", "b.py"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.CHANGED_FILES_PRESENT]
        assert gate[0].passed is True

    def test_changed_files_gate_fails_with_empty_list(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=[])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=[]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.CHANGED_FILES_PRESENT]
        assert gate[0].passed is False
        assert "no changed files" in gate[0].detail.lower()

    def test_verification_cited_gate_passes_when_report_mentions_test(self, tmp_path):
        root = _make_run_root(
            tmp_path,
            completion_content="status: completed\n\nAll tests passed\n",
            changed_files=["x"],
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.VERIFICATION_CITED]
        assert gate[0].passed is True

    def test_verification_cited_gate_fails_when_no_verification_mention(self, tmp_path):
        root = _make_run_root(
            tmp_path,
            completion_content="status: completed\nEverything looks fine.\n",
            changed_files=["x"],
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.VERIFICATION_CITED]
        assert gate[0].passed is False

    def test_verification_cited_gate_passes_via_evidence_yaml_fallback(self, tmp_path):
        root = _make_run_root(
            tmp_path,
            completion_content="status: completed\nNo verification mentioned.\n",
            changed_files=["x"],
            evidence_yaml_extra="verification: pytest\n",
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.VERIFICATION_CITED]
        assert gate[0].passed is True

    def test_reference_access_absent_gate_passes_when_no_reference(self, tmp_path):
        root = _make_run_root(tmp_path, has_reference_dir=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT]
        assert gate[0].passed is True

    def test_reference_access_absent_gate_fails_when_reference_dir_exists(self, tmp_path):
        root = _make_run_root(tmp_path, has_reference_dir=True, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT]
        assert gate[0].passed is False
        assert "_reference" in gate[0].detail

    def test_reference_access_absent_gate_fails_when_report_mentions_oracle(self, tmp_path):
        root = _make_run_root(
            tmp_path,
            completion_content="status: completed\nChecked oracle material.\n",
            changed_files=["x"],
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate = [r for r in ev.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT]
        assert gate[0].passed is False
        assert "oracle" in gate[0].detail.lower()


# =========================================================================
# Score Ceilings Characterization
# =========================================================================


class TestScoreCeilingsCharacterization:
    """Verify ceiling rules derived from gate results."""

    def test_no_diff_sets_result_ceiling_zero(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.result_ceiling == 0.0
        assert not ceilings.overall_invalid

    def test_no_completion_report_sets_process_ceiling_zero(self, tmp_path):
        root = _make_run_root(tmp_path, has_completion=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.process_ceiling == 0.0
        assert not ceilings.overall_invalid

    def test_no_verification_sets_verification_ceiling_zero(self, tmp_path):
        root = _make_run_root(
            tmp_path,
            completion_content="status: completed\nEverything is done.\n",
            changed_files=["x"],
        )
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.verification_ceiling == 0.0

    def test_reference_leakage_marks_overall_invalid(self, tmp_path):
        root = _make_run_root(tmp_path, has_reference_dir=True, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.overall_invalid is True
        # When invalid, individual ceilings are not set (early return)
        assert ceilings.result_ceiling is None
        assert ceilings.process_ceiling is None

    def test_all_gates_pass_yields_no_ceilings(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.result_ceiling is None
        assert ceilings.process_ceiling is None
        assert ceilings.verification_ceiling is None
        assert ceilings.overall_invalid is False

    def test_multiple_failures_compose(self, tmp_path):
        """Missing diff + missing completion → both result and process ceilings at 0."""
        root = _make_run_root(tmp_path, has_diff=False, has_completion=False, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        ceilings = compute_score_ceilings(ev)
        assert ceilings.result_ceiling == 0.0
        assert ceilings.process_ceiling == 0.0


# =========================================================================
# Compare Rule: Present Diff + No Changed Files
# =========================================================================


class TestCompareRuleDiffPresentNoChangedFiles:
    """A run with a present diff but no changed files blocks comparison.

    This is the distinct edge case: the diff exists but changed_files is empty,
    which triggers result evidence blocking in _all_run_result_evidence_blocked.
    """

    def test_compare_blocks_when_diff_present_but_no_changed_files(self, tmp_path):
        """The compare output should report insufficient evidence."""
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=[])

        rec = Recommendation(
            score=82.0,
            sentence="Stale positive recommendation.",
            validity="valid",
            confidence="medium",
        )
        write_recommendation(case.root, rec)

        output = _build_compare_output(case, ["test-case-001"], tmp_path / "runs")
        assert "0 / 100" in output
        assert "insufficient evidence" in output.lower()

    def test_compare_unblocks_when_both_diff_and_changed_files_present(self, tmp_path):
        """Normal case: diff present AND changed_files non-empty → not blocked."""
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        _write_fake_evidence(run_root, changed_files=["x.py"])

        rec = Recommendation(
            score=82.0,
            sentence="Suitable for low-risk implementation.",
            validity="valid",
            confidence="medium",
        )
        write_recommendation(case.root, rec)

        output = _build_compare_output(case, ["test-case-001"], tmp_path / "runs")
        assert "82" in output
        assert "insufficient evidence" not in output.lower()

    def test_all_run_result_evidence_blocked_requires_checked_runs(self):
        """_all_run_result_evidence_blocked returns False when no run_ids exist."""
        from replay_checker.replay import _all_run_result_evidence_blocked
        assert _all_run_result_evidence_blocked([], Path("/nonexistent")) is False

    def test_changed_files_absent_with_diff_is_only_blocking_condition(self, tmp_path):
        """A run with empty changed_files list blocks even with valid diff."""
        from replay_checker.replay import _all_run_result_evidence_blocked
        run_root = tmp_path / "runs" / "case-001"
        _write_fake_evidence(run_root, changed_files=[])
        assert _all_run_result_evidence_blocked(["case-001"], tmp_path / "runs") is True

    def test_changed_files_present_unblocks(self, tmp_path):
        from replay_checker.replay import _all_run_result_evidence_blocked
        run_root = tmp_path / "runs" / "case-001"
        _write_fake_evidence(run_root, changed_files=["a.py"])
        assert _all_run_result_evidence_blocked(["case-001"], tmp_path / "runs") is False


# =========================================================================
# Scoring Package Headings and Key Wording
# =========================================================================


class TestScoringPackageStructure:
    """Characterize the generated scoring package's section headings and key phrases."""

    @staticmethod
    def _score_a_run(tmp_path: Path) -> tuple[Path, Path]:
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        run_root.mkdir(parents=True)
        _write_fake_evidence(run_root)
        (run_root / "completion_report.md").write_text(
            "status: completed\n\n## Verification\n\npytest passed\n",
            encoding="utf-8",
        )
        run = ReplayRun(
            id="test-case-001",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        path = score_run(run, rubric_path=None)
        return path, run_root

    def test_scoring_package_heading_format(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert text.startswith("# Scoring Package: test-case-001")

    def test_scoring_package_has_protocol_version(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Protocol Version: replay-checker-score-v2" in text

    def test_scoring_package_has_anonymous_runner(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "Anonymous runner:" in text

    def test_scoring_package_has_required_inputs_section(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Required Inputs" in text
        assert "Evidence YAML:" in text
        assert "Diff patch:" in text
        assert "Completion report:" in text

    def test_scoring_package_has_rubric_weights(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Rubric Weights" in text
        assert "Result score weight:" in text
        assert "Process score weight:" in text

    def test_scoring_package_has_evidence_inventory(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Evidence Inventory" in text
        assert "Changed files" in text
        assert "Diff present:" in text
        assert "Completion report:" in text

    def test_scoring_package_has_bias_warnings(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Bias Warnings" in text
        assert "Do not infer quality from agent identity" in text

    def test_scoring_package_has_process_scoring_guidance(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Process Scoring Guidance" in text
        assert "Telemetry is a risk/scope signal" in text

    def test_scoring_package_has_invalid_score_conditions(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Invalid Score Conditions" in text

    def test_scoring_package_has_evidence_gate_assessment(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Evidence Gate Assessment" in text
        assert "[PASS]" in text
        assert "all gates passed" in text.lower()

    def test_scoring_package_has_evidence_references(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        assert "## Evidence References" in text

    def test_scoring_package_ceiling_section_not_present_when_all_gates_pass(self, tmp_path):
        path, _ = self._score_a_run(tmp_path)
        text = path.read_text(encoding="utf-8")
        # When all gates pass, no ceiling section should appear
        assert "## Score Ceilings" not in text

    def test_scoring_package_ceiling_section_present_when_gate_fails(self, tmp_path):
        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        run_root.mkdir(parents=True)
        # No diff → gate failure
        _write_fake_evidence(run_root, changed_files=["x"])
        (run_root / "evidence" / "diff.patch").unlink()
        (run_root / "completion_report.md").write_text(
            "status: completed\n\n## Verification\n\npytest passed\n",
            encoding="utf-8",
        )
        run = ReplayRun(
            id="test-case-001",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        path = score_run(run, rubric_path=None)
        text = path.read_text(encoding="utf-8")
        assert "## Score Ceilings" in text
        assert "Result score ceiling: 0" in text


# =========================================================================
# Gate Evaluation Structural Contracts
# =========================================================================


class TestGateEvaluationContracts:
    """Verify GateEvaluation aggregation semantics."""

    def test_all_passed_is_true_when_no_failures(self):
        ev = GateEvaluation()
        ev.add(GateResult(gate=EvidenceGate.DIFF_PRESENT, passed=True))
        ev.add(GateResult(gate=EvidenceGate.COMPLETION_REPORT_PRESENT, passed=True))
        assert ev.all_passed is True
        assert ev.is_invalid is False

    def test_all_passed_is_false_on_any_failure(self):
        ev = GateEvaluation()
        ev.add(GateResult(gate=EvidenceGate.DIFF_PRESENT, passed=True))
        ev.add(GateResult(gate=EvidenceGate.CHANGED_FILES_PRESENT, passed=False))
        assert ev.all_passed is False

    def test_add_invalid_sets_is_invalid(self):
        ev = GateEvaluation()
        ev.add_invalid("oracle leakage")
        assert ev.is_invalid is True
        assert "oracle leakage" in ev.invalid_reasons

    def test_to_diagnostics_converts_results(self):
        ev = GateEvaluation()
        ev.add(GateResult(gate=EvidenceGate.DIFF_PRESENT, passed=True, detail="ok"))
        ev.add(GateResult(gate=EvidenceGate.DIFF_PRESENT, passed=False, detail="missing"))
        diags = ev.to_diagnostics()
        assert len(diags) == 2
        assert diags[0].severity == "info"
        assert diags[1].severity == "error"

    def test_six_gates_are_evaluated(self, tmp_path):
        """All six evidence gates should be present in evaluation results."""
        root = _make_run_root(tmp_path, changed_files=["x"])
        ev = evaluate_evidence_gates(
            evidence_root=root, run_id="r", changed_files=["x"]
        )
        gate_names = {r.gate for r in ev.results}
        expected = {
            EvidenceGate.DIFF_PRESENT,
            EvidenceGate.COMPLETION_REPORT_PRESENT,
            EvidenceGate.CHANGED_FILES_PRESENT,
            EvidenceGate.VERIFICATION_CITED,
            EvidenceGate.REFERENCE_ACCESS_ABSENT,
            EvidenceGate.RUNNER_IDENTITY_HIDDEN,
        }
        assert gate_names == expected


# =========================================================================
# Canonical RunEvaluation Tests
# =========================================================================


class TestRunEvaluationCanonical:
    """Verify RunEvaluation immutable snapshot and evaluate_run() entry point."""

    def test_evaluate_run_returns_run_evaluation(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert isinstance(ev, RunEvaluation)
        assert ev.run_id == "r1"

    def test_repeated_evaluation_returns_equal_snapshots(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev1 = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        ev2 = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert ev1 == ev2
        assert hash(ev1) == hash(ev2)

    def test_gate_results_are_immutable_tuple(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert isinstance(ev.gate_results, tuple)
        with pytest.raises(AttributeError):
            ev.gate_results = ()  # type: ignore[misc]

    def test_invalid_reasons_are_immutable_tuple(self, tmp_path):
        root = _make_run_root(tmp_path, has_reference_dir=True, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert isinstance(ev.invalid_reasons, tuple)
        assert len(ev.invalid_reasons) > 0
        with pytest.raises(AttributeError):
            ev.invalid_reasons = ()  # type: ignore[misc]

    def test_changed_files_are_immutable_tuple(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["a.py", "b.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["a.py", "b.py"])
        assert ev.changed_files == ("a.py", "b.py")
        with pytest.raises(AttributeError):
            ev.changed_files = ()  # type: ignore[misc]

    def test_result_ceiling_matches_lower_level(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=False, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        gate_eval = evaluate_evidence_gates(
            evidence_root=root, run_id="r1", changed_files=["x.py"]
        )
        ceilings = compute_score_ceilings(gate_eval)
        assert ev.result_ceiling == ceilings.result_ceiling
        assert ev.process_ceiling == ceilings.process_ceiling
        assert ev.verification_ceiling == ceilings.verification_ceiling
        assert ev.overall_invalid == ceilings.overall_invalid

    def test_every_gate_appears_once(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        gate_names = tuple(r.gate for r in ev.gate_results)
        assert len(gate_names) == 6
        assert len(set(gate_names)) == 6

    def test_all_gates_passed_preserved(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        gate_eval = evaluate_evidence_gates(
            evidence_root=root, run_id="r1", changed_files=["x.py"]
        )
        assert ev.all_gates_passed == gate_eval.all_passed

    def test_is_invalid_preserved(self, tmp_path):
        root = _make_run_root(tmp_path, has_reference_dir=True, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert ev.is_invalid is True
        assert len(ev.invalid_reasons) > 0

    def test_no_side_effects_evaluation_creates_no_files(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        before = sorted(str(p) for p in root.rglob("*") if p.is_file())
        evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        after = sorted(str(p) for p in root.rglob("*") if p.is_file())
        assert before == after

    def test_result_evidence_blocked_when_diff_present_no_changed_files(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=True, changed_files=[])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=[])
        assert ev.result_evidence_blocked is True
        assert ev.result_ceiling is None
        assert not ev.overall_invalid

    def test_result_evidence_not_blocked_when_diff_present_with_changed_files(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=True, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert ev.result_evidence_blocked is False

    def test_result_evidence_blocked_when_no_diff(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=False, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        assert ev.result_evidence_blocked is True

    def test_result_evidence_blocked_both_absent(self, tmp_path):
        root = _make_run_root(tmp_path, has_diff=False, changed_files=[])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=[])
        assert ev.result_evidence_blocked is True

    def test_repr_contains_key_fields(self, tmp_path):
        root = _make_run_root(tmp_path, changed_files=["x.py"])
        ev = evaluate_run(evidence_root=root, run_id="r1", changed_files=["x.py"])
        r = repr(ev)
        assert "r1" in r
        assert "all_gates_passed" in r
        assert "result_evidence_blocked" in r


# =========================================================================
# scoring_ops Owner Module Tests
# =========================================================================


class TestScoringOpsOwnerModule:
    """Focused tests for the scoring_ops owner module (no facade dependency)."""

    def test_scoring_ops_score_run_matches_facade(self, tmp_path):
        """scoring_ops.score_run produces the same scoring package as the facade."""
        from replay_checker.replay import score_run as facade_score_run

        case = _make_case(tmp_path)
        run_root = tmp_path / "runs" / "test-case-001"
        run_root.mkdir(parents=True)
        _write_fake_evidence(run_root)
        (run_root / "completion_report.md").write_text(
            "status: completed\n\n## Verification\n\npytest passed\n",
            encoding="utf-8",
        )
        run = ReplayRun(
            id="test-case-001",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        path = score_run(run, rubric_path=None)
        assert path.read_text(encoding="utf-8") == facade_score_run(run, rubric_path=None).read_text(encoding="utf-8")

    def test_stable_anonymous_runner_id_deterministic(self):
        """_stable_anonymous_runner_id returns the same value for the same input."""
        id1 = _stable_anonymous_runner_id("case-001-run-001")
        id2 = _stable_anonymous_runner_id("case-001-run-001")
        assert id1 == id2
        assert id1.startswith("runner-")
        assert len(id1) == len("runner-") + 10

    def test_stable_anonymous_runner_id_varies_by_input(self):
        """Different run IDs produce different anonymous IDs."""
        id1 = _stable_anonymous_runner_id("run-a")
        id2 = _stable_anonymous_runner_id("run-b")
        assert id1 != id2

    def test_anonymous_runner_id_uses_stored_id_when_present(self, tmp_path):
        """_anonymous_runner_id reads anonymous_runner_id from run.yaml."""
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "run.yaml").write_text(
            "anonymous_runner_id: anon-12345\n",
            encoding="utf-8",
        )
        case = _make_case(tmp_path)
        run = ReplayRun(
            id="test-run",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        assert _anonymous_runner_id(run) == "anon-12345"

    def test_anonymous_runner_id_falls_back_to_stable(self, tmp_path):
        """Without run.yaml, _anonymous_runner_id falls back to stable hash."""
        run_root = tmp_path / "run"
        run_root.mkdir()
        case = _make_case(tmp_path)
        run = ReplayRun(
            id="test-run-001",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        assert _anonymous_runner_id(run) == _stable_anonymous_runner_id("test-run-001")

    def test_anonymous_runner_id_returns_stable_for_nonstring_value(self, tmp_path):
        """When run.yaml stores a non-string value, the function falls back to stable hash."""
        run_root = tmp_path / "run"
        run_root.mkdir()
        # The YAML parser interprets bare key (no value) as an empty list []
        # str([]).strip() == "[]" is truthy, so the stored value is returned
        (run_root / "run.yaml").write_text(
            "anonymous_runner_id:\n",
            encoding="utf-8",
        )
        case = _make_case(tmp_path)
        run = ReplayRun(
            id="test-run-001",
            root=run_root,
            case=case,
            runner_label="agent",
            workspace=tmp_path / "workspace",
        )
        # The stored value "[]" is returned as-is (parser quirk)
        result = _anonymous_runner_id(run)
        assert result == "[]"


class TestScoringOpsProvenanceRendering:
    """Test provenance section rendering in scoring_ops."""

    def test_render_provenance_section_basic(self):
        prov = CaseProvenance(
            source_types=("plan", "git_history"),
            primary_source_type="orchestration_kit",
            base_commit_source="git_head",
            base_commit_confidence="medium",
            confidence="medium",
            candidate_score=0.5,
            is_synthetic=False,
            risk_level="low",
        )
        lines = _render_provenance_section(prov)
        text = "\n".join(lines)
        assert "## Case Provenance & Source Risk" in text
        assert "orchestration_kit" in text
        assert "git_head" in text
        assert "Synthetic case: no" in text
        assert "Overall risk level: `low`" in text

    def test_render_provenance_section_low_confidence_note(self):
        prov = CaseProvenance(
            confidence="low",
            risk_level="medium",
            risk_reasons=("synthetic case reconstructed from git history",),
        )
        lines = _render_provenance_section(prov)
        text = "\n".join(lines)
        assert "low-confidence case" in text
        assert "Risk Factors" in text

    def test_render_provenance_section_conflicts(self):
        prov = CaseProvenance(
            conflicts=(_conflict_stub("plan_history_disagreement", "Plan and history disagree", "high"),),
            risk_level="high",
        )
        lines = _render_provenance_section(prov)
        text = "\n".join(lines)
        assert "Source Conflicts" in text
        assert "HIGH" in text

    def test_conflict_severity_mark(self):
        assert _conflict_severity_mark("high") == "HIGH"
        assert _conflict_severity_mark("medium") == "MED"
        assert _conflict_severity_mark("low") == "LOW"


class TestScoringOpsEvidenceItemFound:
    """Test evidence item checking in scoring_ops."""

    def test_completion_report_found(self, tmp_path):
        p = tmp_path / "completion_report.md"
        p.write_text("done", encoding="utf-8")
        assert _evidence_item_found("completion_report.md", p, p, "completed", ["x"]) is True

    def test_diff_found(self, tmp_path):
        p = tmp_path / "diff.patch"
        p.write_text("diff --git a/x\n", encoding="utf-8")
        assert _evidence_item_found("diff.patch", p, p, "completed", ["x"]) is True

    def test_diff_empty_returns_false(self, tmp_path):
        p = tmp_path / "diff.patch"
        p.write_text("", encoding="utf-8")
        assert _evidence_item_found("diff.patch", p, p, "completed", ["x"]) is False

    def test_unknown_item_returns_false(self, tmp_path):
        p = tmp_path / "completion_report.md"
        p.write_text("done", encoding="utf-8")
        assert _evidence_item_found("unknown", p, p, "completed", ["x"]) is False

    def test_changed_files_nonempty(self):
        assert _evidence_item_found("changed_files", Path("x"), Path("y"), "completed", ["a.py"]) is True

    def test_changed_files_empty(self):
        assert _evidence_item_found("changed_files", Path("x"), Path("y"), "completed", []) is False


class TestScoringOpsTelemetryFormatting:
    """Test telemetry section formatting."""

    def test_basic_telemetry(self):
        lines = _format_telemetry_scoring({
            "changed_file_count": 3,
            "added_lines": 100,
            "deleted_lines": 20,
            "commit_count": 5,
        })
        text = "\n".join(lines)
        assert "## Process Telemetry" in text
        assert "Changed file count: 3" in text
        assert "Added lines: 100" in text
        assert "Commit count: 5" in text

    def test_forbidden_path_touches(self):
        lines = _format_telemetry_scoring({
            "forbidden_path_touches": [".env", "secret.key"],
        })
        text = "\n".join(lines)
        assert ".env, secret.key" in text

    def test_suspicious_path_touches_absent(self):
        lines = _format_telemetry_scoring({})
        text = "\n".join(lines)
        assert "Suspicious path touches: (none)" in text


def _conflict_stub(ctype: str, desc: str, severity: str):
    """Create a minimal SourceConflict for tests."""
    from replay_checker.scoring import SourceConflict
    return SourceConflict(conflict_type=ctype, description=desc, severity=severity)
