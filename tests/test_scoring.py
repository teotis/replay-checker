"""Tests for evidence validity gates and score ceiling logic in scoring.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from replay_checker.scoring import (
    EvidenceGate,
    GateEvaluation,
    GateResult,
    apply_ceiling,
    compute_score_ceilings,
    evaluate_evidence_gates,
    parse_rubric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence_dir(
    root: Path,
    *,
    include_diff: bool = True,
    include_completion: bool = True,
    include_evidence_yaml: bool = True,
    changed_files: list[str] | None = None,
    completion_text: str = "status: completed\n\nAll tests pass.\n",
    evidence_yaml_text: str = "status: completed\nchanged_files:\n  - src/app.py\n",
    runner_label: str | None = None,
    include_reference_dir: bool = False,
    include_oracle_in_completion: bool = False,
) -> Path:
    """Create a minimal run directory with evidence for testing gates."""
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    if include_diff:
        (evidence_dir / "diff.patch").write_text(
            "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n",
            encoding="utf-8",
        )

    if include_evidence_yaml:
        (evidence_dir / "evidence.yaml").write_text(
            evidence_yaml_text, encoding="utf-8"
        )

    if include_completion:
        text = completion_text
        if include_oracle_in_completion:
            text += "\nAccessed _reference/ for validation.\n"
        (root / "completion_report.md").write_text(text, encoding="utf-8")

    if runner_label:
        run_yaml = root / "run.yaml"
        run_yaml.write_text(
            f"runner_label: {runner_label}\nanonymous_runner_id: runner-abc123\n",
            encoding="utf-8",
        )

    if include_reference_dir:
        (root / "_reference").mkdir(parents=True)
        (root / "_reference" / "expected.patch").write_text(
            "ref diff", encoding="utf-8"
        )

    return root


# ---------------------------------------------------------------------------
# Gate evaluation tests
# ---------------------------------------------------------------------------


def test_all_gates_pass(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\n\nVerified with pytest.\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    assert evaluation.all_passed
    assert not evaluation.is_invalid
    assert len(evaluation.results) == 6
    assert all(r.passed for r in evaluation.results)


def test_diff_missing_gate_fails(tmp_path):
    root = _make_evidence_dir(tmp_path, include_diff=False, changed_files=["src/app.py"])
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    diff_result = next(r for r in evaluation.results if r.gate == EvidenceGate.DIFF_PRESENT)
    assert not diff_result.passed
    assert "missing or empty" in diff_result.detail
    assert not evaluation.all_passed


def test_diff_empty_gate_fails(tmp_path):
    root = _make_evidence_dir(tmp_path, changed_files=["src/app.py"])
    (root / "evidence" / "diff.patch").write_text("", encoding="utf-8")
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    diff_result = next(r for r in evaluation.results if r.gate == EvidenceGate.DIFF_PRESENT)
    assert not diff_result.passed


def test_completion_report_missing_gate_fails(tmp_path):
    root = _make_evidence_dir(tmp_path, include_completion=False, changed_files=["src/app.py"])
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    report_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.COMPLETION_REPORT_PRESENT
    )
    assert not report_result.passed
    assert "missing" in report_result.detail


def test_changed_files_empty_gate_fails(tmp_path):
    root = _make_evidence_dir(tmp_path, changed_files=[])
    evaluation = evaluate_evidence_gates(evidence_root=root, changed_files=[])

    files_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.CHANGED_FILES_PRESENT
    )
    assert not files_result.passed
    assert "no changed files" in files_result.detail


def test_verification_not_cited_gate_fails(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\nEverything is fine.\n",
        evidence_yaml_text="status: completed\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    verif_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.VERIFICATION_CITED
    )
    assert not verif_result.passed


def test_verification_cited_in_completion_passes(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\n\nAll tests pass after verification.\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    verif_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.VERIFICATION_CITED
    )
    assert verif_result.passed


def test_verification_cited_in_evidence_yaml_passes(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\nDone.\n",
        evidence_yaml_text="status: completed\nverification: pass\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    verif_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.VERIFICATION_CITED
    )
    assert verif_result.passed


def test_reference_dir_exists_gate_fails(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        include_reference_dir=True,
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    ref_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT
    )
    assert not ref_result.passed
    assert "_reference/" in ref_result.detail


def test_reference_in_completion_report_gate_fails(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        include_oracle_in_completion=True,
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )

    ref_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.REFERENCE_ACCESS_ABSENT
    )
    assert not ref_result.passed
    assert "oracle" in ref_result.detail or "_reference" in ref_result.detail


def test_runner_identity_hidden_passes(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root,
        runner_label="Claude Opus 4",
        changed_files=["src/app.py"],
    )

    identity_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN
    )
    assert identity_result.passed


def test_runner_identity_exposed_gate_fails(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        runner_label="Claude Opus 4",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root,
        runner_label="Claude Opus 4",
        changed_files=["src/app.py"],
    )

    identity_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN
    )
    assert not identity_result.passed
    assert "Claude Opus 4" in identity_result.detail


def test_runner_identity_in_evidence_yaml_gate_fails(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        evidence_yaml_text="status: completed\nrunner_label: MySecretAgent\n",
        runner_label="MySecretAgent",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root,
        runner_label="MySecretAgent",
        changed_files=["src/app.py"],
    )

    identity_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN
    )
    assert not identity_result.passed


# ---------------------------------------------------------------------------
# Score ceiling tests
# ---------------------------------------------------------------------------


def test_no_diff_caps_result_ceiling(tmp_path):
    root = _make_evidence_dir(tmp_path, include_diff=False, changed_files=["src/app.py"])
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.result_ceiling == 0.0
    assert not ceilings.overall_invalid


def test_no_completion_report_caps_process_ceiling(tmp_path):
    root = _make_evidence_dir(
        tmp_path, include_completion=False, changed_files=["src/app.py"]
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.process_ceiling == 0.0


def test_no_verification_caps_verification_ceiling(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\nDone.\n",
        evidence_yaml_text="status: completed\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.verification_ceiling == 0.0


def test_reference_leakage_marks_invalid(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        include_reference_dir=True,
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.overall_invalid
    assert any("_reference" in r for r in ceilings.reasons)


def test_oracle_in_completion_marks_invalid(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        include_oracle_in_completion=True,
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.overall_invalid
    assert any("oracle" in r for r in ceilings.reasons)


def test_runner_identity_exposed_marks_invalid(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        runner_label="Claude Opus 4",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root,
        runner_label="Claude Opus 4",
        changed_files=["src/app.py"],
    )

    identity_result = next(
        r for r in evaluation.results if r.gate == EvidenceGate.RUNNER_IDENTITY_HIDDEN
    )
    assert not identity_result.passed

    ceilings = compute_score_ceilings(evaluation)
    # Runner identity exposed makes the score invalid — attribution must not stand.
    assert ceilings.overall_invalid
    assert any("runner" in r.lower() or "identity" in r.lower() for r in ceilings.reasons)


def test_all_gates_pass_no_ceilings(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        completion_text="status: completed\n\nVerified with pytest.\n",
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.result_ceiling is None
    assert ceilings.process_ceiling is None
    assert ceilings.verification_ceiling is None
    assert not ceilings.overall_invalid


# ---------------------------------------------------------------------------
# apply_ceiling tests
# ---------------------------------------------------------------------------


def test_apply_ceiling_clamps():
    assert apply_ceiling(80, 60) == 60
    assert apply_ceiling(40, 60) == 40
    assert apply_ceiling(50, None) == 50


# ---------------------------------------------------------------------------
# Rubric parsing tests
# ---------------------------------------------------------------------------


def test_parse_rubric_reads_gates_and_ceilings():
    rubric_path = Path(__file__).resolve().parents[1] / "rubrics" / "default.yaml"
    rubric = parse_rubric(rubric_path)

    assert rubric["result_weight"] == 80
    assert rubric["process_weight"] == 20
    assert "diff.patch" in rubric["minimum_evidence"]
    assert "gates" in rubric
    assert "diff_present" in rubric["gates"]["required"]
    assert "ceilings" in rubric
    assert rubric["ceilings"]["no_diff"]["result_ceiling"] == 0
    assert rubric["ceilings"]["reference_leakage"]["overall_invalid"] is True


# ---------------------------------------------------------------------------
# Multiple failure combinations
# ---------------------------------------------------------------------------


def test_multiple_failures_all_reflected(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        include_diff=False,
        include_completion=False,
        changed_files=[],
    )
    evaluation = evaluate_evidence_gates(evidence_root=root, changed_files=[])
    ceilings = compute_score_ceilings(evaluation)

    assert ceilings.result_ceiling == 0.0
    assert ceilings.process_ceiling == 0.0
    assert not ceilings.overall_invalid
    assert not evaluation.all_passed
    assert len([r for r in evaluation.results if not r.passed]) >= 3


def test_invalid_overrides_ceilings(tmp_path):
    root = _make_evidence_dir(
        tmp_path,
        changed_files=["src/app.py"],
        include_diff=False,
        include_reference_dir=True,
    )
    evaluation = evaluate_evidence_gates(
        evidence_root=root, changed_files=["src/app.py"]
    )
    ceilings = compute_score_ceilings(evaluation)

    # Invalid takes precedence over individual ceilings
    assert ceilings.overall_invalid


# --- Linter score tests from 03-package-linters ---

import subprocess
import sys

from replay_checker.replay import (
    collect_run,
    create_case,
    load_case,
    prepare_run,
    score_run,
)
from replay_checker.linters import lint_score


ROOT = Path(__file__).resolve().parents[1]


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True,
    )


def _make_project(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "docs" / "plans" / "demo" / "packages").mkdir(parents=True)
    (project / "src").mkdir()
    (project / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "docs" / "plans" / "demo" / "INDEX.md").write_text(
        "# Demo Plan\n\n## Goal\nChange VALUE to 2.\n", encoding="utf-8",
    )
    (project / "docs" / "plans" / "demo" / "packages" / "01-change.md").write_text(
        "# Change Package\n\nSet `VALUE = 2`.\n", encoding="utf-8",
    )
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)
    base = _git(["rev-parse", "HEAD"], project).stdout.strip()
    return project, base


def _make_valid_scored_run(tmp_path):
    """Create a valid run with a generated scoring package."""
    project, base = _make_project(tmp_path)
    cases_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case = create_case(
        cases_root=cases_root,
        project_path=project,
        plan_path=project / "docs" / "plans" / "demo" / "INDEX.md",
        base_commit=base,
        case_id="test-case",
        verification_commands=["pytest"],
    )
    run = prepare_run(case, runs_root=runs_root, runner_label="Claude Sonnet 4")
    (run.root / "workspace" / "src" / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    (run.root / "completion_report.md").write_text(
        "status: completed\n\nChanged VALUE to 2.\n", encoding="utf-8",
    )
    collect_run(run)
    score_run(run, rubric_path=ROOT / "rubrics" / "default.yaml")
    return run


def test_valid_scoring_package_lints_clean(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    diagnostics = lint_score(run.root)
    assert diagnostics == [], f"expected clean lint, got: {diagnostics}"


def test_missing_scoring_package_detected(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    (run.root / "scoring_package.md").unlink()
    diagnostics = lint_score(run.root)
    assert any("missing scoring_package.md" in d for d in diagnostics)
    assert len(diagnostics) == 1


def test_missing_evidence_gate_detected(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    path = run.root / "scoring_package.md"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace("## Required Inputs", "## Inputs"),
        encoding="utf-8",
    )
    diagnostics = lint_score(run.root)
    assert any("missing evidence gate" in d for d in diagnostics)


def test_missing_score_ceilings_detected(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    path = run.root / "scoring_package.md"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content.replace("## Rubric Weights", "## Weights"),
        encoding="utf-8",
    )
    diagnostics = lint_score(run.root)
    assert any("missing score ceilings" in d for d in diagnostics)


def test_runner_identity_leak_detected(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    path = run.root / "scoring_package.md"
    content = path.read_text(encoding="utf-8")
    path.write_text(
        content + "\nRunner: Claude Sonnet 4\n",
        encoding="utf-8",
    )
    diagnostics = lint_score(run.root)
    assert any("runner label leakage" in d for d in diagnostics)


def test_cli_lint_score_exits_nonzero_for_bad_package(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "case.yaml").write_text("id: bad\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "lint-score",
            "--run",
            str(bad_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "ERROR:" in result.stderr


def test_cli_lint_score_exits_zero_for_valid_package(tmp_path):
    run = _make_valid_scored_run(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "lint-score",
            "--run",
            str(run.root),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Lint clean" in result.stdout
