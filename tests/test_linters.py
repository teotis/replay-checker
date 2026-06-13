"""Tests for linters.py — scoring and task package structural validation."""
from __future__ import annotations

from pathlib import Path

from replay_checker.linters import lint_score, lint_task


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_scoring_package(
    run_root: Path,
    *,
    sections: dict[str, str] | None = None,
    run_yaml: str | None = None,
    include_protocol: bool = True,
) -> Path:
    """Write a scoring_package.md with specified sections into run_root."""
    sections = sections or {}
    run_root.mkdir(parents=True, exist_ok=True)
    lines = ["# Scoring Package: test-run-001", ""]
    if include_protocol:
        lines.append("Protocol Version: replay-checker-score-v2")
        lines.append("")
    for section, body in sections.items():
        lines.append(f"## {section}")
        lines.append(body)
        lines.append("")
    path = run_root / "scoring_package.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if run_yaml is not None:
        (run_root / "run.yaml").write_text(run_yaml, encoding="utf-8")
    return path


def _minimal_scoring_sections(*, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return all required scoring package sections for a clean lint."""
    base = {
        "Required Inputs": "- Evidence YAML: `evidence.yaml`",
        "Rubric Weights": "- Result: 80\n- Process: 20",
        "Invalid Score Conditions": "- Missing evidence",
        "Evidence Gate Assessment": "- [PASS] diff_present\n- Overall: all gates passed",
        "Evidence References": "- Diff: `diff.patch`",
        "Case Provenance & Source Risk": (
            "- Primary source type: `manual`\n"
            "- Base commit source: `none`\n"
            "- Merged confidence: `medium`\n"
            "- Overall risk level: `low`"
        ),
    }
    if extra:
        base.update(extra)
    return base


def _write_task_package(run_root: Path, content: str) -> Path:
    """Write a TASK.md into run_root."""
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "TASK.md"
    path.write_text(content, encoding="utf-8")
    return path


# =========================================================================
# lint_score Tests
# =========================================================================


class TestLintScoreMissingFile:
    def test_missing_scoring_package(self, tmp_path):
        diags = lint_score(tmp_path)
        codes = [d.code for d in diags]
        assert "missing.scoring_package" in codes


class TestLintScoreRequiredSections:
    def test_missing_protocol_version(self, tmp_path):
        sections = _minimal_scoring_sections()
        _write_scoring_package(
            tmp_path / "run", sections=sections, include_protocol=False,
        )
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.protocol_version" in codes

    def test_missing_required_inputs(self, tmp_path):
        sections = _minimal_scoring_sections()
        del sections["Required Inputs"]
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.required_inputs" in codes

    def test_missing_rubric_weights(self, tmp_path):
        sections = _minimal_scoring_sections()
        del sections["Rubric Weights"]
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.rubric_weights" in codes

    def test_missing_invalid_conditions(self, tmp_path):
        sections = _minimal_scoring_sections()
        del sections["Invalid Score Conditions"]
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.invalid_conditions" in codes

    def test_missing_gate_assessment(self, tmp_path):
        sections = _minimal_scoring_sections()
        del sections["Evidence Gate Assessment"]
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.gate_assessment" in codes

    def test_missing_evidence_references(self, tmp_path):
        sections = _minimal_scoring_sections()
        del sections["Evidence References"]
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.evidence_references" in codes

    def test_clean_scoring_package_passes(self, tmp_path):
        _write_scoring_package(tmp_path / "run", sections=_minimal_scoring_sections())
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert not any(c.startswith("missing.") for c in codes)


class TestLintScoreSecurityChecks:
    def test_runner_label_leakage(self, tmp_path):
        run_yaml = "runner_label: agent-alpha\nanonymous_runner_id: anon-123\n"
        sections = _minimal_scoring_sections()
        sections["Required Inputs"] = "- Runner: agent-alpha"
        _write_scoring_package(tmp_path / "run", sections=sections, run_yaml=run_yaml)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "security.runner_label_leak" in codes

    def test_reference_leakage(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Evidence References"] = "- Diff: _reference/diff.patch"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "security.reference_leak" in codes

    def test_raw_log_leakage(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Evidence References"] = 'raw "role": "user" data'
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "security.raw_log_leak" in codes

    def test_backtick_reference_not_leakage(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Required Inputs"] = "Do not access `_reference/` directory"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "security.reference_leak" not in codes


class TestLintScoreCeilingProjection:
    def test_ceiling_section_missing_value(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Score Ceilings"] = "- Reason: some ceiling applied"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.ceiling_value" in codes

    def test_ceiling_section_with_result_ceiling(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Score Ceilings"] = "- Result score ceiling: 0\n  - No diff present"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.ceiling_value" not in codes

    def test_ceiling_section_with_process_ceiling(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Score Ceilings"] = "- Process score ceiling: 0"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.ceiling_value" not in codes

    def test_ceiling_section_with_verification_ceiling(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Score Ceilings"] = "- Verification score ceiling: 0"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.ceiling_value" not in codes


class TestLintScoreProvenance:
    def test_missing_provenance_non_manual(self, tmp_path):
        sections = _minimal_scoring_sections()
        # Remove the default manual provenance section entirely
        del sections["Case Provenance & Source Risk"]
        sections["Required Inputs"] = "- Source type: git_history"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.case_provenance" in codes

    def test_manual_case_no_provenance_required(self, tmp_path):
        sections = _minimal_scoring_sections()
        sections["Case Provenance & Source Risk"] = "- Primary source type: `manual`"
        _write_scoring_package(tmp_path / "run", sections=sections)
        diags = lint_score(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.case_provenance" not in codes


# =========================================================================
# lint_task Tests
# =========================================================================


class TestLintTaskMissingFile:
    def test_missing_task_file(self, tmp_path):
        diags = lint_task(tmp_path)
        codes = [d.code for d in diags]
        assert "missing.task_file" in codes


class TestLintTaskRequiredSections:
    def test_missing_protocol_version(self, tmp_path):
        content = "## Goal\nReplay a historical situation.\n"
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.protocol_version" in codes

    def test_missing_goal_or_source(self, tmp_path):
        content = "Protocol Version: task-v1\nNo goal here.\n"
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.goal_or_source" in codes

    def test_missing_output_contract(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation.\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.output_contract" in codes

    def test_missing_forbidden_access(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation.\n"
            "## Agent Output Contract\nDo stuff.\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.forbidden_access" in codes

    def test_missing_evidence_requirements(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.evidence_requirements" in codes

    def test_missing_completion_report(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "missing.completion_report_schema" in codes

    def test_clean_task_package_passes(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation and produce changes.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
            "## Verification Commands\n- rtk make test\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        errors = [d for d in diags if d.severity == "error"]
        assert not errors

    def test_task_contract_section_is_validated_when_present(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation and produce changes.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
            "## Task Package Contract\n"
            "```yaml\n"
            "id: TP-weak\n"
            "source_skill: complexity-sweep\n"
            "severity: P1\n"
            "evidence: []\n"
            "affected_paths: []\n"
            "root_cause: Missing canonical contract.\n"
            "proposed_change: Make it better.\n"
            "acceptance_criteria:\n"
            "  - Improve quality.\n"
            "verification_commands:\n"
            "  - echo ok\n"
            "agent_capability: []\n"
            "parallel_safety: unknown\n"
            "dependencies: []\n"
            "blocked_by:\n"
            "  - needs credentials\n"
            "handoff_mode: handoff\n"
            "confidence: medium\n"
            "expected_user_value: \n"
            "falsification:\n"
            "  direct_evidence: false\n"
            "  acceptance_observable: false\n"
            "  verification_proves_completion: false\n"
            "```\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "contract.missing_evidence" in codes
        assert "contract.weak_acceptance_criteria" in codes
        assert "contract.weak_verification" in codes
        assert "contract.blocked_handoff" in codes


class TestLintTaskSecurityChecks:
    def test_reference_leak(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation.\n"
            "## Agent Output Contract\nApply `_reference/diff.patch`\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "security.reference_leak" in codes


class TestLintTaskWarnings:
    def test_depth_warning_for_shallow_generated_task(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical situation and produce changes.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
            "## Verification Commands\n- rtk make test\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "depth.missing_situation_context" in codes
        assert "depth.missing_failure_boundaries" in codes
        assert "depth.missing_observable_acceptance" in codes

    def test_depth_warning_applies_to_case_task_file(self, tmp_path):
        case_root = tmp_path / "case"
        case_root.mkdir()
        (case_root / "task.md").write_text(
            "## Goal\nReplay a historical situation and produce changes.\n",
            encoding="utf-8",
        )

        diags = lint_task(case_root)
        codes = [d.code for d in diags]

        assert "depth.missing_situation_context" in codes
        assert "depth.missing_failure_boundaries" in codes
        assert "depth.missing_observable_acceptance" in codes

    def test_template_goal_warning(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nReplay a historical project situation and produce equivalent changes independently.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "content.template_goal" in codes

    def test_short_goal_warning(self, tmp_path):
        content = (
            "Protocol Version: task-v1\n"
            "## Goal\nFix bug.\n"
            "## Agent Output Contract\nDo stuff.\n"
            "## Forbidden Access\nDo not access _reference/.\n"
            "## Evidence Requirements\nProvide diff.\n"
            "## Completion Report\nProvide completion_report.md\n"
        )
        _write_task_package(tmp_path / "run", content)
        diags = lint_task(tmp_path / "run")
        codes = [d.code for d in diags]
        assert "content.goal_too_short" in codes
