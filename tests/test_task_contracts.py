"""Tests for canonical task package contracts."""

from __future__ import annotations

from replay_checker.core import Diagnostic
from replay_checker.discovery import DiscoveryPackage
from replay_checker.evaluation import TaskOutcomeEntry
from replay_checker.orchestration import generate_package_doc
from replay_checker.packages import ExecutionPackageRenderer, compile_execution_package
from replay_checker.task_contracts import (
    TaskPackageContract,
    TaskPackageFalsification,
    validate_task_contract,
)


def test_execution_package_renders_machine_readable_contract(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        "# Package\n\n"
        "## Goal\n"
        "Add task package contract support with observable validation.\n",
        encoding="utf-8",
    )
    case_task = tmp_path / "task.md"
    case_task.write_text("## Goal\nImplement the contract.\n", encoding="utf-8")

    pkg = compile_execution_package(
        run_id="run-001",
        case_id="case-001",
        case_task_path=str(case_task),
        plan_path=str(plan),
        workspace=str(tmp_path / "workspace"),
        completion_template_path=str(tmp_path / "completion_report_template.md"),
        verification_commands=("rtk python3 -m pytest tests/test_task_contracts.py -q",),
    )

    assert pkg.task_contract.id == "run-001"
    assert pkg.task_contract.source_skill == "replay_checker"
    assert pkg.task_contract.handoff_mode == "codex-retained"
    assert pkg.task_contract.parallel_safety == "single-workspace"
    assert pkg.task_contract.acceptance_criteria

    rendered = ExecutionPackageRenderer().render(pkg)
    assert "## Task Package Contract" in rendered
    assert "```yaml" in rendered
    assert "handoff_mode: codex-retained" in rendered
    assert "verification_commands:" in rendered


def test_execution_package_compiler_uses_prior_outcomes_to_strengthen_acceptance(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(
        "# Package\n\n"
        "## Goal\n"
        "Improve startup performance with observable evidence.\n",
        encoding="utf-8",
    )
    case_task = tmp_path / "task.md"
    case_task.write_text("## Goal\nImprove startup performance.\n", encoding="utf-8")

    pkg = compile_execution_package(
        run_id="run-002",
        case_id="case-002",
        case_task_path=str(case_task),
        plan_path=str(plan),
        workspace=str(tmp_path / "workspace"),
        completion_template_path=str(tmp_path / "completion_report_template.md"),
        verification_commands=("rtk python3 -m pytest tests/test_perf.py -q",),
        task_outcomes=(
            TaskOutcomeEntry(
                package_id="old-package",
                task_contract_id="old-run",
                outcome="blocked",
                blocked_reason="missing baseline measurement",
                acceptance_gaps=("missing measurable baseline",),
                verification_status="weak",
            ),
        ),
    )

    criteria = "\n".join(pkg.task_contract.acceptance_criteria).lower()
    assert "missing measurable baseline" in criteria
    assert "task-specific completion" in criteria

    rendered = ExecutionPackageRenderer().render(pkg)
    assert "missing measurable baseline" in rendered
    assert "prior outcome feedback" in rendered


def test_validate_task_contract_requires_actionable_execution_fields():
    contract = TaskPackageContract(
        id="TP-1",
        source_skill="complexity-sweep",
        severity="P1",
        evidence=("src/replay_checker/packages.py",),
        affected_paths=("src/replay_checker/packages.py",),
        root_cause="Execution packages do not expose a canonical task contract.",
        proposed_change="Add a shared contract object.",
        acceptance_criteria=("Generated packages include a fenced YAML contract.",),
        verification_commands=("rtk python3 -m pytest tests/test_task_contracts.py -q",),
        agent_capability=("local-code-edit",),
        parallel_safety="isolated-paths",
        dependencies=(),
        blocked_by=(),
        handoff_mode="handoff",
        confidence="high",
        expected_user_value="Task packages become routeable and auditable.",
        falsification=TaskPackageFalsification(
            direct_evidence=True,
            may_be_style_preference=False,
            acceptance_observable=True,
            verification_proves_completion=True,
            capability_mismatch=False,
            path_conflict_risk=False,
        ),
    )

    assert validate_task_contract(contract) == []

    weak = TaskPackageContract(
        id="TP-2",
        source_skill="deep-flow-sweep",
        severity="P1",
        evidence=(),
        affected_paths=(),
        root_cause="",
        proposed_change="Make it better.",
        acceptance_criteria=("Improve quality.",),
        verification_commands=("echo ok",),
        agent_capability=(),
        parallel_safety="unknown",
        dependencies=(),
        blocked_by=("needs production credentials",),
        handoff_mode="handoff",
        confidence="medium",
        expected_user_value="",
        falsification=TaskPackageFalsification(
            direct_evidence=False,
            acceptance_observable=False,
            verification_proves_completion=False,
        ),
    )

    diagnostics: list[Diagnostic] = validate_task_contract(weak)
    codes = {d.code for d in diagnostics}
    assert "contract.missing_evidence" in codes
    assert "contract.missing_affected_paths" in codes
    assert "contract.weak_acceptance_criteria" in codes
    assert "contract.weak_verification" in codes
    assert "contract.blocked_handoff" in codes


def test_validate_task_contract_flags_adversarial_hardening_risks():
    contract = TaskPackageContract(
        id="TP-hardening",
        source_skill="deep-flow-sweep",
        severity="P0",
        evidence=("cases/inventory/demo/case-1/_reference/diff.patch",),
        affected_paths=("src/replay_checker/run_ops.py",),
        root_cause="The package can leak oracle evidence.",
        proposed_change="Harden agent-visible task evidence.",
        acceptance_criteria=("Agent-visible files do not expose reference evidence.",),
        verification_commands=("rtk python3 -m pytest tests/test_linters.py -q",),
        agent_capability=("local-code-edit",),
        parallel_safety="isolated-paths",
        dependencies=(),
        blocked_by=(),
        handoff_mode="handoff",
        confidence="high",
        expected_user_value="Execution packages preserve reference isolation.",
        falsification=TaskPackageFalsification(
            direct_evidence=True,
            may_be_style_preference=True,
            acceptance_observable=True,
            verification_proves_completion=True,
            capability_mismatch=False,
            path_conflict_risk=True,
        ),
    )

    codes = {d.code for d in validate_task_contract(contract)}

    assert "contract.reference_evidence_visible" in codes
    assert "contract.style_preference_risk" in codes
    assert "contract.path_conflict_risk" in codes


def test_orchestration_package_doc_renders_task_contract():
    pkg = DiscoveryPackage(
        package_id="01-plan-scan",
        description="Scan plans and produce replay candidates.",
        allowed_paths=("docs/", "plans/"),
        forbidden_paths=("_reference/",),
        dependencies=(),
        verification_commands=("rtk find docs -name '*.md'",),
    )

    rendered = generate_package_doc(pkg)

    assert "## Task Package Contract" in rendered
    assert "id: 01-plan-scan" in rendered
    assert "handoff_mode: orchestration" in rendered
    assert "affected_paths:" in rendered
    assert "verification_commands:" in rendered
