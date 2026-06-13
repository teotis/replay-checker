"""Canonical task package contract and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .core import Diagnostic
from .yaml_lite import coerce_bool_int, emit_yaml, parse_yaml_text


VALID_HANDOFF_MODES = {
    "codex-retained",
    "handoff",
    "orchestration",
    "external-assist",
}

VALID_CONFIDENCE = {"low", "medium", "high"}


@dataclass(frozen=True)
class TaskPackageFalsification:
    """Adversarial checks that keep task packages executable and evidence-led."""

    direct_evidence: bool = False
    may_be_style_preference: bool = False
    acceptance_observable: bool = False
    verification_proves_completion: bool = False
    capability_mismatch: bool = False
    path_conflict_risk: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "direct_evidence": self.direct_evidence,
            "may_be_style_preference": self.may_be_style_preference,
            "acceptance_observable": self.acceptance_observable,
            "verification_proves_completion": self.verification_proves_completion,
            "capability_mismatch": self.capability_mismatch,
            "path_conflict_risk": self.path_conflict_risk,
        }


@dataclass(frozen=True)
class TaskPackageContract:
    """Machine-readable contract shared by package generation and validation."""

    id: str
    source_skill: str
    severity: str
    evidence: tuple[str, ...] = ()
    affected_paths: tuple[str, ...] = ()
    root_cause: str = ""
    proposed_change: str = ""
    acceptance_criteria: tuple[str, ...] = ()
    verification_commands: tuple[str, ...] = ()
    agent_capability: tuple[str, ...] = ()
    parallel_safety: str = "unknown"
    dependencies: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    handoff_mode: str = "codex-retained"
    confidence: str = "medium"
    expected_user_value: str = ""
    falsification: TaskPackageFalsification = field(
        default_factory=TaskPackageFalsification
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_skill": self.source_skill,
            "severity": self.severity,
            "evidence": list(self.evidence),
            "affected_paths": list(self.affected_paths),
            "root_cause": self.root_cause,
            "proposed_change": self.proposed_change,
            "acceptance_criteria": list(self.acceptance_criteria),
            "verification_commands": list(self.verification_commands),
            "agent_capability": list(self.agent_capability),
            "parallel_safety": self.parallel_safety,
            "dependencies": list(self.dependencies),
            "blocked_by": list(self.blocked_by),
            "handoff_mode": self.handoff_mode,
            "confidence": self.confidence,
            "expected_user_value": self.expected_user_value,
            "falsification": self.falsification.to_dict(),
        }

    def to_yaml(self) -> str:
        return emit_yaml(self.to_dict())


def default_execution_contract(
    *,
    run_id: str,
    goal: str,
    workspace: str,
    case_task_path: str,
    plan_path: str,
    verification_commands: tuple[str, ...],
) -> TaskPackageContract:
    """Build a conservative contract projection for generated execution packages."""

    acceptance = (
        "The workspace contains a non-empty result diff implementing the task.",
        "completion_report.md records status, changed files, verification, and blockers.",
    )
    evidence = (case_task_path, plan_path)
    return TaskPackageContract(
        id=run_id,
        source_skill="replay_checker",
        severity="P2",
        evidence=evidence,
        affected_paths=(workspace,),
        root_cause="Replay task reconstructed from case and plan evidence.",
        proposed_change=goal,
        acceptance_criteria=acceptance,
        verification_commands=verification_commands,
        agent_capability=("local-code-edit", "shell-verification"),
        parallel_safety="single-workspace",
        dependencies=(),
        blocked_by=(),
        handoff_mode="codex-retained",
        confidence="medium",
        expected_user_value="A routeable, auditable execution package for replay evaluation.",
        falsification=TaskPackageFalsification(
            direct_evidence=True,
            may_be_style_preference=False,
            acceptance_observable=True,
            verification_proves_completion=bool(verification_commands),
            capability_mismatch=False,
            path_conflict_risk=False,
        ),
    )


def default_orchestration_contract(
    *,
    package_id: str,
    description: str,
    allowed_paths: tuple[str, ...],
    dependencies: tuple[str, ...],
    verification_commands: tuple[str, ...],
    is_manual: bool = False,
) -> TaskPackageContract:
    handoff_mode = "external-assist" if is_manual else "orchestration"
    blocked_by = ("manual gate",) if is_manual else ()
    return TaskPackageContract(
        id=package_id,
        source_skill="replay_checker.orchestration",
        severity="P2",
        evidence=(f"packages/{package_id}.md",),
        affected_paths=allowed_paths,
        root_cause="Generated orchestration package needs explicit routing and validation metadata.",
        proposed_change=description,
        acceptance_criteria=(
            "Package status records changed files, verification, blockers, and recovery hints.",
            "Downstream packages can route from dependencies and parallel-safety metadata.",
        ),
        verification_commands=verification_commands,
        agent_capability=("local-project-inspection", "shell-verification"),
        parallel_safety="depends-on-declared-paths",
        dependencies=dependencies,
        blocked_by=blocked_by,
        handoff_mode=handoff_mode,
        confidence="medium",
        expected_user_value="A routeable package with explicit dependencies, verification, and blocker metadata.",
        falsification=TaskPackageFalsification(
            direct_evidence=True,
            may_be_style_preference=False,
            acceptance_observable=True,
            verification_proves_completion=bool(verification_commands),
            capability_mismatch=is_manual,
            path_conflict_risk=False,
        ),
    )


def validate_task_contract(contract: TaskPackageContract) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    if not contract.id:
        diagnostics.append(Diagnostic("error", "contract.missing_id", "task contract missing id"))
    if not contract.source_skill:
        diagnostics.append(Diagnostic("error", "contract.missing_source_skill", "task contract missing source_skill"))
    if not contract.severity:
        diagnostics.append(Diagnostic("error", "contract.missing_severity", "task contract missing severity"))
    if not contract.evidence:
        diagnostics.append(Diagnostic("error", "contract.missing_evidence", "task contract has no evidence"))
    elif any(_references_oracle_path(item) for item in contract.evidence):
        diagnostics.append(Diagnostic("error", "contract.reference_evidence_visible", "task contract exposes _reference/ oracle evidence to an agent-visible package"))
    if not contract.affected_paths:
        diagnostics.append(Diagnostic("error", "contract.missing_affected_paths", "task contract has no affected_paths"))
    elif any(_references_oracle_path(item) for item in contract.affected_paths):
        diagnostics.append(Diagnostic("error", "contract.reference_path_in_scope", "task contract affected_paths includes forbidden _reference/ scope"))
    if not contract.root_cause.strip():
        diagnostics.append(Diagnostic("warning", "contract.missing_root_cause", "task contract has no root_cause"))
    if not contract.proposed_change.strip():
        diagnostics.append(Diagnostic("error", "contract.missing_proposed_change", "task contract has no proposed_change"))
    if not contract.expected_user_value.strip():
        diagnostics.append(Diagnostic("warning", "contract.missing_user_value", "task contract has no expected_user_value"))

    if not contract.acceptance_criteria:
        diagnostics.append(Diagnostic("error", "contract.missing_acceptance", "task contract has no acceptance_criteria"))
    elif any(_weak_acceptance(item) for item in contract.acceptance_criteria):
        diagnostics.append(Diagnostic("warning", "contract.weak_acceptance_criteria", "acceptance criteria are not clearly observable"))

    if not contract.verification_commands:
        diagnostics.append(Diagnostic("warning", "contract.missing_verification", "task contract has no verification_commands"))
    elif any(_weak_verification(cmd) for cmd in contract.verification_commands):
        diagnostics.append(Diagnostic("warning", "contract.weak_verification", "verification command is too weak to prove completion"))

    if not contract.agent_capability:
        diagnostics.append(Diagnostic("warning", "contract.missing_agent_capability", "task contract has no agent_capability"))
    if contract.handoff_mode not in VALID_HANDOFF_MODES:
        diagnostics.append(Diagnostic("error", "contract.invalid_handoff_mode", f"invalid handoff_mode: {contract.handoff_mode}"))
    if contract.confidence not in VALID_CONFIDENCE:
        diagnostics.append(Diagnostic("error", "contract.invalid_confidence", f"invalid confidence: {contract.confidence}"))

    falsification = contract.falsification
    if not falsification.direct_evidence:
        diagnostics.append(Diagnostic("warning", "contract.indirect_evidence", "falsification pass says direct evidence is missing"))
    if falsification.may_be_style_preference:
        diagnostics.append(Diagnostic("warning", "contract.style_preference_risk", "falsification pass says the task may be a style preference rather than observable result work"))
    if not falsification.acceptance_observable:
        diagnostics.append(Diagnostic("warning", "contract.acceptance_not_observable", "falsification pass says acceptance is not observable"))
    if not falsification.verification_proves_completion:
        diagnostics.append(Diagnostic("warning", "contract.verification_not_proving_completion", "falsification pass says verification does not prove completion"))
    if falsification.capability_mismatch:
        diagnostics.append(Diagnostic("error", "contract.capability_mismatch", "task requires capabilities not available to the selected agent route"))
    if falsification.path_conflict_risk:
        diagnostics.append(Diagnostic("warning", "contract.path_conflict_risk", "falsification pass says affected paths may conflict with other packages or forbidden scopes"))
    if contract.blocked_by and contract.handoff_mode in {"handoff", "orchestration"}:
        diagnostics.append(Diagnostic("warning", "contract.blocked_handoff", "task has blockers but is routed for agent handoff/orchestration"))

    return diagnostics


def task_contract_from_yaml(text: str) -> TaskPackageContract:
    data = parse_yaml_text(text, scalar_parser=coerce_bool_int)
    if not isinstance(data, dict):
        raise ValueError("task contract YAML must be a mapping")

    falsification_data = data.get("falsification", {})
    if not isinstance(falsification_data, dict):
        falsification_data = {}
    falsification = TaskPackageFalsification(
        direct_evidence=_bool(falsification_data.get("direct_evidence", False)),
        may_be_style_preference=_bool(falsification_data.get("may_be_style_preference", False)),
        acceptance_observable=_bool(falsification_data.get("acceptance_observable", False)),
        verification_proves_completion=_bool(falsification_data.get("verification_proves_completion", False)),
        capability_mismatch=_bool(falsification_data.get("capability_mismatch", False)),
        path_conflict_risk=_bool(falsification_data.get("path_conflict_risk", False)),
    )

    return TaskPackageContract(
        id=str(data.get("id", "")),
        source_skill=str(data.get("source_skill", "")),
        severity=str(data.get("severity", "")),
        evidence=_tuple_of_str(data.get("evidence", [])),
        affected_paths=_tuple_of_str(data.get("affected_paths", [])),
        root_cause=str(data.get("root_cause", "")),
        proposed_change=str(data.get("proposed_change", "")),
        acceptance_criteria=_tuple_of_str(data.get("acceptance_criteria", [])),
        verification_commands=_tuple_of_str(data.get("verification_commands", [])),
        agent_capability=_tuple_of_str(data.get("agent_capability", [])),
        parallel_safety=str(data.get("parallel_safety", "")),
        dependencies=_tuple_of_str(data.get("dependencies", [])),
        blocked_by=_tuple_of_str(data.get("blocked_by", [])),
        handoff_mode=str(data.get("handoff_mode", "")),
        confidence=str(data.get("confidence", "")),
        expected_user_value=str(data.get("expected_user_value", "")),
        falsification=falsification,
    )


def validate_task_contract_yaml(text: str) -> list[Diagnostic]:
    try:
        contract = task_contract_from_yaml(text)
    except Exception as exc:
        return [
            Diagnostic(
                "error",
                "contract.invalid_yaml",
                f"task contract YAML could not be parsed: {exc}",
            )
        ]
    return validate_task_contract(contract)


def _weak_acceptance(text: str) -> bool:
    lowered = text.strip().lower()
    if len(lowered) < 20:
        return True
    vague_terms = ("improve quality", "make it better", "clean up", "nice to have")
    return any(term in lowered for term in vague_terms)


def _weak_verification(command: str) -> bool:
    lowered = command.strip().lower()
    weak = {"echo ok", "echo 'ok'", 'echo "ok"', "true"}
    if lowered in weak:
        return True
    return lowered.startswith("echo ") and "pytest" not in lowered and "test" not in lowered


def _references_oracle_path(text: str) -> bool:
    normalized = text.replace("\\", "/").lower()
    return "_reference/" in normalized


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item))
    if isinstance(value, tuple):
        return tuple(str(item) for item in value if str(item))
    if value in ("", None):
        return ()
    return (str(value),)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
