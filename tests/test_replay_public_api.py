"""Public import compatibility tests for replay_checker.replay facade.

Enumerates and verifies that all known imported symbols from
replay_checker.replay are accessible, preserving backward compatibility.
"""
from __future__ import annotations

import importlib
import types

import pytest


# ---------------------------------------------------------------------------
# Facade symbol inventory — every name imported by first-party code
# ---------------------------------------------------------------------------

# Sources enumerated from:
#   tools/replay.py
#   tools/project.py
#   tests/test_agent_workflow.py
#   tests/test_replay_flow.py
#   tests/test_case_dedupe.py
_FACADE_SYMBOLS = [
    # classes
    "ReplayCase",
    "ReplayRun",
    "Evidence",
    "Telemetry",
    "RunHealth",
    "IntakeConfig",
    "PlanPackage",
    # functions used by tools/replay.py CLI
    "batch_intake",
    "collect_run",
    "compare_case",
    "create_case",
    "discover_plan_packages",
    "doctor_runs",
    "intake",
    "inspect_run_dir",
    "load_case",
    "load_run",
    "prepare_run",
    "report_all_cases",
    "score_run",
    # functions used by tools/project.py
    "validate_case",
    # functions used by tests
    "_build_compare_output",
    "_extract_verification",
    "parse_simple_yaml",
    "_write_simple_yaml",
]


class TestFacadeImportSurface:
    """Every symbol in the facade inventory must be importable."""

    @pytest.mark.parametrize("symbol", _FACADE_SYMBOLS)
    def test_symbol_importable(self, symbol: str):
        mod = importlib.import_module("replay_checker.replay")
        assert hasattr(mod, symbol), f"replay_checker.replay.{symbol} not found"
        attr = getattr(mod, symbol)
        assert attr is not None, f"replay_checker.replay.{symbol} is None"

    def test_replay_module_is_importable(self):
        mod = importlib.import_module("replay_checker.replay")
        assert isinstance(mod, types.ModuleType)

    def test_scoring_module_is_importable(self):
        mod = importlib.import_module("replay_checker.scoring")
        assert isinstance(mod, types.ModuleType)

    def test_evaluation_module_is_importable(self):
        mod = importlib.import_module("replay_checker.evaluation")
        assert isinstance(mod, types.ModuleType)

    def test_linters_module_is_importable(self):
        mod = importlib.import_module("replay_checker.linters")
        assert isinstance(mod, types.ModuleType)

    def test_eval_adapter_module_is_importable(self):
        mod = importlib.import_module("replay_checker.eval_adapter")
        assert isinstance(mod, types.ModuleType)

    def test_sources_module_is_importable(self):
        mod = importlib.import_module("replay_checker.sources")
        assert isinstance(mod, types.ModuleType)

    def test_candidates_module_is_importable(self):
        mod = importlib.import_module("replay_checker.candidates")
        assert isinstance(mod, types.ModuleType)

    def test_case_paths_module_is_importable(self):
        mod = importlib.import_module("replay_checker.case_paths")
        assert isinstance(mod, types.ModuleType)

    def test_core_module_is_importable(self):
        mod = importlib.import_module("replay_checker.core")
        assert isinstance(mod, types.ModuleType)

    def test_wizard_module_is_importable(self):
        mod = importlib.import_module("replay_checker.wizard")
        assert isinstance(mod, types.ModuleType)

    def test_agent_workflow_module_is_importable(self):
        mod = importlib.import_module("replay_checker.agent_workflow")
        assert isinstance(mod, types.ModuleType)

    def test_packages_module_is_importable(self):
        mod = importlib.import_module("replay_checker.packages")
        assert isinstance(mod, types.ModuleType)

    def test_orchestration_module_is_importable(self):
        mod = importlib.import_module("replay_checker.orchestration")
        assert isinstance(mod, types.ModuleType)

    def test_discovery_module_is_importable(self):
        mod = importlib.import_module("replay_checker.discovery")
        assert isinstance(mod, types.ModuleType)

    def test_llm_policy_module_is_importable(self):
        mod = importlib.import_module("replay_checker.llm_policy")
        assert isinstance(mod, types.ModuleType)


class TestKeyTypeExports:
    """Key data classes and enums are accessible from their owning modules."""

    def test_evidence_gate_enum_values(self):
        from replay_checker.scoring import EvidenceGate
        values = {g.value for g in EvidenceGate}
        assert values == {
            "diff_present",
            "completion_report_present",
            "changed_files_present",
            "verification_cited",
            "reference_access_absent",
            "runner_identity_hidden",
        }

    def test_eligibility_tier_enum_values(self):
        from replay_checker.evaluation import EligibilityTier
        values = {t.value for t in EligibilityTier}
        assert values == {
            "invalid",
            "failed",
            "partial",
            "solved",
            "excellent",
            "transformative",
        }

    def test_comparison_label_enum_values(self):
        from replay_checker.evaluation import ComparisonLabel
        values = {c.value for c in ComparisonLabel}
        assert values == {
            "much_better",
            "better",
            "tie",
            "worse",
            "much_worse",
            "incomparable",
        }

    def test_lint_task_and_lint_score_callable(self):
        from replay_checker.linters import lint_task, lint_score
        assert callable(lint_task)
        assert callable(lint_score)

    def test_structural_contribution_has_expected_sublevels(self):
        from replay_checker.evaluation import StructuralContribution
        sc = StructuralContribution(
            local_improvement=5.0,
            class_deletion=3.0,
            problem_reframing=2.0,
        )
        assert sc.total == 10.0
        assert sc.has_any is True
        d = sc.to_dict()
        assert "local_improvement" in d
        assert "class_deletion" in d
        assert "problem_reframing" in d


# ---------------------------------------------------------------------------
# Import-cycle and facade-direction guard
# ---------------------------------------------------------------------------

_OWNERSHIP_MODULES = [
    "replay_checker.scoring_ops",
    "replay_checker.reporting",
    "replay_checker.run_ops",
    "replay_checker.case_intake",
    "replay_checker.case_reconstruction",
    "replay_checker.case_validation",
]


class TestImportCycleGuard:
    """Owner modules must not import the replay facade; every module must be
    importable in a fresh interpreter without raising."""

    @pytest.mark.parametrize("module_name", _OWNERSHIP_MODULES)
    def test_owner_module_importable_independently(self, module_name: str):
        """Each owner module should import successfully in isolation."""
        mod = importlib.import_module(module_name)
        assert isinstance(mod, types.ModuleType)

    @pytest.mark.parametrize("module_name", _OWNERSHIP_MODULES)
    def test_owner_module_does_not_import_facade(self, module_name: str):
        """Lifecycle owner modules must not depend on replay_checker.replay."""
        import sys
        import subprocess
        import textwrap
        from pathlib import Path

        src_dir = str(Path(__file__).resolve().parents[1] / "src")

        script = textwrap.dedent(f"""\
            import ast
            import sys
            import importlib
            sys.path.insert(0, "{src_dir}")

            mod = importlib.import_module("{module_name}")
            source_path = getattr(mod, "__file__", None)
            if source_path is None:
                print("SKIP: no source file")
                sys.exit(0)

            with open(source_path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=source_path)

            violations = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imp_mod = node.module
                    elif isinstance(node, ast.Import):
                        imp_mod = node.names[0].name if node.names else ""
                    else:
                        continue
                    if imp_mod == ".replay" or imp_mod == "replay_checker.replay":
                        violations.append(imp_mod)

            if violations:
                print(f"VIOLATION: {module_name} imports the facade: {{violations}}")
                sys.exit(1)
            print(f"OK: {{len(violations)}} violations")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"{module_name} imports the facade:\n{result.stdout}\n{result.stderr}"
        )
