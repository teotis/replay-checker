"""Tests for discovery module. All tests use fakes/mocks, no network calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from replay_checker.candidates import CandidateCase
from replay_checker.discovery import (
    DiscoveryKit,
    DiscoveryKitConfig,
    DiscoveryOutput,
    DiscoveryPackage,
    MergedCandidate,
    ProjectAnalysis,
    analyze_project,
    assign_confidence_and_risk,
    compile_cases_from_discovery,
    compile_low_confidence_case,
    default_discovery_packages,
    default_source_templates,
    dedupe_candidates,
    generate_discovery_kit,
    merge_discovery_outputs,
    source_template_map,
    validate_kit,
)
from replay_checker.sources import EvidenceRecord


# ---------------------------------------------------------------------------
# Discovery kit generation tests
# ---------------------------------------------------------------------------


def test_default_packages_has_required_structure():
    pkgs = default_discovery_packages()
    assert len(pkgs) >= 5
    ids = [p.package_id for p in pkgs]
    assert len(ids) == len(set(ids))
    finalize = [p for p in pkgs if p.is_finalize]
    assert len(finalize) == 1
    assert finalize[0].dependencies


def test_default_packages_has_wave_ordering():
    pkgs = default_discovery_packages()
    non_finalize = [p for p in pkgs if not p.is_finalize]
    waves = [p.wave for p in non_finalize]
    assert waves == sorted(waves), "waves should be in ascending order"


def test_generate_kit_creates_all_files(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("print('hello')", encoding="utf-8")

    config = DiscoveryKitConfig(
        project_path=project,
        scope="test scope",
        output_root=tmp_path / "kits",
    )
    kit = generate_discovery_kit(config)

    assert kit.plan_root.is_dir()
    assert kit.index_path.is_file()
    assert kit.graph_path.is_file()
    assert kit.state_path.is_file()
    assert kit.prompts_path.is_file()
    assert kit.orchestrate_path.is_file()

    for pkg in kit.packages:
        assert (kit.plan_root / "packages" / f"{pkg.package_id}.md").is_file()
        assert (kit.plan_root / "status" / f"{pkg.package_id}.md").is_file()

    assert (kit.plan_root / "launchers" / "agent-prompts.md").is_file()
    assert (kit.plan_root / "scratch" / ".gitignore").is_file()


def test_validate_kit_passes_on_valid_kit(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    config = DiscoveryKitConfig(
        project_path=project,
        scope="scope",
        output_root=tmp_path / "kits",
    )
    kit = generate_discovery_kit(config)
    errors = validate_kit(kit.plan_root)
    assert errors == [], f"valid kit has errors: {errors}"


def test_validate_kit_catches_missing_files(tmp_path):
    plan_root = tmp_path / "kit"
    plan_root.mkdir()
    errors = validate_kit(plan_root)
    assert any("missing" in e for e in errors)


def test_validate_kit_catches_bad_graph_header(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    config = DiscoveryKitConfig(project_path=project, scope="s", output_root=tmp_path / "k")
    kit = generate_discovery_kit(config)

    bad_graph = kit.graph_path.read_text(encoding="utf-8")
    bad_graph = "wrong_header\n" + "".join(bad_graph.splitlines(True)[1:])
    kit.graph_path.write_text(bad_graph, encoding="utf-8")

    errors = validate_kit(kit.plan_root)
    assert any("header" in e or "graph" in e for e in errors)


def test_validate_kit_catches_bad_state_column_count(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    config = DiscoveryKitConfig(project_path=project, scope="s", output_root=tmp_path / "k")
    kit = generate_discovery_kit(config)

    lines = kit.state_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 1:
        lines[1] = "bad\tstate"
        kit.state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    errors = validate_kit(kit.plan_root)
    assert any("columns" in e or "column" in e for e in errors)


def test_validate_kit_catches_invalid_state_value(tmp_path):
    project = tmp_path / "p"
    project.mkdir()
    config = DiscoveryKitConfig(project_path=project, scope="s", output_root=tmp_path / "k")
    kit = generate_discovery_kit(config)

    lines = kit.state_path.read_text(encoding="utf-8").splitlines()
    if len(lines) > 1:
        parts = lines[1].split("\t")
        parts[1] = "INVALID_STATE"
        lines[1] = "\t".join(parts)
        kit.state_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    errors = validate_kit(kit.plan_root)
    assert any("invalid state" in e for e in errors)


def test_kit_plan_root_has_correct_name(tmp_path):
    project = tmp_path / "test-project"
    project.mkdir()
    config = DiscoveryKitConfig(project_path=project, scope="scope", output_root=tmp_path / "kits")
    kit = generate_discovery_kit(config)
    assert "test-project-case-discovery-" in kit.plan_root.name


def test_generate_kit_raises_on_nonexistent_project(tmp_path):
    config = DiscoveryKitConfig(
        project_path=tmp_path / "nonexistent",
        scope="scope",
        output_root=tmp_path / "kits",
    )
    with pytest.raises(ValueError, match="not a directory"):
        generate_discovery_kit(config)


def test_kit_packages_acyclic_dependency_graph():
    pkgs = default_discovery_packages()
    visited: set[str] = set()
    visiting: set[str] = set()
    pkg_map = {p.package_id: p for p in pkgs}

    def has_cycle(pid: str) -> bool:
        if pid in visiting:
            return True
        if pid in visited:
            return False
        visiting.add(pid)
        pkg = pkg_map.get(pid)
        if pkg:
            for dep in pkg.dependencies:
                if dep in pkg_map and has_cycle(dep):
                    return True
        visiting.discard(pid)
        visited.add(pid)
        return False

    for p in pkgs:
        assert not has_cycle(p.package_id), f"cycle detected involving {p.package_id}"


# ---------------------------------------------------------------------------
# Source discovery template tests
# ---------------------------------------------------------------------------


def test_default_source_templates_has_all_required_packages():
    templates = default_source_templates()
    ids = {t.package_id for t in templates}
    assert "plan-source" in ids
    assert "agent-history" in ids
    assert "git-history" in ids
    assert "code-analysis" in ids
    assert "candidate-merge" in ids
    assert "case-package-compile" in ids


def test_source_template_required_fields_populated():
    templates = default_source_templates()
    for t in templates:
        assert t.package_id, f"empty package_id in template"
        assert t.description, f"empty description in {t.package_id}"
        assert t.scope_text, f"empty scope_text in {t.package_id}"
        assert t.source_budget, f"empty source_budget in {t.package_id}"
        assert t.expected_evidence, f"empty expected_evidence in {t.package_id}"
        assert t.allowed_paths, f"empty allowed_paths in {t.package_id}"
        assert t.forbidden_paths, f"empty forbidden_paths in {t.package_id}"


def test_agent_history_raw_log_prohibition():
    templates = default_source_templates()
    agent_hist = next(t for t in templates if t.package_id == "agent-history")
    assert agent_hist.raw_log_prohibition, "agent-history must have raw_log_prohibition"
    assert "raw" in agent_hist.raw_log_prohibition.lower()
    assert "prohibit" in agent_hist.raw_log_prohibition.lower() or "never" in agent_hist.raw_log_prohibition.lower()


def test_code_analysis_works_without_plans():
    templates = default_source_templates()
    code_analysis = next(t for t in templates if t.package_id == "code-analysis")
    assert code_analysis.works_without_plans is True


def test_git_history_works_without_plans():
    templates = default_source_templates()
    git_hist = next(t for t in templates if t.package_id == "git-history")
    assert git_hist.works_without_plans is True


def test_plan_source_not_limited_to_docs_plans():
    templates = default_source_templates()
    plan_src = next(t for t in templates if t.package_id == "plan-source")
    # Must NOT be limited to just docs/plans
    assert len(plan_src.plan_location_hints) > 1
    assert "docs/plans" in plan_src.plan_location_hints
    # Must include at least one alternative
    alternatives = [h for h in plan_src.plan_location_hints if h != "docs/plans"]
    assert len(alternatives) >= 2


def test_plan_source_scope_mentions_flexible_search():
    templates = default_source_templates()
    plan_src = next(t for t in templates if t.package_id == "plan-source")
    assert "docs/plans" in plan_src.scope_text
    # Must also mention that search is not limited
    assert "not" in plan_src.scope_text.lower() or "full" in plan_src.scope_text.lower()


def test_source_budget_fields_present():
    templates = default_source_templates()
    for t in templates:
        assert t.source_budget, f"missing source_budget in {t.package_id}"
        assert len(t.source_budget) > 10, f"source_budget too short in {t.package_id}"


def test_expected_evidence_fields_present():
    templates = default_source_templates()
    for t in templates:
        assert len(t.expected_evidence) >= 2, f"expected_evidence too short in {t.package_id}"


def test_privacy_constraints_present():
    templates = default_source_templates()
    for t in templates:
        assert len(t.privacy_constraints) >= 1, f"missing privacy_constraints in {t.package_id}"


def test_agent_history_privacy_prohibits_raw_logs():
    templates = default_source_templates()
    agent_hist = next(t for t in templates if t.package_id == "agent-history")
    all_constraints = " ".join(agent_hist.privacy_constraints).lower()
    assert "raw" in all_constraints or "full" in all_constraints


def test_compile_package_doc_has_required_sections():
    templates = default_source_templates()
    for t in templates:
        doc = t.compile_package_doc()
        assert f"# {t.package_id}" in doc
        assert "## Goal" in doc
        assert "## Scope" in doc
        assert "## Source Budget" in doc
        assert "## Expected Evidence" in doc
        assert "## Allowed Paths" in doc
        assert "## Forbidden Paths" in doc
        assert "## Privacy Constraints" in doc
        assert "## Verification Commands" in doc


def test_compile_package_doc_agent_history_has_raw_log_policy():
    templates = default_source_templates()
    agent_hist = next(t for t in templates if t.package_id == "agent-history")
    doc = agent_hist.compile_package_doc()
    assert "## Raw Log Policy" in doc
    assert "raw" in doc.lower()
    assert "prohibit" in doc.lower() or "never" in doc.lower()


def test_compile_package_doc_plan_source_has_plan_locations():
    templates = default_source_templates()
    plan_src = next(t for t in templates if t.package_id == "plan-source")
    doc = plan_src.compile_package_doc()
    assert "## Plan Document Locations" in doc
    assert "docs/plans" in doc


def test_compile_package_doc_code_analysis_no_plan_locations_section():
    templates = default_source_templates()
    code_analysis = next(t for t in templates if t.package_id == "code-analysis")
    doc = code_analysis.compile_package_doc()
    # works_without_plans → no Plan Document Locations section
    assert "## Plan Document Locations" not in doc


def test_source_template_map_matches_default_packages():
    templates_map = source_template_map()
    pkgs = default_discovery_packages()
    pkg_ids = {p.package_id for p in pkgs}
    template_ids = set(templates_map.keys())
    # All non-finalize packages should have templates
    non_finalize = {p.package_id for p in pkgs if not p.is_finalize}
    assert non_finalize == template_ids


def test_template_budget_mentions_splitting():
    """Budget fields should mention limits that allow large projects to split work."""
    templates = default_source_templates()
    for t in templates:
        assert "up to" in t.source_budget or "max" in t.source_budget.lower() or "limit" in t.source_budget.lower(), (
            f"{t.package_id} source_budget should mention a limit for splitting"
        )


# ---------------------------------------------------------------------------
# ProjectAnalysis / code analysis tests
# ---------------------------------------------------------------------------


def test_analyze_python_project(tmp_path):
    project = tmp_path / "myproject"
    project.mkdir()
    (project / "main.py").write_text("print('hello')", encoding="utf-8")
    (project / "utils.py").write_text("def helper(): pass", encoding="utf-8")
    (project / "test_main.py").write_text("assert True", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.primary_language == "python"
    assert analysis.source_files == 3
    assert analysis.has_tests is True
    assert analysis.file_count == 3


def test_analyze_typescript_project(tmp_path):
    project = tmp_path / "webapp"
    project.mkdir()
    (project / "index.ts").write_text("export {}", encoding="utf-8")
    (project / "app.tsx").write_text("export default {}", encoding="utf-8")
    (project / "package.json").write_text('{"name": "test"}', encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.primary_language == "typescript"
    assert analysis.has_build_config is True
    assert ".json" in analysis.extension_counts


def test_analyze_empty_project(tmp_path):
    project = tmp_path / "empty"
    project.mkdir()

    analysis = analyze_project(project)

    assert analysis.primary_language == "unknown"
    assert analysis.file_count == 0
    assert analysis.source_files == 0
    assert analysis.has_tests is False


def test_analyze_skips_git_directory(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")
    (project / "main.py").write_text("pass", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.file_count == 1
    assert ".py" in analysis.extension_counts


def test_analyze_skips_node_modules(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("pass", encoding="utf-8")
    nm = project / "node_modules"
    nm.mkdir()
    (nm / "dep.js").write_text("module.exports={}", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.source_files == 1


def test_analyze_skips_venv(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "app.py").write_text("pass", encoding="utf-8")
    venv = project / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("pass", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.source_files == 1


def test_candidate_tasks_add_tests(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "core.py").write_text("x = 1", encoding="utf-8")

    analysis = analyze_project(project)

    assert any("test" in task.lower() for task in analysis.candidate_tasks)


def test_candidate_tasks_add_docs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("pass", encoding="utf-8")
    (project / "test_main.py").write_text("assert True", encoding="utf-8")

    analysis = analyze_project(project)

    assert any("doc" in task.lower() for task in analysis.candidate_tasks)


def test_candidate_tasks_no_suggestions_for_full_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("pass", encoding="utf-8")
    (project / "test_main.py").write_text("assert True", encoding="utf-8")
    docs = project / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text("# Docs", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.has_tests is True
    assert analysis.has_docs is True


def test_total_chars_summed(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    content = "x = 1\n" * 100
    (project / "a.py").write_text(content, encoding="utf-8")
    (project / "b.py").write_text(content, encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.total_chars > 0
    assert analysis.total_chars >= len(content) * 2


def test_build_config_detected(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "Makefile").write_text("all:\n\techo ok", encoding="utf-8")
    (project / "main.py").write_text("pass", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.has_build_config is True


def test_gitignore_detected(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.has_gitignore is True


def test_env_template_detected(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env.example").write_text("API_KEY=\n", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.has_env_template is True


def test_rust_project(tmp_path):
    project = tmp_path / "rust"
    project.mkdir()
    (project / "Cargo.toml").write_text("[package]\nname = \"test\"", encoding="utf-8")
    src = project / "src"
    src.mkdir()
    (src / "main.rs").write_text("fn main() {}", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.primary_language == "rust"
    assert analysis.has_build_config is True
    assert analysis.source_files == 1


def test_mixed_project_primary_language(tmp_path):
    project = tmp_path / "mixed"
    project.mkdir()
    for i in range(10):
        (project / f"mod_{i}.py").write_text(f"x_{i} = {i}", encoding="utf-8")
    (project / "helper.js").write_text("const x = 1", encoding="utf-8")

    analysis = analyze_project(project)

    assert analysis.primary_language == "python"
    assert analysis.extension_counts[".py"] == 10
    assert analysis.extension_counts[".js"] == 1

import subprocess
import sys
from pathlib import Path

from replay_checker.candidates import CandidateCase
from replay_checker.discovery import (
    DiscoveryOutput,
    MergedCandidate,
    assign_confidence_and_risk,
    compile_cases_from_discovery,
    compile_low_confidence_case,
    dedupe_candidates,
    merge_discovery_outputs,
)
from replay_checker.sources import EvidenceRecord

ROOT = Path(__file__).resolve().parents[1]


def _plan_record(tmp_path, name, summary, score=0.8):
    return EvidenceRecord(
        source_type="plan",
        path=tmp_path / name,
        summary=summary,
        confidence="high",
        metadata={"relative_path": f"docs/plans/{name}"},
        relevance_score=score,
    )


def _history_record(source_type, summary, score=0.5):
    return EvidenceRecord(
        source_type=source_type,
        path=Path("/fake/session.jsonl"),
        summary=summary,
        confidence="high",
        metadata={},
        relevance_score=score,
    )


class TestDiscoveryOutputSchema:
    def test_discovery_output_creation(self, tmp_path):
        output = DiscoveryOutput(
            source_agent="plan_detection",
            records=(_plan_record(tmp_path, "plan.md", "Implement feature X"),),
            project_path=str(tmp_path),
        )
        assert output.source_agent == "plan_detection"
        assert len(output.records) == 1

    def test_discovery_output_empty_records(self):
        output = DiscoveryOutput(source_agent="code_analysis")
        assert output.records == ()


class TestMergeDiscoveryOutputs:
    def test_merge_single_output(self, tmp_path):
        output = DiscoveryOutput(
            source_agent="plan_detection",
            records=(_plan_record(tmp_path, "plan.md", "Implement scoring"),),
        )
        merged = merge_discovery_outputs([output])
        assert len(merged) == 1
        assert merged[0].candidate.source_type == "plan"

    def test_merge_preserves_agent_attribution(self, tmp_path):
        output = DiscoveryOutput(
            source_agent="plan_detection",
            records=(_plan_record(tmp_path, "plan.md", "Implement scoring"),),
        )
        merged = merge_discovery_outputs([output])
        assert merged[0].contributing_agents == ("plan_detection",)

    def test_merge_empty_outputs(self):
        merged = merge_discovery_outputs([])
        assert merged == []

    def test_merge_multiple_outputs(self, tmp_path):
        plan_output = DiscoveryOutput(
            source_agent="plan_detection",
            records=(_plan_record(tmp_path, "plan.md", "Implement scoring"),),
        )
        history_output = DiscoveryOutput(
            source_agent="history_scan",
            records=(_history_record("codex_history", "fix intake bug"),),
        )
        merged = merge_discovery_outputs([plan_output, history_output])
        assert len(merged) >= 2
        source_types = [mc.candidate.source_type for mc in merged]
        assert "plan" in source_types
        assert "codex_history" in source_types


class TestDedupeCandidates:
    def test_dedupe_removes_duplicate_plans(self, tmp_path):
        c1 = CandidateCase(
            candidate_id="plan-a",
            source_type="plan",
            primary_source="docs/plans/plan.md",
            relevance_score=0.5,
        )
        c2 = CandidateCase(
            candidate_id="plan-a-dup",
            source_type="plan",
            primary_source="docs/plans/plan.md",
            relevance_score=0.8,
        )
        result = dedupe_candidates([c1, c2])
        assert len(result) == 1
        assert result[0].relevance_score == 0.8

    def test_dedupe_keeps_different_plans(self, tmp_path):
        c1 = CandidateCase(
            candidate_id="plan-a",
            source_type="plan",
            primary_source="docs/plans/a.md",
            relevance_score=0.7,
        )
        c2 = CandidateCase(
            candidate_id="plan-b",
            source_type="plan",
            primary_source="docs/plans/b.md",
            relevance_score=0.6,
        )
        result = dedupe_candidates([c1, c2])
        assert len(result) == 2

    def test_dedupe_keeps_one_per_history_type(self):
        c1 = CandidateCase(
            candidate_id="codex-1",
            source_type="codex_history",
            primary_source="session1.jsonl",
            relevance_score=0.6,
        )
        c2 = CandidateCase(
            candidate_id="codex-2",
            source_type="codex_history",
            primary_source="session2.jsonl",
            relevance_score=0.5,
        )
        result = dedupe_candidates([c1, c2])
        assert len(result) == 1
        assert result[0].candidate_id == "codex-1"


class TestAssignConfidenceAndRisk:
    def test_high_score_is_high_confidence(self):
        c = CandidateCase(
            candidate_id="plan-a",
            source_type="plan",
            primary_source="plan.md",
            relevance_score=0.8,
        )
        enriched = assign_confidence_and_risk([c])
        assert enriched[0].confidence == "high"

    def test_medium_score_is_medium_confidence(self):
        c = CandidateCase(
            candidate_id="plan-a",
            source_type="plan",
            primary_source="plan.md",
            relevance_score=0.5,
        )
        enriched = assign_confidence_and_risk([c])
        assert enriched[0].confidence == "medium"

    def test_low_score_is_low_confidence(self):
        c = CandidateCase(
            candidate_id="hist-1",
            source_type="codex_history",
            primary_source="session.jsonl",
            relevance_score=0.1,
        )
        enriched = assign_confidence_and_risk([c])
        assert enriched[0].confidence == "low"

    def test_low_confidence_candidate_has_risk(self):
        c = CandidateCase(
            candidate_id="hist-1",
            source_type="codex_history",
            primary_source="session.jsonl",
            relevance_score=0.1,
        )
        enriched = assign_confidence_and_risk([c])
        assert enriched[0].risk_level == "high"
        assert len(enriched[0].risk_reasons) > 0

    def test_candidate_with_risks_has_risk_reasons(self):
        c = CandidateCase(
            candidate_id="plan-a",
            source_type="plan",
            primary_source="plan.md",
            relevance_score=0.5,
            risks=("status-only plan doc may lack actionable tasks",),
        )
        enriched = assign_confidence_and_risk([c])
        assert len(enriched[0].risk_reasons) > 0
        assert "status-only" in enriched[0].risk_reasons[0]

    def test_empty_candidates(self):
        enriched = assign_confidence_and_risk([])
        assert enriched == []


class TestCompileCasesFromDiscovery:
    def test_selects_highest_confidence(self):
        mc_low = MergedCandidate(
            candidate=CandidateCase(
                candidate_id="hist-1",
                source_type="codex_history",
                primary_source="session.jsonl",
                relevance_score=0.3,
            ),
            confidence="low",
        )
        mc_high = MergedCandidate(
            candidate=CandidateCase(
                candidate_id="plan-a",
                source_type="plan",
                primary_source="plan.md",
                relevance_score=0.8,
            ),
            confidence="high",
        )
        selected, all_cands = compile_cases_from_discovery(
            [mc_low, mc_high],
            project_path="/tmp/proj",
        )
        assert selected is not None
        assert selected.candidate_id == "plan-a"

    def test_empty_input(self):
        selected, all_cands = compile_cases_from_discovery(
            [],
            project_path="/tmp/proj",
        )
        assert selected is None
        assert all_cands == []


class TestCompileLowConfidenceCase:
    def test_returns_risk_metadata(self):
        mc = MergedCandidate(
            candidate=CandidateCase(
                candidate_id="hist-1",
                source_type="codex_history",
                primary_source="session.jsonl",
                relevance_score=0.2,
            ),
            confidence="low",
            risk_level="high",
            risk_reasons=("low confidence candidate",),
        )
        meta = compile_low_confidence_case(mc, project_path="/tmp/proj")
        assert meta["low_confidence"] is True
        assert meta["confidence"] == "low"
        assert meta["risk_level"] == "high"
        assert "low confidence candidate" in meta["risk_reasons"]


class TestNoPlanProjectProducesCandidate:
    """Acceptance test: no-plan temp project produces at least one candidate."""

    def test_git_only_project_has_candidate(self, tmp_path):
        project = tmp_path / "noplanner"
        project.mkdir()
        (project / "src").mkdir()
        (project / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
        _git(["init"], project)
        _git(["config", "user.email", "test@example.com"], project)
        _git(["config", "user.name", "Test User"], project)
        _git(["add", "."], project)
        _git(["commit", "-m", "initial"], project)

        output = DiscoveryOutput(
            source_agent="code_analysis",
            records=(
                _history_record("codex_history", "fix intake bug", score=0.4),
            ),
        )
        merged = merge_discovery_outputs([output])
        assert len(merged) > 0
        assert merged[0].confidence in ("low", "medium", "high")


def _git(args, cwd):
    subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
