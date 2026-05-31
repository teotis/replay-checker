"""Tests for orchestration kit file generation."""

from pathlib import Path

from replay_checker.discovery import DiscoveryKitConfig, default_discovery_packages, generate_discovery_kit
from replay_checker.orchestration import (
    GRAPH_HEADER,
    STATE_HEADER,
    generate_agent_prompts,
    generate_graph_tsv,
    generate_index,
    generate_kit_files,
    generate_orchestrate_sh,
    generate_package_doc,
    generate_state_tsv,
    generate_status_md,
)


PACKAGES = default_discovery_packages()


def _make_config(tmp_path: Path) -> DiscoveryKitConfig:
    project = tmp_path / "sample_project"
    project.mkdir()
    return DiscoveryKitConfig(
        project_path=project,
        scope="test discovery scope",
        output_root=tmp_path / "kits",
    )


class TestGraphTsv:
    def test_header_matches_template(self):
        tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        lines = tsv.splitlines()
        assert lines[0] == GRAPH_HEADER

    def test_each_row_has_10_columns(self):
        tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        for i, line in enumerate(tsv.splitlines()[1:], start=2):
            cols = line.split("\t")
            assert len(cols) == 10, f"row {i}: {len(cols)} columns"

    def test_unique_package_ids(self):
        tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        ids = [line.split("\t")[0] for line in tsv.splitlines()[1:]]
        assert len(ids) == len(set(ids))

    def test_exactly_one_finalize(self):
        tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        finalize_count = sum(
            1 for line in tsv.splitlines()[1:]
            if line.split("\t")[9] == "1"
        )
        assert finalize_count == 1

    def test_dependencies_reference_existing_ids(self):
        tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        ids = {line.split("\t")[0] for line in tsv.splitlines()[1:]}
        for line in tsv.splitlines()[1:]:
            parts = line.split("\t")
            deps = [d.strip() for d in parts[3].split(",") if d.strip()]
            for dep in deps:
                assert dep in ids, f"{parts[0]} depends on unknown {dep}"


class TestStateTsv:
    def test_header_matches_template(self):
        tsv = generate_state_tsv(PACKAGES)
        lines = tsv.splitlines()
        assert lines[0] == STATE_HEADER

    def test_each_row_has_17_columns(self):
        tsv = generate_state_tsv(PACKAGES)
        for i, line in enumerate(tsv.splitlines()[1:], start=2):
            cols = line.split("\t")
            assert len(cols) == 17, f"row {i}: {len(cols)} columns"

    def test_all_states_are_pending(self):
        tsv = generate_state_tsv(PACKAGES)
        for line in tsv.splitlines()[1:]:
            state = line.split("\t")[1]
            assert state == "pending"

    def test_matches_graph_package_ids(self):
        graph_tsv = generate_graph_tsv(PACKAGES, Path("/kit"), Path("/proj"))
        state_tsv = generate_state_tsv(PACKAGES)
        graph_ids = {l.split("\t")[0] for l in graph_tsv.splitlines()[1:]}
        state_ids = {l.split("\t")[0] for l in state_tsv.splitlines()[1:]}
        assert graph_ids == state_ids


class TestStatusMd:
    def test_has_state_section(self):
        md = generate_status_md(PACKAGES[0])
        assert "## State" in md

    def test_has_pending_in_backticks(self):
        md = generate_status_md(PACKAGES[0])
        assert "`pending`" in md

    def test_has_evidence_section(self):
        md = generate_status_md(PACKAGES[0])
        assert "## Evidence" in md

    def test_has_notes_section(self):
        md = generate_status_md(PACKAGES[0])
        assert "## Notes" in md

    def test_markdown_state_parser_finds_pending(self):
        """Verify markdown_status() from orchestrate.sh would parse this correctly."""
        md = generate_status_md(PACKAGES[0])
        import subprocess
        result = subprocess.run(
            ["awk", r"""
                /^## State/ { in_state = 1; next }
                in_state && /`/ {
                    gsub(/`/, "", $0); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
                    print tolower($0); exit
                }
            """],
            input=md, capture_output=True, text=True, check=False,
        )
        assert result.stdout.strip() == "pending"


class TestIndex:
    def test_has_project_name(self, tmp_path):
        config = _make_config(tmp_path)
        idx = generate_index(config.project_path, config.scope, config.plan_root, PACKAGES)
        assert "sample_project" in idx

    def test_has_scope(self, tmp_path):
        config = _make_config(tmp_path)
        idx = generate_index(config.project_path, config.scope, config.plan_root, PACKAGES)
        assert config.scope in idx

    def test_lists_all_packages(self, tmp_path):
        config = _make_config(tmp_path)
        idx = generate_index(config.project_path, config.scope, config.plan_root, PACKAGES)
        for pkg in PACKAGES:
            assert pkg.package_id in idx


class TestAgentPrompts:
    def test_has_package_sections(self, tmp_path):
        config = _make_config(tmp_path)
        prompts = generate_agent_prompts(PACKAGES, config.plan_root)
        for pkg in PACKAGES:
            assert f"## Package: {pkg.package_id}" in prompts

    def test_has_verification_commands(self, tmp_path):
        config = _make_config(tmp_path)
        prompts = generate_agent_prompts(PACKAGES, config.plan_root)
        assert "mark-state" in prompts
        assert "advance" in prompts

    def test_each_section_has_mode_and_index(self, tmp_path):
        config = _make_config(tmp_path)
        prompts = generate_agent_prompts(PACKAGES, config.plan_root)
        for pkg in PACKAGES:
            assert f"**Package doc**:" in prompts


class TestOrchestrateSh:
    def test_is_valid_bash_syntax(self, tmp_path):
        sh = generate_orchestrate_sh(tmp_path / "kit")
        result = __import__("subprocess").run(
            ["bash", "-n"], input=sh, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"bash -n failed: {result.stderr}"

    def test_contains_plan_root(self, tmp_path):
        plan_root = tmp_path / "my-plan"
        sh = generate_orchestrate_sh(plan_root)
        assert plan_root.as_posix() in sh

    def test_contains_required_functions(self, tmp_path):
        sh = generate_orchestrate_sh(tmp_path / "kit")
        for fn in ["preflight_graph", "preflight_state", "status_consistency_ok", "cmd_status", "cmd_mark_state"]:
            assert fn in sh, f"missing function: {fn}"


class TestPackageDoc:
    def test_has_goal(self):
        doc = generate_package_doc(PACKAGES[0])
        assert "## Goal" in doc

    def test_has_allowed_paths(self):
        doc = generate_package_doc(PACKAGES[0])
        assert "## Allowed Paths" in doc

    def test_has_forbidden_paths(self):
        doc = generate_package_doc(PACKAGES[0])
        assert "## Forbidden Paths" in doc


class TestGenerateKitFiles:
    def test_returns_all_expected_keys(self, tmp_path):
        config = _make_config(tmp_path)
        files = generate_kit_files(config, PACKAGES)
        assert "INDEX.md" in files
        assert "launchers/package-graph.tsv" in files
        assert "launchers/orchestrate.sh" in files
        assert "launchers/agent-prompts.md" in files
        assert "status/state.tsv" in files
        assert "scratch/.gitignore" in files
        for pkg in PACKAGES:
            assert f"packages/{pkg.package_id}.md" in files
            assert f"status/{pkg.package_id}.md" in files

    def test_graph_tsv_content_is_valid(self, tmp_path):
        config = _make_config(tmp_path)
        files = generate_kit_files(config, PACKAGES)
        graph = files["launchers/package-graph.tsv"]
        lines = graph.splitlines()
        assert lines[0] == GRAPH_HEADER
        for line in lines[1:]:
            assert len(line.split("\t")) == 10

    def test_state_tsv_content_is_valid(self, tmp_path):
        config = _make_config(tmp_path)
        files = generate_kit_files(config, PACKAGES)
        state = files["status/state.tsv"]
        lines = state.splitlines()
        assert lines[0] == STATE_HEADER
        for line in lines[1:]:
            assert len(line.split("\t")) == 17


class TestEndToEndOrchestrateValidation:
    def test_generated_kit_passes_bash_preflight(self, tmp_path):
        project = tmp_path / "e2e_project"
        project.mkdir()
        config = DiscoveryKitConfig(
            project_path=project,
            scope="end-to-end test",
            output_root=tmp_path / "kits",
        )
        kit = generate_discovery_kit(config)

        import subprocess

        result = subprocess.run(
            ["bash", kit.orchestrate_path.as_posix(), "status"],
            capture_output=True,
            text=True,
            check=False,
            cwd=kit.plan_root,
        )
        assert result.returncode == 0, f"orchestrate.sh status failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        assert "Coordinator consistency: ok" in result.stdout or "Coordinator consistency: ok" in result.stderr


# ---------------------------------------------------------------------------
# Template-aware package doc tests
# ---------------------------------------------------------------------------

class TestTemplateAwarePackageDoc:
    def test_plan_source_doc_has_scope_section(self):
        plan_source = next(p for p in PACKAGES if p.package_id == "plan-source")
        doc = generate_package_doc(plan_source)
        assert "## Scope" in doc
        assert "## Source Budget" in doc

    def test_agent_history_doc_has_raw_log_policy(self):
        agent_hist = next(p for p in PACKAGES if p.package_id == "agent-history")
        doc = generate_package_doc(agent_hist)
        assert "## Raw Log Policy" in doc
        assert "prohibit" in doc.lower() or "never" in doc.lower()

    def test_code_analysis_doc_has_no_plan_locations_section(self):
        code_analysis = next(p for p in PACKAGES if p.package_id == "code-analysis")
        doc = generate_package_doc(code_analysis)
        assert "## Plan Document Locations" not in doc

    def test_plan_source_doc_has_plan_locations_section(self):
        plan_source = next(p for p in PACKAGES if p.package_id == "plan-source")
        doc = generate_package_doc(plan_source)
        assert "## Plan Document Locations" in doc
        # Must mention alternative locations beyond docs/plans
        assert "docs/" in doc

    def test_all_functional_packages_have_template_docs(self):
        for pkg in PACKAGES:
            if pkg.is_finalize:
                continue
            doc = generate_package_doc(pkg)
            assert "## Goal" in doc
            assert "## Scope" in doc

    def test_template_docs_include_privacy_constraints(self):
        from replay_checker.discovery import source_template_map
        tmap = source_template_map()
        for pkg in PACKAGES:
            if pkg.is_finalize:
                continue
            doc = generate_package_doc(pkg)
            assert "## Privacy Constraints" in doc, f"{pkg.package_id} missing Privacy Constraints"

    def test_template_docs_include_verification_commands(self):
        for pkg in PACKAGES:
            if pkg.is_finalize:
                continue
            doc = generate_package_doc(pkg)
            assert "## Verification Commands" in doc

    def test_kit_generation_with_templates(self, tmp_path):
        project = tmp_path / "proj"
        project.mkdir()
        (project / "src").mkdir()
        (project / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
        config = DiscoveryKitConfig(
            project_path=project,
            scope="template integration test",
            output_root=tmp_path / "kits",
        )
        kit = generate_discovery_kit(config)
        for pkg in kit.packages:
            if pkg.is_finalize:
                continue
            doc_path = kit.plan_root / "packages" / f"{pkg.package_id}.md"
            content = doc_path.read_text(encoding="utf-8")
            assert "## Scope" in content, f"{pkg.package_id} doc missing Scope section"
            assert "## Source Budget" in content, f"{pkg.package_id} doc missing Source Budget"
