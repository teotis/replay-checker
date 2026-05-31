"""Tests for the natural language wizard and plan-case-discovery."""

import subprocess
import sys
from pathlib import Path

import pytest

from replay_checker.wizard import (
    WizardResult,
    generate_plan_name,
    generate_discovery_kit,
    plan_case_discovery,
    validate_project_path,
    wizard,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(args, cwd):
    subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)


def _make_project(tmp_path):
    """Create a minimal git project for testing."""
    project = tmp_path / "target_project"
    project.mkdir()
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
    (project / "docs").mkdir()
    (project / "docs" / "notes.md").write_text("# Notes\n\nSome notes.\n", encoding="utf-8")
    _git(["init"], project)
    _git(["config", "user.email", "test@example.com"], project)
    _git(["config", "user.name", "Test User"], project)
    _git(["add", "."], project)
    _git(["commit", "-m", "initial"], project)
    return project


# ---------------------------------------------------------------------------
# validate_project_path
# ---------------------------------------------------------------------------

def test_validate_project_path_returns_resolved_path(tmp_path):
    project = _make_project(tmp_path)
    result = validate_project_path(project)
    assert result == project.resolve()
    assert result.is_dir()


def test_validate_project_path_raises_for_missing(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        validate_project_path(tmp_path / "nonexistent")


def test_validate_project_path_raises_for_file(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        validate_project_path(f)


# ---------------------------------------------------------------------------
# generate_plan_name
# ---------------------------------------------------------------------------

def test_generate_plan_name_contains_date():
    name = generate_plan_name("find auth bugs")
    assert "2026-" in name  # date prefix
    assert "find_auth_bugs" in name


def test_generate_plan_name_fallback_for_empty_scope():
    name = generate_plan_name("")
    assert name.startswith("2026-")
    assert "discovery" in name


# ---------------------------------------------------------------------------
# generate_discovery_kit
# ---------------------------------------------------------------------------

def test_generate_discovery_kit_creates_all_files(tmp_path):
    project = _make_project(tmp_path)
    result = generate_discovery_kit(project, "find auth bugs")

    kit = result.kit_path
    assert kit.exists()
    assert (kit / "INDEX.md").exists()
    assert (kit / "launchers" / "orchestrate.sh").exists()
    assert (kit / "launchers" / "package-graph.tsv").exists()
    assert (kit / "launchers" / "agent-prompts.md").exists()
    assert (kit / "status" / "state.tsv").exists()
    assert (kit / "packages" / "01-plan-scan.md").exists()
    assert (kit / "packages" / "02-git-discovery.md").exists()
    assert (kit / "packages" / "03-case-merge.md").exists()
    assert (kit / "packages" / "99-finalize.md").exists()
    for pkg_id in ["01-plan-scan", "02-git-discovery", "03-case-merge", "99-finalize"]:
        status_file = kit / "status" / f"{pkg_id}.md"
        assert status_file.exists(), f"missing status file: {pkg_id}.md"
        content = status_file.read_text(encoding="utf-8")
        assert "## State" in content, f"{pkg_id}.md missing ## State section"
        assert "`pending`" in content, f"{pkg_id}.md missing pending state"


def test_generate_discovery_kit_preserves_scope_in_index(tmp_path):
    project = _make_project(tmp_path)
    scope = "refactor the auth module to use JWT tokens"
    result = generate_discovery_kit(project, scope)

    index_text = (result.kit_path / "INDEX.md").read_text(encoding="utf-8")
    assert scope in index_text


def test_generate_discovery_kit_preserves_scope_in_agent_prompts(tmp_path):
    project = _make_project(tmp_path)
    scope = "add unit tests for the API layer"
    result = generate_discovery_kit(project, scope)

    prompts_text = (result.kit_path / "launchers" / "agent-prompts.md").read_text(encoding="utf-8")
    assert scope in prompts_text


def test_generate_discovery_kit_preserves_scope_in_packages(tmp_path):
    project = _make_project(tmp_path)
    scope = "migrate from REST to GraphQL"
    result = generate_discovery_kit(project, scope)

    for pkg_name in ["01-plan-scan", "02-git-discovery", "03-case-merge", "99-finalize"]:
        pkg_text = (result.kit_path / "packages" / f"{pkg_name}.md").read_text(encoding="utf-8")
        assert scope in pkg_text, f"scope missing in {pkg_name}.md"


def test_generate_discovery_kit_records_project_path(tmp_path):
    project = _make_project(tmp_path)
    result = generate_discovery_kit(project, "test scope")

    assert result.project_path == project.resolve()
    assert str(project) in (result.kit_path / "INDEX.md").read_text(encoding="utf-8")


def test_generate_discovery_kit_start_command(tmp_path):
    project = _make_project(tmp_path)
    result = generate_discovery_kit(project, "test scope")

    assert "orchestrate.sh" in result.start_command
    assert "start" in result.start_command


def test_generate_discovery_kit_empty_scope_raises(tmp_path):
    project = _make_project(tmp_path)
    with pytest.raises(ValueError, match="Scope text must not be empty"):
        generate_discovery_kit(project, "")


def test_generate_discovery_kit_whitespace_scope_raises(tmp_path):
    project = _make_project(tmp_path)
    with pytest.raises(ValueError, match="Scope text must not be empty"):
        generate_discovery_kit(project, "   ")


def test_generate_discovery_kit_custom_output_root(tmp_path):
    project = _make_project(tmp_path)
    output_root = tmp_path / "custom_output"
    result = generate_discovery_kit(project, "test scope", output_root=output_root)

    assert result.kit_path.parent == output_root.resolve()
    assert result.kit_path.exists()


def test_generate_discovery_kit_state_tsv_has_all_packages(tmp_path):
    project = _make_project(tmp_path)
    result = generate_discovery_kit(project, "test scope")

    state_text = (result.kit_path / "status" / "state.tsv").read_text(encoding="utf-8")
    assert "01-plan-scan" in state_text
    assert "02-git-discovery" in state_text
    assert "03-case-merge" in state_text
    assert "99-finalize" in state_text


# ---------------------------------------------------------------------------
# wizard (non-interactive)
# ---------------------------------------------------------------------------

def test_wizard_non_interactive_with_scope(tmp_path):
    project = _make_project(tmp_path)
    result = wizard(project, scope="find all bugs", interactive=False)

    assert isinstance(result, WizardResult)
    assert result.kit_path.exists()
    assert result.scope == "find all bugs"


def test_wizard_non_interactive_without_scope_raises(tmp_path):
    project = _make_project(tmp_path)
    with pytest.raises(ValueError, match="Scope is required"):
        wizard(project, scope=None, interactive=False)


# ---------------------------------------------------------------------------
# plan_case_discovery (scriptable)
# ---------------------------------------------------------------------------

def test_plan_case_discovery_generates_kit(tmp_path):
    project = _make_project(tmp_path)
    result = plan_case_discovery(project, "find auth bugs")

    assert result.kit_path.exists()
    assert result.scope == "find auth bugs"


def test_plan_case_discovery_empty_scope_raises(tmp_path):
    project = _make_project(tmp_path)
    with pytest.raises(ValueError, match="Scope must not be empty"):
        plan_case_discovery(project, "")


def test_plan_case_discovery_invalid_project_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        plan_case_discovery(tmp_path / "nonexistent", "test scope")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_cli_plan_case_discovery(tmp_path):
    project = _make_project(tmp_path)
    output = tmp_path / "kit_output"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "plan-case-discovery",
            "--project",
            str(project),
            "--scope",
            "find all auth bugs",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Discovery kit generated:" in result.stdout
    assert "orchestrate.sh" in result.stdout


def test_cli_wizard_non_interactive(tmp_path):
    project = _make_project(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "wizard",
            "--project",
            str(project),
            "--scope",
            "test discovery",
            "--no-interactive",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Discovery kit generated:" in result.stdout


def test_cli_wizard_no_scope_non_interactive(tmp_path):
    project = _make_project(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "wizard",
            "--project",
            str(project),
            "--no-interactive",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr or "Scope is required" in result.stderr


# ---------------------------------------------------------------------------
# No silent agent launch
# ---------------------------------------------------------------------------

def test_wizard_does_not_launch_agents(tmp_path):
    """The wizard should only print a recommended command, not execute it."""
    project = _make_project(tmp_path)

    # Capture output — if wizard tried to run orchestrate.sh, it would fail
    # because the kit is new and state.tsv has no completed packages
    result = wizard(project, scope="find bugs", interactive=False)

    # Verify kit was generated but orchestrate.sh was NOT executed
    state_text = (result.kit_path / "status" / "state.tsv").read_text(encoding="utf-8")
    # All packages should still be in "pending" state
    for line in state_text.splitlines():
        if line.startswith("package_id"):
            continue
        if line.strip():
            parts = line.split("\t")
            assert parts[1] == "pending", "wizard must not execute orchestrate.sh"


# ---------------------------------------------------------------------------
# orchestrate.sh status smoke
# ---------------------------------------------------------------------------

def test_wizard_generated_kit_passes_orchestrate_status(tmp_path):
    """Regression: wizard-generated kit must pass its own orchestrate.sh status."""
    project = _make_project(tmp_path)
    result = generate_discovery_kit(project, "smoke check")

    kit_script = result.kit_path / "launchers" / "orchestrate.sh"
    assert kit_script.exists()

    proc = subprocess.run(
        ["bash", kit_script.as_posix(), "status"],
        capture_output=True,
        text=True,
        check=False,
        cwd=result.kit_path,
    )
    assert proc.returncode == 0, (
        f"orchestrate.sh status failed (rc={proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "Coordinator consistency: ok" in proc.stdout or "Coordinator consistency: ok" in proc.stderr


def test_cli_wizard_generated_kit_passes_orchestrate_status(tmp_path):
    """CLI wizard path: generated kit must pass orchestrate.sh status."""
    project = _make_project(tmp_path)

    gen_result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "wizard",
            "--project",
            str(project),
            "--scope",
            "cli smoke check",
            "--no-interactive",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert gen_result.returncode == 0, gen_result.stderr
    assert "Discovery kit generated:" in gen_result.stdout

    # Find the generated kit root from output
    kit_lines = [l for l in gen_result.stdout.splitlines() if "Discovery kit generated:" in l]
    kit_path = Path(kit_lines[0].split("Discovery kit generated:")[1].strip())
    assert kit_path.exists()

    proc = subprocess.run(
        ["bash", (kit_path / "launchers" / "orchestrate.sh").as_posix(), "status"],
        capture_output=True,
        text=True,
        check=False,
        cwd=kit_path,
    )
    assert proc.returncode == 0, (
        f"orchestrate.sh status failed (rc={proc.returncode}):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "Coordinator consistency: ok" in proc.stdout or "Coordinator consistency: ok" in proc.stderr
