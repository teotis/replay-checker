import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_panel():
    script = ROOT / "tools" / "panel.py"
    spec = importlib.util.spec_from_file_location("panel_tool", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["panel_tool"] = module
    spec.loader.exec_module(module)
    return module


def _copy_and_init(tmp_path, name="Demo Project", package="demo_project"):
    target = tmp_path / "copied_project"
    ignore = shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.egg-info")
    shutil.copytree(ROOT, target, ignore=ignore)
    subprocess.run(
        [sys.executable, "tools/project.py", "init",
         "--name", name, "--package-name", package, "--no-git"],
        cwd=target, text=True, capture_output=True, check=True,
    )
    return target


def test_panel_detects_ready_status():
    """Current repo has project-specific intent."""
    panel = load_panel()
    assert panel.detect_status() == "ready"


def test_panel_distinguishes_seed_pending_ready(tmp_path):
    """Three-level status detection: seed -> pending -> ready."""
    panel = load_panel()

    # current project has already been initialized and customized
    assert panel.detect_status() == "ready"

    # after init it should be pending
    target = _copy_and_init(tmp_path)
    # Reload module so ROOT points to target
    import tools.panel as panel_mod
    original_root = panel_mod.ROOT
    try:
        panel_mod.ROOT = target
        assert panel_mod.detect_status() == "pending"

        # Manually edit AGENTS.md with explicit goals -> ready
        contract = target / "AGENTS.md"
        text = contract.read_text(encoding="utf-8")
        text = text.replace("Initialized, goals pending", "Goal: Build a demo application")
        contract.write_text(text, encoding="utf-8")
        assert panel_mod.detect_status() == "ready"
    finally:
        panel_mod.ROOT = original_root


def test_panel_after_init_shows_project_name(tmp_path):
    """After init, panel should show project name instead of 'Agent Project Seed'."""
    target = _copy_and_init(tmp_path, name="My App", package="my_app")

    result = subprocess.run(
        [sys.executable, "tools/panel.py"],
        cwd=target, text=True, capture_output=True,
    )
    assert result.returncode == 0
    assert "My App" in result.stdout
    assert "Seed Template" not in result.stdout
    assert "Initialized" in result.stdout


def test_panel_shows_package_and_records(tmp_path):
    """Panel should show package name and ledger record count."""
    target = _copy_and_init(tmp_path)

    result = subprocess.run(
        [sys.executable, "tools/panel.py"],
        cwd=target, text=True, capture_output=True,
    )
    assert result.returncode == 0
    assert "demo_project" in result.stdout
    assert "Ledger:" in result.stdout
