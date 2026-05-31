import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_project_tool():
    script = ROOT / "tools" / "project.py"
    spec = importlib.util.spec_from_file_location("project_tool", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["project_tool"] = module
    spec.loader.exec_module(module)
    return module


def test_commit_allowlist_rejects_env_tmp_and_outputs():
    module = load_project_tool()
    changes = [
        module.GitChange("??", "control/ledger.md"),
        module.GitChange("??", ".codex/config.example.toml"),
        module.GitChange("??", ".env"),
        module.GitChange("??", "work/tmp/scratch.txt"),
        module.GitChange("??", "work/out/result.png"),
        module.GitChange("??", "runs/evidence.json"),
        module.GitChange("??", "work/isolated/replay-checker/scratch.txt"),
        module.GitChange("??", "cases/.gitkeep"),
    ]

    allowed, rejected = module.classify_changes(changes)

    assert [change.path for change in allowed] == [
        "control/ledger.md",
        ".codex/config.example.toml",
        "cases/.gitkeep",
    ]
    assert [change.path for change in rejected] == [
        ".env",
        "work/tmp/scratch.txt",
        "work/out/result.png",
        "runs/evidence.json",
        "work/isolated/replay-checker/scratch.txt",
    ]


def test_check_and_sync_agents_are_healthy():
    check = subprocess.run(
        [sys.executable, "tools/project.py", "check", "--skip-git"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert check.returncode == 0, check.stdout + check.stderr

    sync = subprocess.run(
        [sys.executable, "tools/project.py", "sync-agents"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert sync.returncode == 0, sync.stdout + sync.stderr


def test_check_cases_complete_reports_broken_case(tmp_path):
    module = load_project_tool()
    original_root = module.ROOT
    project = tmp_path / "project"
    project.mkdir()
    cases = tmp_path / "cases"
    broken = cases / "broken"
    broken.mkdir(parents=True)
    (broken / "case.yaml").write_text(
        "\n".join([
            "id: broken",
            f"project_path: {project}",
            "plan_path: plan.md",
            "base_commit: abc123",
            "verification_commands:",
            "",
        ]),
        encoding="utf-8",
    )

    result = module.Result()
    try:
        module.ROOT = tmp_path
        module.check_cases_complete(result)
    finally:
        module.ROOT = original_root

    assert result.issues
    assert "case broken incomplete" in result.issues[0]
    assert "base_source" in result.issues[0]


def test_check_orchestration_states_blocks_finalized_with_pending_dependency(tmp_path):
    module = load_project_tool()
    original_root = module.ROOT
    status = tmp_path / "docs" / "plans" / "demo" / "status"
    status.mkdir(parents=True)
    (status / "state.tsv").write_text(
        "\n".join([
            "package_id\tstate",
            "01-work\tpending",
            "99-finalize\tfinalized",
            "",
        ]),
        encoding="utf-8",
    )

    result = module.Result()
    try:
        module.ROOT = tmp_path
        module.check_orchestration_states(result)
    finally:
        module.ROOT = original_root

    assert result.issues == [
        "orchestration demo finalized while dependencies are incomplete: 01-work=pending"
    ]


def _copy_and_init(tmp_path, name="Demo Project", package="demo_project"):
    target = tmp_path / "copied_project"
    ignore = shutil.ignore_patterns(".git", ".pytest_cache", "__pycache__", "*.egg-info")
    shutil.copytree(ROOT, target, ignore=ignore)

    completed = subprocess.run(
        [
            sys.executable,
            "tools/project.py",
            "init",
            "--name", name,
            "--package-name", package,
            "--no-git",
        ],
        cwd=target,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return target


def test_init_project_copy_no_git(tmp_path):
    target = _copy_and_init(tmp_path)
    assert (target / "src" / "demo_project" / "__init__.py").exists()
    assert "demo-project" in (target / "pyproject.toml").read_text(encoding="utf-8")
    assert (target / "AGENTS.md").exists()
    assert (target / ".codex" / "config.example.toml").exists()
    assert (target / "work" / "in" / ".gitkeep").exists()


def test_codex_notify_hook_finds_project_root():
    hook = ROOT / "tools" / "hooks" / "codex_notify.py"
    spec = importlib.util.spec_from_file_location("codex_notify", hook)
    hook_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["codex_notify"] = hook_module
    spec.loader.exec_module(hook_module)

    assert hook_module.find_project_root({"cwd": str(ROOT / "tools" / "hooks")}) == ROOT


def test_init_updates_contract_state_and_ledger(tmp_path):
    target = _copy_and_init(tmp_path, name="My App", package="my_app")

    contract = (target / "AGENTS.md").read_text(encoding="utf-8")
    assert "My App" in contract
    assert "Initialized, goals pending" in contract
    assert "After copying this scaffold" not in contract

    state = (target / "control" / "state.md").read_text(encoding="utf-8")
    assert "My App" in state
    assert "my_app" in state

    ledger = (target / "control" / "ledger.md").read_text(encoding="utf-8")
    assert "Project initialized from seed" in ledger
    assert "My App" in ledger


def test_no_legacy_directories_in_template():
    legacy_dirs = [
        "codex",
        "revision",
        "journal",
        "data",
        "source_material",
        "optional_packs",
        "input",
        "output",
        ".tmp",
    ]
    for relative in legacy_dirs:
        assert not (ROOT / relative).exists(), f"legacy directory remains: {relative}"
