#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTICE = "<!-- Generated from AGENTS.md. Do not edit directly. -->"

REQUIRED_FILES = [
    "control/ledger.md",
    "control/state.md",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".codex/config.example.toml",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    "rubrics/default.yaml",
]

REQUIRED_DIRS = [
    "control",
    "cases",
    "runs",
    "rubrics",
    "reports",
    "work/in",
    "work/out",
    "work/tmp",
    "work/isolated",
    "tools",
    "src",
]

ALLOWED_PREFIXES = (
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "README.md",
    "Makefile",
    "pyproject.toml",
    ".gitignore",
    ".env.example",
    ".codex/",
    ".claude/",
    "cases/",
    "control/",
    "docs/plans/",
    "reports/",
    "rubrics/",
    "tools/",
    "src/",
    "tests/",
    ".tmp/",
    "runs/.gitkeep",
    "work/in/.gitkeep",
    "work/out/.gitkeep",
    "work/tmp/.gitkeep",
    "work/isolated/.gitkeep",
)

REJECT_PREFIXES = (".env", "work/tmp/", ".pytest_cache/", "__pycache__/")

TEXT_SUFFIXES = {".md", ".py", ".toml", ".txt", ".json", ".example", ".gitignore", ".yml", ".yaml"}


@dataclass
class Result:
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class GitChange:
    status: str
    path: str


def slugify_project(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", name.strip()).strip("-").lower()
    return slug or "new-project"


def package_name_from_project(name: str) -> str:
    package = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()
    if not package:
        package = "new_project"
    if package[0].isdigit():
        package = f"p_{package}"
    return package


def render_claude() -> str:
    return f"""@AGENTS.md

# Claude Code adapter

This repository uses `AGENTS.md` as the shared source of truth for AI coding agents.

Claude Code-specific notes:

- Follow `AGENTS.md` first.
- Keep this file short and Claude-specific.
- Do not duplicate shared project rules here.
- When a repeated mistake is discovered, suggest whether it should become a hook, test, lint rule, or CI check instead of adding another reminder here.
- You can configure `.claude/settings.json` Stop hook to call `tools/project.py commit`.
"""


def render_gemini() -> str:
    return f"""# Gemini CLI Entry

{NOTICE}

Shared engineering rules are in:

@./AGENTS.md

## Gemini CLI Notes

- After modifying shared rules, run `python3 tools/project.py sync-agents` and reload context in Gemini CLI.
- Do not copy shared rules into this file.
"""


def expected_agent_files() -> dict[Path, str]:
    return {
        ROOT / "CLAUDE.md": render_claude(),
        ROOT / "GEMINI.md": render_gemini(),
    }


def sync_agents() -> int:
    if not (ROOT / "AGENTS.md").exists():
        print("missing AGENTS.md", file=sys.stderr)
        return 1
    for path, content in expected_agent_files().items():
        path.write_text(content, encoding="utf-8")
    print("Synced CLAUDE.md and GEMINI.md from AGENTS.md.")
    return 0


def check_agent_sync(result: Result) -> None:
    if not (ROOT / "AGENTS.md").exists():
        result.issues.append("missing AGENTS.md")
        return
    for path, expected in expected_agent_files().items():
        if not path.exists():
            result.issues.append(f"missing {path.relative_to(ROOT)}")
        elif path.read_text(encoding="utf-8") != expected:
            result.issues.append(f"{path.relative_to(ROOT)} is not in sync with AGENTS.md")
    if not any("sync" in issue for issue in result.issues):
        result.notices.append("agent entry files are synced")


def check_required(result: Result) -> None:
    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            result.issues.append(f"missing required file: {relative}")
    for relative in REQUIRED_DIRS:
        if not (ROOT / relative).is_dir():
            result.issues.append(f"missing required dir: {relative}")


def check_package_import(result: Result) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        importlib.import_module("replay_checker")
    except Exception as exc:
        result.issues.append(f"cannot import replay_checker: {exc}")
    finally:
        try:
            sys.path.remove(str(ROOT / "src"))
        except ValueError:
            pass


def check_git(result: Result) -> None:
    if not (ROOT / ".git").exists():
        result.warnings.append("git repository is not initialized; run tools/project.py init after copying")
        return
    completed = run_git(["status", "--short"])
    if completed.returncode != 0:
        result.warnings.append("git status failed")


def check_panel_runs(result: Result) -> None:
    """Check that tools/panel.py can execute without errors."""
    panel = ROOT / "tools" / "panel.py"
    if not panel.exists():
        result.issues.append("missing tools/panel.py")
        return
    try:
        completed = subprocess.run(
            [sys.executable, str(panel)],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        if completed.returncode != 0:
            result.issues.append(f"tools/panel.py failed: {completed.stderr.strip()}")
        elif not completed.stdout.strip():
            result.warnings.append("tools/panel.py produced no output")
        else:
            result.notices.append("panel hook is functional")
    except subprocess.TimeoutExpired:
        result.warnings.append("tools/panel.py timed out after 10s")
    except Exception as exc:
        result.warnings.append(f"tools/panel.py error: {exc}")


def check_init_status_consistency(result: Result) -> None:
    """Check that AGENTS.md/state are consistent with each other."""
    agents = ROOT / "AGENTS.md"
    state = ROOT / "control" / "state.md"
    if not agents.exists() or not state.exists():
        return
    agents_text = agents.read_text(encoding="utf-8")
    state_text = state.read_text(encoding="utf-8")
    seed_placeholder = "After copying this scaffold, update this section and"
    contract_initialized = seed_placeholder not in agents_text
    state_initialized = "Initialized at:" in state_text
    if contract_initialized != state_initialized:
        result.warnings.append(
            "init status mismatch: AGENTS.md and state.md disagree on initialization state"
        )
    elif contract_initialized:
        result.notices.append("init status is consistent across AGENTS.md/state")


def check_claude_hook_files(result: Result) -> None:
    """Check that Claude hook files exist if settings.example.json references them."""
    example = ROOT / ".claude" / "settings.example.json"
    if not example.exists():
        return
    hook_script = ROOT / ".claude" / "hooks" / "panel_hook.py"
    if not hook_script.exists():
        result.warnings.append("missing .claude/hooks/panel_hook.py (referenced by settings)")
    else:
        result.notices.append("Claude hook files present")


def check_codex_hook_files(result: Result) -> None:
    """Check that Codex helper files exist."""
    example = ROOT / ".codex" / "config.example.toml"
    notify = ROOT / "tools" / "hooks" / "codex_notify.py"
    panel = ROOT / "tools" / "hooks" / "panel_print.py"
    if not example.exists():
        result.warnings.append("missing .codex/config.example.toml")
    elif not notify.exists():
        result.warnings.append("missing tools/hooks/codex_notify.py (referenced by Codex config example)")
    elif not panel.exists():
        result.warnings.append("missing tools/hooks/panel_print.py")
    else:
        result.notices.append("Codex hook helper files present")


def check_cases_complete(result: Result) -> None:
    """Check committed replay cases have complete contracts before execution."""
    cases_root = ROOT / "cases"
    if not cases_root.exists():
        return

    sys.path.insert(0, str(ROOT / "src"))
    try:
        from replay_checker.replay import load_case, validate_case
    except Exception as exc:
        result.issues.append(f"cannot import case validator: {exc}")
        return
    finally:
        try:
            sys.path.remove(str(ROOT / "src"))
        except ValueError:
            pass

    checked = 0
    for case_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        if not (case_dir / "case.yaml").exists():
            result.issues.append(f"case {case_dir.name} missing case.yaml")
            continue
        checked += 1
        try:
            case = load_case(cases_root, case_dir.name)
            issues = validate_case(case)
        except Exception as exc:
            issues = [str(exc)]
        if issues:
            result.issues.append(f"case {case_dir.name} incomplete: {'; '.join(issues)}")

    if checked:
        result.notices.append(f"replay cases complete: {checked}")


def check_orchestration_states(result: Result) -> None:
    """Block finalized orchestration plans that still have incomplete dependencies."""
    plans_root = ROOT / "docs" / "plans"
    if not plans_root.exists():
        return

    checked = 0
    complete_states = {"completed", "finalized"}
    for state_file in sorted(plans_root.glob("*/status/state.tsv")):
        rows = _read_state_rows(state_file)
        if not rows:
            continue
        finalize = next((row for row in rows if row.get("package_id") == "99-finalize"), None)
        if not finalize or finalize.get("state") not in complete_states:
            continue
        checked += 1
        incomplete = [
            f"{row.get('package_id')}={row.get('state')}"
            for row in rows
            if row.get("package_id") != "99-finalize" and row.get("state") not in complete_states
        ]
        if incomplete:
            plan_name = state_file.parents[1].name
            result.issues.append(
                f"orchestration {plan_name} finalized while dependencies are incomplete: {', '.join(incomplete)}"
            )

    if checked:
        result.notices.append(f"orchestration finalized states checked: {checked}")


def _read_state_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return []
    headers = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        values = line.split("\t")
        rows.append({header: values[index] if index < len(values) else "" for index, header in enumerate(headers)})
    return rows


def check_work_dirs_have_gitkeep(result: Result) -> None:
    """Check that work subdirectories have .gitkeep files."""
    for subdir in ["work/in", "work/out", "work/tmp", "work/isolated", "cases", "runs", "reports"]:
        gitkeep = ROOT / subdir / ".gitkeep"
        if not gitkeep.exists():
            result.warnings.append(f"missing {subdir}/.gitkeep (dir may not survive git clone)")


def check_no_platform_junk_tracked(result: Result) -> None:
    """Check that no platform junk files (._*, .DS_Store) are tracked."""
    if not (ROOT / ".git").exists():
        return
    completed = run_git(["ls-files"])
    if completed.returncode != 0:
        return
    junk = []
    for line in completed.stdout.splitlines():
        name = line.strip()
        if name.startswith("._") or name == ".DS_Store":
            junk.append(name)
    if junk:
        result.warnings.append(f"platform junk files tracked in git: {', '.join(junk)}")


def print_result(result: Result, quiet: bool) -> None:
    if result.issues:
        print("[check] issues:")
        for issue in result.issues:
            print(f"- {issue}")
    if result.warnings:
        print("[check] warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    if result.notices and not quiet:
        print("[check] notices:")
        for notice in result.notices:
            print(f"- {notice}")
    if result.ok and not result.warnings and not quiet:
        print("[check] OK")


def check(args: argparse.Namespace) -> int:
    result = Result()
    check_required(result)
    check_agent_sync(result)
    check_package_import(result)
    check_panel_runs(result)
    check_init_status_consistency(result)
    check_claude_hook_files(result)
    check_codex_hook_files(result)
    check_cases_complete(result)
    check_orchestration_states(result)
    check_work_dirs_have_gitkeep(result)
    if not args.skip_git:
        check_git(result)
        check_no_platform_junk_tracked(result)
    print_result(result, args.quiet)
    return 1 if result.issues else 0


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore"}:
        return True
    return path.suffix in TEXT_SUFFIXES or path.name.endswith(".example")


def replace_text(project_name: str, package_name: str) -> None:
    project_slug = slugify_project(project_name)
    replacements = {
        "Replay Checker": project_name,
        "replay-checker": project_slug,
        "replay_checker": project_slug.replace("-", "_"),
        "replay_checker": package_name,
        "REPLAY_CHECKER": package_name.upper(),
    }
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file() or not is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")


def rename_package(package_name: str) -> None:
    src = ROOT / "src" / "replay_checker"
    dst = ROOT / "src" / package_name
    if package_name == "replay_checker" or not src.exists():
        return
    if dst.exists():
        raise FileExistsError(dst)
    src.rename(dst)


def tracked_initial_files() -> list[str]:
    paths: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel == ".env" or rel.startswith("work/tmp/") and rel != "work/tmp/.gitkeep":
            continue
        if rel.startswith("work/out/") and rel != "work/out/.gitkeep":
            continue
        if "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        paths.append(rel)
    return sorted(paths)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)


def run_git_init(project_name: str) -> None:
    if (ROOT / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=ROOT, check=True)
    files = tracked_initial_files()
    if files:
        subprocess.run(["git", "add", "--", *files], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", f"chore: initialize {project_name}"], cwd=ROOT, check=True)


def init_project(args: argparse.Namespace) -> int:
    package_name = args.package_name or package_name_from_project(args.name)
    replace_text(args.name, package_name)
    rename_package(package_name)
    update_contract(args.name)
    update_state(args.name, package_name)
    append_init_ledger(args.name, package_name)
    activate_settings()
    if not args.no_git:
        run_git_init(args.name)
    print(f"Initialized {args.name} with package {package_name}.")
    return 0


def activate_settings() -> None:
    """Copy settings.example.json to settings.json if settings.json does not exist."""
    example = ROOT / ".claude" / "settings.example.json"
    target = ROOT / ".claude" / "settings.json"
    if example.exists() and not target.exists():
        target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")


def now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


def update_contract(project_name: str) -> None:
    """Replace Current Intent section in AGENTS.md with initialized template."""
    path = ROOT / "AGENTS.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    new_intent = (
        f"**Project**: {project_name}\n"
        f"**Status**: Initialized, goals pending\n"
        f"\n"
        f"Edit this section to specify:\n"
        f"- Project goals\n"
        f"- Non-goals\n"
        f"- Acceptance criteria\n"
    )
    new_text, count = re.subn(
        r"\*\*Project\*\*:.*?\n\*\*Status\*\*:.*?\n\n.*?(?=\n## |\Z)",
        new_intent, text, flags=re.DOTALL
    )
    if count:
        text = new_text
    else:
        text = text.replace(
            "## Project overview\n",
            f"## Current Intent\n\n{new_intent}\n## Project overview\n",
            1,
        )
    path.write_text(text, encoding="utf-8")


def update_state(project_name: str, package_name: str) -> None:
    """Replace state.md with initialized content."""
    path = ROOT / "control" / "state.md"
    content = (
        f"# State\n"
        f"\n"
        f"## Current State\n"
        f"\n"
        f"- Project name: {project_name}\n"
        f"- Package name: {package_name}\n"
        f"- Initialized at: {now_iso()}\n"
        f"- Status: Initialized, goals pending\n"
        f"\n"
        f"## Next Maintenance Action\n"
        f"\n"
        f"- Edit Current Intent in `AGENTS.md` to specify project goals, non-goals, and acceptance criteria.\n"
    )
    path.write_text(content, encoding="utf-8")


def append_init_ledger(project_name: str, package_name: str) -> None:
    """Append an initialization record to ledger.md."""
    path = ROOT / "control" / "ledger.md"
    if not path.exists():
        return
    existing = path.read_text(encoding="utf-8")
    record = (
        f"\n## {now_iso()} - Project initialized from seed\n"
        f"\n"
        f"type: decision\n"
        f"tags: init, scaffold\n"
        f"\n"
        f"summary:\n"
        f"- Initialized project `{project_name}` from Agent Project Seed\n"
        f"- Package name: `{package_name}`\n"
        f"\n"
        f"details:\n"
        f"- Completed: text replacement, package rename, settings activation, contract/state update\n"
        f"- Todo: edit `AGENTS.md` to specify project goals\n"
        f"\n"
        f"links:\n"
        f"- AGENTS.md\n"
        f"- control/state.md\n"
    )
    separator = "\n" if existing.endswith("\n\n") else "\n\n"
    path.write_text(existing.rstrip() + separator + record, encoding="utf-8")


def parse_status_line(line: str) -> GitChange:
    status = line[:2]
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return GitChange(status, path)


def git_changes() -> list[GitChange]:
    completed = run_git(["status", "--porcelain"])
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git status failed")
    return [parse_status_line(line) for line in completed.stdout.splitlines() if line.strip()]


def is_rejected(path: str) -> bool:
    if any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in REJECT_PREFIXES):
        return True
    if path.startswith("work/out/") and path != "work/out/.gitkeep":
        return True
    if path.startswith("runs/") and path != "runs/.gitkeep":
        return True
    return False


def is_allowed(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def classify_changes(changes: list[GitChange]) -> tuple[list[GitChange], list[GitChange]]:
    allowed: list[GitChange] = []
    rejected: list[GitChange] = []
    for change in changes:
        if "U" in change.status or is_rejected(change.path) or not is_allowed(change.path):
            rejected.append(change)
        else:
            allowed.append(change)
    return allowed, rejected


def commit(args: argparse.Namespace) -> int:
    if not (ROOT / ".git").exists():
        print("Not a git repository; run tools/project.py init first.", file=sys.stderr)
        return 1
    changes = git_changes()
    if not changes:
        print("No changes to commit.")
        return 0
    allowed, rejected = classify_changes(changes)
    if rejected:
        print("Refusing auto commit because some changes are unsafe or outside the allowlist:", file=sys.stderr)
        for change in rejected:
            print(f"- {change.status} {change.path}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("Allowed changes:")
        for change in allowed:
            print(f"- {change.status} {change.path}")
        return 0
    add = run_git(["add", "--", *[change.path for change in allowed]])
    if add.returncode != 0:
        print(add.stderr.strip(), file=sys.stderr)
        return add.returncode
    completed = run_git(["commit", "-m", args.message])
    if completed.returncode != 0:
        print(completed.stdout.strip())
        print(completed.stderr.strip(), file=sys.stderr)
        return completed.returncode
    print(completed.stdout.strip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project scaffold command center.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize a copied template")
    init.add_argument("--name", default=ROOT.name)
    init.add_argument("--package-name", default=None)
    init.add_argument("--no-git", action="store_true")

    check_cmd = sub.add_parser("check", help="check scaffold health")
    check_cmd.add_argument("--quiet", action="store_true")
    check_cmd.add_argument("--skip-git", action="store_true")

    sub.add_parser("sync-agents", help="sync thin agent entry files")

    commit_cmd = sub.add_parser("commit", help="safely commit allowed changes")
    commit_cmd.add_argument("--message", default="chore: checkpoint agent work")
    commit_cmd.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    os.chdir(ROOT)
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init":
        return init_project(args)
    if args.command == "check":
        return check(args)
    if args.command == "sync-agents":
        return sync_agents()
    if args.command == "commit":
        return commit(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
