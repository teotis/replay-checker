#!/usr/bin/env python3
"""
Project status panel generator — injected at the start of every conversation.

Three status levels:
  - Seed Template: not yet initialized
  - Initialized, goals pending: init has run, but Current Intent not edited
  - Ready: Current Intent has been customized
"""
from __future__ import annotations

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SEED_PLACEHOLDER = "After copying this scaffold, update this section and"
PENDING_MARKER = "Initialized, goals pending"
CONTRACT_PATH = "AGENTS.md"


def read_project_name() -> str:
    """Read project name from AGENTS.md, falling back to pyproject.toml."""
    contract = ROOT / CONTRACT_PATH
    if contract.exists():
        text = contract.read_text(encoding="utf-8")
        match = re.search(r"\*\*Project\*\*:\s*(.+)", text)
        if match:
            return match.group(1).strip()
    toml = ROOT / "pyproject.toml"
    if toml.exists():
        text = toml.read_text(encoding="utf-8")
        match = re.search(r'^name\s*=\s*"(.+?)"', text, re.MULTILINE)
        if match:
            return match.group(1)
    return ""


def read_package_name() -> str:
    """Infer package name from the directory under src/."""
    src = ROOT / "src"
    if not src.exists():
        return ""
    dirs = [d for d in src.iterdir() if d.is_dir() and not d.name.startswith("_")]
    return dirs[0].name if dirs else ""


def detect_status() -> str:
    """Detect project status: seed / pending / ready."""
    contract = ROOT / CONTRACT_PATH
    if not contract.exists():
        return "seed"
    text = contract.read_text(encoding="utf-8")
    if SEED_PLACEHOLDER in text:
        return "seed"
    if PENDING_MARKER in text:
        return "pending"
    return "ready"


def count_ledger_records() -> int:
    """Count the number of records in ledger.md."""
    ledger = ROOT / "control" / "ledger.md"
    if not ledger.exists():
        return 0
    text = ledger.read_text(encoding="utf-8")
    return len(re.findall(r"^## \d{4}-\d{2}-\d{2}T", text, re.MULTILINE))


def extract_intent_summary() -> str:
    """Extract the project goal line from Current Intent in AGENTS.md."""
    contract = ROOT / CONTRACT_PATH
    if not contract.exists():
        return ""
    text = contract.read_text(encoding="utf-8")
    match = re.search(r"## Current Intent\n\n(.+?)(?=\n## |\Z)", text, re.DOTALL)
    if not match:
        return ""
    body = match.group(1).strip()
    # Skip metadata lines (**Project**: / **Status**:), extract first "- " line
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("- "):
            return line[2:]
    return ""


def extract_next_action() -> str:
    """Extract next action from state.md."""
    state = ROOT / "control" / "state.md"
    if not state.exists():
        return ""
    text = state.read_text(encoding="utf-8")
    match = re.search(r"## Next Maintenance Action\n\n(.+?)(?=\n## |\Z)", text, re.DOTALL)
    if not match:
        return ""
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if line.startswith("- "):
            return line[2:]
    return ""


def git_status_summary() -> str:
    """Get git status summary."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return "unknown"
        changed = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
        return f"{changed} file{'s' if changed != 1 else ''}" if changed else "clean"
    except Exception:
        return "unknown"


STATUS_LABELS = {
    "seed": "Seed Template",
    "pending": "Initialized, goals pending",
    "ready": "Ready",
}


def generate_panel() -> str:
    today = date.today()
    weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][today.weekday()]
    status = detect_status()
    project_name = read_project_name() or "Agent Project Seed"

    if status == "seed":
        return "\n".join([
            f"[{project_name}] {today.isoformat()} ({weekday})",
            f"Status: {STATUS_LABELS[status]}",
            "-> Run `python3 tools/project.py init --name \"your-project-name\"` to start",
        ])

    package = read_package_name()
    records = count_ledger_records()
    git = git_status_summary()
    intent = extract_intent_summary()
    next_action = extract_next_action()

    lines = [
        f"[{project_name}] {today.isoformat()} ({weekday})",
        f"Status: {STATUS_LABELS[status]}",
        f"Git: {git} | Ledger: {records} records | Package: {package}",
    ]
    if intent:
        lines.append(f"Goal: {intent}")
    if next_action:
        lines.append(f"Next: {next_action}")
    return "\n".join(lines)


def main() -> int:
    try:
        print(generate_panel())
    except Exception as exc:
        print(f"[panel] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
