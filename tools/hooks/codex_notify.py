#!/usr/bin/env python3
"""
Codex notify hook: run the scaffold's safe commit at the end of a Codex turn.

Codex notification hooks are normally configured outside the repository, for
example in ~/.codex/config.toml. This script is repository-local so copied
projects keep the same guarded commit behavior as the Claude Code Stop hook.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MARKER = "tools/project.py"
DEFAULT_MESSAGE = "chore: checkpoint agent work"


def log(message: str) -> None:
    print(f"[codex_notify] {message}", file=sys.stderr)


def read_payload() -> dict[str, Any]:
    """Read Codex notification JSON from argv[1] or stdin when available."""
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    if not raw:
        try:
            raw = sys.stdin.read()
        except Exception:
            raw = ""
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log("notification payload was not valid JSON; continuing with cwd fallback")
        return {}
    return payload if isinstance(payload, dict) else {}


def event_name(payload: dict[str, Any]) -> str:
    """Return the event name across known/likely Codex notification fields."""
    for key in ("type", "event", "hook_event_name", "hookEventName"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def candidate_roots(payload: dict[str, Any]) -> list[Path]:
    """Collect possible workspace directories from payload and environment."""
    candidates: list[Path] = []
    for key in ("cwd", "workspace", "workspace_root", "project_root"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))
    for key in ("CODEX_WORKSPACE", "CODEX_CWD", "PWD"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value))
    candidates.append(Path.cwd())
    return candidates


def find_project_root(payload: dict[str, Any]) -> Path | None:
    """Walk upward from candidate directories to find this scaffold."""
    marker = os.environ.get("PANEL_PROJECT_MARKER", MARKER)
    seen: set[Path] = set()
    for start in candidate_roots(payload):
        try:
            candidate = start.resolve()
        except OSError:
            continue
        for _ in range(10):
            if candidate in seen:
                break
            seen.add(candidate)
            if (candidate / marker).exists():
                return candidate
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
    return None


def run_safe_commit(project_root: Path) -> int:
    """Run project.py commit and keep notification hooks non-blocking."""
    command = [
        sys.executable or "python3",
        str(project_root / "tools" / "project.py"),
        "commit",
        "--message",
        os.environ.get("CODEX_SAFE_COMMIT_MESSAGE", DEFAULT_MESSAGE),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log("safe commit timed out after 30s")
        return 0
    except Exception as exc:
        log(f"safe commit failed to start: {exc}")
        return 0

    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    if output:
        log(output)
    return 0


def main() -> int:
    payload = read_payload()
    name = event_name(payload)
    if name and name not in {"agent-turn-complete", "turn-complete", "stop"}:
        return 0
    project_root = find_project_root(payload)
    if project_root is None:
        log("project root not found, skipping")
        return 0
    return run_safe_commit(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
