#!/usr/bin/env python3
"""
Print the repository status panel for tools that cannot inject prompt context.

Claude Code uses .claude/hooks/panel_hook.py to inject this panel automatically.
Codex users can run this command manually or bind it in their shell/editor when
they want the same project snapshot outside the conversation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_project_root(cwd: str) -> Path | None:
    marker = os.environ.get("PANEL_PROJECT_MARKER", "tools/panel.py")
    candidate = Path(cwd).resolve()
    for _ in range(10):
        if (candidate / marker).exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


def main() -> int:
    project_root = find_project_root(os.environ.get("PWD", "") or os.getcwd())
    if project_root is None:
        print("[panel] project root not found", file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable or "python3", str(project_root / "tools" / "panel.py")],
        cwd=project_root,
        text=True,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
