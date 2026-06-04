from __future__ import annotations

import os
import subprocess
from pathlib import Path

DEFAULT_GIT_TIMEOUT = 30


def git_run(
    args: list[str],
    *,
    cwd: str | Path,
    timeout: int = DEFAULT_GIT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured text output."""
    env = {**os.environ, "COPYFILE_DISABLE": "1"}
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def git_output(
    args: list[str],
    *,
    cwd: str | Path,
    timeout: int = DEFAULT_GIT_TIMEOUT,
) -> str:
    """Return stdout for a successful git command, otherwise an empty string."""
    try:
        result = git_run(args, cwd=cwd, timeout=timeout)
        return result.stdout if result.returncode == 0 else ""
    except (subprocess.TimeoutExpired, OSError):
        return ""
