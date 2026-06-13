"""Run lifecycle operations — preparation, loading, collection, health, telemetry.

Owns the full run lifecycle without importing the compatibility facade.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .case_paths import resolve_case_dir
from .core import Diagnostic, format_diagnostics, stable_hash
from .evaluation import read_task_outcomes
from .git_utils import DEFAULT_GIT_TIMEOUT, git_output, git_run
from .packages import ExecutionPackageRenderer, compile_execution_package
from .replay_types import ReplayCase, ReplayRun, RunHealth, Telemetry, Evidence
from .yaml_lite import parse_simple_yaml, write_simple_yaml

_GIT_TIMEOUT = DEFAULT_GIT_TIMEOUT

FORBIDDEN_PATH_PATTERNS = (
    "_reference/", "_reference\\",
    ".git/", ".git\\",
    "cases/", "cases\\",
    "runs/", "runs\\",
)
SUSPICIOUS_PATH_PATTERNS = (
    ".env", "credentials", "secret", "token", "password", "key",
)
SENSITIVE_UNTRACKED_FILENAMES = {
    ".env", ".env.local", ".envrc",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
}
SAFE_ENV_EXAMPLE_FILENAMES = {
    ".env.example", ".env.sample", ".env.template",
}
SENSITIVE_UNTRACKED_SUFFIXES = (
    ".key", ".pem", ".p12", ".pfx",
)
SENSITIVE_UNTRACKED_SUBSTRINGS = (
    "credential", "secret", "token", "password",
)
IGNORED_UNTRACKED_FILENAMES = {
    "completion_report.md",
}


# ---------------------------------------------------------------------------
# Run ID allocation
# ---------------------------------------------------------------------------


def _next_run_id(case_id: str, runs_root: Path) -> str:
    prefix = f"{case_id}-"
    highest = 0
    for path in runs_root.glob(f"{case_id}-*"):
        if not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix):]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return f"{case_id}-{highest + 1:03d}"


# ---------------------------------------------------------------------------
# Workspace creation and fallback chain
# ---------------------------------------------------------------------------


def _cleanup_worktree(workspace: Path, project_path: str) -> None:
    """Remove a git worktree and prune the parent repo's worktree metadata."""
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workspace)],
            cwd=project_path,
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass
    shutil.rmtree(workspace, ignore_errors=True)


def _copy_project_tree(project_path: Path, workspace: Path) -> None:
    """Copy project tree to workspace, excluding .git and forbidden paths.

    Initializes a fresh git repo in the workspace so that collect_run can
    produce a meaningful diff even when the source project has no git history.
    """
    shutil.copytree(
        project_path,
        workspace,
        ignore=shutil.ignore_patterns(
            ".git", ".gitignore", "_reference", ".gradle", ".idea",
            ".claude", ".worktrees", "build",
        ),
        dirs_exist_ok=True,
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    try:
        git_run(["init"], cwd=workspace)
        git_run(["add", "."], cwd=workspace)
        git_run(["commit", "-m", "baseline: copied from project", "--allow-empty"], cwd=workspace)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _extract_commit_fallback(case: ReplayCase, workspace: Path) -> bool:
    """Extract base_commit files via git archive when worktree creation fails.

    Returns True on success, False if the fallback also fails.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    archive_proc: subprocess.Popen[str] | None = None
    tar_proc: subprocess.Popen[str] | None = None
    try:
        archive_env = {**os.environ, "COPYFILE_DISABLE": "1"}
        archive_proc = subprocess.Popen(
            ["git", "archive", case.base_commit],
            cwd=str(case.project_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=archive_env,
        )
        tar_proc = subprocess.Popen(
            ["tar", "-x", "-C", str(workspace)],
            stdin=archive_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if archive_proc.stdout is not None:
            archive_proc.stdout.close()
        tar_proc.communicate(timeout=600)
        archive_proc.wait(timeout=600)
    except Exception:
        for proc in (tar_proc, archive_proc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.communicate(timeout=5)
                except Exception:
                    pass
        return False
    if archive_proc.returncode != 0 or tar_proc.returncode != 0:
        return False
    try:
        git_run(["init"], cwd=workspace)
        git_run(["add", "-A"], cwd=workspace)
        git_run(["commit", "-m", "initial snapshot at base_commit", "--allow-empty"], cwd=workspace)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return True


def _create_worktree(case: ReplayCase, workspace: Path) -> None:
    if not case.base_commit.strip():
        _copy_project_tree(case.project_path, workspace)
        return
    try:
        completed = git_run(
            ["worktree", "add", "--detach", str(workspace), "--", case.base_commit],
            cwd=case.project_path,
            timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("git worktree add timed out")
    if completed.returncode == 0:
        return
    if _extract_commit_fallback(case, workspace):
        return
    _copy_project_tree(case.project_path, workspace)


# ---------------------------------------------------------------------------
# Run preparation
# ---------------------------------------------------------------------------


def _write_completion_template(run: ReplayRun) -> None:
    lines = [
        "# Completion Report",
        "",
        "Fill this out after the agent run. The status line is required for evidence collection.",
        "",
        "status: ",
        "",
        "## Summary",
        "",
        "What was accomplished?",
        "",
        "## Changes Made",
        "",
        "List files changed and why.",
        "Run `git add -A` in the workspace before finishing so that new files appear in the diff.",
        "",
        "## Verification",
        "",
        "What verification commands were run and what was their output?",
        "",
        "## Issues / Notes",
        "",
        "Any blockers, risks, or notes for the scorer.",
    ]
    (run.root / "completion_report_template.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_task(run: ReplayRun) -> None:
    pkg = compile_execution_package(
        run_id=run.id,
        case_id=run.case.id,
        case_task_path=str(run.case.root / "task.md"),
        plan_path=str(run.case.plan_path),
        workspace=str(run.workspace),
        completion_template_path=str(run.root / "completion_report_template.md"),
        verification_commands=run.case.verification_commands,
        task_outcomes=read_task_outcomes(run.case.root),
    )
    renderer = ExecutionPackageRenderer()
    text = renderer.render(pkg)
    (run.root / "TASK.md").write_text(text, encoding="utf-8")


def validate_case(case: ReplayCase) -> list[Diagnostic]:
    """Return missing/incomplete field diagnostics for a case.

    An empty list means the case is complete.
    """
    diagnostics: list[Diagnostic] = []
    if not case.base_source:
        diagnostics.append(Diagnostic("error", "missing_field.base_source", "base_source is empty"))
    if not case.base_confidence or case.base_confidence == "unknown":
        diagnostics.append(Diagnostic("error", "missing_field.base_confidence", "base_confidence is unknown or empty"))
    if not case.base_commit and case.source_type not in ("no_git", "empty_history", "manual"):
        diagnostics.append(Diagnostic("error", "missing_field.base_commit", "base_commit is empty for git-based source"))
    if not case.source_type:
        diagnostics.append(Diagnostic("error", "missing_field.source_type", "source_type is empty"))
    if not case.source_path and case.source_type not in ("git_history", "no_git", "empty_history"):
        diagnostics.append(Diagnostic("error", "missing_field.source_path", "source_path is empty"))
    if not case.selection_reason:
        diagnostics.append(Diagnostic("error", "missing_field.selection_reason", "selection_reason is empty"))
    if not case.plan_path and case.source_type in ("orchestration_kit", "handoff_plan"):
        diagnostics.append(Diagnostic("error", "missing_field.plan_path", "plan_path is required for orchestration_kit/handoff_plan"))
    evidence_md = case.root / "evidence_sources.md"
    if not evidence_md.exists():
        diagnostics.append(Diagnostic("error", "missing_file.evidence_sources", "evidence_sources.md not found"))
    return diagnostics


def _stable_anonymous_runner_id(run_id: str) -> str:
    return f"runner-{stable_hash(run_id, length=10)}"


def prepare_run(case: ReplayCase, *, runs_root: str | Path, runner_label: str) -> ReplayRun:
    diagnostics = validate_case(case)
    if diagnostics:
        raise ValueError(
            f"Case {case.id} is incomplete — missing: {format_diagnostics(diagnostics, separator=', ')}. "
            "Required fields: base_source, base_confidence, source_type, source_path, selection_reason, evidence_sources.md"
        )
    runs = Path(runs_root)
    runs.mkdir(parents=True, exist_ok=True)
    while True:
        run_id = _next_run_id(case.id, runs)
        run_root = runs / run_id
        workspace = run_root / "workspace"
        try:
            run_root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    _create_worktree(case, workspace)
    try:
        run = ReplayRun(run_id, run_root, case, runner_label, workspace)
        write_simple_yaml(
            run_root / "run.yaml",
            {
                "id": run.id,
                "case_id": case.id,
                "runner_label": runner_label,
                "anonymous_runner_id": _stable_anonymous_runner_id(run.id),
                "workspace": workspace,
                "base_commit": case.base_commit,
                "status": "prepared",
            },
        )
        _write_completion_template(run)
        _write_run_task(run)
    except Exception:
        _cleanup_worktree(workspace, case.project_path)
        shutil.rmtree(run_root, ignore_errors=True)
        raise
    return run


# ---------------------------------------------------------------------------
# Case loading (needed by load_run, avoids importing facade)
# ---------------------------------------------------------------------------


def load_case(cases_root: str | Path, case_id: str) -> ReplayCase:
    root = resolve_case_dir(cases_root, case_id)
    data = parse_simple_yaml(root / "case.yaml")
    if not data.get("id"):
        raise ValueError(f"case.yaml at {root} is missing required field 'id'")
    commands = tuple(data.get("verification_commands") or [])
    return ReplayCase(
        id=str(data["id"]),
        root=root,
        project_path=Path(str(data.get("project_path", ""))),
        plan_path=Path(str(data.get("plan_path", ""))),
        base_commit=str(data.get("base_commit", "")),
        verification_commands=commands,
        base_source=str(data.get("base_source", "")),
        base_confidence=str(data.get("base_confidence", "unknown")),
        source_type=str(data.get("source_type", "manual")),
        source_path=str(data.get("source_path", "")),
        selection_reason=str(data.get("selection_reason", "")),
        synthetic_case=str(data.get("synthetic_case", "false")).lower() == "true",
        evidence_sources=tuple(data.get("evidence_sources") or []),
        selected_candidate_id=str(data.get("selected_candidate_id", "")),
        candidate_score=float(str(data.get("candidate_score", "0"))),
        candidate_source_type=str(data.get("candidate_source_type", "")),
        reference_target=str(data.get("reference_target", "")),
    )


# ---------------------------------------------------------------------------
# Run loading and inspection
# ---------------------------------------------------------------------------


def load_run(runs_root: str | Path, run_id: str, *, cases_root: str | Path) -> ReplayRun:
    root = Path(runs_root) / run_id
    run_yaml = root / "run.yaml"
    if not run_yaml.exists():
        raise FileNotFoundError(f"run.yaml not found at {run_yaml}")
    data = parse_simple_yaml(run_yaml)
    case_id = str(data.get("case_id", ""))
    if not case_id:
        raise ValueError(f"run.yaml at {root} is missing required field 'case_id'")
    case = load_case(cases_root, case_id)
    workspace = _resolve_workspace_path(root, data)
    return ReplayRun(
        id=str(data.get("id", run_id)),
        root=root,
        case=case,
        runner_label=str(data.get("runner_label", "")),
        workspace=workspace,
    )


def inspect_run_dir(run_root: str | Path) -> RunHealth:
    """Inspect a run directory and derive a canonical health status from evidence."""
    root = Path(run_root)
    data = parse_simple_yaml(root / "run.yaml") if (root / "run.yaml").exists() else {}
    run_id = str(data.get("id", root.name))
    workspace = _resolve_workspace_path(root, data)
    workspace_exists = workspace.is_dir()
    completion_ws = workspace / "completion_report.md"
    completion_legacy = root / "completion_report.md"
    completion = completion_ws if completion_ws.exists() else completion_legacy
    raw_status = _completion_status(completion)

    raw_base = data.get("base_commit", "")
    base = "" if isinstance(raw_base, list) else str(raw_base).strip()
    if base == "[]":
        base = ""
    diff = ""
    changed_files: tuple[str, ...] = ()
    if workspace_exists:
        diff_ref = base
        if not diff_ref:
            head = _git_capture(workspace, ["rev-parse", "HEAD"]).strip()
            diff_ref = head if head else ""
        diff = _git_capture(workspace, ["diff", diff_ref, "--binary"]) if diff_ref else _git_capture(workspace, ["diff", "--binary"])
        changed_output = _git_capture(workspace, ["diff", "--name-only", diff_ref]) if diff_ref else _git_capture(workspace, ["diff", "--name-only"])
        changed_files = tuple(f for f in changed_output.strip().splitlines() if f)

    status, reason = _derive_canonical_run_status(
        raw_status=raw_status,
        workspace_exists=workspace_exists,
        completion_report_exists=completion.exists(),
        has_result_diff=bool(diff.strip()) and bool(changed_files),
        completion_text=_read_text_if_exists(completion),
    )
    return RunHealth(
        run_id=run_id,
        status=status,
        raw_status=raw_status,
        workspace=workspace,
        workspace_exists=workspace_exists,
        completion_report_exists=completion.exists(),
        has_result_diff=bool(diff.strip()) and bool(changed_files),
        changed_files=changed_files,
        reason=reason,
    )


def doctor_runs(runs_root: str | Path, *, runner_label: str | None = None) -> tuple[RunHealth, ...]:
    root = Path(runs_root)
    if not root.is_dir():
        return ()
    reports: list[RunHealth] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "run.yaml").exists()):
        data = parse_simple_yaml(run_dir / "run.yaml")
        if runner_label is not None and str(data.get("runner_label", "")) != runner_label:
            continue
        reports.append(inspect_run_dir(run_dir))
    return tuple(reports)


def _resolve_workspace_path(run_root: Path, data: dict[str, object]) -> Path:
    raw = str(data.get("workspace", "workspace")).strip() or "workspace"
    workspace = Path(raw)
    if workspace.is_absolute():
        return workspace
    if raw.startswith("runs/") or raw.startswith("runs\\"):
        return run_root.parent.parent / workspace
    return run_root / workspace


# ---------------------------------------------------------------------------
# Telemetry collection
# ---------------------------------------------------------------------------


def _parse_diff_stat(workspace: Path, base_commit: str) -> tuple[int, int]:
    stat = _git_capture(workspace, ["diff", "--stat", base_commit]) if base_commit else _git_capture(workspace, ["diff", "--stat"])
    if not stat.strip():
        return (0, 0)
    added = 0
    deleted = 0
    for part in stat.split(","):
        part = part.strip()
        if "insertion" in part:
            try:
                added = int(part.split()[0])
            except (ValueError, IndexError):
                pass
        elif "deletion" in part:
            try:
                deleted = int(part.split()[0])
            except (ValueError, IndexError):
                pass
    return (added, deleted)


def _detect_forbidden_paths(changed_files: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for f in changed_files:
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern in f:
                hits.append(f)
                break
    return hits


def _detect_suspicious_paths(changed_files: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for f in changed_files:
        fname = f.split("/")[-1].lower()
        for pattern in SUSPICIOUS_PATH_PATTERNS:
            if pattern in fname:
                hits.append(f)
                break
    return hits


def _is_sensitive_untracked_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    lower_path = normalized.lower()
    if any(pattern.replace("\\", "/") in lower_path for pattern in FORBIDDEN_PATH_PATTERNS):
        return True
    parts = [part.lower() for part in normalized.split("/") if part]
    if ".ssh" in parts:
        return True
    fname = parts[-1] if parts else lower_path
    if fname in SAFE_ENV_EXAMPLE_FILENAMES:
        return False
    if fname in SENSITIVE_UNTRACKED_FILENAMES:
        return True
    if fname.startswith(".env."):
        return True
    if fname.endswith(SENSITIVE_UNTRACKED_SUFFIXES):
        return True
    return any(pattern in fname for pattern in SENSITIVE_UNTRACKED_SUBSTRINGS)


def _partition_collectable_untracked(paths: list[str]) -> tuple[list[str], list[str]]:
    collectable: list[str] = []
    skipped: list[str] = []
    for path in paths:
        fname = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if fname in IGNORED_UNTRACKED_FILENAMES:
            continue
        if _is_sensitive_untracked_path(path):
            skipped.append(path)
        else:
            collectable.append(path)
    return collectable, skipped


def _count_untracked(workspace: Path) -> int:
    output = _git_capture(workspace, ["ls-files", "--others", "--exclude-standard"])
    return len([line for line in output.strip().splitlines() if line])


def _count_workspace_commits(workspace: Path, base_commit: str) -> int:
    if not base_commit:
        return 0
    output = _git_capture(
        workspace, ["rev-list", "--count", f"{base_commit}..HEAD"]
    )
    try:
        return int(output.strip()) if output.strip() else 0
    except ValueError:
        return 0


def _detect_rollback_signal(workspace: Path) -> str:
    reflog = _git_capture(workspace, ["reflog", "--oneline", "-5"])
    if not reflog.strip():
        return ""
    signals: list[str] = []
    for line in reflog.splitlines():
        lower = line.lower()
        if "reset" in lower:
            signals.append("reset")
        if "checkout" in lower:
            signals.append("checkout")
        if "rebase" in lower:
            signals.append("rebase")
    return ", ".join(sorted(set(signals))) if signals else ""


def _gather_telemetry(
    run: ReplayRun,
    base_commit: str,
    changed_files: tuple[str, ...],
    diff: str,
) -> Telemetry:
    workspace = run.workspace
    added_lines, deleted_lines = _parse_diff_stat(workspace, base_commit)
    forbidden = _detect_forbidden_paths(changed_files)
    suspicious = _detect_suspicious_paths(changed_files)
    untracked_count = _count_untracked(workspace)
    commit_count = _count_workspace_commits(workspace, base_commit)
    rollback = _detect_rollback_signal(workspace)
    return Telemetry(
        changed_file_count=len(changed_files),
        added_lines=added_lines,
        deleted_lines=deleted_lines,
        forbidden_path_touches=tuple(forbidden),
        suspicious_path_touches=tuple(suspicious),
        untracked_file_count=untracked_count,
        commit_count=commit_count,
        rollback_signal=rollback,
    )


def _write_telemetry_block(
    evidence_data: dict[str, object],
    telemetry: Telemetry,
) -> None:
    evidence_data["telemetry"] = {
        "changed_file_count": telemetry.changed_file_count,
        "added_lines": telemetry.added_lines,
        "deleted_lines": telemetry.deleted_lines,
        "forbidden_path_touches": list(telemetry.forbidden_path_touches),
        "suspicious_path_touches": list(telemetry.suspicious_path_touches),
        "untracked_file_count": telemetry.untracked_file_count,
        "commit_count": telemetry.commit_count,
        "rollback_signal": telemetry.rollback_signal,
    }


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_capture(cwd: Path, args: list[str]) -> str:
    return git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


def _parse_changed_files(status: str) -> list[str]:
    files = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Run metadata reading
# ---------------------------------------------------------------------------


def _read_run_base(run: ReplayRun) -> str:
    """Read the original base_commit saved at prepare-run time."""
    run_yaml = run.root / "run.yaml"
    if run_yaml.exists():
        data = parse_simple_yaml(run_yaml)
        base = data.get("base_commit", "")
        if isinstance(base, list):
            return ""
        base = str(base).strip()
        if base and base != "[]":
            return base
    return ""


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------


def _completion_status(path: Path) -> str:
    if not path.exists():
        return "missing-completion-report"
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^status:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else "reported"


def _normalize_status(raw_status: str) -> str:
    """Map common variant status labels to canonical forms."""
    s = raw_status.lower().strip()
    if s == "complete":
        return "completed"
    if s == "pass":
        return "completed"
    if s == "complete-already-implemented":
        return "completed"
    return s


def _completion_declares_noop(text: str) -> bool:
    lowered = text.lower()
    patterns = (
        "no-op", "no changes needed", "already implemented",
        "already contained", "already had", "base commit already",
        "base commit 已", "已含完整实现",
    )
    return any(pattern in lowered for pattern in patterns)


def _derive_canonical_run_status(
    *,
    raw_status: str,
    workspace_exists: bool,
    completion_report_exists: bool,
    has_result_diff: bool,
    completion_text: str,
) -> tuple[str, str]:
    status = _normalize_status(raw_status)
    if not workspace_exists:
        return ("failed", "workspace missing")
    if status in {"failed", "invalid", "stale"}:
        return (status, f"completion report status is {status}")
    if status in {"partial", "blocked"}:
        return ("partial", f"completion report status is {status}")
    if not completion_report_exists or status == "missing-completion-report":
        return ("failed", "completion_report.md missing")
    if status == "reported":
        return ("partial", "completion report has no explicit status line")
    if status in {"completed", "solved"}:
        if has_result_diff:
            return ("completed", "completed report and non-empty result diff")
        if _completion_declares_noop(completion_text):
            return ("completed", "completed report declares no-op/base already satisfied")
        return ("failed", "completed report has no result diff")
    return ("partial", f"unrecognized completion status: {raw_status}")


def _update_run_yaml_status(run_root: Path, health: RunHealth) -> None:
    run_yaml = run_root / "run.yaml"
    if not run_yaml.exists():
        return
    data = parse_simple_yaml(run_yaml)
    data["status"] = health.status
    data["raw_status"] = health.raw_status
    data["updated_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    data["workspace_exists"] = "yes" if health.workspace_exists else "no"
    data["completion_report_exists"] = "yes" if health.completion_report_exists else "no"
    data["has_result_diff"] = "yes" if health.has_result_diff else "no"
    data["changed_file_count"] = len(health.changed_files)
    data["status_reason"] = health.reason
    write_simple_yaml(run_yaml, data)


# ---------------------------------------------------------------------------
# Core run collection
# ---------------------------------------------------------------------------


def collect_run(run: ReplayRun) -> Evidence:
    evidence_dir = run.root / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    saved_base = _read_run_base(run) or run.case.base_commit
    base = saved_base

    if not base:
        head = _git_capture(run.workspace, ["rev-parse", "HEAD"]).strip()
        diff_ref = head if head else ""
    else:
        diff_ref = base

    diff = _git_capture(run.workspace, ["diff", diff_ref, "--binary"]) if diff_ref else _git_capture(run.workspace, ["diff", "--binary"])
    changed_output = _git_capture(run.workspace, ["diff", "--name-only", diff_ref]) if diff_ref else _git_capture(run.workspace, ["diff", "--name-only"])

    untracked_raw = _git_capture(run.workspace, ["ls-files", "--others", "--exclude-standard"])
    untracked_files = [f for f in untracked_raw.strip().splitlines() if f]
    collectable_untracked, skipped_untracked = _partition_collectable_untracked(untracked_files)
    if untracked_files:
        if collectable_untracked:
            _git_capture(run.workspace, ["add", "--intent-to-add", "--", *collectable_untracked])
        diff = _git_capture(run.workspace, ["diff", diff_ref, "--binary"]) if diff_ref else _git_capture(run.workspace, ["diff", "--binary"])
        changed_output = _git_capture(run.workspace, ["diff", "--name-only", diff_ref]) if diff_ref else _git_capture(run.workspace, ["diff", "--name-only"])

    changed_files = tuple(f for f in changed_output.strip().splitlines() if f)
    diff_path = evidence_dir / "diff.patch"
    diff_path.write_text(diff, encoding="utf-8")
    completion_ws = run.workspace / "completion_report.md"
    completion_legacy = run.root / "completion_report.md"
    completion = completion_ws if completion_ws.exists() else completion_legacy
    run_status = _completion_status(completion)

    missing: list[str] = []
    if run_status == "missing-completion-report":
        missing.append("completion_report.md")
    if not diff.strip():
        missing.append("diff.patch (empty)")
    if not changed_files:
        missing.append("changed_files (none)")

    telemetry = _gather_telemetry(run, diff_ref, changed_files, diff)

    evidence_data: dict[str, object] = {
        "run_id": run.id,
        "status": run_status,
        "changed_files": list(changed_files),
        "diff_path": diff_path,
        "missing_fields": missing,
        "skipped_untracked_files": list(skipped_untracked),
        "skipped_untracked_file_count": len(skipped_untracked),
    }
    _write_telemetry_block(evidence_data, telemetry)
    write_simple_yaml(evidence_dir / "evidence.yaml", evidence_data)
    _update_run_yaml_status(run.root, inspect_run_dir(run.root))
    return Evidence(run.id, run_status, changed_files, diff_path, tuple(missing))
