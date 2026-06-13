"""Shared replay dataclasses used across lifecycle modules.

These types are dependency-light — they import only stdlib and contain no
lifecycle or scoring logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PlanPackage:
    plan_path: Path
    title: str
    package_count: int


@dataclass(frozen=True)
class ReplayCase:
    id: str
    root: Path
    project_path: Path
    plan_path: Path
    base_commit: str
    verification_commands: tuple[str, ...] = ()
    base_source: str = ""
    base_confidence: str = "unknown"
    source_type: str = "manual"
    source_path: str = ""
    selection_reason: str = ""
    synthetic_case: bool = False
    evidence_sources: tuple[str, ...] = ()
    selected_candidate_id: str = ""
    candidate_score: float = 0.0
    candidate_source_type: str = ""
    reference_target: str = ""
    depth_score: int = 0
    depth_level: str = ""
    episode_label: str = ""


@dataclass(frozen=True)
class ReplayRun:
    id: str
    root: Path
    case: ReplayCase
    runner_label: str
    workspace: Path


@dataclass(frozen=True)
class Evidence:
    run_id: str
    status: str
    changed_files: tuple[str, ...] = field(default_factory=tuple)
    diff_path: Path | None = None
    missing_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Telemetry:
    changed_file_count: int
    added_lines: int
    deleted_lines: int
    forbidden_path_touches: tuple[str, ...]
    suspicious_path_touches: tuple[str, ...]
    untracked_file_count: int
    commit_count: int
    rollback_signal: str


@dataclass(frozen=True)
class RunHealth:
    run_id: str
    status: str
    raw_status: str
    workspace: Path
    workspace_exists: bool
    completion_report_exists: bool
    has_result_diff: bool
    changed_files: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class IntakeConfig:
    """Bundled parameters for replay case intake operations."""
    project_path: Path
    cases_root: Path
    codex_history_roots: tuple[Path, ...] = ()
    claude_history_roots: tuple[Path, ...] = ()
    allow_duplicate: bool = False
