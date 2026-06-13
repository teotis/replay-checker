"""Replay Checker — compatibility facade.

All lifecycle implementation lives in owning modules.  This facade
re-exports the public surface for backward-compatible imports only.
"""
from __future__ import annotations

from typing import Any

from .replay_types import (
    Evidence,
    IntakeConfig,
    PlanPackage,
    ReplayCase,
    ReplayRun,
    RunHealth,
    Telemetry,
)

# -- intake (case_intake.py) --
from .case_intake import (  # noqa: F401
    batch_intake,
    create_case,
    discover_plan_packages,
    intake,
    load_case,
    _extract_verification,
    _first_heading,
)

# -- run lifecycle (run_ops.py) --
from .run_ops import (  # noqa: F401
    collect_run,
    doctor_runs,
    inspect_run_dir,
    load_run,
    prepare_run,
    validate_case,
)

# -- scoring (scoring_ops.py) --
from .scoring_ops import score_run  # noqa: F401

# -- reporting (reporting.py) --
from .reporting import (  # noqa: F401
    compare_case,
    report_all_cases,
    _build_compare_output,
    _all_run_result_evidence_blocked,
)

# -- scoring evaluation (scoring.py) --
from .scoring import (  # noqa: F401
    CaseProvenance,
    EvidenceGate,
    RunEvaluation,
    build_case_provenance,
    evaluate_run,
    parse_rubric,
)

# -- yaml (yaml_lite.py) --
from .yaml_lite import parse_simple_yaml, write_simple_yaml  # noqa: F401

_write_simple_yaml = write_simple_yaml

# -- constants kept for backward compatibility --
_GIT_TIMEOUT: int = 30

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
    ".env",
    ".env.local",
    ".envrc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}
SAFE_ENV_EXAMPLE_FILENAMES = {
    ".env.example",
    ".env.sample",
    ".env.template",
}
SENSITIVE_UNTRACKED_SUFFIXES = (
    ".key",
    ".pem",
    ".p12",
    ".pfx",
)
SENSITIVE_UNTRACKED_SUBSTRINGS = (
    "credential",
    "secret",
    "token",
    "password",
)
IGNORED_UNTRACKED_FILENAMES = {
    "completion_report.md",
}
