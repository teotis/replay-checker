from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .case_dedupe import find_all_duplicates
from .case_paths import iter_case_dirs
from .case_validation import case_quality_gate_issues
from .core import Diagnostic
from .run_ops import load_case
from .yaml_lite import parse_simple_yaml


@dataclass(frozen=True)
class CaseAuditIssue:
    case_id: str
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class CaseAuditReport:
    total_cases: int
    issues: tuple[CaseAuditIssue, ...]
    blocking_case_ids: tuple[str, ...]
    low_confidence_case_ids: tuple[str, ...]
    duplicate_case_ids: tuple[str, ...]
    removed_case_ids: tuple[str, ...]

    @property
    def has_blocking_issues(self) -> bool:
        return bool(self.blocking_case_ids)


def audit_cases(
    cases_root: str | Path,
    *,
    prune_unusable: bool = False,
    prune_low_confidence: bool = False,
    prune_duplicates: bool = False,
    dry_run: bool = False,
) -> CaseAuditReport:
    """Audit case inventory quality and optionally remove policy failures."""
    root = Path(cases_root)
    case_dirs = iter_case_dirs(root)
    issues: list[CaseAuditIssue] = []
    blocking_ids: set[str] = set()
    low_confidence_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    dirs_by_id = {path.name: path for path in case_dirs}

    for case_dir in case_dirs:
        case_id = case_dir.name
        try:
            case = load_case(root, case_id)
            for diagnostic in case_quality_gate_issues(case):
                issues.append(_issue_from_diagnostic(case_id, diagnostic))
                if diagnostic.severity == "error":
                    blocking_ids.add(case_id)
            metadata = parse_simple_yaml(case_dir / "case.yaml")
        except Exception as exc:
            issues.append(CaseAuditIssue(case_id, "error", "case.load_failed", str(exc)))
            blocking_ids.add(case_id)
            continue

        if str(metadata.get("base_confidence", "")).strip().lower() == "low":
            issues.append(CaseAuditIssue(
                case_id,
                "warning",
                "case.low_confidence",
                "base_confidence is low",
            ))
            low_confidence_ids.add(case_id)

        reachability_issue = _base_reachability_issue(case_id, metadata)
        if reachability_issue is not None:
            issues.append(reachability_issue)
            blocking_ids.add(case_id)

    exact_clusters, _likely_pairs = find_all_duplicates(root)
    for _kept, dropped, reasons in exact_clusters:
        duplicate_ids.add(dropped)
        issues.append(CaseAuditIssue(
            dropped,
            "warning",
            "case.exact_duplicate",
            f"exact duplicate reference: {', '.join(reasons)}",
        ))

    remove_ids: set[str] = set()
    if prune_unusable:
        remove_ids.update(blocking_ids)
    if prune_low_confidence:
        remove_ids.update(low_confidence_ids)
    if prune_duplicates:
        remove_ids.update(duplicate_ids)

    removed = tuple(sorted(remove_ids))
    if not dry_run:
        for case_id in removed:
            case_dir = dirs_by_id.get(case_id)
            if case_dir is not None:
                shutil.rmtree(case_dir, ignore_errors=True)

    return CaseAuditReport(
        total_cases=len(case_dirs),
        issues=tuple(issues),
        blocking_case_ids=tuple(sorted(blocking_ids)),
        low_confidence_case_ids=tuple(sorted(low_confidence_ids)),
        duplicate_case_ids=tuple(sorted(duplicate_ids)),
        removed_case_ids=removed,
    )


def _issue_from_diagnostic(case_id: str, diagnostic: Diagnostic) -> CaseAuditIssue:
    return CaseAuditIssue(
        case_id=case_id,
        severity=diagnostic.severity,
        code=diagnostic.code,
        message=diagnostic.message,
    )


def _base_reachability_issue(case_id: str, metadata: dict[str, object]) -> CaseAuditIssue | None:
    project = Path(str(metadata.get("project_path", "")))
    if not project.exists():
        return CaseAuditIssue(case_id, "error", "case.project_missing", f"project path not found: {project}")

    base = str(metadata.get("base_commit", "")).strip()
    if not base:
        return None
    if not (project / ".git").exists():
        return None

    try:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{base}^{{commit}}"],
            cwd=project,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return CaseAuditIssue(case_id, "error", "case.base_commit_unresolved", base)
    if result.returncode != 0:
        return CaseAuditIssue(case_id, "error", "case.base_commit_unresolved", base)
    return None
