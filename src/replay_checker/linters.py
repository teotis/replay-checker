from __future__ import annotations

import os
import re
from pathlib import Path

from .core import Diagnostic
from .replay import parse_simple_yaml


def _safe_is_file(path: Path) -> bool:
    """Check if a file exists at the exact given path.

    On case-insensitive filesystems (macOS APFS), Path.exists() may return True
    for different casings. This checks the actual directory listing for an exact match.
    """
    if not path.is_file():
        return False
    # Verify the name matches exactly via directory listing
    parent = path.parent
    if not parent.is_dir():
        return False
    return path.name in os.listdir(parent)


def lint_task(run_root: Path) -> list[Diagnostic]:
    """Validate a task package directory for structural completeness.

    Checks run-level (TASK.md) and case-level (task.md) independently.
    Returns a list of Diagnostic objects (empty = lint clean).
    """
    diagnostics: list[Diagnostic] = []

    task_path = run_root / "TASK.md"
    case_path = run_root / "task.md"

    has_run_task = _safe_is_file(task_path)
    has_case_task = _safe_is_file(case_path)

    if not has_run_task and not has_case_task:
        diagnostics.append(Diagnostic("error", "missing.task_file", "missing task file: no TASK.md or task.md found in directory"))
        return diagnostics

    if has_run_task:
        content = task_path.read_text(encoding="utf-8", errors="ignore")

        if "Protocol Version:" not in content:
            diagnostics.append(Diagnostic("error", "missing.protocol_version", "missing protocol version: no 'Protocol Version:' line found in TASK.md"))

        if not _has_heading_content(content, ("goal", "source", "reconstructed task")):
            diagnostics.append(Diagnostic("error", "missing.goal_or_source", "no goal or source statement: expected a goal, source, or reconstructed task section"))

        if "## Agent Output Contract" not in content:
            diagnostics.append(Diagnostic("error", "missing.output_contract", "missing output contract: TASK.md should contain '## Agent Output Contract' section"))

        if "## Forbidden Access" not in content:
            diagnostics.append(Diagnostic("error", "missing.forbidden_access", "missing forbidden access section: TASK.md should contain '## Forbidden Access' section"))

        if "## Evidence Requirements" not in content:
            diagnostics.append(Diagnostic("error", "missing.evidence_requirements", "missing evidence requirements section: TASK.md should contain '## Evidence Requirements' section"))

        if not _has_completion_report(content):
            diagnostics.append(Diagnostic("error", "missing.completion_report_schema", "missing completion report schema: TASK.md should reference a completion_report file or contain a '## Completion Report' section"))

        if _references_leaked(content):
            diagnostics.append(Diagnostic("error", "security.reference_leak", "reference leak: task file contains leaked _reference/ content (diff/patch data) that must not be visible to the executing agent"))

        if "## Verification Commands" in content:
            for entry in _extract_verification_entries(content):
                if _looks_like_prose(entry):
                    diagnostics.append(Diagnostic("warning", "content.prose_verification", f"non-command verification entry: '{entry}' under ## Verification Commands is not a runnable shell command"))

        # Goal quality check
        if "## Goal" in content:
            goal_text = content.split("## Goal")[1].split("##")[0].strip()
            if len(goal_text) < 20:
                diagnostics.append(Diagnostic("warning", "content.goal_too_short", f"goal 过于简略：{len(goal_text)} 字符（建议 ≥20）"))
            if _is_template_goal(goal_text):
                diagnostics.append(Diagnostic("warning", "content.template_goal", "template goal detected: the goal is the default package-compiler template — the case was likely generated without extracting a concrete task description from the source plan"))
        elif "## Goal" not in content and "## Source" not in content:
            diagnostics.append(Diagnostic("error", "missing.goal_or_source", "no goal or source section found in TASK.md"))

    if has_case_task:
        case_content = case_path.read_text(encoding="utf-8", errors="ignore")

        if not _has_heading_content(case_content, ("source", "reconstructed task", "goal")):
            diagnostics.append(Diagnostic("error", "missing.goal_or_source", "case task.md has no source or goal statement"))

        if _has_base_commit(case_path):
            base = _read_base_commit(case_path)
            if base and not _is_valid_base(base):
                diagnostics.append(Diagnostic("error", "content.invalid_base_commit", f"base_commit in case.yaml is invalid: {base}"))

        if _references_leaked(case_content):
            diagnostics.append(Diagnostic("error", "security.reference_leak", "reference leak: case task.md contains leaked _reference/ content (diff/patch data) that must not be visible to the executing agent"))

    return diagnostics


def lint_score(run_root: Path) -> list[Diagnostic]:
    """Validate a scoring package for structural completeness.

    Checks scoring_package.md and run.yaml for required sections
    and detects runner label leakage.

    Returns a list of Diagnostic objects (empty = lint clean).
    """
    diagnostics: list[Diagnostic] = []

    scoring_path = run_root / "scoring_package.md"
    if not scoring_path.is_file():
        diagnostics.append(Diagnostic("error", "missing.scoring_package", "missing scoring_package.md"))
        return diagnostics

    content = scoring_path.read_text(encoding="utf-8", errors="ignore")
    run_yaml_path = run_root / "run.yaml"
    has_run_yaml = run_yaml_path.is_file()

    if "Protocol Version:" not in content:
        diagnostics.append(Diagnostic("error", "missing.protocol_version", "missing protocol version: no 'Protocol Version:' line found in scoring package"))

    if "## Required Inputs" not in content:
        diagnostics.append(Diagnostic("error", "missing.required_inputs", "missing evidence gate: scoring package should contain '## Required Inputs' section"))

    if "## Rubric Weights" not in content:
        diagnostics.append(Diagnostic("error", "missing.rubric_weights", "missing score ceilings: scoring package should contain '## Rubric Weights' section"))

    if "## Invalid Score Conditions" not in content:
        diagnostics.append(Diagnostic("error", "missing.invalid_conditions", "missing invalid score conditions: scoring package should contain '## Invalid Score Conditions' section"))

    if has_run_yaml:
        runner_label = parse_simple_yaml(run_yaml_path).get("runner_label", "")
        if runner_label and str(runner_label) in content:
            diagnostics.append(Diagnostic("error", "security.runner_label_leak", f"runner label leakage: '{runner_label}' found in scoring package; runner identity must be anonymized"))

    if "## Evidence References" not in content:
        diagnostics.append(Diagnostic("error", "missing.evidence_references", "missing evidence references: scoring package should contain '## Evidence References' section"))

    # Check provenance for non-manual cases
    is_manual = _scoring_package_is_manual(content, run_root)
    if not is_manual:
        if "## Case Provenance & Source Risk" not in content:
            diagnostics.append(Diagnostic("error", "missing.case_provenance", "missing case provenance: non-manual scoring package should contain '## Case Provenance & Source Risk' section"))
        else:
            _check_provenance_content(content, diagnostics)

    # Check for raw history log leakage
    _check_raw_log_leakage(content, diagnostics)

    # Check for reference/oracle leakage in scoring package
    _check_scoring_reference_leakage(content, diagnostics)

    return diagnostics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_heading_content(content: str, keywords: tuple[str, ...]) -> bool:
    """Check if any H1 or H2 heading contains one of the given keywords."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") or stripped.startswith("## "):
            heading_lower = stripped.lower()
            if any(kw in heading_lower for kw in keywords):
                return True
    return False


def _has_completion_report(content: str) -> bool:
    """Check if the task file references a completion report template."""
    # Check for completion_report as a filename with extension or as a section heading
    if re.search(r"completion_report", content):
        return True
    if "## completion report" in content.lower():
        return True
    return False


def _references_leaked(content: str) -> bool:
    """Detect leaked _reference/ content (diff data) in agent-visible text.

    Does not flag `_reference/` in Forbidden Access / policy statements.
    """
    for line in content.splitlines():
        line_stripped = line.strip().lower()
        # Skip lines that are policy/warning statements
        if any(kw in line_stripped for kw in ("do not", "must not", "forbidden", "should not")):
            continue
        if re.search(r"_reference[/\\]", line):
            return True
    return False


def _is_valid_base(base: str) -> bool:
    """Check whether a base commit looks like a valid git SHA."""
    return bool(re.match(r"^[0-9a-f]{6,40}$", base))


def _has_base_commit(case_path: Path) -> bool:
    """Check if the task file has an explicit base commit reference."""
    content = case_path.read_text(encoding="utf-8", errors="ignore")
    return bool(re.search(r"base commit|Base commit|base_commit", content, re.IGNORECASE))


def _read_base_commit(case_path: Path) -> str:
    """Extract the base commit value from case.yaml in the parent directory."""
    case_yaml = case_path.parent / "case.yaml"
    if case_yaml.exists():
        data = parse_simple_yaml(case_yaml)
        return str(data.get("base_commit", ""))
    return ""


def _extract_verification_entries(content: str) -> list[str]:
    """Extract bullet entries from the ## Verification Commands section."""
    in_section = False
    entries: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## verification commands":
            in_section = True
            continue
        if in_section and (stripped.startswith("## ") or stripped.startswith("# ")):
            break
        if in_section and stripped.startswith("- "):
            # Strip leading "- " and backticks
            entry = stripped[2:].strip().strip("`")
            if entry:
                entries.append(entry)
    return entries


def _scoring_package_is_manual(content: str, run_root: Path) -> bool:
    """Check if the scoring package is for a manual case."""
    if "Primary source type: `manual`" in content:
        return True
    if "source_type: manual" in content.lower():
        return True
    case_yaml = run_root.parent.parent / "cases"
    return False


def _check_provenance_content(content: str, diagnostics: list[Diagnostic]) -> None:
    """Validate provenance section content in scoring package."""
    required_fields = [
        ("Primary source type", "missing.primary_source_type", "missing primary source type in provenance section"),
        ("Base commit source", "missing.base_commit_source", "missing base commit source in provenance section"),
        ("Merged confidence", "missing.merged_confidence", "missing merged confidence in provenance section"),
        ("Overall risk level", "missing.overall_risk_level", "missing overall risk level in provenance section"),
    ]
    for field, code, message in required_fields:
        if field not in content:
            diagnostics.append(Diagnostic("error", code, message))


def _check_raw_log_leakage(content: str, diagnostics: list[Diagnostic]) -> None:
    """Detect raw conversation log content leaked into scoring package."""
    raw_indicators = [
        r'"role"\s*:\s*"user"',
        r'"role"\s*:\s*"assistant"',
        r'"content"\s*:\s*"',
        r'"type"\s*:\s*"message"',
    ]
    for pattern in raw_indicators:
        if re.search(pattern, content):
            diagnostics.append(Diagnostic(
                "error", "security.raw_log_leak",
                f"raw log leakage: scoring package contains raw conversation data (pattern: {pattern})",
            ))
            break


def _check_scoring_reference_leakage(content: str, diagnostics: list[Diagnostic]) -> None:
    """Detect reference/oracle content leaked into scoring package.

    _reference/ inside backticks (`_reference/`) is a policy/documentation
    reference, not leaked data. Only flag bare path uses.
    """
    for line in content.splitlines():
        line_stripped = line.strip().lower()
        # Bare _reference/ outside backticks is leakage
        if re.search(r"(?<!`)_reference[/\\]", line):
            diagnostics.append(Diagnostic(
                "error", "security.reference_leak",
                "reference leak: scoring package contains leaked _reference/ content",
            ))
            break


_TEMPLATE_GOALS = (
    "Replay a historical project situation and produce equivalent changes independently.",
)


def _is_template_goal(text: str) -> bool:
    """Detect the default package-compiler goal that signals a hollow task."""
    return text.strip() in _TEMPLATE_GOALS


def _looks_like_prose(text: str) -> bool:
    """Heuristic: detect natural-language prose masquerading as a shell command.

    Returns True if the text reads like an English sentence rather than a
    runnable shell command.  Common false-positive guards: paths with '/',
    CLI tools like 'rtk', flags like '--', and short tokens joined by ':' or
    '='.
    """
    words = text.split()
    if not words:
        return False

    # Short single-token entries (paths, flags, tool invocations) are fine
    if len(words) <= 2:
        return False

    # Contains shell syntax → definitely not prose
    if any(ch in text for ch in ("|", ";", "&&", "||", ">>", "$(")):
        return False

    first = words[0]
    # Articles / pronouns at start almost always mean prose
    if first.lower() in ("the", "a", "an", "any", "all", "every", "each", "make"):
        return True

    # "should" / "must" / "can" mid-sentence is a strong prose signal
    lower_words = {w.lower() for w in words}
    if lower_words & {"should", "must", "might", "could", "would", "please"}:
        return True

    # "to <verb>" infinitive phrase → prose
    if len(words) >= 3 and words[0].lower() == "to" and words[1].isalpha():
        return True

    # Sentences that are mostly lowercase alpha words (≥ 3 words, all alpha)
    # with no digits, flags, or path separators are almost always prose.
    alpha_words = [w for w in words if w.isalpha()]
    if len(alpha_words) >= 3 and len(alpha_words) == len(words) and all(w.islower() for w in alpha_words):
        return True

    # Capitalized natural-language sentence (3+ words, first word capitalized,
    # rest all lowercase alpha).  Shell commands almost never start this way.
    if (
        len(words) >= 3
        and len(alpha_words) == len(words)
        and words[0][0].isupper()
        and all(w.islower() for w in words[1:])
    ):
        return True

    return False
