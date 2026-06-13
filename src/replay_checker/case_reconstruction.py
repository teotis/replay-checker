from __future__ import annotations

import re
from pathlib import Path

from .git_utils import git_output
from .case_depth import SituationProfile
from .packages import _extract_goal_from_plan
from .replay_types import ReplayCase
from .sources import EvidenceRecord
from .yaml_lite import write_simple_yaml

_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".rs", ".go", ".java",
    ".rb", ".c", ".cpp", ".h", ".hpp", ".swift", ".kt", ".scala", ".cs",
    ".php", ".r", ".m", ".mm",
})

_TEST_NAMES: frozenset[str] = frozenset({"test", "tests", "spec", "__tests__", "testing"})

_DOC_EXTENSIONS: frozenset[str] = frozenset({".md", ".rst", ".txt", ".adoc"})

_CONFIG_NAMES: frozenset[str] = frozenset({
    "pyproject.toml", "setup.cfg", "setup.py", "package.json",
    "cargo.toml", "go.mod", "makefile", "tox.ini", ".gitignore",
    "dockerfile", "docker-compose.yml", "cmakelists.txt",
})

_GIT_TIMEOUT = 30


def _git_output(args: list[str], *, cwd: Path) -> str:
    return git_output(args, cwd=cwd, timeout=_GIT_TIMEOUT)


def _git_head(project_path: Path) -> str:
    return _git_output(["rev-parse", "HEAD"], cwd=project_path).strip()


def _first_heading(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.parent.name


def _find_reference_target(project_path: Path, base_commit: str, base_source: str) -> str | None:
    """Find the target commit whose diff serves as oracle evidence."""
    if not base_commit:
        return None

    if base_source == "synthetic_target_parent":
        output = _git_output(
            ["log", "--no-merges", "--format=%H", f"{base_commit}..HEAD", "--"],
            cwd=project_path,
        )
        commits = [c for c in output.strip().splitlines() if c]
        return commits[-1] if commits else None

    if base_source == "synthetic_first_commit":
        return base_commit

    if base_source == "plan_first_commit_parent":
        output = _git_output(
            ["log", "--no-merges", "--format=%H", f"{base_commit}..HEAD", "--"],
            cwd=project_path,
        )
        commits = [c for c in output.strip().splitlines() if c]
        return commits[-1] if commits else None

    if base_source == "plan_first_commit":
        head = _git_head(project_path)
        return head if head and head != base_commit else None

    if base_source in ("user_supplied", "manual", "head_fallback", "head_dirty", "commit", "plan_state_base"):
        head = _git_head(project_path)
        return head if head and head != base_commit else None

    return None


def _save_reference_evidence(case: ReplayCase, project_path: Path) -> None:
    """Persist reference/oracle evidence for a case outside agent-visible paths."""
    if not case.base_commit:
        return
    target = case.reference_target or _find_reference_target(project_path, case.base_commit, case.base_source)
    if not target or target == case.base_commit:
        return

    log_output = _git_output(["log", "--format=%s%n%b", "-1", target], cwd=project_path)
    commit_message = _compact_reference_commit_message(log_output)
    diff_output = _git_output(
        ["diff", case.base_commit, target, "--binary"],
        cwd=project_path,
    )

    changed_output = _git_output(
        ["diff", "--name-only", case.base_commit, target],
        cwd=project_path,
    )
    changed_files = [f for f in changed_output.strip().splitlines() if f]

    ref_dir = case.root / "_reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    _MAX_DIFF_BYTES = 10 * 1024 * 1024
    diff_bytes = diff_output.encode("utf-8", errors="replace")
    if len(diff_bytes) > _MAX_DIFF_BYTES:
        (ref_dir / "diff.patch").write_text(
            diff_output[:_MAX_DIFF_BYTES] + "\n... [truncated — reference diff exceeds 10 MB]\n",
            encoding="utf-8",
        )
    else:
        (ref_dir / "diff.patch").write_text(diff_output, encoding="utf-8")

    write_simple_yaml(
        ref_dir / "reference_metadata.yaml",
        {
            "target_commit": target,
            "base_commit": case.base_commit,
            "commit_message": commit_message,
            "changed_files": changed_files,
        },
    )


def _compact_reference_commit_message(log_output: str) -> str:
    return " ".join(line.strip() for line in log_output.splitlines() if line.strip())


def _classify_changed_files(files: list[str]) -> dict[str, list[str]]:
    """Classify file paths into source/test/config/doc/other categories."""
    categories: dict[str, list[str]] = {"source": [], "test": [], "config": [], "doc": [], "other": []}
    for f in files:
        ext = Path(f).suffix.lower()
        parts_lower = tuple(p.lower() for p in Path(f).parts)

        if any(ind in parts_lower for ind in _TEST_NAMES):
            categories["test"].append(f)
        elif Path(f).name.lower() in _CONFIG_NAMES:
            categories["config"].append(f)
        elif ext in _DOC_EXTENSIONS:
            categories["doc"].append(f)
        elif ext in _SOURCE_EXTENSIONS:
            categories["source"].append(f)
        else:
            categories["other"].append(f)
    return {k: v for k, v in categories.items() if v}


def _is_test_path(path: str) -> bool:
    """Check if a path looks like a test file or lives in a test directory."""
    name = Path(path).name.lower()
    parts_lower = tuple(p.lower() for p in Path(path).parts)
    if name.startswith("test_") or name.endswith("_test.py") or ".test." in name or ".spec." in name:
        return True
    return any(ind in parts_lower for ind in _TEST_NAMES)


def _is_code_like(text: str) -> bool:
    """Heuristic: return True if *text* looks like code rather than prose."""
    lines = text.strip().splitlines()
    if not lines:
        return False
    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("def ", "class ", "function ", "import ", "from ",
                                "const ", "let ", "var ", "fn ", "pub ")):
            code_indicators += 1
        elif stripped.startswith(("+", "-", "@@")) and len(stripped) > 1:
            code_indicators += 2
        elif stripped.startswith(("{", "}", "```", "<!--")):
            code_indicators += 1
        elif re.search(r"^[a-zA-Z_]\w*\s*[=:]\s*", stripped) and len(stripped) > 10:
            code_indicators += 1
    return code_indicators >= 2


def _body_contains_code(lines: list[str]) -> bool:
    """Check if any line in the body looks like source code."""
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("def ", "class ", "function ", "import ", "from ",
                                "const ", "let ", "var ", "fn ", "pub ")):
            code_lines += 1
        elif re.search(r"^[a-zA-Z_]\w*\s*[=:]\s*", stripped) and len(stripped) > 15:
            code_lines += 1
    return code_lines >= 2


def _find_verification_hints(
    project_path: Path,
    changed_files: list[str],
) -> list[str]:
    """Find verification-related signals from project structure and changed files."""
    hints: list[str] = []

    test_files = [f for f in changed_files if _is_test_path(f)]
    if test_files:
        if len(test_files) == 1:
            hints.append(f"Test file is part of the change: `{test_files[0]}`")
        else:
            hints.append(f"Test files are part of the change ({len(test_files)} files)")

    for test_dir_name in ("tests", "test", "spec"):
        test_dir = project_path / test_dir_name
        if test_dir.is_dir():
            hints.append(f"Test directory `{test_dir_name}/` exists in project")
            break

    for config_name in ("Makefile", "pyproject.toml", "package.json", "Cargo.toml"):
        if (project_path / config_name).exists():
            content = ""
            try:
                content = (project_path / config_name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                pass
            if "pytest" in content or "test" in content.lower():
                hints.append(f"Build config `{config_name}` contains test configuration")

    for ci_path in (".github/workflows", ".gitlab-ci.yml", "Jenkinsfile"):
        full = project_path / ci_path
        if full.exists():
            hints.append(f"CI configuration found: `{ci_path}`")
            break

    return hints


def _find_doc_nearby(
    project_path: Path,
    changed_files: list[str],
) -> list[str]:
    """Find documentation headings or README files near changed paths."""
    hints: list[str] = []
    seen_dirs: set[str] = set()

    for f in changed_files:
        parent = str(Path(f).parent)
        if parent in seen_dirs:
            continue
        seen_dirs.add(parent)
        for doc_name in ("README.md", "README", "README.rst"):
            doc_path = project_path / parent / doc_name
            if doc_path.is_file():
                hints.append(f"Documentation: `{parent}/{doc_name}`")
                break

    plans_dir = project_path / "docs" / "plans"
    if plans_dir.is_dir():
        index_files = list(plans_dir.glob("**/INDEX.md"))
        if index_files:
            hints.append("Project contains orchestration plan documentation")

    return hints


def _extract_goal_section(plan_path: Path) -> str:
    """Extract the Goal section from an orchestration kit INDEX.md."""
    if not plan_path.exists():
        return ""
    try:
        text = plan_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    in_goal = False
    goal_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## Goal"):
            in_goal = True
            continue
        if in_goal:
            if line.startswith("## "):
                break
            goal_lines.append(line)

    while goal_lines and not goal_lines[0].strip():
        goal_lines.pop(0)
    while goal_lines and not goal_lines[-1].strip():
        goal_lines.pop()

    return "\n".join(goal_lines).strip()


def _build_reconstruction_context(
    case: ReplayCase,
    project_path: Path,
    evidence_records: list[EvidenceRecord],
) -> dict[str, object]:
    """Build deterministic context for synthetic task reconstruction."""
    target = _find_reference_target(project_path, case.base_commit, case.base_source)
    ctx: dict[str, object] = {
        "target_commit": target or "",
        "commit_subject": "",
        "commit_body": "",
        "changed_files": [],
        "file_categories": {},
        "test_signals": [],
        "verification_hints": [],
        "doc_hints": [],
        "risk_notes": [],
        "scope_summary": "unknown scope",
    }

    if not target:
        ctx["risk_notes"] = ["No target commit identified"]
        return ctx

    full_msg = _git_output(["log", "--format=%B", "-1", target], cwd=project_path)
    msg_lines = full_msg.strip().splitlines()
    ctx["commit_subject"] = msg_lines[0] if msg_lines else ""
    body = "\n".join(msg_lines[1:]).strip() if len(msg_lines) > 1 else ""

    if body:
        safe_body_lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("Signed-off-by:", "Co-authored-by:", "Reviewed-by:", "Acked-by:", "Tested-by:", "Change-Id:", "Closes #", "Fixes #", "Refs #", "See also:", "CR:", "Differential Revision:")):
                continue
            if stripped.startswith(("diff ", "index ", "---", "+++", "@@")):
                break
            safe_body_lines.append(stripped)
        if safe_body_lines and not _body_contains_code(safe_body_lines):
            ctx["commit_body"] = " ".join(safe_body_lines)[:500]
        else:
            ctx["commit_body"] = ""

    if not ctx["commit_body"]:
        ctx["risk_notes"] = ["Commit message only — no detailed body available"]

    changed_output = _git_output(
        ["diff-tree", "--no-commit-id", "--name-only", "-r", target],
        cwd=project_path,
    )
    changed_files = [f for f in changed_output.strip().splitlines() if f]
    ctx["changed_files"] = changed_files

    if not changed_files:
        ctx["risk_notes"] = [*(ctx["risk_notes"] if isinstance(ctx["risk_notes"], list) else []), "No changed files detected"]
    else:
        categories = _classify_changed_files(changed_files)
        ctx["file_categories"] = categories
        scope_parts: list[str] = []
        if categories.get("source"):
            scope_parts.append(f"{len(categories['source'])} source file(s)")
        if categories.get("test"):
            scope_parts.append(f"{len(categories['test'])} test file(s)")
        if categories.get("config"):
            scope_parts.append(f"{len(categories['config'])} config file(s)")
        if categories.get("doc"):
            scope_parts.append(f"{len(categories['doc'])} doc file(s)")
        if categories.get("other"):
            scope_parts.append(f"{len(categories['other'])} other file(s)")
        ctx["scope_summary"] = ", ".join(scope_parts) if scope_parts else "unknown scope"

    ctx["test_signals"] = [f for f in changed_files if _is_test_path(f)]
    ctx["verification_hints"] = _find_verification_hints(project_path, changed_files)
    ctx["doc_hints"] = _find_doc_nearby(project_path, changed_files)

    conversation = [r for r in evidence_records if r.source_type in ("codex_history", "claude_history")]
    if conversation:
        ctx["conversation_summaries"] = [r.summary[:500] for r in conversation[:3]]

    risk_notes: list[str] = list(ctx.get("risk_notes", []))
    if not risk_notes and not ctx.get("test_signals") and not ctx.get("verification_hints"):
        risk_notes.append("No verification hints found — task scope is inferred from commit message only")
    if len(changed_files) == 1:
        risk_notes.append("Single file change — limited scope signals")
    ctx["risk_notes"] = risk_notes

    return ctx


def _write_case_task(
    case: ReplayCase,
    evidence_records: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
    situation_profile: SituationProfile | None = None,
) -> None:
    """Write task.md — the contract that drives agent execution."""
    lines = [
        f"# Replay Case: {case.id}",
        "",
        "This case captures a historical project situation for agent evaluation.",
        "",
        "Do not launch or control an agent automatically from this case.",
        "Replay Checker prepares the contract and evidence paths; the user chooses the agent platform and model.",
        "",
        "## Source",
        f"- Project: `{case.project_path}`",
    ]

    if case.base_commit:
        lines.append(f"- Base commit: `{case.base_commit}`")
    else:
        lines.append("- Base commit: (none — working from current project state)")

    if case.synthetic_case:
        lines.extend([
            f"- Source type: `synthetic` (generated from git history, not a human-written plan)",
            f"- Selection reason: {case.selection_reason}",
            "",
            "## Note",
            "This is a synthetic case reconstructed from git history.",
            "Reference evidence (commit diff) is stored separately and not shown to the executing agent.",
        ])
    elif case.source_type in ("no_git", "empty_history"):
        lines.extend([
            f"- Source type: `{case.source_type}`",
            f"- Selection reason: {case.selection_reason}",
            "",
            "## Note",
            "This case has no git history baseline. The agent should work from the current project state.",
        ])
    else:
        lines.extend([
            f"- Plan package: `{case.plan_path}`",
            f"- Source type: {case.source_type}",
        ])

        if case.plan_path:
            goal_text = _extract_goal_section(Path(case.plan_path)) or _extract_goal_from_plan(str(case.plan_path))
            if goal_text:
                lines.extend(["", "## Goal", goal_text])

    if case.verification_commands:
        lines.extend([
            "",
            "## Verification Commands",
            *[f"- `{command}`" for command in case.verification_commands],
        ])

    _append_situation_profile(lines, situation_profile)
    _append_conversation_evidence(lines, evidence_records)

    (case.root / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_synthetic_task(
    case: ReplayCase,
    project_path: Path,
    evidence_records: tuple[EvidenceRecord, ...] | list[EvidenceRecord] = (),
    situation_profile: SituationProfile | None = None,
) -> None:
    """Write a richer task.md for synthetic cases using safe local evidence."""
    ctx = _build_reconstruction_context(case, project_path, list(evidence_records))

    lines = [
        f"# Replay Case: {case.id}",
        "",
        "This case captures a historical project situation for agent evaluation.",
        "",
        "Do not launch or control an agent automatically from this case.",
        "Replay Checker prepares the contract and evidence paths; the user chooses the agent platform and model.",
        "",
        "## Source",
        f"- Project: `{case.project_path}`",
    ]
    if case.base_commit:
        lines.append(f"- Base commit: `{case.base_commit}`")
    else:
        lines.append("- Base commit: (none — working from current project state)")
    lines.extend([
        f"- Source type: `synthetic` (generated from git history, not a human-written plan)",
        f"- Selection reason: {case.selection_reason}",
        "",
        "## Reconstructed Task",
    ])

    commit_subject = str(ctx.get("commit_subject", ""))
    commit_body = str(ctx.get("commit_body", ""))
    if commit_subject:
        goal_text = commit_subject
        if commit_body:
            goal_text = f"{commit_subject}\n\nAdditional context: {commit_body}"
        lines.extend([
            "",
            "## Goal",
            "",
            goal_text,
            "",
            f"Starting from base commit `{case.base_commit}`, implement the changes described by this commit message.",
            "The original implementation exists as a later commit — produce equivalent changes independently.",
            "",
            "## Reconstructed Task",
            "",
            "### Objective",
            "",
            f"The historical commit message was: **{commit_subject}**",
        ])
        if commit_body:
            lines.extend([
                "",
                f"Additional context: _{commit_body}_",
            ])
    else:
        lines.extend([
            "",
            "## Goal",
            "",
            "Review the project state at the base commit and identify meaningful improvements.",
            "",
            "## Reconstructed Task",
            "",
            "### Objective",
            "",
            "No commit message available. Review the project state at the base commit and identify meaningful improvements.",
        ])

    changed_files: list[str] = list(ctx.get("changed_files", []))
    file_categories: dict[str, list[str]] = dict(ctx.get("file_categories", {}))
    if changed_files:
        lines.extend([
            "",
            "### Scope Hints",
            "",
            f"The target change affects {ctx.get('scope_summary', 'unknown scope')}:",
            "",
        ])
        for cat, files in file_categories.items():
            cat_label = {"source": "Source", "test": "Test", "config": "Config", "doc": "Documentation", "other": "Other"}.get(cat, cat.title())
            lines.append(f"**{cat_label}**:")
            for f in files:
                lines.append(f"- `{f}`")
            lines.append("")

    verification_hints: list[str] = list(ctx.get("verification_hints", []))
    test_signals: list[str] = list(ctx.get("test_signals", []))
    doc_hints: list[str] = list(ctx.get("doc_hints", []))
    if verification_hints or test_signals:
        lines.extend([
            "### Verification Hints",
            "",
        ])
        for hint in verification_hints:
            lines.append(f"- {hint}")
        if test_signals:
            if len(test_signals) == 1:
                lines.append(f"- Changed file includes test: `{test_signals[0]}`")
            else:
                lines.append(f"- {len(test_signals)} test files are part of the change")
        lines.append("")

    if doc_hints:
        lines.extend([
            "### Documentation Nearby",
            "",
        ])
        for hint in doc_hints:
            lines.append(f"- {hint}")
        lines.append("")

    risk_notes: list[str] = list(ctx.get("risk_notes", []))
    if risk_notes:
        lines.extend([
            "### Reconstruction Confidence",
            "",
        ])
        for risk in risk_notes:
            lines.append(f"- {risk}")
        lines.append("")

    _append_conversation_evidence(lines, evidence_records)
    _append_situation_profile(lines, situation_profile)

    (case.root / "task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_situation_profile(
    lines: list[str],
    profile: SituationProfile | None,
) -> None:
    if profile is None:
        return
    if not any((
        profile.problem_context,
        profile.constraints,
        profile.failure_boundaries,
        profile.observable_acceptance,
    )):
        return

    lines.extend([
        "",
        "## Situation Context",
        "",
        profile.problem_context or "No bounded situation context was extracted.",
        "",
        f"- Depth: `{profile.depth_level}` ({profile.depth_score}/100)",
        f"- Episode: `{profile.episode_label}`",
    ])
    if profile.anchor_paths:
        lines.append(f"- Anchor paths: {', '.join(f'`{p}`' for p in profile.anchor_paths[:5])}")

    if profile.constraints:
        lines.extend([
            "",
            "## Constraints",
            "",
            profile.constraints,
        ])

    lines.extend([
        "",
        "## Failure Boundaries",
        "",
        profile.failure_boundaries or "No explicit failure boundary was extracted; treat completion claims conservatively.",
        "",
        "## Observable Acceptance Signals",
        "",
        profile.observable_acceptance or "No task-specific verification signal was extracted; cite concrete evidence before claiming completion.",
    ])


def _append_conversation_evidence(
    lines: list[str],
    records: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
) -> None:
    conversation = [
        record for record in records
        if record.source_type in {"codex_history", "claude_history"}
    ]
    if not conversation:
        return
    lines.extend([
        "",
        "## Conversation Evidence",
        "Bounded summaries from local agent history. Raw logs are not included.",
    ])
    for record in conversation[:5]:
        lines.append(f"- `{record.source_type}`: {record.summary}")
