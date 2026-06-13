"""Adapter that turns skill eval definitions into replay cases."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .case_paths import case_inventory_root
from .core import sanitize_slug, stable_hash
from .git_utils import git_output
from .yaml_lite import write_simple_yaml


@dataclass(frozen=True)
class SkillEvalCase:
    skill_name: str
    eval_id: int
    prompt: str
    expected_output: str
    assertions: list[str]
    files: list[str]
    evals_path: Path
    workspace_files: list[str] = field(default_factory=list)


def load_skill_evals(skill_repo_path: str | Path) -> list[SkillEvalCase]:
    """Read all skill evals from <repo>/skills/<skill_name>/evals/evals.json."""
    repo = Path(skill_repo_path).resolve()
    cases: list[SkillEvalCase] = []
    for evals_json in sorted(repo.glob("skills/*/evals/evals.json")):
        skill_dir = evals_json.parents[1]
        skill_name = skill_dir.name
        data = json.loads(evals_json.read_text(encoding="utf-8"))
        for raw_eval in data.get("evals", []):
            cases.append(
                SkillEvalCase(
                    skill_name=skill_name,
                    eval_id=int(raw_eval["id"]),
                    prompt=str(raw_eval.get("prompt", "")),
                    expected_output=str(raw_eval.get("expected_output", "")),
                    assertions=_normalize_assertions(raw_eval.get("assertions", [])),
                    files=[str(item) for item in raw_eval.get("files", [])],
                    evals_path=evals_json,
                    workspace_files=_discover_workspace_files(skill_dir),
                )
            )
    return cases


def write_skill_eval_case(eval_case: SkillEvalCase, cases_root: str | Path, skill_repo: str | Path) -> Path:
    """Write case.yaml, task.md, evidence_sources.md, and eval_rubric.yaml."""
    repo = Path(skill_repo).resolve()
    case_id = _case_id(eval_case)
    case_root = case_inventory_root(cases_root, repo) / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    base_commit, base_source, base_confidence = _repo_base(repo)
    evidence_sources = [f"skill_eval: {eval_case.skill_name} eval {eval_case.eval_id}"]

    write_simple_yaml(
        case_root / "case.yaml",
        {
            "id": case_id,
            "project_path": repo,
            "plan_path": eval_case.evals_path,
            "base_commit": base_commit,
            "base_source": base_source,
            "base_confidence": base_confidence,
            "source_type": "skill_eval",
            "source_path": eval_case.evals_path,
            "selection_reason": f"Skill eval {eval_case.skill_name} #{eval_case.eval_id}",
            "synthetic_case": "false",
            "evidence_sources": evidence_sources,
            "verification_commands": [],
            "skill_name": eval_case.skill_name,
            "eval_id": str(eval_case.eval_id),
            "skill_repo": repo,
            "prompt_preview": eval_case.prompt[:120],
            "assertion_count": str(len(eval_case.assertions)),
            "workspace_files": eval_case.workspace_files,
        },
    )
    (case_root / "task.md").write_text(_build_task_md(eval_case), encoding="utf-8")
    (case_root / "evidence_sources.md").write_text(_build_evidence_sources_md(eval_case), encoding="utf-8")
    write_simple_yaml(case_root / "eval_rubric.yaml", generate_skill_rubric(eval_case))
    return case_root


def generate_skill_rubric(eval_case: SkillEvalCase) -> dict[str, Any]:
    return {
        "result_weight": "80",
        "process_weight": "20",
        "expected_output": eval_case.expected_output,
        "criteria": [f"assertion: {assertion}" for assertion in eval_case.assertions],
    }


def eval_to_case_data(eval_case: SkillEvalCase, cases_root: str | Path) -> dict[str, Any]:
    case_id = _case_id(eval_case)
    evals_path = Path(eval_case.evals_path)
    repo = evals_path.parents[3] if len(evals_path.parents) > 3 else evals_path.parent
    return {
        "id": case_id,
        "source_type": "skill_eval",
        "skill_name": eval_case.skill_name,
        "eval_id": eval_case.eval_id,
        "prompt": eval_case.prompt,
        "expected_output": eval_case.expected_output,
        "assertions": eval_case.assertions,
        "files": eval_case.files,
        "workspace_files": eval_case.workspace_files,
        "root": case_inventory_root(cases_root, repo) / case_id,
    }


def _normalize_assertions(raw: object) -> list[str]:
    assertions: list[str] = []
    if not isinstance(raw, list):
        return assertions
    for item in raw:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
        else:
            text = str(item).strip()
        if text:
            assertions.append(text)
    return assertions


def _discover_workspace_files(skill_dir: Path) -> list[str]:
    files: list[str] = []
    for subdir in ("references", "scripts"):
        root = skill_dir / subdir
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(f"{subdir}/{path.relative_to(root).as_posix()}")
    return files


def _repo_base(repo: Path) -> tuple[str, str, str]:
    if not (repo / ".git").exists():
        return ("", "no_git", "low")
    output = git_output(["rev-parse", "HEAD"], cwd=repo)
    if not output:
        return ("", "head_unavailable", "low")
    return (output.strip(), "skill_repo_head", "high")


def _build_task_md(eval_case: SkillEvalCase) -> str:
    lines = [
        f"# Skill Eval: {eval_case.skill_name} #{eval_case.eval_id}",
        "",
        "This case is generated from a skill evaluation definition.",
        "The prompt below is the primary task to execute.",
        "",
        "## Prompt",
        "",
        eval_case.prompt,
        "",
        "## Expected Output",
        "",
        eval_case.expected_output,
    ]
    if eval_case.workspace_files:
        lines.extend([
            "",
            "## Workspace Context Files",
            "The following files from the skill repository may be useful:",
            *[f"- `{path}`" for path in eval_case.workspace_files],
        ])
    lines.append("")
    return "\n".join(lines)


def _build_evidence_sources_md(eval_case: SkillEvalCase) -> str:
    return "\n".join([
        f"# Evidence Sources: {_case_id(eval_case)}",
        "",
        "These are bounded summaries. Raw private logs are not copied into the case.",
        "",
        f"- `skill_eval` confidence=high: {eval_case.skill_name} eval {eval_case.eval_id}",
        f"  - path: `{eval_case.evals_path}`",
        "",
    ])


def dedupe_eval_cases(cases_root: str | Path, *, dry_run: bool = False) -> list[tuple[str, str]]:
    """Remove duplicate eval cases, keeping the newest version per skill-eval group.

    Two cases are duplicates if they share the same ``{skill_name}-eval{eval_id}``
    prefix but differ in their trailing hash suffix (i.e. the prompt changed).

    Returns a list of ``(kept_id, removed_id)`` pairs for audit.
    """
    root = Path(cases_root)
    if not root.exists():
        return []

    # Group case dirs by their normalized prefix (without the trailing hash).
    groups: dict[str, list[Path]] = defaultdict(list)
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        prefix = _eval_case_prefix(child.name)
        if prefix:
            groups[prefix].append(child)

    removed: list[tuple[str, str]] = []
    for prefix, dirs in groups.items():
        if len(dirs) < 2:
            continue
        # Sort by directory mtime descending — newest first.
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        keep = dirs[0]
        for obsolete in dirs[1:]:
            kept_name = keep.name
            removed_name = obsolete.name
            if not dry_run:
                if not _is_safe_case_dir(root, obsolete):
                    continue
                shutil.rmtree(obsolete)
            removed.append((kept_name, removed_name))

    return removed


def _is_safe_case_dir(root: Path, candidate: Path) -> bool:
    if candidate.is_symlink():
        return False
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return candidate.is_dir()


def _eval_case_prefix(name: str) -> str | None:
    """Extract the ``{skill}-eval{N}`` prefix from a case directory name.

    Returns *None* when the name does not look like an eval case.
    """
    # Find the last `-` that separates the eval id from the trailing hash.
    parts = name.rsplit("-", 1)
    if len(parts) != 2:
        return None
    prefix, suffix = parts
    # The suffix must look like a hex hash (at least 4 chars).
    if len(suffix) < 4 or not all(c in "0123456789abcdef" for c in suffix):
        return None
    # The prefix must contain "-eval".
    if "-eval" not in prefix:
        return None
    return prefix


def _case_id(eval_case: SkillEvalCase) -> str:
    slug = sanitize_slug(f"{eval_case.skill_name}-eval{eval_case.eval_id}")
    digest = stable_hash(f"{eval_case.skill_name}:{eval_case.eval_id}:{eval_case.prompt}")
    return f"{slug}-{digest}"
