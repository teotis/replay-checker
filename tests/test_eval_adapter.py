from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from replay_checker.replay import load_case, parse_simple_yaml, validate_case


ROOT = Path(__file__).resolve().parents[1]


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=True)


def _make_skill_repo(base: Path, skill_name: str, evals: list[dict], *, references: list[str] | None = None) -> Path:
    skill_dir = base / "skills" / skill_name
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"# {skill_name}\n", encoding="utf-8")
    (evals_dir / "evals.json").write_text(json.dumps({"skill_name": skill_name, "evals": evals}), encoding="utf-8")
    if references:
        ref_dir = skill_dir / "references"
        ref_dir.mkdir(parents=True)
        for ref in references:
            (ref_dir / ref).write_text(f"# {ref}\n", encoding="utf-8")
    _git(["init"], base)
    _git(["config", "user.email", "test@example.com"], base)
    _git(["config", "user.name", "Test User"], base)
    _git(["add", "."], base)
    _git(["commit", "-m", "add skill evals"], base)
    return base


def _sample_evals() -> list[dict]:
    return [
        {
            "id": 1,
            "prompt": "Implement feature X with correct error handling.",
            "expected_output": "Produces working code with proper error messages.",
            "files": [],
            "assertions": [
                {"text": "Handles edge case Y."},
                {"text": "Returns correct error code."},
            ],
        },
        {
            "id": 2,
            "prompt": "Refactor module Z for better testability.",
            "expected_output": "Module is refactored with dependency injection.",
            "files": [],
            "assertions": ["All existing tests pass."],
        },
    ]


def test_load_skill_evals_normalizes_assertions_and_reference_files(tmp_path):
    from replay_checker.eval_adapter import load_skill_evals

    repo = _make_skill_repo(tmp_path / "skill_repo", "my-skill", _sample_evals(), references=["guide.md"])

    cases = load_skill_evals(repo)

    assert len(cases) == 2
    assert cases[0].skill_name == "my-skill"
    assert cases[0].eval_id == 1
    assert cases[0].assertions == ["Handles edge case Y.", "Returns correct error code."]
    assert cases[0].workspace_files == ["references/guide.md"]
    assert cases[1].assertions == ["All existing tests pass."]


def test_write_skill_eval_case_uses_complete_replay_contract(tmp_path):
    from replay_checker.eval_adapter import load_skill_evals, write_skill_eval_case

    repo = _make_skill_repo(tmp_path / "skill_repo", "my-skill", _sample_evals())
    cases_root = tmp_path / "cases"
    eval_case = load_skill_evals(repo)[0]

    case_root = write_skill_eval_case(eval_case, cases_root, repo)

    case_yaml = parse_simple_yaml(case_root / "case.yaml")
    assert case_yaml["source_type"] == "skill_eval"
    assert case_yaml["project_path"] == str(repo.resolve())
    assert case_yaml["plan_path"].endswith("skills/my-skill/evals/evals.json")
    assert case_yaml["base_source"] == "skill_repo_head"
    assert case_yaml["base_confidence"] == "high"
    assert case_yaml["source_path"].endswith("skills/my-skill/evals/evals.json")
    assert case_yaml["selection_reason"] == "Skill eval my-skill #1"
    assert case_yaml["synthetic_case"] == "false"
    assert case_yaml["verification_commands"] == []
    assert "skill_eval: my-skill eval 1" in case_yaml["evidence_sources"]
    assert (case_root / "task.md").exists()
    assert (case_root / "evidence_sources.md").exists()
    assert (case_root / "eval_rubric.yaml").exists()

    loaded = load_case(cases_root, case_root.name)
    assert validate_case(loaded) == []


def test_skill_eval_rubric_contains_assertions_not_verification_commands(tmp_path):
    from replay_checker.eval_adapter import load_skill_evals, write_skill_eval_case

    repo = _make_skill_repo(tmp_path / "skill_repo", "my-skill", _sample_evals())
    case_root = write_skill_eval_case(load_skill_evals(repo)[0], tmp_path / "cases", repo)

    rubric = parse_simple_yaml(case_root / "eval_rubric.yaml")
    assert rubric["result_weight"] == "80"
    assert rubric["process_weight"] == "20"
    assert rubric["criteria"] == [
        "assertion: Handles edge case Y.",
        "assertion: Returns correct error code.",
    ]


def test_extract_skill_evals_cli_filters_and_reports_summary(tmp_path):
    repo = _make_skill_repo(tmp_path / "skill_repo", "my-skill", _sample_evals())
    _make_skill_repo(repo, "other-skill", [{"id": 1, "prompt": "Other", "assertions": []}])
    cases_root = tmp_path / "cases"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "replay.py"),
            "extract-skill-evals",
            "--skill-repo",
            str(repo),
            "--cases-root",
            str(cases_root),
            "--skills",
            "my-skill",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Created case:" in result.stdout
    assert "(my-skill eval 1)" in result.stdout
    assert "Extracted 2 cases from 1 skills." in result.stdout
    assert len(list(cases_root.glob("*/case.yaml"))) == 2
