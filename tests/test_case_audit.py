from __future__ import annotations

from pathlib import Path

from replay_checker.case_audit import audit_cases
from replay_checker.yaml_lite import write_simple_yaml


def _write_case(
    root: Path,
    case_id: str,
    *,
    base_confidence: str = "high",
    verification_commands: tuple[str, ...] = ("make test",),
) -> Path:
    case_dir = root / "inventory" / "demo" / case_id
    case_dir.mkdir(parents=True)
    write_simple_yaml(
        case_dir / "case.yaml",
        {
            "id": case_id,
            "project_path": str(root),
            "plan_path": str(root / "plan.md"),
            "base_commit": "abc123",
            "base_source": "plan_state_base" if base_confidence != "low" else "head_dirty",
            "base_confidence": base_confidence,
            "source_type": "orchestration_kit",
            "source_path": str(root / "plan.md"),
            "selection_reason": "test case",
            "synthetic_case": "false",
            "evidence_sources": ["plan: test"],
            "verification_commands": list(verification_commands),
        },
    )
    (case_dir / "task.md").write_text(
        "# Replay Case\n\n## Goal\n\nImplement a useful behavior with tests.\n",
        encoding="utf-8",
    )
    (case_dir / "evidence_sources.md").write_text("- plan: test\n", encoding="utf-8")
    ref = case_dir / "_reference"
    ref.mkdir()
    (ref / "diff.patch").write_text("diff --git a/src/app.py b/src/app.py\n", encoding="utf-8")
    write_simple_yaml(ref / "reference_metadata.yaml", {"changed_files": ["src/app.py"]})
    return case_dir


def test_audit_cases_reports_low_confidence_cases(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    _write_case(cases_root, "solid-case")
    _write_case(cases_root, "low-case", base_confidence="low")

    report = audit_cases(cases_root)

    assert report.total_cases == 2
    assert report.low_confidence_case_ids == ("low-case",)
    assert report.blocking_case_ids == ()


def test_audit_cases_can_prune_low_confidence_cases(tmp_path: Path) -> None:
    cases_root = tmp_path / "cases"
    _write_case(cases_root, "solid-case")
    low_case = _write_case(cases_root, "low-case", base_confidence="low")

    report = audit_cases(cases_root, prune_low_confidence=True)

    assert report.removed_case_ids == ("low-case",)
    assert not low_case.exists()
    assert (cases_root / "inventory" / "demo" / "solid-case").exists()


def test_audit_cases_cli_prunes_low_confidence_cases(tmp_path: Path, capsys) -> None:
    from tools.replay import build_parser

    cases_root = tmp_path / "cases"
    _write_case(cases_root, "solid-case")
    low_case = _write_case(cases_root, "low-case", base_confidence="low")

    parser = build_parser()
    args = parser.parse_args([
        "audit-cases",
        "--cases-root",
        str(cases_root),
        "--prune-low-confidence",
    ])

    assert args.func(args) == 0
    output = capsys.readouterr().out
    assert "Low-confidence cases: 1" in output
    assert "Removed cases: 1" in output
    assert not low_case.exists()
