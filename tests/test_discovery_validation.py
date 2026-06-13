"""Tests for discovery orchestration kit projection validation."""

from __future__ import annotations

from pathlib import Path

from replay_checker.discovery.validation import validate_kit


GRAPH_HEADER = (
    "package_id\tpackage_doc\tstatus_file\tdependencies\tdependency_type\t"
    "wave\tbranch\tworktree\tmanual\tfinalize\n"
)
STATE_HEADER = (
    "package_id\tstate\tlaunched_at\tcompleted_at\tagent\tbranch\tworktree\t"
    "base_commit\tcommit_hash\tverification\tintegration\tcleanup\tlast_error\t"
    "failed_command\tconflict_files\tlog_summary\trecovery_hint\n"
)


def _write_kit(root: Path, *, graph_rows: list[str] | None = None) -> Path:
    kit = root / "kit"
    for directory in ("launchers", "status", "packages", "scratch"):
        (kit / directory).mkdir(parents=True, exist_ok=True)
    (kit / "INDEX.md").write_text("# Kit\n", encoding="utf-8")
    rows = graph_rows or [
        "01-plan\tpackages/01-plan.md\tstatus/01-plan.md\t\tstatus\t1\tbranch/01\t/tmp/01\t0\t0",
        "02-merge\tpackages/02-merge.md\tstatus/02-merge.md\t01-plan\tstatus\t2\tbranch/02\t/tmp/02\t0\t0",
        "99-finalize\tpackages/99-finalize.md\tstatus/99-finalize.md\t01-plan,02-merge\tstatus\tfinal\tbranch/99\t/tmp/99\t0\t1",
    ]
    (kit / "launchers" / "package-graph.tsv").write_text(
        GRAPH_HEADER + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    (kit / "status" / "state.tsv").write_text(
        STATE_HEADER
        + "01-plan\tpending\t\t\t\t\t\t\t\tpending\tpending\tpending\t\t\t\t\t\n"
        + "02-merge\tpending\t\t\t\t\t\t\t\tpending\tpending\tpending\t\t\t\t\t\n"
        + "99-finalize\tpending\t\t\t\t\t\t\t\tpending\tpending\tpending\t\t\t\t\t\n",
        encoding="utf-8",
    )
    (kit / "launchers" / "agent-prompts.md").write_text(
        "# Agent Prompts\n\n"
        "## Package: 01-plan - Plan scan\n\n"
        "## Package: 02-merge - Merge\n\n"
        "## Package: 99-finalize - Finalize\n",
        encoding="utf-8",
    )
    for package_id in ("01-plan", "02-merge", "99-finalize"):
        (kit / "packages" / f"{package_id}.md").write_text(
            f"# {package_id}\n", encoding="utf-8"
        )
        (kit / "status" / f"{package_id}.md").write_text(
            f"# {package_id} Status\n\n## State\n\n`pending`\n",
            encoding="utf-8",
        )
    return kit


def test_validate_kit_rejects_missing_prompt_heading(tmp_path):
    kit = _write_kit(tmp_path)
    (kit / "launchers" / "agent-prompts.md").write_text(
        "# Agent Prompts\n\n"
        "## Package: 01-plan - Plan scan\n\n"
        "## Package: 99-finalize - Finalize\n",
        encoding="utf-8",
    )

    errors = validate_kit(kit)

    assert "agent-prompts.md missing package headings: 02-merge" in errors


def test_validate_kit_rejects_duplicate_and_unknown_prompt_headings(tmp_path):
    kit = _write_kit(tmp_path)
    (kit / "launchers" / "agent-prompts.md").write_text(
        "# Agent Prompts\n\n"
        "## Package: 01-plan - Plan scan\n\n"
        "## Package: 01-plan - Duplicate\n\n"
        "## Package: 02-merge - Merge\n\n"
        "## Package: 03-extra - Extra\n\n"
        "## Package: 99-finalize - Finalize\n",
        encoding="utf-8",
    )

    errors = validate_kit(kit)

    assert "agent-prompts.md duplicate package headings: 01-plan" in errors
    assert "agent-prompts.md has unknown package headings: 03-extra" in errors


def test_validate_kit_rejects_unknown_dependencies_and_cycles(tmp_path):
    kit = _write_kit(
        tmp_path,
        graph_rows=[
            "01-plan\tpackages/01-plan.md\tstatus/01-plan.md\t02-merge,missing\tstatus\t1\tbranch/01\t/tmp/01\t0\t0",
            "02-merge\tpackages/02-merge.md\tstatus/02-merge.md\t01-plan\tstatus\t2\tbranch/02\t/tmp/02\t0\t0",
            "99-finalize\tpackages/99-finalize.md\tstatus/99-finalize.md\t01-plan,02-merge\tstatus\tfinal\tbranch/99\t/tmp/99\t0\t1",
        ],
    )

    errors = validate_kit(kit)

    assert "package-graph.tsv package 01-plan has unknown dependency: missing" in errors
    assert "package-graph.tsv dependency graph contains a cycle" in errors


def test_validate_kit_rejects_missing_package_doc_and_status_drift(tmp_path):
    kit = _write_kit(tmp_path)
    (kit / "packages" / "02-merge.md").unlink()
    (kit / "status" / "state.tsv").write_text(
        STATE_HEADER
        + "01-plan\tpending\t\t\t\t\t\t\t\tpending\tpending\tpending\t\t\t\t\t\n"
        + "02-merge\tcompleted\t\t\t\t\t\t\t\tpassed\tpending\tpending\t\t\t\t\t\n"
        + "99-finalize\tpending\t\t\t\t\t\t\t\tpending\tpending\tpending\t\t\t\t\t\n",
        encoding="utf-8",
    )

    errors = validate_kit(kit)

    assert "missing package doc: packages/02-merge.md" in errors
    assert "status file status/02-merge.md state `pending` disagrees with state.tsv `completed`" in errors
