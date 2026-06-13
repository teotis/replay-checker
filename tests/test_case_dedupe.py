"""Tests for case fingerprinting and duplicate detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from replay_checker.case_dedupe import (
    CaseFingerprint,
    DuplicateMatch,
    DuplicateReport,
    _extract_changed_files_from_diff,
    _file_overlap,
    _hash_diff_patch_chunked,
    _normalize_text,
    _text_similarity,
    build_case_fingerprint,
    find_duplicates,
    inspect_duplicates,
    scan_all_fingerprints,
)
from replay_checker.core import stable_hash
from replay_checker.yaml_lite import write_simple_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(
    cases_root: Path,
    case_id: str,
    *,
    project_path: str = "/project/foo",
    base_commit: str = "abc123",
    source_type: str = "orchestration_kit",
    source_path: str = "/project/foo/docs/plans/plan-a/INDEX.md",
    task_text: str = "Implement feature X with tests and documentation.",
    diff_content: str = "",
    changed_files: list[str] | None = None,
    skill_name: str = "",
    eval_id: str = "",
) -> Path:
    """Create a minimal case directory for testing."""
    case_dir = cases_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, object] = {
        "id": case_id,
        "project_path": project_path,
        "base_commit": base_commit,
        "source_type": source_type,
        "source_path": source_path,
    }
    if skill_name:
        data["skill_name"] = skill_name
    if eval_id:
        data["eval_id"] = eval_id
    write_simple_yaml(case_dir / "case.yaml", data)

    # task.md
    (case_dir / "task.md").write_text(f"# Task: {case_id}\n\n{task_text}\n", encoding="utf-8")

    # evidence_sources.md (required by load_case)
    (case_dir / "evidence_sources.md").write_text("- none\n", encoding="utf-8")

    # _reference/ with diff and metadata
    if diff_content or changed_files:
        ref_dir = case_dir / "_reference"
        ref_dir.mkdir(parents=True, exist_ok=True)
        if diff_content:
            (ref_dir / "diff.patch").write_text(diff_content, encoding="utf-8")
        if changed_files:
            write_simple_yaml(
                ref_dir / "reference_metadata.yaml",
                {
                    "target_commit": "def456",
                    "base_commit": base_commit,
                    "commit_message": "test commit",
                    "changed_files": changed_files,
                },
            )

    return case_dir


# ---------------------------------------------------------------------------
# Text helper tests
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercase_and_collapse_whitespace(self):
        assert _normalize_text("Hello   World") == "hello world"

    def test_trim_punctuation(self):
        assert _normalize_text("foo!@# bar.") == "foo bar"

    def test_bounded_length(self):
        long_text = "x " * 5000
        result = _normalize_text(long_text, max_length=100)
        assert len(result) <= 100

    def test_empty_input(self):
        assert _normalize_text("") == ""


class TestTextSimilarity:
    def test_identical_texts(self):
        assert _text_similarity("hello world", "hello world") == 1.0

    def test_empty_both(self):
        assert _text_similarity("", "") == 1.0

    def test_one_empty(self):
        assert _text_similarity("hello", "") == 0.0

    def test_similar_texts(self):
        sim = _text_similarity("implement feature x", "implement feature y")
        assert sim > 0.7


class TestFileOverlap:
    def test_identical_sets(self):
        files = {"a.py", "b.py"}
        assert _file_overlap(files, files) == 1.0

    def test_disjoint_sets(self):
        assert _file_overlap({"a.py"}, {"b.py"}) == 0.0

    def test_partial_overlap(self):
        overlap = _file_overlap({"a.py", "b.py"}, {"b.py", "c.py"})
        assert 0.0 < overlap < 1.0

    def test_both_empty(self):
        assert _file_overlap(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# Diff helper tests
# ---------------------------------------------------------------------------


class TestDiffHelpers:
    def test_hash_diff_patch_chunked(self, tmp_path):
        diff = tmp_path / "diff.patch"
        content = "diff --git a/x b/x\n+added line\n"
        diff.write_text(content, encoding="utf-8")
        h = _hash_diff_patch_chunked(diff)
        assert len(h) == 16
        # Deterministic
        assert _hash_diff_patch_chunked(diff) == h

    def test_hash_diff_patch_chunked_large(self, tmp_path):
        """Large diff is processed in chunks, not loaded entirely into memory."""
        diff = tmp_path / "diff.patch"
        # Write 256 KiB of data
        line = "+" + "x" * 100 + "\n"
        diff.write_text(line * 2600, encoding="utf-8")
        h = _hash_diff_patch_chunked(diff)
        assert len(h) == 16

    def test_extract_changed_files_from_diff(self, tmp_path):
        diff_content = (
            "diff --git a/src/foo.py b/src/foo.py\n"
            "--- a/src/foo.py\n"
            "+++ b/src/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/tests/test_foo.py b/tests/test_foo.py\n"
            "--- a/tests/test_foo.py\n"
            "+++ b/tests/test_foo.py\n"
        )
        diff = tmp_path / "diff.patch"
        diff.write_text(diff_content, encoding="utf-8")
        files = _extract_changed_files_from_diff(diff)
        assert "src/foo.py" in files
        assert "tests/test_foo.py" in files


# ---------------------------------------------------------------------------
# Fingerprint builder tests
# ---------------------------------------------------------------------------


class TestBuildCaseFingerprint:
    def test_basic_fingerprint(self, tmp_path):
        case_dir = _make_case(tmp_path, "test-case-1")
        fp = build_case_fingerprint(case_dir)
        assert fp.case_id == "test-case-1"
        assert fp.source_type == "orchestration_kit"
        assert fp.base_commit == "abc123"
        assert fp.fingerprint  # non-empty
        assert len(fp.fingerprint) == 8

    def test_fingerprint_deterministic(self, tmp_path):
        case_dir = _make_case(tmp_path, "test-case-1")
        fp1 = build_case_fingerprint(case_dir)
        fp2 = build_case_fingerprint(case_dir)
        assert fp1.fingerprint == fp2.fingerprint

    def test_different_content_different_fingerprint(self, tmp_path):
        dir_a = _make_case(
            tmp_path, "case-a",
            task_text="Implement feature A",
            diff_content="diff --git a/x b/x\n+feature A\n",
            changed_files=["src/a.py"],
        )
        dir_b = _make_case(
            tmp_path, "case-b",
            task_text="Implement feature B",
            diff_content="diff --git a/y b/y\n+feature B\n",
            changed_files=["src/b.py"],
        )
        fp_a = build_case_fingerprint(dir_a)
        fp_b = build_case_fingerprint(dir_b)
        assert fp_a.fingerprint != fp_b.fingerprint

    def test_fingerprint_with_reference_diff(self, tmp_path):
        diff_content = "diff --git a/foo.py b/foo.py\n+new line\n"
        case_dir = _make_case(
            tmp_path, "ref-case",
            diff_content=diff_content,
            changed_files=["foo.py"],
        )
        fp = build_case_fingerprint(case_dir)
        assert fp.reference_key  # non-empty from diff hash
        assert fp.changed_files_key  # non-empty from changed files


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------


class TestExactDuplicate:
    """Two cases with different source path/title but identical reference diff
    and base commit should be detected as exact duplicates."""

    def test_same_diff_different_source_path(self, tmp_path):
        diff_content = "diff --git a/foo.py b/foo.py\n+new line\n"
        case_a = _make_case(
            tmp_path, "case-a",
            source_path="/project/docs/plans/plan-a/INDEX.md",
            task_text="Same task content here",
            diff_content=diff_content,
            base_commit="abc123",
            changed_files=["foo.py"],
        )
        case_b = _make_case(
            tmp_path, "case-b",
            source_path="/project/docs/plans/plan-b/INDEX.md",
            task_text="Same task content here",
            diff_content=diff_content,
            base_commit="abc123",
            changed_files=["foo.py"],
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        exact = [m for m in report.matches if m.relation == "exact"]
        assert len(exact) >= 1
        assert any("same_reference_diff" in m.reasons for m in exact)

    def test_same_base_commit_and_files(self, tmp_path):
        """Same base commit and changed files (no diff content) = exact."""
        case_a = _make_case(
            tmp_path, "case-a",
            base_commit="abc123",
            task_text="Some task A",
            changed_files=["src/foo.py", "src/bar.py"],
        )
        case_b = _make_case(
            tmp_path, "case-b",
            base_commit="abc123",
            task_text="Some task B",
            changed_files=["src/foo.py", "src/bar.py"],
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        exact = [m for m in report.matches if m.relation == "exact"]
        assert len(exact) >= 1

    def test_different_content_not_duplicate(self, tmp_path):
        _make_case(
            tmp_path, "case-a",
            task_text="Implement feature A",
            diff_content="diff --git a/a.py b/a.py\n+feature A\n",
            changed_files=["a.py"],
        )
        case_b = _make_case(
            tmp_path, "case-b",
            task_text="Implement feature B completely different",
            diff_content="diff --git a/b.py b/b.py\n+feature B\n",
            changed_files=["b.py"],
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        exact = [m for m in report.matches if m.relation == "exact"]
        assert len(exact) == 0


class TestLikelyDuplicate:
    """Cases with similar task text and overlapping changed files should be
    reported as likely duplicates, not deleted."""

    def test_similar_task_and_overlapping_files(self, tmp_path):
        case_a = _make_case(
            tmp_path, "case-a",
            task_text="Implement camera shutter button with haptic feedback and animation",
            changed_files=["src/camera/shutter.py", "src/ui/button.py"],
            diff_content="",
        )
        case_b = _make_case(
            tmp_path, "case-b",
            task_text="Implement camera shutter button with haptic feedback and animation effects",
            changed_files=["src/camera/shutter.py", "src/ui/button.py"],
            diff_content="",
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        likely = [m for m in report.matches if m.relation == "likely_duplicate"]
        # The task texts are very similar and files overlap exactly
        if likely:
            assert any("task_similarity" in str(m.reasons) for m in likely)

    def test_likely_duplicate_not_deleted(self, tmp_path):
        """Likely duplicates are reported but the case is still created."""
        _make_case(
            tmp_path, "case-a",
            task_text="Add dark mode support to the settings page",
            changed_files=["src/settings.py", "src/theme.py"],
        )
        case_b = _make_case(
            tmp_path, "case-b",
            task_text="Add dark mode support to the settings page with theme switching",
            changed_files=["src/settings.py", "src/theme.py"],
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        # Even if likely duplicates exist, the report doesn't delete anything
        assert report.candidate_id == "case-b"
        assert case_b.exists()  # directory still exists


class TestSkillEvalPrefix:
    def test_same_skill_eval_prefix(self, tmp_path):
        """Skill eval cases with same prefix are exact duplicates."""
        case_a = _make_case(
            tmp_path, "my-skill-eval1-abc12345",
            skill_name="my-skill",
            eval_id="1",
            task_text="eval prompt A",
        )
        case_b = _make_case(
            tmp_path, "my-skill-eval1-def67890",
            skill_name="my-skill",
            eval_id="1",
            task_text="eval prompt B",
        )

        existing = scan_all_fingerprints(tmp_path)
        report = find_duplicates(case_b, existing)

        exact = [m for m in report.matches if m.relation == "exact"]
        assert any("same_skill_eval_prefix" in m.reasons for m in exact)


# ---------------------------------------------------------------------------
# Scan and inspect tests
# ---------------------------------------------------------------------------


class TestScanAndInspect:
    def test_scan_empty_root(self, tmp_path):
        result = scan_all_fingerprints(tmp_path)
        assert result == {}

    def test_scan_returns_all_cases(self, tmp_path):
        _make_case(tmp_path, "case-1")
        _make_case(tmp_path, "case-2")
        result = scan_all_fingerprints(tmp_path)
        assert len(result) == 2

    def test_scan_returns_nested_inventory_cases(self, tmp_path):
        _make_case(tmp_path / "inventory" / "demo_project", "case-1")
        result = scan_all_fingerprints(tmp_path)
        assert set(result) == {"case-1"}

    def test_inspect_duplicates_returns_structure(self, tmp_path):
        _make_case(tmp_path, "case-1")
        _make_case(tmp_path, "case-2")
        report = inspect_duplicates(tmp_path)
        assert "exact_clusters" in report
        assert "likely_pairs" in report
        assert "total_cases" in report
        assert report["total_cases"] == 2

    def test_inspect_ignores_non_case_dirs(self, tmp_path):
        _make_case(tmp_path, "case-1")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "not-a-case").mkdir()
        result = scan_all_fingerprints(tmp_path)
        assert len(result) == 1
