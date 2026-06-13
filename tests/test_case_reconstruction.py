from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from replay_checker.case_reconstruction import (
    _append_conversation_evidence,
    _body_contains_code,
    _classify_changed_files,
    _compact_reference_commit_message,
    _extract_goal_section,
    _find_doc_nearby,
    _find_verification_hints,
    _is_code_like,
    _is_test_path,
)
from replay_checker.sources import EvidenceRecord


class TestClassifyChangedFiles:
    def test_source_files(self):
        result = _classify_changed_files(["src/main.py", "lib/core.rs"])
        assert "source" in result
        assert "src/main.py" in result["source"]

    def test_test_files(self):
        result = _classify_changed_files(["tests/test_foo.py", "spec/bar_spec.js"])
        assert "test" in result
        assert len(result["test"]) == 2

    def test_config_files(self):
        result = _classify_changed_files(["pyproject.toml", "Makefile"])
        assert "config" in result
        assert len(result["config"]) == 2

    def test_doc_files(self):
        result = _classify_changed_files(["README.md", "docs/guide.rst"])
        assert "doc" in result
        assert len(result["doc"]) == 2

    def test_other_files(self):
        result = _classify_changed_files(["image.png", "data.csv"])
        assert "other" in result

    def test_mixed_files(self):
        result = _classify_changed_files(["src/app.py", "tests/test_app.py", "README.md"])
        assert "source" in result
        assert "test" in result
        assert "doc" in result

    def test_empty_list(self):
        result = _classify_changed_files([])
        assert result == {}


class TestIsTestPath:
    def test_test_prefix(self):
        assert _is_test_path("tests/test_foo.py") is True

    def test_test_suffix(self):
        assert _is_test_path("src/foo_test.py") is True

    def test_spec_in_name(self):
        assert _is_test_path("src/foo.spec.js") is True

    def test_test_dir_name(self):
        assert _is_test_path("test/unit.py") is True

    def test_not_test(self):
        assert _is_test_path("src/main.py") is False


class TestIsCodeLike:
    def test_code_indicators(self):
        assert _is_code_like("def foo():\n    return True\nclass Bar:\n    pass") is True

    def test_prose(self):
        assert _is_code_like("This is a commit message describing the change.") is False

    def test_empty(self):
        assert _is_code_like("") is False

    def test_single_line(self):
        assert _is_code_like("import os") is False


class TestBodyContainsCode:
    def test_code_lines(self):
        assert _body_contains_code(["def foo():", "return True", "class Bar:", "pass"]) is True

    def test_prose_lines(self):
        assert _body_contains_code(["This is a description", "of the changes."]) is False


class TestExtractGoalSection:
    def test_valid_goal(self, tmp_path):
        p = tmp_path / "INDEX.md"
        p.write_text("# Plan\n\n## Goal\n\nImplement feature X\n\n## Steps\n\n1. Do stuff\n", encoding="utf-8")
        goal = _extract_goal_section(p)
        assert "Implement feature X" in goal

    def test_no_goal_section(self, tmp_path):
        p = tmp_path / "INDEX.md"
        p.write_text("# Plan\n\n## Steps\n\n1. Do stuff\n", encoding="utf-8")
        assert _extract_goal_section(p) == ""

    def test_nonexistent_file(self):
        assert _extract_goal_section(Path("/nonexistent/file.md")) == ""


class TestFindVerificationHints:
    def test_test_file_in_changes(self, tmp_path):
        hints = _find_verification_hints(tmp_path, ["tests/test_foo.py"])
        assert any("test" in h.lower() for h in hints)

    def test_no_test_files(self, tmp_path):
        hints = _find_verification_hints(tmp_path, ["src/main.py"])
        assert not any("Test file is part" in h for h in hints)

    def test_test_dir_exists(self, tmp_path):
        (tmp_path / "tests").mkdir()
        hints = _find_verification_hints(tmp_path, [])
        assert any("tests/" in h for h in hints)


class TestFindDocNearby:
    def test_readme_nearby(self, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "README.md").write_text("# Source", encoding="utf-8")
        hints = _find_doc_nearby(tmp_path, ["src/app.py"])
        assert any("README.md" in h for h in hints)

    def test_no_docs(self, tmp_path):
        hints = _find_doc_nearby(tmp_path, ["src/app.py"])
        assert isinstance(hints, list)


class TestAppendConversationEvidence:
    def test_with_conversation_records(self):
        records = [
            EvidenceRecord(source_type="plan", confidence="high", path=Path("/tmp"), summary="plan summary"),
            EvidenceRecord(source_type="codex_history", confidence="medium", path=Path("/tmp"), summary="codex summary"),
        ]
        lines = ["# Task", "## Source"]
        _append_conversation_evidence(lines, records)
        assert any("Conversation Evidence" in l for l in lines)
        assert any("codex summary" in l for l in lines)

    def test_without_conversation_records(self):
        records = [
            EvidenceRecord(source_type="plan", confidence="high", path=Path("/tmp"), summary="plan summary"),
        ]
        lines = ["# Task"]
        _append_conversation_evidence(lines, records)
        assert not any("Conversation Evidence" in l for l in lines)

    def test_empty_records(self):
        lines = ["# Task"]
        _append_conversation_evidence(lines, [])
        assert lines == ["# Task"]


class TestCompactReferenceCommitMessage:
    def test_multiline(self):
        msg = "feat: add feature\n\nThis is a longer\nmessage body"
        compact = _compact_reference_commit_message(msg)
        assert "\n" not in compact
        assert "feat: add feature" in compact

    def test_empty(self):
        assert _compact_reference_commit_message("") == ""

    def test_single_line(self):
        assert _compact_reference_commit_message("simple message") == "simple message"
