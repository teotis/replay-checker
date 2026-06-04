from __future__ import annotations

import subprocess

from replay_checker.git_utils import git_output
from replay_checker.yaml_lite import (
    coerce_bool_int,
    coerce_nested_int,
    emit_yaml,
    parse_yaml_text,
)


def test_yaml_lite_preserves_evaluation_leaf_strings_by_default():
    text = emit_yaml({"flag": True, "score": 42.5})

    assert text == "flag: true\nscore: 42.5"
    assert parse_yaml_text(text) == {"flag": "true", "score": "42.5"}


def test_yaml_lite_parses_replay_nested_lists_and_nested_ints():
    text = "\n".join(
        [
            "id: case-1",
            "telemetry:",
            "  changed_file_count: 2",
            "  forbidden_path_touches:",
            "    - _reference/diff.patch",
            "changed_files:",
            "  - src/example.py",
            "verification_commands:",
            "  - 1",
        ]
    )

    assert parse_yaml_text(text, scalar_parser=coerce_nested_int) == {
        "id": "case-1",
        "telemetry": {
            "changed_file_count": 2,
            "forbidden_path_touches": ["_reference/diff.patch"],
        },
        "changed_files": ["src/example.py"],
        "verification_commands": ["1"],
    }


def test_yaml_lite_parses_rubric_bool_int_and_nested_lists():
    text = "\n".join(
        [
            "result_weight: 80",
            "gates:",
            "  required:",
            "    - diff_present",
            "ceilings:",
            "  runner_identity_exposed:",
            "    overall_invalid: true",
        ]
    )

    assert parse_yaml_text(text, scalar_parser=coerce_bool_int) == {
        "result_weight": 80,
        "gates": {"required": ["diff_present"]},
        "ceilings": {"runner_identity_exposed": {"overall_invalid": True}},
    }


def test_yaml_lite_parses_inline_lists_when_enabled():
    text = 'criteria: ["result", "process"]'

    assert parse_yaml_text(text, parse_inline_lists=True) == {
        "criteria": ["result", "process"],
    }


def test_yaml_lite_round_trips_list_of_dicts():
    items = [
        {"run_id": "run-a", "total_score": 12.5},
        {"run_id": "run-b", "total_score": 8.0},
    ]

    parsed = parse_yaml_text(emit_yaml(items))

    assert parsed == [
        {"run_id": "run-a", "total_score": "12.5"},
        {"run_id": "run-b", "total_score": "8.0"},
    ]


def test_git_output_returns_stdout_only_for_success(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc\n", stderr="")

    monkeypatch.setattr("replay_checker.git_utils.subprocess.run", fake_run)

    assert git_output(["rev-parse", "HEAD"], cwd="/repo") == "abc\n"
    assert calls[0][0] == ["git", "rev-parse", "HEAD"]
    assert calls[0][1]["cwd"] == "/repo"


def test_git_output_returns_empty_string_on_failure(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="nope\n", stderr="bad")

    monkeypatch.setattr("replay_checker.git_utils.subprocess.run", fake_run)

    assert git_output(["status"], cwd="/repo") == ""
