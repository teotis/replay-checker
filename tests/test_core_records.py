from replay_checker.core import (
    ApiDisabledError,
    ProjectPaths,
    assert_api_enabled,
    atomic_write_json,
    build_output_name,
    load_env,
    read_json,
    sanitize_slug,
)
from replay_checker.records import Ledger, Manifest, QCIssue, Record, run_qc


def test_paths_and_output_name():
    paths = ProjectPaths()

    assert paths.input_dir.as_posix().endswith("work/in")
    assert paths.output_dir.as_posix().endswith("work/out")
    assert sanitize_slug(' a/b c:*?"<>| ') == "a_b_c"
    assert build_output_name("Series X", "topic y", {"value": 1}, date="20260514").startswith(
        "Series_X_20260514_topic_y_"
    )


def test_atomic_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "data.json"
    atomic_write_json(path, {"ok": True})
    assert read_json(path) == {"ok": True}


def test_env_gate(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DEMO_API_ENABLED=1\nDEMO_API_KEY=secret\n", encoding="utf-8")
    environ = {}
    load_env(env_file, environ=environ)

    assert assert_api_enabled("DEMO", allow_api=True, environ=environ) == "secret"
    try:
        assert_api_enabled("DEMO", allow_api=False, environ=environ)
    except ApiDisabledError:
        pass
    else:
        raise AssertionError("API should require CLI allow flag")


def test_ledger_appends_unified_record(tmp_path):
    ledger_path = tmp_path / "ledger.md"
    ledger = Ledger(ledger_path)
    ledger.append(Record(type="request", title="Capture habit", summary=("record durable requirements",)))
    ledger.append(Record(type="risk", title="Check output noise", details=("do not commit work/out files",)))

    text = ledger_path.read_text(encoding="utf-8")
    assert "type: request" in text
    assert "record durable requirements" in text
    assert "type: risk" in text


def test_manifest_roundtrip(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(title="demo", status="done", outputs=["work/out/demo.txt"])
    manifest.save(path)

    loaded = Manifest.load(path)

    assert loaded.title == "demo"
    assert loaded.status == "done"
    assert loaded.outputs == ["work/out/demo.txt"]


def test_run_qc_aggregates_errors_and_warnings():
    def warning_check(_target):
        return [QCIssue(severity="warning", check="warn", detail="soft")]

    def failing_check(_target):
        raise ValueError("boom")

    result = run_qc("target", [("warning", warning_check), ("failing", failing_check)])

    assert not result.passed
    assert len(result.warnings) == 1
    assert len(result.errors) == 1
    assert result.errors[0].check == "failing"
