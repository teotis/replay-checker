"""Tests for llm_policy module. All tests use fakes/mocks, no network calls."""

from pathlib import Path

from replay_checker.llm_policy import (
    LLMPolicy,
    LLMConfig,
    evaluate_llm_policy,
    load_llm_config,
    summarize_upload_scope,
)


def test_default_config_is_local_only():
    config = LLMConfig()
    assert config.enabled is False
    assert config.local_only is True
    assert config.upload_scope == "summary"


def test_config_from_dict_full():
    data = {
        "llm_enabled": True,
        "llm_provider": "openai",
        "llm_api_key_env": "MY_API_KEY",
        "max_upload_chars": 8000,
        "max_upload_files": 5,
        "local_only": False,
        "upload_scope": "bounded_snippets",
        "allowed_extensions": [".py", ".js"],
    }
    config = LLMConfig.from_dict(data)
    assert config.enabled is True
    assert config.provider == "openai"
    assert config.api_key_env == "MY_API_KEY"
    assert config.max_upload_chars == 8000
    assert config.max_upload_files == 5
    assert config.local_only is False
    assert config.upload_scope == "bounded_snippets"
    assert config.allowed_extensions == (".py", ".js")


def test_policy_from_config_local_only():
    config = LLMConfig(local_only=True)
    policy = LLMPolicy.from_config(config)
    assert policy.allowed is False
    assert policy.local_only is True
    assert "local_only" in policy.restrictions[0]


def test_policy_from_config_disabled():
    config = LLMConfig(enabled=False, local_only=False)
    policy = LLMPolicy.from_config(config)
    assert policy.allowed is False
    assert "llm_enabled" in policy.restrictions[0]


def test_policy_from_config_opt_out(tmp_path):
    config = LLMConfig(enabled=True, local_only=False, provider="openai")
    policy = LLMPolicy.from_config(config, environ={"REPLAY_CHECKER_LLM_OPT_OUT": "1"})
    assert policy.allowed is False
    assert "opt-out" in policy.restrictions[0]


def test_policy_from_config_enabled_with_key():
    config = LLMConfig(enabled=True, local_only=False, provider="openai", api_key_env="MY_KEY")
    policy = LLMPolicy.from_config(config, environ={"MY_KEY": "sk-test-123"})
    assert policy.allowed is True
    assert policy.provider == "openai"
    assert policy.max_upload_chars == 4000


def test_policy_from_config_enabled_no_key():
    config = LLMConfig(enabled=True, local_only=False, provider="anthropic")
    policy = LLMPolicy.from_config(config, environ={})
    assert policy.allowed is False
    assert "no API key" in policy.restrictions[0]


def test_load_config_with_json(tmp_path):
    config_path = tmp_path / "replay-checker.json"
    config_path.write_text(
        '{"llm_enabled": true, "llm_provider": "anthropic", "local_only": false}',
        encoding="utf-8",
    )
    config = load_llm_config(tmp_path)
    assert config.enabled is True
    assert config.provider == "anthropic"
    assert config.local_only is False


def test_load_config_without_json(tmp_path):
    (tmp_path / "other.json").write_text("{}", encoding="utf-8")
    config = load_llm_config(tmp_path)
    assert config.enabled is False
    assert config.local_only is True


def test_load_config_invalid_json(tmp_path):
    config_path = tmp_path / "replay-checker.json"
    config_path.write_text("not-json", encoding="utf-8")
    config = load_llm_config(tmp_path)
    assert config.enabled is False
    assert config.local_only is True


def test_evaluate_policy_no_config(tmp_path):
    policy = evaluate_llm_policy(tmp_path, environ={})
    assert policy.allowed is False
    assert policy.local_only is True


def test_summarize_scope_local_only():
    policy = LLMPolicy(allowed=False, local_only=True, upload_scope="none",
                        provider="", max_upload_chars=0, max_upload_files=0)
    text = summarize_upload_scope(policy)
    assert "local-only" in text.lower() or "No data" in text


def test_summarize_scope_with_upload():
    policy = LLMPolicy(allowed=True, local_only=False, upload_scope="bounded_snippets",
                        provider="openai", max_upload_chars=8000, max_upload_files=10)
    text = summarize_upload_scope(policy)
    assert "8000" in text
    assert "Full project" in text


def test_upload_scope_never_includes_full_project():
    policy = LLMPolicy(allowed=True, local_only=False, upload_scope="bounded_snippets",
                        provider="anthropic", max_upload_chars=4000, max_upload_files=5)
    summary = summarize_upload_scope(policy)
    assert "Full project contents are never uploaded" in summary


def test_config_extends_default_extensions():
    data = {"llm_enabled": False, "allowed_extensions": [".rs"]}
    config = LLMConfig.from_dict(data)
    assert config.allowed_extensions == (".rs",)
