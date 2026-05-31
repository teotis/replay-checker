"""LLM authorization policy for Replay Checker.

Defines when and how external LLM calls are permitted:
- Default is local-only (no LLM calls).
- LLM-enabled analysis requires an explicit config field in the project.
- Upload scope is bounded to project summary plus bounded snippets.
- Opt-out is always available via config or environment variable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping

# Default upload scope limits
_DEFAULT_MAX_CHARS = 4000
_DEFAULT_MAX_FILES = 10
_DEFAULT_ALLOWED_EXTENSIONS = (".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb")


@dataclass(frozen=True)
class LLMConfig:
    """Parsed LLM policy configuration from a project."""

    enabled: bool = False
    provider: str = ""
    api_key_env: str = ""
    max_upload_chars: int = _DEFAULT_MAX_CHARS
    max_upload_files: int = _DEFAULT_MAX_FILES
    allowed_extensions: tuple[str, ...] = _DEFAULT_ALLOWED_EXTENSIONS
    local_only: bool = True
    upload_scope: str = "summary"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        enabled = data.get("llm_enabled", False)
        provider = str(data.get("llm_provider", "")).strip()
        api_key_env = str(data.get("llm_api_key_env", "")).strip()
        local_only = bool(data.get("local_only", not enabled))

        raw_extensions = data.get("allowed_extensions")
        if isinstance(raw_extensions, (list, tuple)):
            allowed_extensions = tuple(str(e) for e in raw_extensions)
        else:
            allowed_extensions = _DEFAULT_ALLOWED_EXTENSIONS

        return cls(
            enabled=bool(enabled),
            provider=provider,
            api_key_env=api_key_env,
            max_upload_chars=int(data.get("max_upload_chars", _DEFAULT_MAX_CHARS)),
            max_upload_files=int(data.get("max_upload_files", _DEFAULT_MAX_FILES)),
            allowed_extensions=allowed_extensions,
            local_only=bool(local_only),
            upload_scope=str(data.get("upload_scope", "summary")),
        )


@dataclass(frozen=True)
class LLMPolicy:
    """Evaluated LLM policy for a project."""

    allowed: bool
    provider: str
    upload_scope: str
    max_upload_chars: int
    max_upload_files: int
    local_only: bool
    restrictions: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: LLMConfig, *, environ: MutableMapping[str, str] | None = None) -> LLMPolicy:
        env = environ if environ is not None else os.environ

        if config.local_only:
            return cls(
                allowed=False,
                provider="",
                upload_scope="none",
                max_upload_chars=0,
                max_upload_files=0,
                local_only=True,
                restrictions=("local_only mode enabled",),
            )

        opt_out = env.get("REPLAY_CHECKER_LLM_OPT_OUT", "").strip().lower() in ("1", "true", "yes")
        if opt_out:
            return cls(
                allowed=False,
                provider=config.provider,
                upload_scope="none",
                max_upload_chars=0,
                max_upload_files=0,
                local_only=True,
                restrictions=("opt-out via REPLAY_CHECKER_LLM_OPT_OUT",),
            )

        if not config.enabled:
            return cls(
                allowed=False,
                provider=config.provider,
                upload_scope="none",
                max_upload_chars=0,
                max_upload_files=0,
                local_only=True,
                restrictions=("llm_enabled not set in project config",),
            )

        has_key = False
        if config.api_key_env:
            has_key = bool(env.get(config.api_key_env))
        else:
            has_key = bool(env.get("OPENAI_API_KEY") or env.get("ANTHROPIC_API_KEY"))

        restrictions: list[str] = []
        if not has_key:
            restrictions.append("no API key found in environment")

        return cls(
            allowed=config.enabled and has_key,
            provider=config.provider,
            upload_scope=config.upload_scope,
            max_upload_chars=config.max_upload_chars,
            max_upload_files=config.max_upload_files,
            local_only=False,
            restrictions=tuple(restrictions),
        )


def load_llm_config(project_path: str | Path) -> LLMConfig:
    """Load LLM configuration from a project directory.

    Returns LLMConfig with local_only=True if no config file exists.
    """
    project = Path(project_path)
    for name in ("replay-checker.json", ".replay-checker.json", ".replay-checker.yaml"):
        path = project / name
        if path.exists():
            return _parse_config_file(path)
    return LLMConfig()


def evaluate_llm_policy(
    project_path: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> LLMPolicy:
    """Evaluate the full LLM policy for a project."""
    config = load_llm_config(project_path)
    return LLMPolicy.from_config(config, environ=environ)


def summarize_upload_scope(policy: LLMPolicy) -> str:
    """Return a human-readable description of what data can be uploaded."""
    if not policy.allowed or policy.local_only:
        return "No data will be uploaded. Analysis is local-only."

    parts = [
        f"Upload scope: {policy.upload_scope}",
        f"Max characters: {policy.max_upload_chars}",
        f"Max files: {policy.max_upload_files}",
        "Full project contents are never uploaded.",
    ]
    return "\n".join(parts)


def _parse_config_file(path: Path) -> LLMConfig:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == ".json":
        try:
            data = json.loads(text)
            return LLMConfig.from_dict(data)
        except (json.JSONDecodeError, TypeError):
            return LLMConfig()
    return LLMConfig()
