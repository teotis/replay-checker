from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, MutableMapping


ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "gbk")


class ApiDisabledError(RuntimeError):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sanitize_slug(value: str, fallback: str = "unnamed") -> str:
    text = value.strip()
    text = re.sub(r'[/\\:*?"<>|\s]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def stable_hash(content: Any, length: int = 8) -> str:
    if isinstance(content, str):
        payload = content
    else:
        payload = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def build_output_name(series: str, topic: str, content: Any = "", *, date: str | None = None) -> str:
    date_str = date or datetime.now().strftime("%Y%m%d")
    digest = stable_hash({"series": series, "topic": topic, "content": content})
    return f"{sanitize_slug(series)}_{date_str}_{sanitize_slug(topic)}_{digest}"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = project_root()

    @property
    def control_dir(self) -> Path:
        return self.root / "control"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def input_dir(self) -> Path:
        return self.work_dir / "in"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "out"

    @property
    def tmp_dir(self) -> Path:
        return self.work_dir / "tmp"

    @property
    def ledger_path(self) -> Path:
        return self.control_dir / "ledger.md"

    def ensure_dirs(self) -> None:
        for path in [self.control_dir, self.input_dir, self.output_dir, self.tmp_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def create_output_dir(self, name: str) -> Path:
        out = self.output_dir / sanitize_slug(name)
        out.mkdir(parents=True, exist_ok=True)
        return out


def read_text_with_fallback(path: str | Path) -> str:
    target = Path(path)
    for encoding in ENCODING_FALLBACKS:
        try:
            return target.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return target.read_text(encoding="utf-8", errors="ignore")


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
        os.replace(tmp_name, target)
    except Exception:
        try:
            os.unlink(tmp_name)
        finally:
            raise


def read_json(path: str | Path) -> Any:
    return json.loads(read_text_with_fallback(path))


def atomic_write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=indent) + "\n")


def load_env(path: str | Path = ".env", *, environ: MutableMapping[str, str] | None = None) -> dict[str, str]:
    env = environ if environ is not None else os.environ
    loaded: dict[str, str] = {}
    target = Path(path)
    if not target.exists():
        return loaded
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        env.setdefault(key, value)
    return loaded


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def assert_api_enabled(
    prefix: str = "REPLAY_CHECKER",
    *,
    allow_api: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> str:
    env = environ if environ is not None else os.environ
    enabled_name = f"{prefix}_API_ENABLED"
    key_name = f"{prefix}_API_KEY"
    if not allow_api:
        raise ApiDisabledError("CLI flag did not allow external API calls")
    if not truthy(env.get(enabled_name)):
        raise ApiDisabledError(f"{enabled_name} is not enabled")
    key = env.get(key_name) or env.get("OPENAI_API_KEY") or ""
    if not key:
        raise ApiDisabledError(f"{key_name} or OPENAI_API_KEY is missing")
    return key
