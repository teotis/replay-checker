from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .core import atomic_write_json, atomic_write_text, read_json


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Record:
    type: str
    title: str
    summary: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    created_at: str = field(default_factory=now_iso)

    def render_markdown(self) -> str:
        def block(name: str, values: tuple[str, ...]) -> list[str]:
            if not values:
                return []
            return [f"{name}:", *[f"- {value}" for value in values], ""]

        tags = ", ".join(self.tags) if self.tags else ""
        lines = [
            f"## {self.created_at} - {self.title}",
            "",
            f"type: {self.type}",
            f"tags: {tags}",
            "",
        ]
        lines += block("summary", self.summary)
        lines += block("details", self.details)
        lines += block("links", self.links)
        return "\n".join(lines).rstrip() + "\n"


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: Record) -> Path:
        existing = self.path.read_text(encoding="utf-8") if self.path.exists() else "# Ledger\n\n"
        separator = "\n" if existing.endswith("\n\n") else "\n\n"
        atomic_write_text(self.path, existing.rstrip() + separator + record.render_markdown())
        return self.path


@dataclass
class Manifest:
    title: str = ""
    status: str = "planned"
    created_at: str = field(default_factory=now_iso)
    sources: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        known = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        target = Path(path)
        if not target.exists():
            return cls()
        return cls.from_dict(read_json(target))


@dataclass(frozen=True)
class QCIssue:
    severity: str = "error"
    check: str = ""
    detail: str = ""


@dataclass
class QCResult:
    checks_run: list[str] = field(default_factory=list)
    issues: list[QCIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[QCIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[QCIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks_run": self.checks_run,
            "issues": [dataclasses.asdict(issue) for issue in self.issues],
        }


Check = Callable[[Any], Iterable[QCIssue]]


def run_qc(target: Any, checks: Iterable[tuple[str, Check]]) -> QCResult:
    result = QCResult()
    for name, check in checks:
        result.checks_run.append(name)
        try:
            result.issues.extend(check(target))
        except Exception as exc:
            result.issues.append(QCIssue("error", name, f"QC check raised {type(exc).__name__}: {exc}"))
    return result
