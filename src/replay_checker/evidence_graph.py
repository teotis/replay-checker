"""Evidence graph helpers for candidate ranking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sources import EvidenceRecord


@dataclass(frozen=True)
class EvidenceGraph:
    """Small graph projection over records, source types, and mentioned paths."""

    path_source_types: dict[str, frozenset[str]]

    def connected_paths_for(self, record: EvidenceRecord) -> tuple[str, ...]:
        paths = _record_paths(record)
        return tuple(
            path
            for path in paths
            if len(self.path_source_types.get(path, frozenset())) >= 2
        )

    def bonus_for(self, record: EvidenceRecord) -> tuple[float, tuple[str, ...]]:
        connected = self.connected_paths_for(record)
        if not connected:
            return 0.0, ()
        shown = connected[0]
        sources = " + ".join(_ordered_sources(self.path_source_types.get(shown, frozenset())))
        return (
            0.08,
            (f"evidence graph episode support: {sources} on {shown}",),
        )


def build_evidence_graph(records: list[EvidenceRecord]) -> EvidenceGraph:
    path_source_types: dict[str, set[str]] = {}
    for record in records:
        for path in _record_paths(record):
            path_source_types.setdefault(path, set()).add(record.source_type)
    return EvidenceGraph(
        path_source_types={
            path: frozenset(source_types)
            for path, source_types in path_source_types.items()
        }
    )


def _record_paths(record: EvidenceRecord) -> tuple[str, ...]:
    raw = record.metadata.get("mentioned_files", "")
    paths = [part.strip() for part in raw.split(",") if part.strip()]
    paths.extend(_extract_path_mentions(record.summary))
    return tuple(dict.fromkeys(_normalize_path(path) for path in paths if path))


def _extract_path_mentions(text: str) -> list[str]:
    matches = re.findall(r"(?:^|\s|`)([\w./-]+\.\w{1,10})(?:\s|$|`|[.,:;])", text)
    return [match.strip("./") for match in matches]


def _normalize_path(path: str) -> str:
    return path.strip().replace("\\", "/").lstrip("./")


def _ordered_sources(source_types: frozenset[str]) -> tuple[str, ...]:
    preferred = ("plan", "codex_history", "claude_history", "git_history")
    ordered = [source for source in preferred if source in source_types]
    ordered.extend(sorted(source for source in source_types if source not in preferred))
    return tuple(ordered)
