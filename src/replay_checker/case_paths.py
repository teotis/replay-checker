from __future__ import annotations

from pathlib import Path

from .core import sanitize_slug


CASE_NAMESPACE_DIRS = ("inventory", "examples", "archive")


def case_inventory_root(cases_root: str | Path, project_path: str | Path) -> Path:
    """Return the default inventory bucket for cases from a project."""
    project = Path(project_path)
    return Path(cases_root) / "inventory" / sanitize_slug(project.name)


def is_case_dir(path: Path) -> bool:
    return path.is_dir() and (path / "case.yaml").is_file()


def iter_case_dirs(cases_root: str | Path) -> tuple[Path, ...]:
    """Return case directories from legacy flat roots and nested inventories."""
    root = Path(cases_root)
    if not root.is_dir():
        return ()

    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen and is_case_dir(path):
            seen.add(resolved)
            found.append(path)

    for child in sorted(root.iterdir()):
        if child.name.startswith("."):
            continue
        if child.name in CASE_NAMESPACE_DIRS:
            continue
        add(child)

    for namespace in CASE_NAMESPACE_DIRS:
        namespace_root = root / namespace
        if not namespace_root.is_dir():
            continue
        for case_yaml in sorted(namespace_root.rglob("case.yaml")):
            if any(part.startswith(".") for part in case_yaml.parts):
                continue
            add(case_yaml.parent)

    return tuple(found)


def resolve_case_dir(cases_root: str | Path, case_id: str) -> Path:
    """Resolve a case id across legacy and nested case layouts."""
    root = Path(cases_root)
    legacy = root / case_id
    if (legacy / "case.yaml").is_file():
        return legacy

    matches = tuple(path for path in iter_case_dirs(root) if path.name == case_id)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = ", ".join(path.as_posix() for path in matches)
        raise ValueError(f"case id {case_id!r} is ambiguous: {paths}")
    return legacy
