"""Project code structure analysis (no-plan path)."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path

from .models import ProjectAnalysis

# File extension → language mapping (ordered by specificity)
_LANG_EXTENSIONS: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".h": "c",
    ".sh": "shell", ".bash": "shell",
    ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".toml": "toml",
    ".md": "markdown", ".txt": "text",
    ".sql": "sql", ".css": "css", ".html": "html",
}

_BUILD_INDICATORS = {
    "pyproject.toml", "setup.py", "setup.cfg", "cargo.toml",
    "go.mod", "package.json", "gemfile", "build.gradle", "pom.xml",
    "makefile", "cmakelists.txt",
}

_TEST_DIR_NAMES = {"tests", "test", "__tests__", "spec"}
_DOC_DIR_NAMES = {"docs", "doc", "documentation"}


def analyze_project(project_path: str | Path) -> ProjectAnalysis:
    """Analyze a project directory for code structure and metadata."""
    root = Path(project_path).resolve()
    skip = {".git", ".venv", "node_modules", "__pycache__", ".tox", "work", "venv"}

    extension_counts: dict[str, int] = {}
    source_files = 0
    total_chars = 0
    has_tests = False
    has_docs = False
    has_build_config = False
    has_gitignore = False
    has_env_template = False

    for path in _safe_iterdir(root, skip):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        extension_counts[ext] = extension_counts.get(ext, 0) + 1
        if ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb", ".c", ".cpp", ".h"}:
            source_files += 1
            try:
                total_chars += path.stat().st_size
            except OSError:
                pass
        lower = path.name.lower()
        if _is_test_file(path):
            has_tests = True
        if _is_doc_file(path):
            has_docs = True
        if lower in _BUILD_INDICATORS:
            has_build_config = True
        if lower == ".gitignore":
            has_gitignore = True
        if lower in (".env.example", ".env.sample", ".env.template"):
            has_env_template = True

    primary_language = _detect_primary_language(extension_counts)
    candidate_tasks = tuple(_suggest_candidates(
        root, primary_language, source_files, has_tests, has_docs, has_build_config,
    ))
    return ProjectAnalysis(
        primary_language=primary_language,
        file_count=sum(extension_counts.values()),
        extension_counts=dict(extension_counts),
        has_tests=has_tests,
        has_docs=has_docs,
        has_build_config=has_build_config,
        has_gitignore=has_gitignore,
        has_env_template=has_env_template,
        source_files=source_files,
        total_chars=total_chars,
        candidate_tasks=candidate_tasks,
    )


def _safe_iterdir(root: Path, skip: set[str]) -> list[Path]:
    result: list[Path] = []
    try:
        for entry in root.iterdir():
            if entry.name in skip:
                continue
            if entry.is_dir():
                if not entry.name.startswith("."):
                    result.extend(_safe_iterdir(entry, skip))
            else:
                result.append(entry)
    except PermissionError:
        pass
    return result


def _detect_primary_language(extension_counts: dict[str, int]) -> str:
    _CODE_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".java", ".rb", ".c", ".cpp", ".h"}
    code_exts = {k: v for k, v in extension_counts.items() if k in _CODE_EXTENSIONS}
    if not code_exts:
        return "unknown"
    best_ext = max(code_exts, key=code_exts.get)
    return _LANG_EXTENSIONS.get(best_ext, "unknown")


def _is_test_file(path: Path) -> bool:
    lower = path.name.lower()
    if lower.startswith("test_") or lower.endswith("_test.py") or lower.endswith(".test.js") or lower.endswith(".test.ts") or lower.endswith(".spec.js") or lower.endswith(".spec.ts"):
        return True
    parts = path.parts
    return any(part.lower() in _TEST_DIR_NAMES for part in parts)


def _is_doc_file(path: Path) -> bool:
    parts = path.parts
    return any(part.lower() in _DOC_DIR_NAMES for part in parts)


def _suggest_candidates(
    root: Path,
    language: str,
    source_files: int,
    has_tests: bool,
    has_docs: bool,
    has_build_config: bool,
) -> list[str]:
    tasks: list[str] = []
    if source_files > 0 and not has_tests:
        tasks.append("add tests for existing source code")
    if not has_docs:
        tasks.append("add project documentation")
    if not has_build_config:
        tasks.append("add build/packaging configuration")
    if has_tests and source_files > 5:
        tasks.append("improve test coverage for complex modules")
    if has_build_config and not has_docs:
        tasks.append("add developer documentation for build system")
    if language == "python":
        tasks.append("verify type annotations and linting")
    return tasks
