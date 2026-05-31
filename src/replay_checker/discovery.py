"""Case-discovery orchestration kit generator.

Generates a self-contained orchestration kit that external agents can execute
to discover replayable cases from a project directory and natural-language scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .candidates import CandidateCase, build_case_candidates, select_case_candidate
from .sources import (
    EvidenceRecord,
    EvidenceSourceConfig,
    discover_evidence_sources,
    score_evidence_records,
)


@dataclass(frozen=True)
class DiscoveryPackage:
    package_id: str
    description: str
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    dependency_type: str = "status"
    wave: int = 1
    branch_suffix: str = ""
    verification_commands: tuple[str, ...] = ()
    is_manual: bool = False
    is_finalize: bool = False


# ---------------------------------------------------------------------------
# Source discovery package templates
# ---------------------------------------------------------------------------

# Fallback plan document search locations.  Generated kits never assume only
# ``docs/plans``; agents are instructed to search the full list.
_DEFAULT_PLAN_LOCATION_HINTS: tuple[str, ...] = (
    "docs/plans", "docs", "plans", "specifications", "specs",
    "notes", "rfc", ".playbook",
)


@dataclass(frozen=True)
class SourceDiscoveryTemplate:
    """Rich template for a single source-discovery package.

    Fields beyond the basic ``DiscoveryPackage`` carry the extra metadata that
    generated kit agents need: scope text, source budget, expected evidence
    list, privacy constraints, plan-location hints, and workload budget.
    """

    package_id: str
    description: str
    scope_text: str
    source_budget: str
    expected_evidence: tuple[str, ...] = ()
    privacy_constraints: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    forbidden_paths: tuple[str, ...] = ()
    verification_commands: tuple[str, ...] = ()
    tail_call_advance: bool = True
    plan_location_hints: tuple[str, ...] = _DEFAULT_PLAN_LOCATION_HINTS
    raw_log_prohibition: str = ""
    works_without_plans: bool = False

    def compile_package_doc(self) -> str:
        """Render a complete package document for an executing agent."""
        sections: list[str] = [
            f"# {self.package_id}",
            "",
            "## Goal",
            "",
            self.description,
            "",
            "## Scope",
            "",
            self.scope_text,
            "",
        ]

        if self.plan_location_hints and not self.works_without_plans:
            sections += [
                "## Plan Document Locations",
                "",
                "Search these directories for plan/spec documents (not limited to any single path):",
                "",
            ]
            for hint in self.plan_location_hints:
                sections.append(f"- `{hint}/`")
            sections.append("")

        sections += [
            "## Source Budget",
            "",
            self.source_budget,
            "",
        ]

        if self.expected_evidence:
            sections += [
                "## Expected Evidence",
                "",
            ]
            for ev in self.expected_evidence:
                sections.append(f"- {ev}")
            sections.append("")

        sections += [
            "## Allowed Paths",
            "",
        ]
        for p in self.allowed_paths:
            sections.append(f"- `{p}`")
        sections += [
            "",
            "## Forbidden Paths",
            "",
        ]
        for p in self.forbidden_paths:
            sections.append(f"- `{p}`")
        sections.append("")

        if self.privacy_constraints:
            sections += [
                "## Privacy Constraints",
                "",
            ]
            for c in self.privacy_constraints:
                sections.append(f"- {c}")
            sections.append("")

        if self.raw_log_prohibition:
            sections += [
                "## Raw Log Policy",
                "",
                self.raw_log_prohibition,
                "",
            ]

        sections += [
            "## Verification Commands",
            "",
            "```bash",
        ]
        if self.verification_commands:
            for cmd in self.verification_commands:
                sections.append(cmd)
        else:
            sections.append(f"echo '{self.package_id} ok'")
        sections += [
            "```",
            "",
        ]

        return "\n".join(sections) + "\n"


def default_source_templates() -> tuple[SourceDiscoveryTemplate, ...]:
    """Return source-discovery templates for all package classes.

    Templates are designed for large real-world projects:
    - Plan search is NOT limited to ``docs/plans``.
    - Agent-history templates explicitly prohibit copying raw logs.
    - Code-analysis works when no plan docs exist.
    - Budget fields allow large projects to split work.
    """
    return (
        SourceDiscoveryTemplate(
            package_id="plan-source",
            description="Discover and analyze plan documents, task specifications, and project documentation.",
            scope_text=(
                "Search the full project tree for plan/spec documents. "
                "Do not limit search to ``docs/plans`` — also check ``docs/``, "
                "``plans/``, ``specifications/``, ``specs/``, ``notes/``, "
                "``rfc/``, and ``.playbook/``.  Each document found should be "
                "read and summarized (title, first heading, key goals)."
            ),
            source_budget="Read up to 200 plan documents; skip files larger than 200 KB.",
            expected_evidence=(
                "List of plan document paths with summaries",
                "Task keyword presence per document",
                "Confidence score per document",
            ),
            privacy_constraints=(
                "Do not copy plan document contents into case artifacts.",
                "Store only paths and summaries.",
            ),
            allowed_paths=("docs/", "plans/", "specifications/", "specs/", "notes/", "rfc/", ".playbook/", "*.md", "README.md"),
            forbidden_paths=("_reference/",),
            verification_commands=("find . -name '*.md' -path '*/docs/*' -o -name '*.md' -path '*/plans/*' | head -20",),
            plan_location_hints=_DEFAULT_PLAN_LOCATION_HINTS,
            works_without_plans=False,
        ),
        SourceDiscoveryTemplate(
            package_id="agent-history",
            description="Discover bounded local agent conversation history summaries.",
            scope_text=(
                "Scan local agent session directories (``~/.claude/``, ``~/.codex/``) "
                "for conversations mentioning this project.  Produce bounded summaries "
                "(max 500 chars per source) of relevant user turns."
            ),
            source_budget="Scan up to 500 session files; summarize up to 3 relevant snippets per file.",
            expected_evidence=(
                "List of relevant session file paths",
                "Bounded conversation summaries (max 500 chars each)",
                "Task signal classification per session",
            ),
            privacy_constraints=(
                "Do not copy raw conversation logs into case artifacts.",
                "Summaries only — never embed full message history.",
                "Skip files in temporary or cache directories.",
            ),
            allowed_paths=("~/.claude/", "~/.codex/"),
            forbidden_paths=("_reference/",),
            verification_commands=(),
            raw_log_prohibition=(
                "Raw log copying is strictly prohibited.  Agents must extract "
                "bounded summaries from conversation files and store only those "
                "summaries.  Full JSONL or text logs must never be embedded in "
                "case packages, evidence packs, or any committed artifact."
            ),
            plan_location_hints=(),
            works_without_plans=True,
        ),
        SourceDiscoveryTemplate(
            package_id="git-history",
            description="Discover replayable situations from git commit history and diffs.",
            scope_text=(
                "Analyze git log for task-like commits (feat, fix, refactor, test). "
                "For each candidate commit: extract the diff stats, changed files, "
                "and any test signal.  Focus on commits with measurable code changes."
            ),
            source_budget="Examine up to 200 commits; include commits with >5 lines changed.",
            expected_evidence=(
                "Commit SHA, message, and stats",
                "Changed file list per commit",
                "Diff hunks for key files",
                "Test signal (passing/failing) if available",
            ),
            privacy_constraints=(
                "Do not copy binary diffs or large generated files.",
                "Diff hunks capped at 2000 chars each.",
            ),
            allowed_paths=(".git/",),
            forbidden_paths=("_reference/",),
            verification_commands=("git log --oneline -20",),
            plan_location_hints=(),
            works_without_plans=True,
        ),
        SourceDiscoveryTemplate(
            package_id="code-analysis",
            description="Analyze codebase structure and generate discovery candidates from code patterns.",
            scope_text=(
                "Scan the project source tree for entry points, public APIs, "
                "and test files.  Generate discovery candidates from code structure "
                "without requiring plan documents.  When plan docs exist, cross-"
                "reference them with code structure to improve confidence."
            ),
            source_budget="Scan up to 500 source files; extract structure from the first 100 meaningful modules.",
            expected_evidence=(
                "Module/file structure summary",
                "Public API surface",
                "Test coverage indicators",
                "Cross-reference with plan docs (if any)",
            ),
            privacy_constraints=(
                "Do not embed source code in case artifacts — only file paths and summaries.",
                "Respect .gitignore and .ignore patterns.",
            ),
            allowed_paths=("src/", "lib/", "app/", "tests/", "test/"),
            forbidden_paths=("_reference/",),
            verification_commands=(),
            plan_location_hints=(),
            works_without_plans=True,
        ),
        SourceDiscoveryTemplate(
            package_id="candidate-merge",
            description="Merge source discovery candidates, rank by relevance, and select best case candidates.",
            scope_text=(
                "Combine candidates from plan-source, agent-history, git-history, "
                "and code-analysis.  Deduplicate, rank by relevance score, and "
                "select the top candidates for case compilation."
            ),
            source_budget="Process up to 200 candidates; output top 20 ranked results.",
            expected_evidence=(
                "Merged candidate list with deduplication",
                "Relevance ranking per candidate",
                "Selected top-N candidates for case compilation",
            ),
            privacy_constraints=(
                "Do not expose runner identity or agent model information.",
                "Store only candidate metadata, not raw source content.",
            ),
            allowed_paths=("cases/", "work/"),
            forbidden_paths=("_reference/",),
            verification_commands=(),
            plan_location_hints=(),
            works_without_plans=False,
        ),
        SourceDiscoveryTemplate(
            package_id="case-package-compile",
            description="Compile selected candidates into replayable case packages with task contracts.",
            scope_text=(
                "For each selected candidate, generate a case package with: "
                "task contract, allowed/forbidden paths, evidence gates, and "
                "verification commands.  Packages must be self-contained and "
                "platform-neutral."
            ),
            source_budget="Compile up to 20 case packages; each package under 50 KB.",
            expected_evidence=(
                "Case directory with task.md and case.yaml",
                "Evidence source manifest",
                "Verification command list",
            ),
            privacy_constraints=(
                "Do not embed reference/oracle evidence in execution packages.",
                "Anonymize runner identity with stable IDs.",
                "Do not include scoring details in execution packages.",
            ),
            allowed_paths=("cases/", "runs/", "rubrics/"),
            forbidden_paths=("_reference/",),
            verification_commands=("python3 -c 'from replay_checker.packages import *'",),
            plan_location_hints=(),
            works_without_plans=False,
        ),
    )


def source_template_map() -> dict[str, SourceDiscoveryTemplate]:
    """Return a package_id → template lookup."""
    return {t.package_id: t for t in default_source_templates()}


@dataclass(frozen=True)
class DiscoveryKitConfig:
    project_path: Path
    scope: str
    output_root: Path
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self, "timestamp", datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            )

    @property
    def slug(self) -> str:
        import re
        raw = self.project_path.name
        slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
        return slug or "project"

    @property
    def plan_dir_name(self) -> str:
        return f"{self.slug}-case-discovery-{self.timestamp}"

    @property
    def plan_root(self) -> Path:
        return self.output_root / self.plan_dir_name


@dataclass
class DiscoveryKit:
    plan_root: Path
    project_path: Path
    scope: str
    packages: tuple[DiscoveryPackage, ...]

    @property
    def graph_path(self) -> Path:
        return self.plan_root / "launchers" / "package-graph.tsv"

    @property
    def state_path(self) -> Path:
        return self.plan_root / "status" / "state.tsv"

    @property
    def orchestrate_path(self) -> Path:
        return self.plan_root / "launchers" / "orchestrate.sh"

    @property
    def index_path(self) -> Path:
        return self.plan_root / "INDEX.md"

    @property
    def prompts_path(self) -> Path:
        return self.plan_root / "launchers" / "agent-prompts.md"


def default_discovery_packages() -> tuple[DiscoveryPackage, ...]:
    return (
        DiscoveryPackage(
            package_id="plan-source",
            description="Discover and analyze plan documents, task specifications, and project documentation.",
            allowed_paths=("docs/", "plans/", "*.md", "README.md"),
            forbidden_paths=("_reference/",),
            dependencies=(),
            dependency_type="status",
            wave=1,
            verification_commands=("find docs/ -name '*.md' -type f",),
        ),
        DiscoveryPackage(
            package_id="agent-history",
            description="Discover bounded local agent conversation history summaries.",
            allowed_paths=("~/.claude/", "~/.codex/"),
            forbidden_paths=("_reference/",),
            dependencies=(),
            dependency_type="status",
            wave=1,
            verification_commands=(),
        ),
        DiscoveryPackage(
            package_id="git-history",
            description="Discover replayable situations from git commit history and diffs.",
            allowed_paths=(".git/",),
            forbidden_paths=("_reference/",),
            dependencies=(),
            dependency_type="status",
            wave=1,
            verification_commands=("git log --oneline -20",),
        ),
        DiscoveryPackage(
            package_id="code-analysis",
            description="Analyze codebase structure and generate discovery candidates from code patterns.",
            allowed_paths=("src/", "lib/", "app/", "tests/"),
            forbidden_paths=("_reference/",),
            dependencies=("plan-source", "agent-history", "git-history"),
            dependency_type="status",
            wave=2,
            verification_commands=(),
        ),
        DiscoveryPackage(
            package_id="candidate-merge",
            description="Merge source discovery candidates, rank by relevance, and select best case candidates.",
            allowed_paths=("cases/", "work/"),
            forbidden_paths=("_reference/",),
            dependencies=("code-analysis",),
            dependency_type="status",
            wave=3,
            verification_commands=(),
        ),
        DiscoveryPackage(
            package_id="case-package-compile",
            description="Compile selected candidates into replayable case packages with task contracts.",
            allowed_paths=("cases/", "runs/", "rubrics/"),
            forbidden_paths=("_reference/",),
            dependencies=("candidate-merge",),
            dependency_type="status",
            wave=4,
            verification_commands=("python3 -c 'from replay_checker.packages import *'",),
        ),
        DiscoveryPackage(
            package_id="finalize",
            description="Merge discovery branches, verify integration, and produce final report.",
            allowed_paths=("docs/", "control/"),
            forbidden_paths=("_reference/",),
            dependencies=(
                "plan-source", "agent-history", "git-history",
                "code-analysis", "candidate-merge", "case-package-compile",
            ),
            dependency_type="status+code",
            wave="final",
            is_manual=False,
            is_finalize=True,
        ),
    )


def generate_discovery_kit(config: DiscoveryKitConfig) -> DiscoveryKit:
    from .orchestration import generate_kit_files

    project = config.project_path.resolve()
    if not project.is_dir():
        raise ValueError(f"Project path is not a directory: {project}")

    packages = default_discovery_packages()
    kit_files = generate_kit_files(config, packages)
    plan_root = config.plan_root

    for path, content in kit_files.items():
        full = plan_root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")

    plan_root.chmod(0o755)

    return DiscoveryKit(
        plan_root=plan_root,
        project_path=project,
        scope=config.scope,
        packages=packages,
    )


def validate_kit(plan_root: Path) -> list[str]:
    errors: list[str] = []

    required_dirs = ["launchers", "status", "packages", "scratch"]
    for d in required_dirs:
        if not (plan_root / d).is_dir():
            errors.append(f"missing directory: {d}")

    required_files = ["INDEX.md", "launchers/package-graph.tsv", "status/state.tsv", "launchers/agent-prompts.md"]
    for f in required_files:
        if not (plan_root / f).is_file():
            errors.append(f"missing file: {f}")

    graph_path = plan_root / "launchers" / "package-graph.tsv"
    if graph_path.is_file():
        lines = [l for l in graph_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            header = lines[0]
            expected_header_cols = ("package_id", "package_doc", "status_file", "dependencies",
                                    "dependency_type", "wave", "branch", "worktree", "manual", "finalize")
            actual_header_cols = header.split("\t")
            if tuple(actual_header_cols) != expected_header_cols:
                errors.append("package-graph.tsv header does not match expected format")
            expected_cols = 10
            for i, line in enumerate(lines[1:], start=2):
                cols = line.count("\t") + 1
                if cols != expected_cols:
                    errors.append(f"package-graph.tsv row {i}: {cols} columns, expected {expected_cols}")

            ids = [line.split("\t")[0] for line in lines[1:]]
            if len(ids) != len(set(ids)):
                errors.append("package-graph.tsv contains duplicate package IDs")
            finalize_count = sum(1 for line in lines[1:] if len(line.split("\t")) > 9 and line.split("\t")[9] == "1")
            if finalize_count != 1:
                errors.append(f"package-graph.tsv: expected exactly 1 finalize row, got {finalize_count}")

    state_path = plan_root / "status" / "state.tsv"
    if state_path.is_file():
        valid_states = {"pending", "ready", "manual_required", "launched", "in_progress", "completed", "blocked", "stale", "invalid", "finalizing", "finalized"}
        lines = [l for l in state_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            for i, line in enumerate(lines[1:], start=2):
                cols = line.split("\t")
                if len(cols) != 17:
                    errors.append(f"state.tsv row {i}: {len(cols)} columns, expected 17")
                elif cols[1] not in valid_states:
                    errors.append(f"state.tsv row {i}: invalid state '{cols[1]}'")

            graph_ids = set()
            if graph_path.is_file():
                glines = [l for l in graph_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                graph_ids = {l.split("\t")[0] for l in glines[1:]}

            state_ids = {line.split("\t")[0] for line in lines[1:]}
            extra = state_ids - graph_ids
            if extra:
                errors.append(f"state.tsv has unknown packages: {', '.join(sorted(extra))}")
            missing = graph_ids - state_ids
            if missing:
                errors.append(f"state.tsv missing packages: {', '.join(sorted(missing))}")

    status_dir = plan_root / "status"
    if status_dir.is_dir():
        graph_path = plan_root / "launchers" / "package-graph.tsv"
        if graph_path.is_file():
            glines = [l for l in graph_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            for gline in glines[1:]:
                parts = gline.split("\t")
                if len(parts) >= 3:
                    status_rel = parts[2]
                    status_file = plan_root / status_rel
                    if not status_file.is_file():
                        errors.append(f"missing status file: {status_rel}")
                    else:
                        content = status_file.read_text(encoding="utf-8")
                        if "## State" not in content:
                            errors.append(f"status file {status_rel}: missing ## State section")
                        elif "`pending`" not in content.lower() and "`completed`" not in content.lower():
                            errors.append(f"status file {status_rel}: missing backtick-wrapped state value")

    return errors


# ---------------------------------------------------------------------------
# Code analysis scaffolding (no-plan path)
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class ProjectAnalysis:
    """Result of analyzing a project's source code structure."""

    primary_language: str
    file_count: int
    extension_counts: dict[str, int] = field(default_factory=dict)
    has_tests: bool = False
    has_docs: bool = False
    has_build_config: bool = False
    has_gitignore: bool = False
    has_env_template: bool = False
    source_files: int = 0
    total_chars: int = 0
    candidate_tasks: tuple[str, ...] = ()


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


# ---------------------------------------------------------------------------
# Discovery output schema and merge logic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryOutput:
    """Schema for the output of a single discovery source.

    Each discovery agent (plan detection, code analysis, history scan)
    produces a DiscoveryOutput containing the evidence records it found.
    """
    source_agent: str
    records: tuple[EvidenceRecord, ...] = ()
    project_path: str = ""


@dataclass(frozen=True)
class MergedCandidate:
    """A candidate enriched with confidence and risk metadata after merge.

    Wraps CandidateCase with additional merge-specific information:
    - Which discovery agents contributed to this candidate
    - Computed confidence level and risk annotations
    - Whether this candidate is low-confidence (still generates a case)
    """
    candidate: CandidateCase
    contributing_agents: tuple[str, ...] = ()
    confidence: str = "medium"
    risk_level: str = "low"
    risk_reasons: tuple[str, ...] = ()


def merge_discovery_outputs(
    outputs: list[DiscoveryOutput],
) -> list[MergedCandidate]:
    """Merge multiple discovery outputs into a single ranked candidate list.

    Combines evidence records from all sources, deduplicates candidates,
    and assigns confidence/risk metadata to each.
    """
    all_records: list[EvidenceRecord] = []
    # Map evidence record path → contributing agents
    record_agent_map: dict[str, str] = {}

    for output in outputs:
        for record in output.records:
            key = record.path.as_posix()
            if key not in record_agent_map:
                record_agent_map[key] = output.source_agent
            all_records.append(record)

    scored = score_evidence_records(all_records) if all_records else []
    candidates = build_case_candidates(scored)
    deduped = dedupe_candidates(candidates)

    # Build a map from primary_source → contributing agents for candidates
    source_agents: dict[str, tuple[str, ...]] = {}
    for c in candidates:
        key = c.primary_source
        source_agents.setdefault(key, ())

    # Enrich source_agents from the record_agent_map
    agent_set_by_source: dict[str, set[str]] = {}
    for rec in all_records:
        source_key = _record_primary_source(rec)
        agent_set_by_source.setdefault(source_key, set()).add(
            record_agent_map.get(rec.path.as_posix(), rec.source_type)
        )
    source_agents = {k: tuple(v) for k, v in agent_set_by_source.items()}

    enriched = assign_confidence_and_risk(deduped, source_agents)
    enriched.sort(key=lambda c: c.candidate.relevance_score, reverse=True)
    return enriched


def _record_primary_source(record: EvidenceRecord) -> str:
    return record.metadata.get("relative_path", str(record.path))


def dedupe_candidates(candidates: list[CandidateCase]) -> list[CandidateCase]:
    """Remove duplicate candidates, keeping the highest-scored version.

    Two plan candidates are duplicates if they reference the same plan file.
    History candidates are deduplicated by source_type (already grouped by
    build_case_candidates).
    """
    seen_plan: dict[str, CandidateCase] = {}
    history_types: dict[str, CandidateCase] = {}
    result: list[CandidateCase] = []

    for c in candidates:
        if c.source_type == "plan":
            key = c.primary_source
            if key not in seen_plan or c.relevance_score > seen_plan[key].relevance_score:
                seen_plan[key] = c
        else:
            # History candidates are already grouped by source_type
            if c.source_type not in history_types:
                history_types[c.source_type] = c

    result.extend(seen_plan.values())
    result.extend(history_types.values())
    return result


def assign_confidence_and_risk(
    candidates: list[CandidateCase],
    agent_map: dict[str, list[str]] | None = None,
) -> list[MergedCandidate]:
    """Compute confidence and risk level for each candidate.

    Confidence is based on:
    - Source type (plan > code_analysis > history)
    - Base commit availability
    - Number of contributing discovery agents
    - Task signal quality

    Risk is based on:
    - Confidence level
    - Presence of risk markers from the candidate
    - Supporting source count
    """
    result: list[MergedCandidate] = []
    for c in candidates:
        confidence = _compute_confidence(c, agent_map)
        risk_level, risk_reasons = _compute_risk(c, confidence, agent_map)
        contributing = tuple(
            _agents_for_candidate(c, agent_map)
        )
        result.append(MergedCandidate(
            candidate=c,
            contributing_agents=contributing,
            confidence=confidence,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
        ))
    return result


def compile_cases_from_discovery(
    merged_candidates: list[MergedCandidate],
    *,
    project_path: str | Path,
    selected_id: str = "",
) -> tuple[CandidateCase | None, list[CandidateCase]]:
    """Select the best candidate and prepare case compilation data.

    Returns (selected, all_candidates) where selected is the highest-confidence
    candidate to build a case from, and all_candidates is the full deduplicated
    list.

    Low-confidence candidates are not excluded — they still produce cases
    with explicit risk metadata.
    """
    if not merged_candidates:
        return None, []

    sorted_by_confidence = sorted(
        merged_candidates,
        key=lambda mc: (
            _confidence_rank(mc.confidence),
            mc.candidate.relevance_score,
        ),
        reverse=True,
    )

    selected_mc = sorted_by_confidence[0]
    all_candidates = [mc.candidate for mc in merged_candidates]
    return selected_mc.candidate, all_candidates


def compile_low_confidence_case(
    candidate: MergedCandidate,
    *,
    project_path: str | Path,
) -> dict[str, object]:
    """Build case metadata for a low-confidence candidate.

    Returns a dict with explicit risk markers that the case writer
    should include. Does not create files — the caller decides
    whether to persist the case.
    """
    return {
        "candidate_id": candidate.candidate.candidate_id,
        "source_type": candidate.candidate.source_type,
        "confidence": candidate.confidence,
        "risk_level": candidate.risk_level,
        "risk_reasons": list(candidate.risk_reasons),
        "relevance_score": candidate.candidate.relevance_score,
        "primary_source": candidate.candidate.primary_source,
        "selection_reasons": list(candidate.candidate.selection_reasons),
        "low_confidence": True,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def _confidence_rank(confidence: str) -> int:
    return _CONFIDENCE_RANK.get(confidence, 0)


def _compute_confidence(
    candidate: CandidateCase,
    agent_map: dict[str, list[str]] | None,
) -> str:
    """Score-based confidence assignment."""
    score = candidate.relevance_score

    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _compute_risk(
    candidate: CandidateCase,
    confidence: str,
    agent_map: dict[str, list[str]] | None,
) -> tuple[str, tuple[str, ...]]:
    """Compute risk level and reasons for a candidate."""
    reasons: list[str] = []

    if confidence == "low":
        reasons.append("low confidence candidate")
    if candidate.risks:
        reasons.extend(candidate.risks)
    if not candidate.supporting_sources and candidate.source_type != "plan":
        reasons.append("no supporting sources")
    if candidate.base_confidence == "low":
        reasons.append("low base commit confidence")

    if not reasons:
        return "low", ()
    if confidence == "low":
        return "high", tuple(reasons)
    return "medium", tuple(reasons)


def _agents_for_candidate(
    candidate: CandidateCase,
    agent_map: dict[str, tuple[str, ...]] | None,
) -> tuple[str, ...]:
    """Find which discovery agents contributed to this candidate."""
    if not agent_map:
        return (candidate.source_type,)
    key = candidate.primary_source
    agents = agent_map.get(key)
    return agents if agents else (candidate.source_type,)
