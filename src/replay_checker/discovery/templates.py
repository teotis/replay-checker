"""Source-discovery package templates and default package definitions."""

from .models import (
    DiscoveryPackage,
    SourceDiscoveryTemplate,
    _DEFAULT_PLAN_LOCATION_HINTS,
)


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
