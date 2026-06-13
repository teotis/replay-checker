"""Core dataclasses for case-discovery orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..candidates import CandidateCase
from ..sources import EvidenceRecord
from ..task_contracts import default_orchestration_contract

# Fallback plan document search locations.  Generated kits never assume only
# ``docs/plans``; agents are instructed to search the full list.
_DEFAULT_PLAN_LOCATION_HINTS: tuple[str, ...] = (
    "docs/plans", "docs", "plans", "specifications", "specs",
    "notes", "rfc", ".playbook",
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
            "## Task Package Contract",
            "",
            "```yaml",
            default_orchestration_contract(
                package_id=self.package_id,
                description=self.description,
                allowed_paths=self.allowed_paths,
                dependencies=(),
                verification_commands=self.verification_commands,
            ).to_yaml(),
            "```",
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
