"""Case-discovery orchestration kit generator.

Generates a self-contained orchestration kit that external agents can execute
to discover replayable cases from a project directory and natural-language scope.
"""

from __future__ import annotations

from .code_analysis import ProjectAnalysis, analyze_project
from .merge import (
    MergedCandidate,
    assign_confidence_and_risk,
    compile_cases_from_discovery,
    compile_low_confidence_case,
    dedupe_candidates,
    merge_discovery_outputs,
)
from .models import (
    DiscoveryKit,
    DiscoveryKitConfig,
    DiscoveryOutput,
    DiscoveryPackage,
    SourceDiscoveryTemplate,
)
from .templates import (
    default_discovery_packages,
    default_source_templates,
    source_template_map,
)
from .validation import validate_kit


def generate_discovery_kit(config: DiscoveryKitConfig) -> DiscoveryKit:
    from ..orchestration import generate_kit_files

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


__all__ = [
    "DiscoveryPackage",
    "DiscoveryKitConfig",
    "DiscoveryKit",
    "DiscoveryOutput",
    "MergedCandidate",
    "ProjectAnalysis",
    "SourceDiscoveryTemplate",
    "analyze_project",
    "assign_confidence_and_risk",
    "compile_cases_from_discovery",
    "compile_low_confidence_case",
    "default_discovery_packages",
    "default_source_templates",
    "dedupe_candidates",
    "generate_discovery_kit",
    "merge_discovery_outputs",
    "source_template_map",
    "validate_kit",
]
