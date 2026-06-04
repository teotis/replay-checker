from __future__ import annotations

from dataclasses import dataclass, field


PROTOCOL_VERSION = "replay-checker-task-v2"


@dataclass(frozen=True)
class InputContract:
    workspace: str
    case_task_path: str
    completion_template_path: str


@dataclass(frozen=True)
class OutputContract:
    modify_files_inside_workspace: str
    write_completion_report: str
    include_status_summary: str


@dataclass(frozen=True)
class ExecutionPackage:
    protocol_version: str
    run_id: str
    goal: str
    starting_point: str
    allowed_scope: str
    expected_output: str
    verification_contract: str
    completion_report_schema: str
    forbidden_access: str
    ambiguity_handling: str
    source_material: str
    input_contract: InputContract
    output_contract: OutputContract
    runner_instructions: str
    verification_commands: tuple[str, ...] = ()
    conversation_evidence: str = ""


class ExecutionPackageCompiler:
    def __init__(
        self,
        *,
        run_id: str,
        case_id: str,
        case_task_path: str,
        plan_path: str,
        workspace: str,
        completion_template_path: str,
        verification_commands: tuple[str, ...] | list[str] = (),
    ) -> None:
        self._run_id = run_id
        self._case_id = case_id
        self._case_task_path = case_task_path
        self._plan_path = plan_path
        self._workspace = workspace
        self._completion_template_path = completion_template_path
        self._verification_commands = tuple(verification_commands)

    def compile(self) -> ExecutionPackage:
        return ExecutionPackage(
            protocol_version=PROTOCOL_VERSION,
            run_id=self._run_id,
            goal=self._build_goal(),
            starting_point=self._build_starting_point(),
            allowed_scope=self._build_allowed_scope(),
            expected_output=self._build_expected_output(),
            verification_contract=self._build_verification_contract(),
            completion_report_schema=self._build_completion_report_schema(),
            forbidden_access=self._build_forbidden_access(),
            ambiguity_handling=self._build_ambiguity_handling(),
            source_material=self._build_source_material(),
            input_contract=InputContract(
                workspace=self._workspace,
                case_task_path=self._case_task_path,
                completion_template_path=self._completion_template_path,
            ),
            output_contract=OutputContract(
                modify_files_inside_workspace=self._workspace,
                write_completion_report="completion_report.md in the workspace root",
                include_status_summary="status, summary, changed files, verification, and issues/notes",
            ),
            runner_instructions=self._build_runner_instructions(),
            verification_commands=self._verification_commands,
        )

    def _build_goal(self) -> str:
        extracted = _extract_goal_from_plan(self._plan_path)
        if extracted:
            return extracted
        extracted = _extract_goal_from_case_task(self._case_task_path)
        if extracted:
            return extracted
        return "Replay a historical project situation and produce equivalent changes independently."

    def _build_starting_point(self) -> str:
        return f"Workspace at `{self._workspace}`, starting from the base commit state."

    def _build_allowed_scope(self) -> str:
        return (
            "Work only inside the workspace.\n"
            "Follow the historical plan package and current case task.\n"
            "Leave enough process notes for later scoring, but do not include secrets."
        )

    def _build_expected_output(self) -> str:
        return (
            "A working workspace with file changes implementing the task.\n"
            "A completed `completion_report.md` in the workspace root."
        )

    def _build_verification_contract(self) -> str:
        commands = list(self._verification_commands)
        if not commands:
            return "No verification commands specified."
        return "Run the following commands and record their output:\n" + "\n".join(
            f"- `{cmd}`" for cmd in commands
        )

    def _build_completion_report_schema(self) -> str:
        return (
            "The completion report must include:\n"
            "- status (required)\n"
            "- summary of what was accomplished\n"
            "- list of changed files and why\n"
            "- verification command output\n"
            "- any blockers, risks, or notes for the scorer"
        )

    def _build_forbidden_access(self) -> str:
        return (
            "- Do not read `_reference/` directories or reference/oracle evidence.\n"
            "- Do not use runner identity, model identity, or hidden prompts as evidence.\n"
            "- Do not add secrets, credentials, or private environment files."
        )

    def _build_ambiguity_handling(self) -> str:
        return (
            "If the task is ambiguous, proceed with the most reasonable interpretation.\n"
            "Document your interpretation in the completion report's issues/notes section."
        )

    def _build_source_material(self) -> str:
        return (
            f"- Case task: `{self._case_task_path}`\n"
            f"- Plan package: `{self._plan_path}`"
        )

    def _build_runner_instructions(self) -> str:
        lines = [
            f"Copy TASK.md into your chosen agent, or open this workspace in that agent manually.",
            f"Replay Checker does not control the agent process in the default V1 flow.",
            "",
            f"Workspace: `{self._workspace}`",
            f"Case task: `{self._case_task_path}`",
            f"Completion template: `{self._completion_template_path}`",
        ]
        return "\n".join(lines)


class ExecutionPackageRenderer:
    def render(self, pkg: ExecutionPackage) -> str:
        sections = [
            self._header(pkg),
            self._source_section(pkg),
            self._input_section(pkg),
            self._goal_section(pkg),
            self._starting_point_section(pkg),
            self._allowed_scope_section(pkg),
            self._expected_output_section(pkg),
            self._agent_output_contract_section(pkg),
            self._verification_contract_section(pkg),
            self._completion_report_schema_section(pkg),
            self._forbidden_access_section(pkg),
            self._evidence_requirements_section(pkg),
            self._ambiguity_handling_section(pkg),
            self._verification_commands_section(pkg),
        ]
        if pkg.conversation_evidence:
            sections.append(self._conversation_evidence_section(pkg))
        return "\n".join(sections)

    def _header(self, pkg: ExecutionPackage) -> str:
        return (
            f"# Execution Package: {pkg.run_id}\n"
            f"\n"
            f"Protocol Version: {pkg.protocol_version}\n"
        )

    def _source_section(self, pkg: ExecutionPackage) -> str:
        return f"## Source\n{pkg.source_material}\n"

    def _input_section(self, pkg: ExecutionPackage) -> str:
        lines = [
            "## Inputs",
            f"- Workspace: `{pkg.input_contract.workspace}`",
            f"- Case task: `{pkg.input_contract.case_task_path}`",
            f"- Completion template: `{pkg.input_contract.completion_template_path}`",
        ]
        return "\n".join(lines) + "\n"

    def _goal_section(self, pkg: ExecutionPackage) -> str:
        return f"## Goal\n{pkg.goal}\n"

    def _starting_point_section(self, pkg: ExecutionPackage) -> str:
        return f"## Starting Point\n{pkg.starting_point}\n"

    def _allowed_scope_section(self, pkg: ExecutionPackage) -> str:
        return f"## Allowed Scope\n{pkg.allowed_scope}\n"

    def _expected_output_section(self, pkg: ExecutionPackage) -> str:
        return f"## Expected Output\n{pkg.expected_output}\n"

    def _verification_contract_section(self, pkg: ExecutionPackage) -> str:
        return f"## Verification Contract\n{pkg.verification_contract}\n"

    def _completion_report_schema_section(self, pkg: ExecutionPackage) -> str:
        return f"## Completion Report Schema\n{pkg.completion_report_schema}\n"

    def _forbidden_access_section(self, pkg: ExecutionPackage) -> str:
        return f"## Forbidden Access\n{pkg.forbidden_access}\n"

    def _agent_output_contract_section(self, pkg: ExecutionPackage) -> str:
        return (
            "## Agent Output Contract\n"
            "- Modify files only inside the workspace.\n"
            f"- Write `{pkg.output_contract.write_completion_report}`.\n"
            f"- Include {pkg.output_contract.include_status_summary} in the report.\n"
        )

    def _evidence_requirements_section(self, pkg: ExecutionPackage) -> str:
        lines = [
            "## Evidence Requirements",
            "- A non-empty git diff in the workspace.",
            "- Stage all changes (`git add -A`) before finishing so that new files appear in the diff.",
            "- A completed `completion_report.md`.",
            "- Verification command output or an explicit explanation if verification cannot run.",
        ]
        return "\n".join(lines) + "\n"

    def _ambiguity_handling_section(self, pkg: ExecutionPackage) -> str:
        return f"## Ambiguity Handling\n{pkg.ambiguity_handling}\n"

    def _verification_commands_section(self, pkg: ExecutionPackage) -> str:
        lines = ["## Verification Commands"]
        if pkg.verification_commands:
            lines.extend(f"- `{cmd}`" for cmd in pkg.verification_commands)
        else:
            lines.append("- (none specified)")
        return "\n".join(lines) + "\n"

    def _conversation_evidence_section(self, pkg: ExecutionPackage) -> str:
        return f"## Conversation Evidence\n{pkg.conversation_evidence}\n"


def compile_execution_package(
    *,
    run_id: str,
    case_id: str,
    case_task_path: str,
    plan_path: str,
    workspace: str,
    completion_template_path: str,
    verification_commands: tuple[str, ...] | list[str] = (),
) -> ExecutionPackage:
    compiler = ExecutionPackageCompiler(
        run_id=run_id,
        case_id=case_id,
        case_task_path=case_task_path,
        plan_path=plan_path,
        workspace=workspace,
        completion_template_path=completion_template_path,
        verification_commands=verification_commands,
    )
    return compiler.compile()


def _extract_goal_from_plan(plan_path: str) -> str:
    """Extract a concrete goal summary from a plan markdown file.

    Priority order:
    1. ## Title section (most common in orchestration-kit packages)
    2. ## Goal / ## Task / ## Objective section
    3. Document title (# heading)

    Returns "" when the plan file is unreadable or empty.
    """
    from pathlib import Path

    path = Path(plan_path)
    if not path.is_file():
        return ""

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""

    if not text.strip():
        return ""

    lines = text.splitlines()

    # Priority 1: ## Title section
    title_text = _extract_section(lines, ("title", "标题"))
    if title_text:
        return title_text

    # Priority 2: Explicit goal/task/objective section
    goal_text = _extract_section(lines, ("goal", "task", "objective", "purpose", "概述", "目标", "任务"))
    if goal_text:
        return goal_text

    # Priority 3: Document title
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            # Strip "Package: NN-" prefix if present
            if title.startswith("Package:"):
                title = title.split("-", 1)[-1].strip() if "-" in title else title[len("Package:"):].strip()
            if len(title) >= 10:
                return title
            break

    return ""


def _extract_goal_from_case_task(case_task_path: str) -> str:
    """Extract an execution-facing goal from a rendered case task."""
    from pathlib import Path

    path = Path(case_task_path)
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""

    goal = _extract_section(lines, ("goal", "objective", "reconstructed task"))
    if goal:
        return goal

    for i, line in enumerate(lines):
        if line.strip().lower() == "### objective":
            parts: list[str] = []
            for subsequent in lines[i + 1:]:
                stripped = subsequent.strip()
                if stripped.startswith("#"):
                    break
                if stripped:
                    parts.append(stripped)
                elif parts:
                    break
            return " ".join(parts)

    return ""


def _extract_section(lines: list[str], keywords: tuple[str, ...]) -> str:
    """Extract the first paragraph body from a ## section matching any keyword."""
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("## ") and any(kw in stripped for kw in keywords):
            parts: list[str] = []
            for j in range(i + 1, len(lines)):
                subsequent = lines[j].strip()
                if subsequent.startswith("#"):
                    break
                if subsequent:
                    parts.append(subsequent)
                elif parts:
                    break
            if parts:
                return " ".join(parts)
            break
    return ""
