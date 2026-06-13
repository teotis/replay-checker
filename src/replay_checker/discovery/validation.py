"""Validation of generated discovery orchestration kits."""

from pathlib import Path
import re


def validate_kit(plan_root: Path) -> list[str]:
    errors: list[str] = []
    graph_rows: list[dict[str, str]] = []
    state_rows: dict[str, dict[str, str]] = {}

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
                parts = line.split("\t")
                if len(parts) != expected_cols:
                    errors.append(f"package-graph.tsv row {i}: {len(parts)} columns, expected {expected_cols}")
                    continue
                graph_rows.append(dict(zip(expected_header_cols, parts)))

            ids = [row["package_id"] for row in graph_rows]
            if len(ids) != len(set(ids)):
                errors.append("package-graph.tsv contains duplicate package IDs")
            finalize_count = sum(1 for row in graph_rows if row["finalize"] == "1")
            if finalize_count != 1:
                errors.append(f"package-graph.tsv: expected exactly 1 finalize row, got {finalize_count}")
            errors.extend(_validate_graph_dependencies(graph_rows))
            errors.extend(_validate_package_docs(plan_root, graph_rows))

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
                else:
                    state_rows[cols[0]] = {"state": cols[1]}

            graph_ids = set()
            if graph_rows:
                graph_ids = {row["package_id"] for row in graph_rows}

            state_ids = {line.split("\t")[0] for line in lines[1:]}
            extra = state_ids - graph_ids
            if extra:
                errors.append(f"state.tsv has unknown packages: {', '.join(sorted(extra))}")
            missing = graph_ids - state_ids
            if missing:
                errors.append(f"state.tsv missing packages: {', '.join(sorted(missing))}")

    prompts_path = plan_root / "launchers" / "agent-prompts.md"
    if prompts_path.is_file() and graph_rows:
        errors.extend(_validate_prompt_headings(prompts_path, graph_rows))

    status_dir = plan_root / "status"
    if status_dir.is_dir():
        for row in graph_rows:
            status_rel = row["status_file"]
            status_file = plan_root / status_rel
            if not status_file.is_file():
                errors.append(f"missing status file: {status_rel}")
                continue
            content = status_file.read_text(encoding="utf-8")
            status_state = _extract_markdown_state(content)
            if "## State" not in content:
                errors.append(f"status file {status_rel}: missing ## State section")
            elif not status_state:
                errors.append(f"status file {status_rel}: missing backtick-wrapped state value")
            elif row["package_id"] in state_rows and status_state != state_rows[row["package_id"]]["state"]:
                errors.append(
                    f"status file {status_rel} state `{status_state}` disagrees with state.tsv `{state_rows[row['package_id']]['state']}`"
                )

    return errors


def _validate_package_docs(plan_root: Path, graph_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    for row in graph_rows:
        package_rel = row["package_doc"]
        if not (plan_root / package_rel).is_file():
            errors.append(f"missing package doc: {package_rel}")
    return errors


def _validate_prompt_headings(prompts_path: Path, graph_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    graph_ids = {row["package_id"] for row in graph_rows}
    headings: list[str] = []
    malformed: list[str] = []
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("## Package:"):
            continue
        match = re.match(r"^## Package: ([^\s]+) - .+$", line.strip())
        if not match:
            malformed.append(line.strip())
            continue
        headings.append(match.group(1))

    heading_ids = set(headings)
    missing = sorted(graph_ids - heading_ids)
    unknown = sorted(heading_ids - graph_ids)
    duplicates = sorted({package_id for package_id in headings if headings.count(package_id) > 1})
    if missing:
        errors.append(f"agent-prompts.md missing package headings: {', '.join(missing)}")
    if unknown:
        errors.append(f"agent-prompts.md has unknown package headings: {', '.join(unknown)}")
    if duplicates:
        errors.append(f"agent-prompts.md duplicate package headings: {', '.join(duplicates)}")
    if malformed:
        errors.append(f"agent-prompts.md has malformed package headings: {', '.join(malformed)}")
    return errors


def _validate_graph_dependencies(graph_rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    graph_ids = {row["package_id"] for row in graph_rows}
    adjacency: dict[str, list[str]] = {}
    for row in graph_rows:
        package_id = row["package_id"]
        dependencies = _split_dependencies(row["dependencies"])
        adjacency[package_id] = [dep for dep in dependencies if dep in graph_ids]
        for dep in dependencies:
            if dep not in graph_ids:
                errors.append(f"package-graph.tsv package {package_id} has unknown dependency: {dep}")
    if _has_cycle(adjacency):
        errors.append("package-graph.tsv dependency graph contains a cycle")
    return errors


def _split_dependencies(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _has_cycle(adjacency: dict[str, list[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dep in adjacency.get(node, []):
            if visit(dep):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _extract_markdown_state(content: str) -> str:
    in_state = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.lower() == "## state":
            in_state = True
            continue
        if in_state and stripped.startswith("#"):
            break
        if in_state:
            match = re.search(r"`([^`]+)`", stripped)
            if match:
                return match.group(1)
    return ""
