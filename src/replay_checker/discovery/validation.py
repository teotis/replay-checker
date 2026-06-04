"""Validation of generated discovery orchestration kits."""

from pathlib import Path


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
