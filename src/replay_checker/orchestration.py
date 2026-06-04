"""Orchestration kit file generation.

Generates all files for a case-discovery orchestration kit: INDEX, package-graph,
state.tsv, status markdown files, agent prompts, and orchestrate.sh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .discovery import DiscoveryKitConfig, DiscoveryPackage, SourceDiscoveryTemplate, source_template_map


GRAPH_HEADER = "package_id\tpackage_doc\tstatus_file\tdependencies\tdependency_type\twave\tbranch\tworktree\tmanual\tfinalize"
STATE_HEADER = "package_id\tstate\tlaunched_at\tcompleted_at\tagent\tbranch\tworktree\tbase_commit\tcommit_hash\tverification\tintegration\tcleanup\tlast_error\tfailed_command\tconflict_files\tlog_summary\trecovery_hint"
VALID_STATES = (
    "pending", "ready", "manual_required", "launched", "in_progress",
    "completed", "blocked", "stale", "invalid", "finalizing", "finalized",
)


def generate_graph_tsv(
    packages: Sequence[DiscoveryPackage],
    plan_root: Path,
    project_root: Path,
) -> str:
    lines = [GRAPH_HEADER]
    for pkg in packages:
        package_doc = f"packages/{pkg.package_id}.md"
        status_file = f"status/{pkg.package_id}.md"
        deps = ",".join(pkg.dependencies)
        branch = f"discovery/{plan_root.name}/{pkg.package_id}"
        worktree = (project_root / ".replay-kit" / plan_root.name / pkg.package_id).as_posix()
        manual = "1" if pkg.is_manual else "0"
        finalize = "1" if pkg.is_finalize else "0"
        line = "\t".join([
            pkg.package_id, package_doc, status_file, deps,
            pkg.dependency_type, str(pkg.wave), branch, worktree,
            manual, finalize,
        ])
        lines.append(line)
    return "\n".join(lines) + "\n"


def generate_state_tsv(packages: Sequence[DiscoveryPackage]) -> str:
    lines = [STATE_HEADER]
    for pkg in packages:
        cols = [
            pkg.package_id, "pending", "", "", "", "", "", "", "",
            "pending", "pending", "pending", "", "", "", "", "",
        ]
        lines.append("\t".join(cols))
    return "\n".join(lines) + "\n"


def generate_status_md(pkg: DiscoveryPackage) -> str:
    return f"""# {pkg.package_id} Status

## State

`pending`

## Evidence

- Worktree:
- Branch:
- Base commit:
- Commit hash:
- Changed files:
- Verification:

## Notes

- Risks:
- Blockers:
- Recovery hint:
"""


def generate_index(
    project_path: Path,
    scope: str,
    plan_root: Path,
    packages: Sequence[DiscoveryPackage],
) -> str:
    lines = [
        "# Case Discovery Orchestration Kit",
        "",
        "## Goal",
        "",
        f"Discover replayable historical situations from project `{project_path}` "
        f"using scope: {scope}.",
        "",
        "## User Entry Points",
        "",
        "- Manual: copy prompts from `launchers/agent-prompts.md` into any agent platform.",
        f"- Script: run `bash {plan_root}/launchers/orchestrate.sh start`.",
        "- Status: run `bash orchestrate.sh status`.",
        "",
        "## Dependency Graph",
        "",
        "| Package | Depends On | Dependency Type | Wave |",
        "|---|---|---|---|",
    ]
    for pkg in packages:
        deps = ", ".join(pkg.dependencies) if pkg.dependencies else "none"
        lines.append(f"| {pkg.package_id} | {deps} | {pkg.dependency_type} | {pkg.wave} |")
    lines += [
        "",
        "## Merge Strategy",
        "",
        "Functional merge order follows wave number: 1, 2, 3, 4, then finalize.",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_agent_prompts(
    packages: Sequence[DiscoveryPackage],
    plan_root: Path,
) -> str:
    sections = [
        "# Agent Prompts",
        "",
        "Copy the relevant prompt into an agent, or let `orchestrate.sh start/advance` launch it for Claude Code.",
        "",
    ]
    for pkg in packages:
        label = pkg.package_id
        sections += [
            f"## Package: {label} - {pkg.description}",
            "",
            "---",
            "",
            "**Mode**: package executor",
            f"**INDEX**: `{plan_root}/INDEX.md`",
            f"**Package doc**: `{plan_root}/packages/{pkg.package_id}.md`",
            f"**Coordinator status**: `{plan_root}/status/{pkg.package_id}.md`",
            f"**Coordinator state**: `{plan_root}/status/state.tsv`",
            f"**Scratch path**: run `bash {plan_root}/launchers/orchestrate.sh scratch-path {pkg.package_id}`",
            f"**Orchestrator**: `{plan_root}/launchers/orchestrate.sh`",
            "",
            f"Read INDEX and package doc. {pkg.description}",
            "",
            "**Verification**:",
            "",
            "```bash",
            f"echo '{pkg.package_id} verification'",
            "```",
            "",
            "Update coordinator status with evidence, then:",
            "",
            "```bash",
            f"bash {plan_root}/launchers/orchestrate.sh mark-state {pkg.package_id} completed --commit <commit-sha> --verification \"{pkg.package_id}: passed\"",
            f"bash {plan_root}/launchers/orchestrate.sh advance --from {pkg.package_id}",
            "```",
            "",
            f"For a blocker, use `mark-state {pkg.package_id} blocked --error ... --failed-command ... --recovery-hint ...`.",
            "",
            "---",
            "",
        ]
    return "\n".join(sections) + "\n"


def generate_package_doc(pkg: DiscoveryPackage) -> str:
    """Generate a package document, using the rich template when available."""
    templates = source_template_map()
    template = templates.get(pkg.package_id)
    if template is not None:
        return template.compile_package_doc()

    lines = [
        f"# {pkg.package_id}",
        "",
        "## Goal",
        "",
        pkg.description,
        "",
    ]
    if pkg.dependencies:
        lines += [
            "## Dependencies",
            "",
        ]
        for dep in pkg.dependencies:
            lines.append(f"- `{dep}`")
        lines.append("")

    lines += [
        "## Allowed Paths",
        "",
    ]
    for p in pkg.allowed_paths:
        lines.append(f"- `{p}`")
    lines.append("")

    lines += [
        "## Forbidden Paths",
        "",
    ]
    for p in pkg.forbidden_paths:
        lines.append(f"- `{p}`")
    lines += [
        "",
        "## Verification Commands",
        "",
        "```bash",
    ]
    if pkg.verification_commands:
        for cmd in pkg.verification_commands:
            lines.append(cmd)
    else:
        lines.append(f"echo '{pkg.package_id} ok'")
    lines += [
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_orchestrate_sh(plan_root: Path) -> str:
    return _ORCHESTRATE_TEMPLATE.replace("{{PLAN_ROOT}}", plan_root.as_posix())


_ORCHESTRATE_TEMPLATE = r'''#!/usr/bin/env bash
# orchestrate.sh — generated by replay_checker discovery kit
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_ROOT="{{PLAN_ROOT}}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GRAPH="$PLAN_ROOT/launchers/package-graph.tsv"
STATE="$PLAN_ROOT/status/state.tsv"
EVENTS="$PLAN_ROOT/status/events.jsonl"
SCRATCH="$PLAN_ROOT/scratch"
PROMPTS="$PLAN_ROOT/launchers/agent-prompts.md"
LOCK_DIR="$PLAN_ROOT/status/.orchestrate.lock"
MAX_PARALLEL="${ORCHESTRATION_MAX_PARALLEL:-10}"
STATE_HEADER="package_id	state	launched_at	completed_at	agent	branch	worktree	base_commit	commit_hash	verification	integration	cleanup	last_error	failed_command	conflict_files	log_summary	recovery_hint"
GRAPH_HEADER="package_id	package_doc	status_file	dependencies	dependency_type	wave	branch	worktree	manual	finalize"

log() { printf '[orchestrate] %s\n' "$*" >&2; }
die() { printf '[orchestrate] ERROR: %s\n' "$*" >&2; exit 1; }
timestamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
json_escape() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  value="${value//$'\t'/\\t}"
  value="${value//$'\n'/\\n}"
  value="${value//$'\r'/\\r}"
  printf '%s' "$value"
}
json_pair() {
  local key="$1" value="$2"
  printf ',"%s":"%s"' "$(json_escape "$key")" "$(json_escape "$value")"
}
tsv_safe() {
  local value="$1"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  value="${value//$'\r'/ }"
  printf '%s' "$value" | sed 's/[[:space:]][[:space:]]*/ /g; s/^[[:space:]]*//; s/[[:space:]]*$//'
}
emit_event() {
  local event="$1" package_id="${2:-}" extra="${3:-}"
  mkdir -p "$(dirname "$EVENTS")"
  printf '{"ts":"%s","event":"%s","package_id":"%s"%s}\n' \
    "$(timestamp)" "$(json_escape "$event")" "$(json_escape "$package_id")" "$extra" >> "$EVENTS"
}
ensure_scratch_root() {
  mkdir -p "$SCRATCH"
  if [ ! -f "$SCRATCH/.gitignore" ]; then
    printf '*\n!.gitignore\n' > "$SCRATCH/.gitignore"
  fi
}
scratch_path_for() { printf '%s/%s\n' "$SCRATCH" "$1"; }

valid_state() {
  case "$1" in
    pending|ready|manual_required|launched|in_progress|completed|blocked|stale|invalid|finalizing|finalized) return 0 ;;
    *) return 1 ;;
  esac
}

graph_field() {
  local package_id="$1" column="$2"
  awk -F '\t' -v id="$package_id" -v col="$column" '
    FNR == 1 { for (i = 1; i <= NF; i++) idx[$i] = i; next }
    $1 == id { print $idx[col]; found = 1; exit }
    END { if (!found) exit 1 }
  ' "$GRAPH"
}

state_field() {
  local package_id="$1" column="$2"
  awk -F '\t' -v id="$package_id" -v col="$column" '
    FNR == 1 { for (i = 1; i <= NF; i++) idx[$i] = i; next }
    $1 == id { print $idx[col]; found = 1; exit }
    END { if (!found) exit 1 }
  ' "$STATE"
}

all_package_ids() { awk -F '\t' 'FNR > 1 && NF { print $1 }' "$GRAPH"; }
functional_package_ids() { awk -F '\t' 'FNR > 1 && NF && $10 != "1" { print $1 }' "$GRAPH"; }
finalize_package_id() { awk -F '\t' 'FNR > 1 && NF && $10 == "1" { print $1; exit }' "$GRAPH"; }

status_file_for() {
  local package_id="$1" rel
  rel="$(graph_field "$package_id" status_file)"
  printf '%s/%s\n' "$PLAN_ROOT" "$rel"
}

markdown_status() {
  local package_id="$1" file line value
  file="$(status_file_for "$package_id")"
  [ -f "$file" ] || { printf 'missing\n'; return; }
  line="$(awk '
    /^## State/ { in_state = 1; next }
    in_state && /`/ {
      gsub(/`/, "", $0); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      print tolower($0); exit
    }
    /\*\*Status\*\*:/ {
      sub(/^.*\*\*Status\*\*:[[:space:]]*/, "", $0)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
      print tolower($0); exit
    }
  ' "$file")"
  value="${line:-unknown}"
  case "$value" in
    done|complete) printf 'completed\n' ;;
    *) printf '%s\n' "$value" ;;
  esac
}

sync_markdown_state() {
  local package_id="$1" new_state="$2" file tmp
  file="$(status_file_for "$package_id")"
  [ -f "$file" ] || return 0
  tmp="$(mktemp)"
  awk -v new_state="$new_state" '
    /^## State/ { print; in_state = 1; changed = 0; next }
    in_state && !changed && /`/ {
      print "`" new_state "`"; changed = 1; in_state = 0; next
    }
    /\*\*Status\*\*:/ {
      print "**Status**: " new_state; changed = 1; next
    }
    { print }
    END {
      if (!changed) { print ""; print "## State"; print ""; print "`" new_state "`" }
    }
  ' "$file" > "$tmp"
  mv "$tmp" "$file"
}

preflight_files() {
  [ -f "$PLAN_ROOT/INDEX.md" ] || die "missing INDEX.md"
  [ -f "$GRAPH" ] || die "missing package graph: $GRAPH"
  [ -f "$STATE" ] || die "missing state ledger: $STATE"
  [ -f "$PROMPTS" ] || die "missing agent prompts: $PROMPTS"
}

preflight_graph() {
  preflight_files
  [ "$(head -n 1 "$GRAPH")" = "$GRAPH_HEADER" ] || die "invalid graph header"
  awk -F '\t' '
    FNR == 1 { next } NF == 0 { next }
    NF != 10 { printf("graph row has %d fields, expected 10: %s\n", NF, $0) > "/dev/stderr"; bad = 1 }
    seen[$1]++ { printf("duplicate package id: %s\n", $1) > "/dev/stderr"; bad = 1 }
    $10 == "1" { finalize++ }
    END { if (finalize != 1) { printf("expected exactly one finalize row, got %d\n", finalize + 0) > "/dev/stderr"; bad = 1 }; exit bad ? 1 : 0 }
  ' "$GRAPH" || die "graph validation failed"

  awk -F '\t' '
    FNR == 1 { next } NF == 0 { next }
    { ids[$1] = 1; deps[$1] = $4 }
    END {
      for (id in ids) {
        split(deps[id], arr, ",")
        for (i in arr) {
          dep = arr[i]; gsub(/^[[:space:]]+|[[:space:]]+$/, "", dep)
          if (dep == "") continue
          if (!(dep in ids)) { printf("missing dependency: %s -> %s\n", id, dep) > "/dev/stderr"; bad = 1 }
          if (dep == id) { printf("self dependency: %s\n", id) > "/dev/stderr"; bad = 1 }
        }
      }
      exit bad ? 1 : 0
    }
  ' "$GRAPH" || die "graph dependency validation failed"
}

preflight_state() {
  [ -s "$STATE" ] || die "state ledger is empty"
  [ "$(head -n 1 "$STATE")" = "$STATE_HEADER" ] || die "invalid state header"
  awk -F '\t' '
    FNR == NR {
      if (FNR > 1 && NF) ids[$1] = 1
      next
    }
    FNR == 1 { next } NF == 0 { next }
    {
      if (NF != 17) { printf("state row has %d fields, expected 17: %s\n", NF, $0) > "/dev/stderr"; bad = 1 }
      if (!($1 in ids)) { printf("state has unknown package: %s\n", $1) > "/dev/stderr"; bad = 1 }
      seen[$1] = 1
      if ($2 !~ /^(pending|ready|manual_required|launched|in_progress|completed|blocked|stale|invalid|finalizing|finalized)$/) {
        printf("invalid state for %s: %s\n", $1, $2) > "/dev/stderr"; bad = 1
      }
    }
    END {
      for (id in ids) { if (!(id in seen)) { printf("state missing package: %s\n", id) > "/dev/stderr"; bad = 1 } }
      exit bad ? 1 : 0
    }
  ' "$GRAPH" "$STATE" || die "state validation failed"
}

preflight_all() { preflight_graph; preflight_state; }

status_consistency_ok() {
  local id state md
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    state="$(state_field "$id" state)"
    md="$(markdown_status "$id")"
    case "$md" in
      missing|unknown) printf 'INVALID: %s markdown status is %s\n' "$id" "$md" >&2; return 1 ;;
    esac
    if [ "$state" = "completed" ] && [ "$md" != "completed" ]; then
      printf 'INVALID: %s state completed but markdown is %s\n' "$id" "$md" >&2; return 1
    fi
    if [ "$md" = "completed" ] && [ "$state" != "completed" ] && [ "$state" != "finalized" ]; then
      printf 'INVALID: %s markdown completed but state is %s\n' "$id" "$state" >&2; return 1
    fi
  done < <(all_package_ids)
  return 0
}

set_state_fields() {
  local package_id="$1" new_state="$2"
  local launched_at="${3:-__KEEP__}" completed_at="${4:-__KEEP__}" agent="${5:-__KEEP__}"
  local branch="${6:-__KEEP__}" worktree="${7:-__KEEP__}" base_commit="${8:-__KEEP__}"
  local commit_hash="${9:-__KEEP__}" verification="${10:-__KEEP__}" integration="${11:-__KEEP__}"
  local cleanup="${12:-__KEEP__}" last_error="${13:-__KEEP__}" failed_command="${14:-__KEEP__}"
  local conflict_files="${15:-__KEEP__}" log_summary="${16:-__KEEP__}" recovery_hint="${17:-__KEEP__}"
  local now tmp old_state event_extra

  valid_state "$new_state" || die "invalid state: $new_state"
  old_state="$(state_field "$package_id" state || true)"
  now="$(timestamp)"
  tmp="$(mktemp)"
  awk -F '\t' -v OFS='\t' \
    -v id="$package_id" -v new_state="$new_state" -v now="$now" \
    -v launched="$launched_at" -v completed="$completed_at" -v agent="$agent" \
    -v branch="$branch" -v worktree="$worktree" -v base="$base_commit" \
    -v commit="$commit_hash" -v verification="$verification" -v integration="$integration" \
    -v cleanup="$cleanup" -v last_error="$last_error" -v failed_command="$failed_command" \
    -v conflict_files="$conflict_files" -v log_summary="$log_summary" -v recovery_hint="$recovery_hint" '
      FNR == 1 { for (i = 1; i <= NF; i++) idx[$i] = i; print; next }
      $1 == id {
        $idx["state"] = new_state
        if (launched == "__NOW__" || ((new_state == "launched" || new_state == "in_progress" || new_state == "finalizing") && launched == "__KEEP__")) $idx["launched_at"] = now
        else if (launched != "__KEEP__") $idx["launched_at"] = launched
        if (completed == "__NOW__" || ((new_state == "completed" || new_state == "blocked" || new_state == "stale" || new_state == "invalid" || new_state == "finalized") && completed == "__KEEP__")) $idx["completed_at"] = now
        else if (completed != "__KEEP__") $idx["completed_at"] = completed
        if (agent != "__KEEP__") $idx["agent"] = agent
        if (branch != "__KEEP__") $idx["branch"] = branch
        if (worktree != "__KEEP__") $idx["worktree"] = worktree
        if (base != "__KEEP__") $idx["base_commit"] = base
        if (commit != "__KEEP__") $idx["commit_hash"] = commit
        if (verification != "__KEEP__") $idx["verification"] = verification
        if (integration != "__KEEP__") $idx["integration"] = integration
        if (cleanup != "__KEEP__") $idx["cleanup"] = cleanup
        if (last_error != "__KEEP__") $idx["last_error"] = last_error
        else if (new_state == "completed" || new_state == "finalized") $idx["last_error"] = ""
        if (failed_command != "__KEEP__") $idx["failed_command"] = failed_command
        else if (new_state == "completed" || new_state == "finalized") $idx["failed_command"] = ""
        if (conflict_files != "__KEEP__") $idx["conflict_files"] = conflict_files
        else if (new_state == "completed" || new_state == "finalized") $idx["conflict_files"] = ""
        if (log_summary != "__KEEP__") $idx["log_summary"] = log_summary
        else if (new_state == "completed" || new_state == "finalized") $idx["log_summary"] = ""
        if (recovery_hint != "__KEEP__") $idx["recovery_hint"] = recovery_hint
        else if (new_state == "completed" || new_state == "finalized") $idx["recovery_hint"] = ""
        touched = 1
      }
      { print }
      END { if (!touched) exit 1 }
    ' "$STATE" > "$tmp" || { rm -f "$tmp"; die "unknown package in state ledger: $package_id"; }
  mv "$tmp" "$STATE"
  sync_markdown_state "$package_id" "$new_state"
  emit_event "state_changed" "$package_id" "$(json_pair "old_state" "$old_state")$(json_pair "new_state" "$new_state")"
}

set_error_state() {
  local package_id="$1" new_state="$2" message="$3" failed_command="${4:-}" recovery_hint="${5:-}"
  message="$(tsv_safe "$message")"
  failed_command="$(tsv_safe "$failed_command")"
  recovery_hint="$(tsv_safe "$recovery_hint")"
  set_state_fields "$package_id" "$new_state" "__KEEP__" "__NOW__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "$message" "$failed_command" "" "" "$recovery_hint"
}

deps_completed() {
  local package_id="$1" deps dep state
  deps="$(graph_field "$package_id" dependencies || true)"
  [ -z "$deps" ] && return 0
  IFS=',' read -r -a dep_array <<< "$deps"
  for dep in "${dep_array[@]}"; do
    dep="${dep#"${dep%%[![:space:]]*}"}"
    dep="${dep%"${dep##*[![:space:]]}"}"
    [ -z "$dep" ] && continue
    state="$(state_field "$dep" state)"
    [ "$state" = "completed" ] || [ "$state" = "finalized" ] || return 1
  done
  return 0
}

any_bad_terminal_state() {
  awk -F '\t' 'FNR > 1 && ($2 == "blocked" || $2 == "stale" || $2 == "invalid") { print $1 ":" $2 }' "$STATE"
}

all_functional_completed() {
  local id
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    [ "$(state_field "$id" state)" = "completed" ] || return 1
  done < <(functional_package_ids)
  return 0
}

ready_packages() {
  local id state manual
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    state="$(state_field "$id" state)"
    manual="$(graph_field "$id" manual)"
    if { [ "$state" = "pending" ] || [ "$state" = "ready" ]; } && [ "$manual" != "1" ] && deps_completed "$id"; then
      printf '%s\n' "$id"
    fi
  done < <(functional_package_ids)
}

prompt_for_package() {
  local package_id="$1"
  awk -v id="$package_id" '
    $0 ~ "^## Package: " id "([[:space:]-]|$)" { found = 1; print; next }
    found && /^## Package: / { exit }
    found { print }
  ' "$PROMPTS"
}

print_agents_command() {
  printf '\nView background sessions with:\n  claude agents --cwd "%s"\n' "$REPO_ROOT"
}

cmd_status() {
  preflight_all
  printf '%-36s %-12s %-54s %-22s %-14s %-30s %-30s %s\n' "PACKAGE" "STATE" "BRANCH" "VERIFICATION" "INTEGRATION" "LAST_ERROR" "FAILED_COMMAND" "RECOVERY_HINT"
  awk -F '\t' 'FNR > 1 {
    printf "%-36s %-12s %-54s %-22s %-14s %-30s %-30s %s\n", $1, $2, $6, $10, $11, $13, $14, $17
  }' "$STATE"
  if status_consistency_ok; then
    printf '\nCoordinator consistency: ok\n'
  else
    printf '\nCoordinator consistency: invalid\n'
    exit 1
  fi
}

cmd_mark_state() {
  local package_id="${1:-}" new_state="${2:-}"
  local base="__KEEP__" commit="__KEEP__" verification="__KEEP__" integration="__KEEP__"
  local cleanup="__KEEP__" error="__KEEP__" failed_command="__KEEP__"
  local conflict_files="__KEEP__" log_summary="__KEEP__" recovery_hint="__KEEP__"
  [ -n "$package_id" ] || die "usage: mark-state <package-id> <state>"
  [ -n "$new_state" ] || die "usage: mark-state <package-id> <state>"
  shift 2 || true
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --base) shift; base="${1:-}" ;;
      --commit) shift; commit="${1:-}" ;;
      --verification) shift; verification="${1:-}" ;;
      --integration) shift; integration="${1:-}" ;;
      --cleanup) shift; cleanup="${1:-}" ;;
      --error) shift; error="${1:-}" ;;
      --failed-command) shift; failed_command="${1:-}" ;;
      --conflict-files) shift; conflict_files="${1:-}" ;;
      --log-summary) shift; log_summary="${1:-}" ;;
      --recovery-hint) shift; recovery_hint="${1:-}" ;;
      *) die "unknown mark-state option: $1" ;;
    esac
    shift || true
  done
  preflight_all
  [ "$verification" = "__KEEP__" ] || verification="$(tsv_safe "$verification")"
  [ "$integration" = "__KEEP__" ] || integration="$(tsv_safe "$integration")"
  [ "$cleanup" = "__KEEP__" ] || cleanup="$(tsv_safe "$cleanup")"
  [ "$error" = "__KEEP__" ] || error="$(tsv_safe "$error")"
  [ "$failed_command" = "__KEEP__" ] || failed_command="$(tsv_safe "$failed_command")"
  [ "$conflict_files" = "__KEEP__" ] || conflict_files="$(tsv_safe "$conflict_files")"
  [ "$log_summary" = "__KEEP__" ] || log_summary="$(tsv_safe "$log_summary")"
  [ "$recovery_hint" = "__KEEP__" ] || recovery_hint="$(tsv_safe "$recovery_hint")"
  set_state_fields "$package_id" "$new_state" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "__KEEP__" "$base" "$commit" "$verification" "$integration" "$cleanup" "$error" "$failed_command" "$conflict_files" "$log_summary" "$recovery_hint"
  log "marked $package_id as $new_state"
}

cmd_scratch_path() {
  local package_id="${1:-}"
  [ -n "$package_id" ] || die "usage: scratch-path <package-id>"
  preflight_graph
  graph_field "$package_id" package_doc >/dev/null || die "unknown package: $package_id"
  ensure_scratch_root
  local path
  path="$(scratch_path_for "$package_id")"
  mkdir -p "$path"
  emit_event "scratch_path_requested" "$package_id" "$(json_pair "path" "$path")"
  printf '%s\n' "$path"
}

verify_scratch_files() {
  local package_id="$1"
  local scratch_dir file_count
  scratch_dir="$(scratch_path_for "$package_id")"
  if [ ! -d "$scratch_dir" ]; then
    printf '%s scratch directory missing: %s\n' "$package_id" "$scratch_dir" >&2
    return 1
  fi
  file_count="$(find "$scratch_dir" -type f ! -name '.gitignore' 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${file_count:-0}" -eq 0 ]; then
    printf '%s scratch directory is empty: %s\n' "$package_id" "$scratch_dir" >&2
    return 1
  fi
  return 0
}

cmd_verify_package() {
  local package_id="${1:-}" id bad=0
  [ -n "$package_id" ] || die "usage: verify-package <package-id>"
  preflight_all
  verify_scratch_files "$package_id" || bad=1
  [ "$bad" -eq 0 ] && printf 'verify-package %s: ok\n' "$package_id"
  return "$bad"
}

cmd_verify_finalize() {
  local id bad=0
  preflight_all
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    if ! verify_scratch_files "$id"; then
      bad=1
    fi
  done < <(functional_package_ids)
  [ "$bad" -eq 0 ] || die "verify-finalize: some packages have missing or empty scratch directories"
  printf 'verify-finalize: ok\n'
}

usage() {
  cat <<USAGE
Usage: bash launchers/orchestrate.sh <command>

Commands:
  status
  mark-state <package-id> <state> [options]
  scratch-path <package-id>
  verify-package <package-id>
  verify-finalize
USAGE
}

case "${1:-}" in
  status)   cmd_status ;;
  mark-state) shift; cmd_mark_state "$@" ;;
  scratch-path) shift; cmd_scratch_path "${1:-}" ;;
  verify-package) shift; cmd_verify_package "${1:-}" ;;
  verify-finalize) cmd_verify_finalize ;;
  *)        usage; exit 1 ;;
esac
'''


def generate_kit_files(
    config: DiscoveryKitConfig,
    packages: Sequence[DiscoveryPackage],
) -> dict[str, str]:
    plan_root = config.plan_root
    project_root = config.project_path.resolve()

    files: dict[str, str] = {
        "INDEX.md": generate_index(project_path=project_root, scope=config.scope, plan_root=plan_root, packages=packages),
        "launchers/package-graph.tsv": generate_graph_tsv(packages, plan_root, project_root),
        "launchers/orchestrate.sh": generate_orchestrate_sh(plan_root),
        "launchers/agent-prompts.md": generate_agent_prompts(packages, plan_root),
        "status/state.tsv": generate_state_tsv(packages),
        "status/events.jsonl": "",
        "scratch/.gitignore": "*\n!.gitignore\n",
    }

    for pkg in packages:
        files[f"packages/{pkg.package_id}.md"] = generate_package_doc(pkg)
        files[f"status/{pkg.package_id}.md"] = generate_status_md(pkg)

    return files
