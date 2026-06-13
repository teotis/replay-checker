#!/usr/bin/env bash
#
# run_concurrent.sh - Concurrent case processing launcher
#
# Usage:
#   ./tools/run_concurrent.sh                    # generate manifest + dry-run
#   ./tools/run_concurrent.sh run                # generate + execute
#   ./tools/run_concurrent.sh run --project open_camera  # filter by project
#   ./tools/run_concurrent.sh status             # show manifest status
#   ./tools/run_concurrent.sh run --dry-run      # generate + preview only
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST="$ROOT/work/concurrent_task_manifest.json"
CONCURRENCY="${MAX_CONCURRENCY:-8}"

cd "$ROOT"

usage() {
    echo "Usage: $0 [run|status|generate] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  run         Generate manifest and execute tasks (default)"
    echo "  status      Show manifest progress"
    echo "  generate    Generate manifest only"
    echo ""
    echo "Options:"
    echo "  --project NAME        Filter to a specific project"
    echo "  --dry-run             Preview without executing"
    echo "  --max-concurrency N   Override concurrency limit (default: $CONCURRENCY)"
    echo ""
    echo "Environment:"
    echo "  MAX_CONCURRENCY=$CONCURRENCY (override via env)"
    exit 1
}

CMD="run"
PROJECT_FILTER=""
DRY_RUN=""
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        run|status|generate) CMD="$arg" ;;
        --project) shift_next=1 ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --max-concurrency) shift_next_cc=1 ;;
        --help|-h) usage ;;
        *)
            if [ "${shift_next:-}" = "1" ]; then
                PROJECT_FILTER="$arg"
                shift_next=0
            elif [ "${shift_next_cc:-}" = "1" ]; then
                CONCURRENCY="$arg"
                shift_next_cc=0
            else
                EXTRA_ARGS="$EXTRA_ARGS $arg"
            fi
            ;;
    esac
done

echo "=== Replay Checker: Concurrent Task Dispatcher ==="
echo "Command:     $CMD"
echo "Concurrency: $CONCURRENCY"
[ -n "$PROJECT_FILTER" ] && echo "Project:     $PROJECT_FILTER"
echo ""

# Step 1: Generate manifest
echo "[1/2] Generating task manifest..."
GEN_ARGS="generate-manifest --max-concurrency $CONCURRENCY"
[ -n "$PROJECT_FILTER" ] && GEN_ARGS="$GEN_ARGS --project $PROJECT_FILTER"

python3 tools/concurrent_tasks.py $GEN_ARGS

if [ "$CMD" = "generate" ]; then
    echo "Manifest written to: $MANIFEST"
    exit 0
fi

# Step 2: Run or status
echo ""
echo "[2/2] Executing..."
RUN_ARGS="run --manifest $MANIFEST $DRY_RUN --max-concurrency $CONCURRENCY"

python3 tools/concurrent_tasks.py $RUN_ARGS

echo ""
echo "=== Done ==="
