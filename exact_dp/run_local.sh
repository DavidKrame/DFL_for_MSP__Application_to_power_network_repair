#!/usr/bin/env bash
# Exact DP on the toy (5-bus) network, local run. Solves both variants
# (unbudgeted, and budgeted with phi = 0.3) at repair-cost scale s = 100,
# seed 42 and T = 14, with scipy's HiGHS for the stage LPs. Results go to
# exact_dp/results/local/{unbudgeted,budgeted}/exact_dp_solution.json.
#
# Usage (from the repository root):
#   bash exact_dp/run_local.sh [extra exact_dp_toy.py arguments]
# Extra arguments are appended to the command below, so a repeated option
# overrides the value set here (e.g. --variant unbudgeted).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=python3
OUTPUT_DIR="$SCRIPT_DIR/results/local"

cd "$PROJECT_DIR"

echo "Running exact toy DP locally"
echo "  Python : $($PYTHON --version)"
echo "  Output : $OUTPUT_DIR"

"$PYTHON" -u "$SCRIPT_DIR/exact_dp_toy.py" \
    --variant both \
    --seed 42 \
    --horizon 14 \
    --repair-base-scale 100 \
    --budget-fraction 0.3 \
    --lp-solver highs \
    --output-dir "$OUTPUT_DIR" \
    "$@"
