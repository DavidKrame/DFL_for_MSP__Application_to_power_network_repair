#!/usr/bin/env bash
# Quick end-to-end check of a checkout, meant to run on a laptop in about a minute:
#   1. installation check (imports, network data, solvers)
#   2. unit tests
#   3. a short DFL run on the toy (5-bus) network: 10 training scenarios, 10 epochs
#   4. a small SAA solve (wait-and-see form, 4 scenarios)
#   5. the exact DP of the toy network
#   6. the exact-DP optimum, compared with the stored one in experiments/
#   7. the reported numbers, recomputed from experiments/ (needs internet once)
#   8. every figure, redrawn into a temporary folder
#   9. the redrawn figures, compared with figures/
#  10. (only with --with-sddip) a 5-iteration SDDiP run on the toy network; the first
#      call installs Julia and SDDP.jl through juliacall (internet, several minutes)
#
# Outputs go to a temporary folder. Outside it, the script writes only the PGLib case14
# file to ~/.cache/pglib_opf/; matplotlib and PyTorch may add caches of their own.
#
# Usage, from anywhere, with the Python environment active:
#   bash scripts/quick_check.sh [--with-sddip]
# Otherwise name the interpreter: PYTHON=.venv/bin/python bash scripts/quick_check.sh

set -uo pipefail

WITH_SDDIP=0
for arg in "$@"; do
    case "$arg" in
        --with-sddip) WITH_SDDIP=1 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
# Every step runs from the work dir, so turn a relative interpreter path into an
# absolute one first.
if [[ "$PYTHON" == */* && "$PYTHON" != /* ]]; then
    PYTHON="$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")"
fi
WORK="$(mktemp -d "${TMPDIR:-/tmp}/quick_check.XXXXXX")"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1

echo "Repository : $REPO"
echo "Python     : $("$PYTHON" --version 2>&1) ($(command -v "$PYTHON"))"
echo "Work dir   : $WORK"

N_PASS=0
N_FAIL=0
FAILED_LIST=""

# run <label> <command...>: runs the command from the work dir, logs its output,
# prints ok/FAILED with the elapsed time and the log tail on failure.
run() {
    local label="$1"; shift
    local log="$WORK/$(echo "$label" | tr -c 'A-Za-z0-9\n' '_').log"
    local t0; t0=$(date +%s)
    printf '\n[%s] %s ... ' "$(date '+%H:%M:%S')" "$label"
    if (cd "$WORK" && "$@") > "$log" 2>&1; then
        printf 'ok (%ds)\n' $(( $(date +%s) - t0 ))
        N_PASS=$(( N_PASS + 1 ))
    else
        printf 'FAILED (%ds)\n' $(( $(date +%s) - t0 ))
        tail -n 25 "$log" | sed 's/^/    /'
        echo "    full log: $log"
        N_FAIL=$(( N_FAIL + 1 ))
        FAILED_LIST="$FAILED_LIST  failed: $label\n"
    fi
}

run "installation check" "$PYTHON" "$REPO/verify_installation.py"

run "unit tests" "$PYTHON" -m pytest -q -p no:cacheprovider --rootdir "$REPO" "$REPO/tests"

run "DFL quick run (toy)" "$PYTHON" "$REPO/Pseudo_DFL_main.py" --test \
    --output-dir "$WORK/dfl_test"

run "SAA wait-and-see (toy, 4 scenarios)" "$PYTHON" "$REPO/main.py" --mode ws \
    --n-scenarios 4 --evaluate --n-test 4 --no-plots --output-dir "$WORK/saa_ws"

run "exact DP (toy)" "$PYTHON" "$REPO/exact_dp/exact_dp_toy.py" --variant both \
    --seed 42 --horizon 14 --repair-base-scale 100 --budget-fraction 0.3 \
    --lp-solver highs --output-dir "$WORK/exact_dp"

run "exact DP matches the stored optimum" "$PYTHON" - "$REPO" "$WORK/exact_dp" <<'EOF'
import json
import sys
from pathlib import Path

repo, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
for variant in ("unbudgeted", "budgeted"):
    stored = json.loads((repo / "experiments/exact_dp_toy" / variant
                         / "exact_dp_solution.json").read_text())
    rerun = json.loads((new_dir / variant / "exact_dp_solution.json").read_text())
    a = stored["solution"]["exact_optimal_value"]
    b = rerun["solution"]["exact_optimal_value"]
    print(f"{variant:10s} stored {a:,.4f}  rerun {b:,.4f}")
    if abs(a - b) > 1e-6 * abs(a):
        sys.exit(f"{variant}: exact optimum differs from the stored value")
EOF

run "reported numbers" "$PYTHON" "$REPO/paper/reproduce_paper_numbers.py"

run "figures" "$PYTHON" "$REPO/paper/make_paper_figures.py" --out "$WORK/figures"

run "figures match figures/" "$PYTHON" - "$REPO/figures" "$WORK/figures" <<'EOF'
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

stored_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
stored = sorted(p.relative_to(stored_dir) for p in stored_dir.rglob("*.png"))
missing = [p for p in stored if not (new_dir / p).exists()]
if missing:
    sys.exit(f"not redrawn: {missing}")
n_same = 0
for rel in stored:
    a, b = mpimg.imread(stored_dir / rel), mpimg.imread(new_dir / rel)
    if a.shape != b.shape:
        print(f"{rel}: canvas {b.shape[1]}x{b.shape[0]} px instead of {a.shape[1]}x{a.shape[0]} px")
        continue
    changed = float(np.any(a != b, axis=-1).mean())
    n_same += changed == 0
    print(f"{rel}: " + ("identical" if changed == 0 else f"{100 * changed:.2f}% of pixels differ"))
print(f"{n_same} of {len(stored)} figures are pixel-identical. Other matplotlib builds render text "
      "slightly differently, which shifts glyphs and the canvas size by a few pixels; the plotted "
      "data are the same. Compare the folders by eye if in doubt.")
EOF

if [[ "$WITH_SDDIP" == 1 ]]; then
    run "SDDiP quick run (toy, 5 iterations)" "$PYTHON" "$REPO/SDDiP_main.py" --test \
        --no-plots --output-dir "$WORK/sddip_test"
fi

echo
echo "------------------------------------------------------------"
echo "Passed: $N_PASS   Failed: $N_FAIL"
printf "%b" "$FAILED_LIST"
echo "Logs and outputs: $WORK"
[[ "$N_FAIL" -eq 0 ]]
