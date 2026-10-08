#!/usr/bin/env bash
# =============================================================================
# smoke_5_saa_singlesolve_highs.sh: SAA single solve on case14 with HiGHS
#
# Purpose : check on HiGHS the SAA solve settings that
#           scripts/slurm/{unbudgeted,budgeted}/saa.sh use. With
#           --checkpoint-interval 0, main.py passes no checkpoint callback and
#           the MIP is solved in one uninterrupted run; the solver returns its
#           incumbent at the time limit. A chunked solve (the main.py default,
#           --checkpoint-interval 600) restarts the solver every 600 s and
#           repeats presolve each time, which on a large MIP can leave no time
#           for branch and bound. Exercises:
#             - --checkpoint-interval 0  => single solve, no chunked re-presolve
#             - --mip-gap 0.10           => the gap used in saa.sh
#             - exit-code propagation    => a failed run exits non-zero, so a
#               calling wrapper's rc check reports the failure
#
# Runs on case14 in Wait-and-See mode (--mode ws) with 5 scenarios, so the MIP
# solves in seconds (saa.sh solves the scenario tree, --mode tree). HiGHS
# counterpart of smoke_6 (Gurobi): same test, separate output dir, so the two
# solvers can be compared afterward.
#
# Expected runtime: ~2–8 min on a rome node.
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_5_saa_singlesolve_highs.sh
# =============================================================================
#SBATCH --job-name=smoke5_saa_ss_highs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke5_saa_ss_highs_%j.out
#SBATCH --error=logs/smoke5_saa_ss_highs_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# No solver is set, so the solver_config.py default (appsi_highs) applies.
source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dm %02ds" $(( s/60 )) $(( s%60 )); }

SEED=42
EVAL_SEED=1000
N_EVAL=20
ROOT="experiments/smoke_snellius/saa_ss_highs"
rm -rf "$ROOT"; mkdir -p "$ROOT"
RUNLOG="$ROOT/run.log"

echo "============================================================"
echo "SMOKE 5 — SAA single uninterrupted solve (case14, HiGHS)"
$PYTHON -c "import solver_config as s; print(' ', s.describe())"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

# ── Part A: single uninterrupted solve produces a solution ───────────────────
echo ""
echo "[$(ts)] ── A. SAA case14, --checkpoint-interval 0 (single solve) ─────"
T0=$(date +%s)
$PYTHON main.py \
    --network case14 \
    --mode ws \
    --n-scenarios 5 \
    --seed "$SEED" \
    --time-limit 300 \
    --mip-gap 0.10 \
    --checkpoint-interval 0 \
    --output-dir "$ROOT/SAA" \
    --evaluate --eval-seed "$EVAL_SEED" --n-test "$N_EVAL" \
    --no-plots 2>&1 | tee "$RUNLOG"
echo "[$(ts)] solve returned  $(elapsed $(( $(date +%s)-T0 )))"

if [ ! -f "$ROOT/SAA/solution.json" ]; then
    echo "  FAIL: solution.json was not written (single solve produced no incumbent)"
    exit 1
fi
# EDOPFProblem.solve prints "Checkpoint interval:" only when main.py passes a
# checkpoint callback, which for HiGHS means a chunked solve.
if grep -q "Checkpoint interval:" "$RUNLOG"; then
    echo "  FAIL: chunked checkpointing was used (expected a single uninterrupted solve)"
    exit 1
fi
echo "  PASS: solution.json written; no chunk restarts (single solve)."

# ── Part B: a real failure exits non-zero (exit-code propagation) ────────────
# The network name contains no "case<N>", so create_network raises ValueError
# and main.py exits with status 1. (A name such as case15 would resolve to the
# closest PGLib case and not fail.)
echo ""
echo "[$(ts)] ── B. exit-code propagation (deliberate failure) ────────────"
set +e
$PYTHON main.py --network NOPE_invalid_network --mode ws --n-scenarios 2 \
    --time-limit 30 --checkpoint-interval 0 --no-plots \
    --output-dir "$ROOT/should_fail" > "$ROOT/fail.log" 2>&1
rc=$?
set -e
if [ $rc -eq 0 ]; then
    echo "  FAIL: an invalid run exited 0 — a calling wrapper would report a false 'DONE'"
    exit 1
fi
echo "  PASS: invalid run exited non-zero (rc=$rc) — a calling wrapper's rc check will report FAILED."

echo ""
echo "============================================================"
echo "SMOKE 5 complete — SAA single solve OK (HiGHS)."
echo "Results in $ROOT"
echo "============================================================"
