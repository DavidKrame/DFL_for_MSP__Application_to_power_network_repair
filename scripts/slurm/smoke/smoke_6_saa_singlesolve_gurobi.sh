#!/usr/bin/env bash
# =============================================================================
# smoke_6_saa_singlesolve_gurobi.sh: SAA single solve on case14 with Gurobi
#
# Purpose : the smoke_5 test on Gurobi, the default solver of the experiment
#           scripts in scripts/slurm/unbudgeted/ and scripts/slurm/budgeted/.
#           One uninterrupted solve (--checkpoint-interval 0) on case14; Gurobi
#           returns its incumbent at the time limit. Exercises the Gurobi MIP
#           path (Layer A), the rollout LPs through gurobipy (Layer C) and:
#             - --checkpoint-interval 0  => single solve, no chunked re-presolve
#             - --mip-gap 0.10           => the gap used in saa.sh
#             - exit-code propagation    => a failed run exits non-zero
#
# Gurobi counterpart of smoke_5 (HiGHS): same test, separate output dir, so the
# two solvers can be compared afterward (e.g. objective, solve time).
#
# Expected runtime: ~2–8 min on a rome node.
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_6_saa_singlesolve_gurobi.sh
# =============================================================================
#SBATCH --job-name=smoke6_saa_ss_gurobi
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke6_saa_ss_gurobi_%j.out
#SBATCH --error=logs/smoke6_saa_ss_gurobi_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Gurobi/10.0.1-GCCcore-12.3.0
# Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
# otherwise the cluster's default Gurobi license applies.
[[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"

source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
export ED_OPF_SOLVER=gurobi          # SAA MIP (Layer A) and rollout LPs (Layer C) on Gurobi
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dm %02ds" $(( s/60 )) $(( s%60 )); }

SEED=42
EVAL_SEED=1000
N_EVAL=20
ROOT="experiments/smoke_snellius/saa_ss_gurobi"
rm -rf "$ROOT"; mkdir -p "$ROOT"
RUNLOG="$ROOT/run.log"

echo "============================================================"
echo "SMOKE 6 — SAA single uninterrupted solve (case14, Gurobi)"
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
# checkpoint callback, which for Gurobi means a chunked solve.
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
echo "SMOKE 6 complete — SAA single solve OK (Gurobi)."
echo "Results in $ROOT"
echo "============================================================"
