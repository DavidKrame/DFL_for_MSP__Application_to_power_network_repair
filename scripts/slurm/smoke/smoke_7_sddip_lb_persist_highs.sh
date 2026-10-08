#!/usr/bin/env bash
# =============================================================================
# smoke_7_sddip_lb_persist_highs.sh: SDDiP pre-simulation write on toy with HiGHS
#
# Purpose : check that the SDDiP bounds reach disk before the policy simulation.
#           --hard-time-limit bounds training only; the Monte Carlo simulation
#           of the trained policy (--n-simulations replications) has no time
#           limit, so a job can reach its SLURM time limit during it. After
#           training, SDDiPSolver.solve therefore writes sddip_solution.json with
#           the lower and upper bounds, the gap and the convergence history from
#           the training log. When the simulation finishes, SDDiP_main.py
#           overwrites the file with the full solution, which adds the repair and
#           dispatch statistics. Exercises:
#             - the pre-simulation write (log line "Pre-simulation solution persisted")
#             - the post-simulation overwrite of sddip_solution.json
#             - `python3 -u` unbuffered output, so run.log fills as the run goes
#
# Runs on toy (small and fast); the pre-simulation write does not depend on the
# network, so this covers case14 as well. HiGHS counterpart of smoke_8 (Gurobi):
# same test, separate output dir, for comparison.
#
# The first Julia call precompiles the packages (~3–4 min); once they are
# compiled into the Julia depot, later runs skip this step.
# Expected runtime: ~8–20 min on a rome node (Julia startup dominates).
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_7_sddip_lb_persist_highs.sh
# =============================================================================
#SBATCH --job-name=smoke7_sddip_lb_highs
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke7_sddip_lb_highs_%j.out
#SBATCH --error=logs/smoke7_sddip_lb_highs_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dm %02ds" $(( s/60 )) $(( s%60 )); }

SEED=42
ROOT="experiments/smoke_snellius/sddip_lb_highs"
rm -rf "$ROOT"; mkdir -p "$ROOT"
RUNLOG="$ROOT/run.log"

echo "============================================================"
echo "SMOKE 7 — SDDiP pre-simulation save (toy, HiGHS)"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── SDDiP toy (10 iters; Julia startup may take ~3–4 min) ────"
T0=$(date +%s)
$PYTHON -u SDDiP_main.py \
    --network toy \
    --solver highs \
    --iterations 10 \
    --n-simulations 5 \
    --hard-time-limit 600 \
    --seed "$SEED" \
    --output-dir "$ROOT" \
    --no-plots 2>&1 | tee "$RUNLOG"
echo "[$(ts)] solve returned  $(elapsed $(( $(date +%s)-T0 )))"

# ── Assertions ───────────────────────────────────────────────────────────────
# SDDiPSolver.solve logs "Pre-simulation solution persisted (LB=..., UB=...,
# gap=...)" once the bounds are written, before the policy simulation starts.
if ! grep -q "Pre-simulation solution persisted" "$RUNLOG"; then
    echo "  FAIL: the pre-simulation write of sddip_solution.json did not run"
    exit 1
fi
if [ ! -f "$ROOT/sddip_solution.json" ]; then
    echo "  FAIL: sddip_solution.json was not written"
    exit 1
fi
$PYTHON - "$ROOT/sddip_solution.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
lb, ub = d.get("lower_bound"), d.get("upper_bound")
assert isinstance(lb, (int, float)) and lb == lb, f"lower_bound missing/NaN: {lb!r}"
ub_str = f"${ub:,.2f}" if isinstance(ub, (int, float)) else str(ub)
print(f"  PASS: LB persisted pre-sim; final sddip_solution.json "
      f"lower_bound=${lb:,.2f}, upper_bound={ub_str} (overwritten post-sim).")
PY

echo ""
echo "============================================================"
echo "SMOKE 7 complete — SDDiP pre-simulation save OK (HiGHS)."
echo "Results in $ROOT"
echo "============================================================"
