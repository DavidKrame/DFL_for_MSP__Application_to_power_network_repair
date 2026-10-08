#!/usr/bin/env bash
# =============================================================================
# smoke_8_sddip_lb_persist_gurobi.sh: SDDiP pre-simulation write on toy with Gurobi
#
# Purpose : the smoke_7 test with Gurobi as the SDDP.jl optimizer (Gurobi.jl,
#           Layer B). Checks that training runs under Gurobi, that
#           sddip_solution.json is written with the bounds from the training log
#           before the policy simulation, and that the final file holds a
#           numeric lower bound. Runs on toy; the pre-simulation write does not
#           depend on the network.
#
# Gurobi counterpart of smoke_7 (HiGHS): same test, separate output dir, for
# comparison. Needs Gurobi.jl in the Julia environment (added by
# scripts/slurm/install_env.sh and scripts/slurm/install_deps.sh); if it is
# missing, the script exits with the command to add it (smoke_7 covers the HiGHS
# SDDiP path).
#
# The first Julia call precompiles the packages (~3–4 min); once they are
# compiled into the Julia depot, later runs skip this step.
# Expected runtime: ~8–25 min on a rome node (Julia startup dominates).
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_8_sddip_lb_persist_gurobi.sh
# =============================================================================
#SBATCH --job-name=smoke8_sddip_lb_gurobi
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke8_sddip_lb_gurobi_%j.out
#SBATCH --error=logs/smoke8_sddip_lb_gurobi_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Gurobi/10.0.1-GCCcore-12.3.0   # sets GUROBI_HOME and the license for Gurobi.jl
# Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
# otherwise the cluster's default Gurobi license applies.
[[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"

source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
export ED_OPF_SOLVER=gurobi
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ts()      { date '+%H:%M:%S'; }
elapsed() { local s=$1; printf "%02dm %02ds" $(( s/60 )) $(( s%60 )); }

SEED=42
ROOT="experiments/smoke_snellius/sddip_lb_gurobi"
rm -rf "$ROOT"; mkdir -p "$ROOT"
RUNLOG="$ROOT/run.log"

echo "============================================================"
echo "SMOKE 8 — SDDiP pre-simulation save (toy, Gurobi)   [OPTIONAL]"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── Pre-flight: is Gurobi.jl available in the Julia env? ─────"
if ! $PYTHON -c "from juliacall import Main as jl; jl.seval('using Gurobi')" 2>/dev/null; then
    echo ""
    echo "  Gurobi.jl is NOT installed in the Julia environment. Add it once:"
    echo "    python -c \"import juliapkg; juliapkg.add('Gurobi', '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()\""
    echo "  then resubmit this script. (Use smoke_7 for the HiGHS SDDiP path.)"
    exit 1
fi
echo "[$(ts)] Gurobi.jl is available."

echo ""
echo "[$(ts)] ── SDDiP toy on Gurobi (10 iters) ───────────────────────────"
T0=$(date +%s)
$PYTHON -u SDDiP_main.py \
    --network toy \
    --solver gurobi \
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
echo "SMOKE 8 complete — SDDiP pre-simulation save OK (Gurobi)."
echo "Results in $ROOT"
echo "============================================================"
