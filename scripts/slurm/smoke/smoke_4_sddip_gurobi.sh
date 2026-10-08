#!/usr/bin/env bash
# =============================================================================
# smoke_4_sddip_gurobi.sh: small SDDiP run on toy with Gurobi (Layer B)
#
# Purpose : the smoke_3 run with Gurobi as the SDDP.jl optimizer (through
#           Gurobi.jl). Together with smoke_2 (Layers A and C) it runs every
#           layer of solver_config.py on Gurobi.
#
# Requires Gurobi.jl in the Julia environment. scripts/slurm/install_env.sh
# (fresh setup) and scripts/slurm/install_deps.sh (existing venv) both add it.
# It is not in juliapkg.json, so HiGHS-only Julia environments never build it.
# If it is missing, the pre-flight check below exits with the command to add it
# (the bridge raises an error with the same command). Run that command with the
# Gurobi module loaded, since Gurobi.jl builds against GUROBI_HOME:
#   python -c "import juliapkg; juliapkg.add('Gurobi', \
#       '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()"
#
# Expected runtime: ~10–25 min (the first Gurobi.jl precompile can be slow).
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_4_sddip_gurobi.sh
# =============================================================================
#SBATCH --job-name=smoke4_sddip_grb
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoke4_sddip_grb_%j.out
#SBATCH --error=logs/smoke4_sddip_grb_%j.err

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
ROOT="experiments/smoke_snellius/sddip_gurobi"
mkdir -p "$ROOT"

echo "============================================================"
echo "SMOKE 4 — tiny SDDiP (Gurobi) on toy   [OPTIONAL]"
echo "Node   : $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "[$(ts)] ── Pre-flight: is Gurobi.jl available in the Julia env? ─────"
if ! $PYTHON -c "from juliacall import Main as jl; jl.seval('using Gurobi')" 2>/dev/null; then
    echo ""
    echo "  Gurobi.jl is NOT installed in the Julia environment. Add it once:"
    echo "    python -c \"import juliapkg; juliapkg.add('Gurobi', '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()\""
    echo "  then resubmit this script. (Use smoke_3 for the HiGHS SDDiP path.)"
    exit 1
fi
echo "[$(ts)] Gurobi.jl is available."

echo ""
echo "[$(ts)] ── SDDiP toy on Gurobi (5 iters) ────────────────────────────"
T0=$(date +%s)
$PYTHON SDDiP_main.py \
    --network toy \
    --solver gurobi \
    --iterations 5 \
    --hard-time-limit 600 \
    --seed "$SEED" \
    --output-dir "$ROOT" \
    --no-plots
echo "[$(ts)] DONE  SDDiP  $(elapsed $(( $(date +%s)-T0 )))"

echo ""
echo "============================================================"
echo "SMOKE 4 complete — SDDiP / SDDP.jl (Gurobi) OK."
echo "Results in $ROOT"
echo "============================================================"
