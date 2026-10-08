#!/usr/bin/env bash
# =============================================================================
# smoke_0_env.sh: environment and solver check (Snellius)
#
# Purpose : check on a compute node that the venv, the core Python modules and
#           the Python-side solver backends work before any long job is
#           submitted. Run it first after installing or updating dependencies.
#
# Checks  : verify_installation.py (core imports, toy model build without a
#           solve, Pyomo MIP solver probe, gurobipy), the solver_config.py
#           selection by default and with ED_OPF_SOLVER=gurobi, a Gurobi license
#           check through gurobipy, the numpy/scipy/pandas/pandapower versions,
#           and loading case14 from PGLib-OPF. The Gurobi module is loaded so the
#           Gurobi backends can be probed. The Julia side (SDDP.jl) is not checked
#           here; smoke_3 and smoke_4 cover it.
#
# Expected runtime: ~2–5 min.
#
# Submit with:
#   sbatch scripts/slurm/smoke/smoke_0_env.sh
# =============================================================================
#SBATCH --job-name=smoke0_env
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --output=logs/smoke0_env_%j.out
#SBATCH --error=logs/smoke0_env_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
# The checks below probe the Gurobi backends, so the module is always loaded here.
module load Gurobi/10.0.1-GCCcore-12.3.0
# Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
# otherwise the cluster's default Gurobi license applies.
[[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"

source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

echo "============================================================"
echo "SMOKE 0 — environment + solver availability"
echo "Node   : $(hostname)"
echo "CPUs   : ${SLURM_CPUS_PER_TASK:-?}"
echo "Python : $($PYTHON --version)"
echo "Started: $(date)"
echo "============================================================"

echo ""
echo "── Full installation check (imports + Pyomo solver probe + gurobipy) ──"
$PYTHON verify_installation.py

echo ""
echo "── Central solver selection (default vs. ED_OPF_SOLVER=gurobi) ──"
$PYTHON -c "import solver_config as s; print(' ', s.describe())"
ED_OPF_SOLVER=gurobi $PYTHON -c "import solver_config as s; print(' ', s.describe())"

echo ""
echo "── Gurobi license handshake (gurobipy) ──"
$PYTHON -c "import gurobipy as gp; m = gp.Model(); m.setParam('OutputFlag', 0); print('  gurobipy', '.'.join(map(str, gp.gurobi.version())), '+ license OK')"

echo ""
echo "── Numeric stack versions + PGLib network load (case14) ──"
# case14 is loaded through pandapower's from_mpc, which raises "output array is
# read-only" under incompatible pandapower/pandas/numpy versions (see the pins in
# requirements.txt).
$PYTHON -c "import numpy,scipy,pandas,pandapower as pp; print(f'  numpy {numpy.__version__} | scipy {scipy.__version__} | pandas {pandas.__version__} | pandapower {pp.__version__}')"
$PYTHON -c "from network_data import create_network; create_network('case14'); print('  case14 PGLib load OK')"

echo ""
echo "============================================================"
echo "SMOKE 0 complete — environment looks good."
echo "============================================================"
