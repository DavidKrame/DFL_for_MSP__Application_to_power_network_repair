#!/usr/bin/env bash
# =============================================================================
# install_deps.sh: add missing packages to the existing project venv
#
# Installs the requirements.txt packages that are missing from .venv (created by
# install_env.sh), plus gurobipy and Gurobi.jl, then checks the main imports.
# Run it interactively on Snellius, not through sbatch: it finds the repository
# from its own path, and under sbatch that path is SLURM's copy of the script.
#
# Usage:
#   bash scripts/slurm/install_deps.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Gurobi/10.0.1-GCCcore-12.3.0   # sets GUROBI_HOME for gurobipy and Gurobi.jl

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python : $(python3 --version)"
echo "Venv   : $VENV_DIR"
echo "Reqs   : $REQUIREMENTS"
echo ""
echo "Installing missing packages (pip will skip already-installed ones)..."
echo ""

pip install --upgrade pip
pip install -r "$REQUIREMENTS"

# --- Gurobi backend (all layers) ----------------------------------------------
# gurobipy (Layers A and C in solver_config.py) is pinned to the Snellius
# Gurobi/10.0.1 module so that the license check is version-compatible.
# Gurobi.jl (Layer B, SDDiP) is not in juliapkg.json, so this script installs it
# into the Julia environment; it builds against GUROBI_HOME from the module loaded
# above. Solving needs a valid Gurobi license (site or academic) at run time;
# the experiment scripts point GRB_LICENSE_FILE at ~/gurobi.lic when that file
# exists and otherwise use the cluster's default Gurobi license.
echo ""
echo "Installing gurobipy (Layers A + C)..."
pip install gurobipy==10.0.1

echo ""
echo "Adding Gurobi.jl to the Julia environment (Layer B / SDDiP)..."
python3 -c "import juliapkg; juliapkg.add('Gurobi', '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()"
python3 -c "from juliacall import Main as jl; jl.seval('using Gurobi'); print('  Gurobi.jl ready')"

echo ""
echo "Done. Verifying key imports..."
python3 -c "import torch; print(f'  torch     {torch.__version__}')"
python3 -c "import pyomo; print(f'  pyomo     {pyomo.__version__}')"
python3 -c "import highspy; print(f'  highspy   ok')"
python3 -c "import numpy as np; print(f'  numpy     {np.__version__}')"
python3 -c "import gurobipy as gp; print('  gurobipy ', '.'.join(map(str, gp.gurobi.version())))"
echo "All OK (HiGHS + Gurobi, all layers)."
