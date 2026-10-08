#!/usr/bin/env bash
# =============================================================================
# sensitivity_plot.sh: run scripts/sensitivity_plot.py once the sweep is done.
# submit_sensitivity.sh submits it with a dependency on the sweep array. Submit
# it by hand (from the repository root) only after all sweep tasks have ended.
# =============================================================================
#SBATCH --job-name=dfl_sensitivity_plot
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dfl_sensitivity_plot_%j.out
#SBATCH --error=logs/dfl_sensitivity_plot_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection. SOLVER defaults to gurobi; when it contains "gurobi" the
# Gurobi module is loaded. ED_OPF_SOLVER is exported for the Python code.
# Override at submission with
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/sensitivity/sensitivity_plot.sh
# sensitivity_plot.py only reads evaluation.json files, so the solver does not
# affect its output.
SOLVER="${SOLVER:-gurobi}"
if [[ "$SOLVER" == *gurobi* ]]; then
    module load Gurobi/10.0.1-GCCcore-12.3.0
    # Use ~/gurobi.lic (e.g. a Gurobi WLS license) when present instead of the
    # cluster's default Gurobi license
    [[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi
export ED_OPF_SOLVER="$SOLVER"
export MPLBACKEND=Agg

cd "$PROJECT_DIR"

python3 scripts/sensitivity_plot.py \
    --base "$PROJECT_DIR/experiments/sensitivity" \
    --out  "$PROJECT_DIR/experiments/sensitivity/sensitivity_curve.png"

echo "Figure and table saved to $PROJECT_DIR/experiments/sensitivity/"
