#!/usr/bin/env bash
# =============================================================================
# dfl_aggregate.sh: aggregate the multi-seed DFL results, no repair budget
#
# Runs aggregate_multiseed.py on the dfl_multiseed.sh output in
# experiments/paper_unbudgeted/dfl_scaling_multiseed. For every
# <folder>/dfl_train<N>/ with at least one seed_<s>/evaluation.json, it writes
# <folder>/dfl_train<N>/evaluation.json. Its DFL entry holds the mean and std of
# the per-seed DFL mean costs; the baseline entries are copied from the lowest
# init seed's file, since they do not depend on the init seed. With --plot it then
# draws the DFL mean cost (with a std band) against n_train, once per folder
# (<folder>/convergence_multiseed.png) and once with all folders side by side
# (convergence_multiseed.png directly in dfl_scaling_multiseed/).
#
# Submit after the 15-task dfl_multiseed.sh array has finished (from the
# repository root):
#   JOB=$(sbatch --parsable scripts/slurm/unbudgeted/dfl_multiseed.sh)
#   sbatch --dependency=afterok:$JOB scripts/slurm/unbudgeted/dfl_aggregate.sh
# =============================================================================
#SBATCH --job-name=paper_dfl_scaling_agg
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/unb_paper_dfl_scaling_agg_%j.out
#SBATCH --error=logs/unb_paper_dfl_scaling_agg_%j.err

set -uo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
source "$VENV_DIR/bin/activate"

# Solver selection, as in the other experiment scripts. SOLVER defaults to gurobi
# and is exported as ED_OPF_SOLVER for solver_config.py; a name that contains
# "gurobi" also loads the Gurobi module. aggregate_multiseed.py solves no
# optimization problems, so the choice has no effect on this job. Override at
# submission, e.g.
#   sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/dfl_aggregate.sh
SOLVER="${SOLVER:-gurobi}"
if [[ "$SOLVER" == *gurobi* ]]; then
    module load Gurobi/10.0.1-GCCcore-12.3.0
    # Use a personal Gurobi license (e.g. WLS) at ~/gurobi.lic when present;
    # otherwise the cluster's default Gurobi license applies.
    [[ -f "$HOME/gurobi.lic" ]] && export GRB_LICENSE_FILE="$HOME/gurobi.lic"
fi
export ED_OPF_SOLVER="$SOLVER"
export MPLBACKEND=Agg
PYTHON=python3
cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"

ROOT="experiments/paper_unbudgeted/dfl_scaling_multiseed"

echo "============================================================"
echo "DFL scaling aggregation"
echo "  Root    : ${ROOT}"
echo "Started   : $(date)"
echo "============================================================"
echo ""

$PYTHON aggregate_multiseed.py --root "$ROOT" --plot

echo ""
echo "============================================================"
echo "Aggregation complete. Figures written to ${ROOT}/."
echo "============================================================"
