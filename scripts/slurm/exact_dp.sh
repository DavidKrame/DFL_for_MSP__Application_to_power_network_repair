#!/usr/bin/env bash
# =============================================================================
# exact_dp.sh: exact DP on the toy (5-bus) network
#
# Solves both variants (unbudgeted, and budgeted with phi = 0.3) at repair-cost
# scale s = 100, seed 42 and T = 14, and compares each with the saved toy SDDiP
# solution in experiments/paper_unbudgeted/toy/SDDiP/ or
# experiments/paper_budget0.3/toy/SDDiP/ when that file exists.
#
# The only solver calls are the stage DC-OPF LPs (128 per variant), which run
# on scipy's HiGHS (--lp-solver highs). Unlike the scripts in
# scripts/slurm/unbudgeted/ and scripts/slurm/budgeted/, this job has no SOLVER
# switch and loads no Gurobi module.
#
# Output: experiments/exact_dp_toy/run_<jobid>/{unbudgeted,budgeted}/exact_dp_solution.json
# paper/reproduce_paper_numbers.py reads
# experiments/exact_dp_toy/{unbudgeted,budgeted}/exact_dp_solution.json instead.
#
# Submit from the repository root (needs the .venv of scripts/slurm/install_env.sh):
#   sbatch scripts/slurm/exact_dp.sh [extra exact_dp_toy.py arguments]
# Extra arguments are appended to the command below, so a repeated option
# overrides the value set here (e.g. --variant unbudgeted).
# =============================================================================
#SBATCH --job-name=exact_dp_toy
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --output=logs/exact_dp_toy_%j.out
#SBATCH --error=logs/exact_dp_toy_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/.venv"
SCRIPT_DIR="$PROJECT_DIR/exact_dp"
OUTPUT_DIR="$PROJECT_DIR/experiments/exact_dp_toy/run_$SLURM_JOB_ID"

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source "$VENV_DIR/bin/activate"
# Sets the Layer C (stage LP) default of solver_config.py. The DP's stage LPs
# follow --lp-solver instead, which exact_dp_toy.py passes to StagewiseOPF.
export ED_OPF_LP_SOLVER=highs

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Exact DP validation — toy network"
echo "Node       : $(hostname)"
echo "Job        : $SLURM_JOB_ID"
echo "Python     : $(python3 --version)"
echo "Project    : $PROJECT_DIR"
echo "Output     : $OUTPUT_DIR"
echo "Started    : $(date)"
echo "============================================================"

python3 -u "$SCRIPT_DIR/exact_dp_toy.py" \
    --variant both \
    --seed 42 \
    --horizon 14 \
    --repair-base-scale 100 \
    --budget-fraction 0.3 \
    --lp-solver highs \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "============================================================"
echo "Completed  : $(date)"
echo "Results    : $OUTPUT_DIR"
echo "============================================================"
