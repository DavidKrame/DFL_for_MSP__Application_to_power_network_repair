#!/usr/bin/env bash
# =============================================================================
# submit_sensitivity.sh: submit the sensitivity sweep array and the plot job
# that runs after it, on Snellius.
#
# Usage (from the repository root on Snellius):
#   bash scripts/slurm/sensitivity/submit_sensitivity.sh
#
# What it does:
#   1. Submits sensitivity_sweep.sh (array 0-11, 12 scales).
#   2. Submits sensitivity_plot.sh with --dependency=afterany, so the plot runs
#      once all 12 array tasks have finished, whether they succeeded or failed.
# =============================================================================

set -euo pipefail

SCRIPTS="scripts/slurm/sensitivity"

SWEEP_JOB=$(sbatch --parsable "$SCRIPTS/sensitivity_sweep.sh")
echo "Submitted sweep:  job ${SWEEP_JOB}  (array 0-11, 12 scales)"

PLOT_JOB=$(sbatch --parsable \
    --dependency=afterany:"${SWEEP_JOB}" \
    "$SCRIPTS/sensitivity_plot.sh")
echo "Submitted plot:   job ${PLOT_JOB}  (runs after ${SWEEP_JOB} finishes)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Results at:    experiments/sensitivity/sensitivity_curve.png  (once plot job completes)"
