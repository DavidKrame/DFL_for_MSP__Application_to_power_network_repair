#!/usr/bin/env bash
# =============================================================================
# submit_all.sh: submit the Snellius smoke suite (smoke_0 to smoke_8)
#
# The jobs are independent SLURM jobs, so they queue and run in parallel; each
# has an SBATCH time limit between 15 and 45 min. smoke_4 and smoke_8 run SDDiP
# on Gurobi and need Gurobi.jl, which scripts/slurm/install_env.sh and
# scripts/slurm/install_deps.sh install; if it is missing, they exit with the
# command to add it.
#
# smoke_0 checks the environment. 1–4 run each layer of solver_config.py on
# HiGHS and on Gurobi (A: Pyomo MIPs, B: SDDiP through SDDP.jl, C: stage DC-OPF
# LPs). 5–8 check the SAA single-solve settings and the SDDiP pre-simulation
# write, one job per solver so the two can be compared afterward (each writes
# its own directory under experiments/smoke_snellius/):
#   5 = SAA single-solve (HiGHS)   6 = SAA single-solve (Gurobi)
#   7 = SDDiP persist-LB (HiGHS)   8 = SDDiP persist-LB (Gurobi)
#
# Run with bash from the repository root; the script calls sbatch itself, and
# the jobs use the submission directory as the project directory:
#   bash scripts/slurm/smoke/submit_all.sh
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

for job in smoke_0_env smoke_1_highs_pipeline smoke_2_gurobi_pipeline \
           smoke_3_sddip_highs smoke_4_sddip_gurobi \
           smoke_5_saa_singlesolve_highs smoke_6_saa_singlesolve_gurobi \
           smoke_7_sddip_lb_persist_highs smoke_8_sddip_lb_persist_gurobi; do
    echo "Submitting $job ..."
    sbatch "$HERE/$job.sh"
done

echo ""
echo "Submitted smoke 0–8. Monitor with:  squeue -u \$USER"
