#!/bin/bash
# =============================================================================
# install_env.sh: create the project venv on Snellius
#
# Creates .venv in the repository root and installs requirements.txt, gurobipy
# and the Julia packages for SDDiP (juliapkg.json plus Gurobi.jl). The SLURM job
# scripts in scripts/slurm/ activate this venv; install_deps.sh adds packages to
# an existing one.
#
# Submit from the repository root:
#   sbatch scripts/slurm/install_env.sh
# =============================================================================
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=install_env
#SBATCH --output=logs/install_%j.out
#SBATCH --error=logs/install_%j.err

set -euo pipefail

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Gurobi/10.0.1-GCCcore-12.3.0

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

python3 -m venv .venv
source .venv/bin/activate

pip3 install --upgrade pip
pip3 install -r requirements.txt

# gurobipy matching the version of the loaded Gurobi module, for the Pyomo MIPs
# and the stage LPs (Layers A and C in solver_config.py)
pip3 install gurobipy==10.0.1

# Importing juliacall installs and precompiles the Julia packages listed in
# juliapkg.json (SDDP.jl, JuMP, HiGHS, JSON) when they are missing.
python3 -c "from juliacall import Main as jl; print('Julia ready')"

# Install Gurobi.jl into the Julia environment so that SDDiP (Layer B) can also
# run on Gurobi. It is left out of juliapkg.json so that HiGHS-only setups need
# no Gurobi install. Gurobi.jl builds against GUROBI_HOME, which the Gurobi
# module loaded above sets.
python3 -c "import juliapkg; juliapkg.add('Gurobi', '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()"
# Precompile Gurobi.jl here so that the first SDDiP run on Gurobi starts fast.
python3 -c "from juliacall import Main as jl; jl.seval('using Gurobi'); print('Gurobi.jl ready')"

echo "Installation complete! (HiGHS + Gurobi, all layers)"
