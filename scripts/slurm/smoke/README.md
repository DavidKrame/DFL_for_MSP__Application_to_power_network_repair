# Smoke tests for the cluster setup

These nine short SLURM jobs check a new installation on a compute node before any long experiment job is submitted. Each job runs a small problem through one code path with one solver. The outputs are disposable and go to `experiments/smoke_snellius/`, which is gitignored. The jobs show that the code runs, not that its results are meaningful.

The jobs follow the conventions of the experiment scripts described in [`scripts/README.md`](../../README.md#common-conventions): partition `rome` on Snellius, the modules `2023` and `Python/3.11.3-GCCcore-12.3.0`, the environment `.venv` in the submission directory, and log files in `logs/`.

**Contents**

1. [Solver layers](#solver-layers)
2. [What each job checks](#what-each-job-checks)
3. [Resources, logs and outputs](#resources-logs-and-outputs)
4. [Before submitting](#before-submitting)
5. [Submitting](#submitting)
6. [Reading the results](#reading-the-results)
7. [Adding Gurobi.jl](#adding-gurobijl)
8. [Switching solvers in the experiment runs](#switching-solvers-in-the-experiment-runs)
9. [Cleaning up](#cleaning-up)

## Solver layers

[`solver_config.py`](../../../solver_config.py) selects a solver for three places in the code, and the jobs test each of them on HiGHS and on Gurobi:

| Layer | Where a solver is used | HiGHS | Gurobi |
|---|---|---|---|
| A | Pyomo MIPs: DFL training labels and SAA | `appsi_highs` | `gurobi` |
| B | SDDiP subproblems in SDDP.jl, called from Python through juliacall | HiGHS.jl | Gurobi.jl |
| C | stage DC-OPF LPs in every policy rollout | SciPy `linprog(method="highs")` | gurobipy |

With no environment variable set, all three layers use HiGHS. `ED_OPF_SOLVER=gurobi` moves all three to Gurobi.

## What each job checks

`toy` is the synthetic 5-bus network of the paper.

| Job | Solver | Network | Checks |
|---|---|---|---|
| `smoke_0_env.sh` | probes both | toy, case14 | environment, solver selection, Gurobi license, case14 load |
| `smoke_1_highs_pipeline.sh` | HiGHS | toy | SAA (wait-and-see) and the DFL pipeline, layers A and C |
| `smoke_2_gurobi_pipeline.sh` | Gurobi | toy | the same on Gurobi |
| `smoke_3_sddip_highs.sh` | HiGHS | toy | SDDiP through SDDP.jl, layer B |
| `smoke_4_sddip_gurobi.sh` | Gurobi | toy | the same with Gurobi.jl |
| `smoke_5_saa_singlesolve_highs.sh` | HiGHS | case14 | the SAA solve settings of `saa.sh`, and the exit status of a failed run |
| `smoke_6_saa_singlesolve_gurobi.sh` | Gurobi | case14 | the same on Gurobi |
| `smoke_7_sddip_lb_persist_highs.sh` | HiGHS | toy | SDDiP bounds written to disk before the policy simulation |
| `smoke_8_sddip_lb_persist_gurobi.sh` | Gurobi | toy | the same with Gurobi.jl |

### smoke_0: environment

The job runs these steps in order and stops at the first failure:

1. `verify_installation.py`: core imports, the toy network, model parameters and scenarios, an extensive-form MIP that is built but not solved, and a probe of the Pyomo MIP solvers and of gurobipy. Its summary reports `Total: 7/7 tests passed`.
2. The solver selection, printed once with the defaults and once with `ED_OPF_SOLVER=gurobi`:

   ```text
   solver_config: MIP=appsi_highs  LP=highs  SDDiP=highs  (master=<unset>)
   solver_config: MIP=gurobi  LP=gurobi  SDDiP=gurobi  (master='gurobi')
   ```

3. A Gurobi license check: gurobipy creates a model and prints its version followed by `+ license OK`.
4. The versions of NumPy, SciPy, pandas and pandapower, then case14 loaded from PGLib-OPF (`case14 PGLib load OK`). Incompatible pandapower, pandas or NumPy versions make this load fail. For this reason, `requirements.txt` pins pandapower 3.2.1 and keeps pandas below 3.

The job always loads the Gurobi module, so it needs Gurobi. It does not start Julia, which the SDDiP jobs cover.

### smoke_1 and smoke_2: SAA and DFL pipeline

Both run the SAA and DFL entry points end to end on toy. smoke_1 runs:

```bash
python3 main.py --network toy --mode ws --n-scenarios 5 --seed 42 --time-limit 60 \
    --output-dir experiments/smoke_snellius/highs/SAA --evaluate --eval-seed 1000 --n-test 20 --no-plots
python3 Pseudo_DFL_main.py --network toy --n-train 60 --n-eval 20 --eval-seed 1000 --seed 42 \
    --cache-path experiments/smoke_snellius/highs/cache/training_data.pkl \
    --saa-dir experiments/smoke_snellius/highs/SAA --output-dir experiments/smoke_snellius/highs/DFL
```

- The first command solves the wait-and-see SAA, one MIP over 5 independent scenarios (layer A), and rolls its repair schedule out on 20 test scenarios (layer C). The scenario-tree SAA of `saa.sh`, with 16,384 sample paths, is far too large for a smoke test.
- The second command runs the whole DFL pipeline: 60 label MIPs solved in parallel worker processes (layer A), 6 training epochs, the evaluation of DFL, AlwaysRepair and NoRepair on the same 20 test scenarios (layer C), the SAA costs read from the first command, and the plots.
- `--cache-path` gives the job its own label cache, so it neither reads nor writes the shared `data_cache*/` folders. On the first run, the solver under test solves the labels. A rerun reuses that cache. Delete `experiments/smoke_snellius/<solver>/cache/` to have the labels solved again.
- smoke_1 sets no solver, so the code defaults (HiGHS) apply. smoke_2 contains the `SOLVER` block of the experiment scripts (default `gurobi`) and writes to `experiments/smoke_snellius/$SOLVER/`. For example, `sbatch --export=ALL,SOLVER=gurobi_direct scripts/slurm/smoke/smoke_2_gurobi_pipeline.sh` checks Pyomo's in-memory Gurobi interface and writes to `experiments/smoke_snellius/gurobi_direct/`.

### smoke_3 and smoke_4: SDDiP

smoke_3 runs:

```bash
python3 SDDiP_main.py --network toy --solver highs --iterations 5 --hard-time-limit 600 --seed 42 \
    --output-dir experiments/smoke_snellius/sddip_highs --no-plots
```

- This command checks the Julia side on its own. It starts Julia through juliacall, loads SDDP.jl, JuMP, HiGHS.jl and JSON.jl, builds the model, trains for 5 iterations (at most 600 s), and then runs the 500 default policy simulations and writes the result files.
- The first Julia call on a new installation precompiles the packages, which takes about 3 to 4 minutes. Later runs reuse the compiled packages in the Julia depot.
- toy has 7 failure-prone components, so SDDiP enumerates all $`2^7 = 128`$ failure outcomes per stage.
- smoke_4 runs the same command with `--solver gurobi` into `experiments/smoke_snellius/sddip_gurobi/`. It first checks that Gurobi.jl loads. If it does not, the job prints the command from [Adding Gurobi.jl](#adding-gurobijl) and exits with status 1.

### smoke_5 and smoke_6: SAA single solve

Both check the solve settings of `saa.sh` on case14, in wait-and-see mode so that the MIP solves in seconds. smoke_5 runs:

```bash
python3 main.py --network case14 --mode ws --n-scenarios 5 --seed 42 --time-limit 300 --mip-gap 0.10 \
    --checkpoint-interval 0 --output-dir experiments/smoke_snellius/saa_ss_highs/SAA \
    --evaluate --eval-seed 1000 --n-test 20 --no-plots
```

- Part A passes when `solution.json` is written and the run log `run.log` has no line `Checkpoint interval:`. `main.py` prints that line only for a chunked solve. Chunking is its default (`--checkpoint-interval 600`) and restarts the solver at every interval, repeating presolve each time. With `--checkpoint-interval 0`, as in `saa.sh`, the MIP is solved in one uninterrupted run.
- Part B runs `main.py --network NOPE_invalid_network ...` and passes when it exits with a non-zero status. `saa.sh` ends with the exit status of `main.py`, so a failed SAA run shows up as a failed task.
- smoke_5 sets no solver and so runs on HiGHS. smoke_6 loads the Gurobi module and sets `ED_OPF_SOLVER=gurobi`, so the MIP and the rollout LPs run on Gurobi. It writes to `experiments/smoke_snellius/saa_ss_gurobi/`.
- Both delete their output folder at the start.

### smoke_7 and smoke_8: SDDiP bounds before the simulation

smoke_7 runs:

```bash
python3 -u SDDiP_main.py --network toy --solver highs --iterations 10 --n-simulations 5 \
    --hard-time-limit 600 --seed 42 --output-dir experiments/smoke_snellius/sddip_lb_highs --no-plots
```

`--hard-time-limit` bounds training only. The policy simulation that follows has no limit, so a long SDDiP job can reach its SLURM time limit during the simulation. For that case, `SDDiP_main.py` writes `sddip_solution.json` right after training, with the lower and upper bounds, the gap and the convergence history. When the simulation finishes, it overwrites the file with the full solution, which adds the repair and dispatch statistics.

- The job passes when the log contains `Pre-simulation solution persisted`, which is printed after the early write, and the final `sddip_solution.json` holds a numeric `lower_bound`.
- `python3 -u` writes the output unbuffered, so `run.log` fills while the job runs.
- The early write does not depend on the network, so toy covers case14 as well.
- smoke_8 runs the same command with `--solver gurobi` into `experiments/smoke_snellius/sddip_lb_gurobi/`, after the Gurobi.jl check of smoke_4.
- Both start by deleting their output folder.

## Resources, logs and outputs

All jobs request partition `rome`, 1 node and 1 task. The expected runtimes are the estimates in the script headers. The first SDDiP job on a new installation also precompiles the Julia packages.

| Job | CPUs | Memory | Time limit | Expected runtime | Log files (`.out` and `.err`) | Output folder under `experiments/smoke_snellius/` |
|---|---|---|---|---|---|---|
| smoke_0 | 8 | 16G | 00:15:00 | 2–5 min | `logs/smoke0_env_<jobid>` | none |
| smoke_1 | 16 | 32G | 00:30:00 | 5–15 min | `logs/smoke1_highs_<jobid>` | `highs/` |
| smoke_2 | 16 | 32G | 00:30:00 | 5–15 min | `logs/smoke2_gurobi_<jobid>` | `<SOLVER>/`, by default `gurobi/` |
| smoke_3 | 16 | 32G | 00:40:00 | 10–20 min | `logs/smoke3_sddip_<jobid>` | `sddip_highs/` |
| smoke_4 | 16 | 32G | 00:40:00 | 10–25 min | `logs/smoke4_sddip_grb_<jobid>` | `sddip_gurobi/` |
| smoke_5 | 16 | 32G | 00:30:00 | 2–8 min | `logs/smoke5_saa_ss_highs_<jobid>` | `saa_ss_highs/` |
| smoke_6 | 16 | 32G | 00:30:00 | 2–8 min | `logs/smoke6_saa_ss_gurobi_<jobid>` | `saa_ss_gurobi/` |
| smoke_7 | 16 | 32G | 00:40:00 | 8–20 min | `logs/smoke7_sddip_lb_highs_<jobid>` | `sddip_lb_highs/` |
| smoke_8 | 16 | 32G | 00:45:00 | 8–25 min | `logs/smoke8_sddip_lb_gurobi_<jobid>` | `sddip_lb_gurobi/` |

## Before submitting

1. Install the environment from the repository root, with `sbatch scripts/slurm/install_env.sh` for a new setup or `bash scripts/slurm/install_deps.sh` on a login node to complete an existing `.venv`. Both install HiGHS and Gurobi for all three layers: the packages of `requirements.txt`, gurobipy 10.0.1, Julia with SDDP.jl, JuMP, HiGHS.jl and JSON.jl, and Gurobi.jl. See [Environment setup](../../README.md#environment-setup).
2. Submit from the repository root, where `.venv/` and `logs/` are. SLURM opens the log files before the job starts, so `logs/` must exist. The repository ships it with a `.gitkeep` file.
3. smoke_0, smoke_5 and smoke_6 load case14, which is downloaded from PGLib-OPF into `~/.cache/pglib_opf/` on first use. If the compute nodes have no internet access, fetch the file on a login node first. This command pins it to PGLib-OPF v23.07, the version used for the stored results:

   ```bash
   mkdir -p ~/.cache/pglib_opf
   curl -L -o ~/.cache/pglib_opf/pglib_opf_case14_ieee.m \
       https://raw.githubusercontent.com/power-grid-lib/pglib-opf/v23.07/pglib_opf_case14_ieee.m
   ```

4. The Gurobi jobs (smoke_0, 2, 4, 6 and 8) load the module `Gurobi/10.0.1-GCCcore-12.3.0` and need a Gurobi license. They use `~/gurobi.lic` when that file exists and the cluster's default license otherwise.

## Submitting

From the repository root:

```bash
bash scripts/slurm/smoke/submit_all.sh                  # all nine; they queue and run in parallel
sbatch scripts/slurm/smoke/smoke_0_env.sh               # or one job at a time
squeue -u $USER
```

`submit_all.sh` calls `sbatch` for each job itself, so start it with `bash`, not `sbatch`. The jobs use the submission directory as the project directory.

Start with smoke_0 and fix any problem it reports before looking at the other jobs. Then go through smoke_1 and smoke_2 (the pipeline on each solver), smoke_3 and smoke_4 (SDDiP on each solver), and smoke_5 to smoke_8 (the SAA and SDDiP settings of the experiment scripts). On an installation without Gurobi, run only smoke_1, smoke_3, smoke_5 and smoke_7, because smoke_0 also needs Gurobi.

## Reading the results

Each job writes `logs/<name>_<jobid>.out` (stdout) and `logs/<name>_<jobid>.err` (stderr), with the names in the table above.

```bash
tail -n 4 logs/smoke*_*.out                                        # finished jobs end with "SMOKE <n> complete"
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed     # FAILED or TIMEOUT for a failed job
```

- A job that passes ends with a banner that starts with `SMOKE <n> complete`. The scripts use `set -euo pipefail`, so a failing command stops the job. The `.out` file then ends at that step, the Python traceback is in the `.err` file, and `sacct` shows the job as `FAILED` with a non-zero exit code. smoke_5 to smoke_8 also print a `PASS:` or `FAIL:` line for each of their checks.
- smoke_1, smoke_2, smoke_5 and smoke_6 show which solver ran by printing the resolved selection in their header, for example `solver_config: MIP=appsi_highs  LP=highs  SDDiP=highs  (master=<unset>)` on HiGHS and `solver_config: MIP=gurobi  LP=gurobi  SDDiP=gurobi  (master='gurobi')` on Gurobi. The SDDiP jobs print `Optimizer:` followed by `highs` or `gurobi` in the SDDiP header.
- When case14 is loaded (smoke_0, smoke_5, smoke_6), pandapower prints `There are 3 branches which are considered as trafos ...`. This message is expected.
- smoke_1 and smoke_2 write `SAA/solution.json`, the DFL plots, and `DFL/evaluation.json` with the entries `Pseudo_DFL`, `AlwaysRepair`, `NoRepair` and `SAA`. The SDDiP jobs write `sddip_solution.json`, `cuts.json` and the other SDDiP output files. smoke_5 to smoke_8 also keep their console output in `run.log`.

The HiGHS and Gurobi jobs of a pair write separate folders, so the two solvers can be compared afterwards. With the environment active, run from the repository root:

```bash
python - <<'EOF'
import json
base = "experiments/smoke_snellius"
for s in ("highs", "gurobi"):
    m = json.load(open(f"{base}/saa_ss_{s}/SAA/solution.json"))
    md = m["solver_metadata"]
    print(f"SAA   {s:6s} objective {m['objective_value']:,.0f}  gap {md['mip_gap_achieved']}  solve {md['solve_time']:.1f} s")
for s in ("highs", "gurobi"):
    d = json.load(open(f"{base}/sddip_lb_{s}/sddip_solution.json"))
    print(f"SDDiP {s:6s} lower bound {d['lower_bound']:,.0f}  iterations {d['iterations']}  time {d['solve_time']:.0f} s")
EOF
```

## Adding Gurobi.jl

Gurobi.jl is not listed in `juliapkg.json`, so an installation that uses only HiGHS never needs Gurobi. `install_env.sh` and `install_deps.sh` add it. If it is missing, smoke_4 and smoke_8 stop and print the command below, and `SDDiP_main.py --solver gurobi` raises an error whose message contains it. To add it, run these commands once on a login node, from the repository root:

```bash
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Gurobi/10.0.1-GCCcore-12.3.0          # sets GUROBI_HOME
source .venv/bin/activate
python -c "import juliapkg; juliapkg.add('Gurobi', '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()"
python -c "from juliacall import Main as jl; jl.seval('using Gurobi')"     # precompile once
```

Then resubmit `smoke_4_sddip_gurobi.sh` and `smoke_8_sddip_lb_persist_gurobi.sh`.

- Run the commands from the repository root. juliapkg finds the repository's `juliapkg.json` through the working directory. A resolve started elsewhere rebuilds the Julia project without SDDP.jl, JuMP, HiGHS.jl and JSON.jl, and the next SDDiP run then has to resolve again.
- `juliapkg.add` records Gurobi.jl in the Julia project of the active environment (`.venv/julia_env`), not in the repository's `juliapkg.json`.
- By default, Gurobi.jl downloads its own Gurobi library. If the license does not cover that version, build Gurobi.jl against the module's installation instead, using `GUROBI_HOME` together with `GUROBI_JL_USE_GUROBI_JLL=false` as described in the [Gurobi.jl README](https://github.com/jump-dev/Gurobi.jl).
- The reported SDDiP runs used HiGHS and do not need Gurobi.jl.

## Switching solvers in the experiment runs

The experiment scripts in `scripts/slurm/unbudgeted/` and `scripts/slurm/budgeted/` select the solver of the whole job at submission. Their variable `SOLVER` defaults to `gurobi` and is exported as `ED_OPF_SOLVER`. A value that contains `gurobi` also loads the Gurobi module. `sensitivity_sweep.sh` defaults to `appsi_highs`.

```bash
sbatch scripts/slurm/unbudgeted/saa.sh                                           # default: Gurobi in every layer
sbatch --export=ALL,SOLVER=appsi_highs scripts/slurm/unbudgeted/dfl_scaling.sh   # HiGHS in every layer
sbatch --export=ALL,SOLVER=gurobi,ED_OPF_LP_SOLVER=highs scripts/slurm/unbudgeted/saa.sh   # one layer changed
```

- Pass `SOLVER` through `--export=ALL,SOLVER=...`. On Snellius, the prefix form `SOLVER=appsi_highs sbatch ...` did not reach the job, which then ran with the default `gurobi`.
- `ED_OPF_MIP_SOLVER`, `ED_OPF_LP_SOLVER` and `ED_OPF_SDDIP_SOLVER` override layers A, C and B one at a time. `--export=ALL` forwards them from the login shell as well, so unset them unless they are intended.
- Before a long run on a solver, run its smoke jobs: smoke_1, 3, 5 and 7 for HiGHS, smoke_2, 4, 6 and 8 for Gurobi.

The reported runs used HiGHS for the DFL jobs (`dfl_scaling.sh`, `dfl_multiseed.sh`) and the SDDiP jobs, all submitted with `--export=ALL,SOLVER=appsi_highs`. The SAA jobs used the default, Gurobi. The section [Solver selection](../../README.md#solver-selection) of `scripts/README.md` lists the settings of every job.

## Cleaning up

- `experiments/smoke_snellius/` is gitignored and can be deleted at any time.
- smoke_5 to smoke_8 delete their own output folder when they start. smoke_1 and smoke_2 keep their label cache between runs, and smoke_3 and smoke_4 overwrite their files.
- The SDDiP jobs append the SDDP.jl training log to `SDDP.log` in the repository root, which is also gitignored.
