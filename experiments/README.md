# Stored results

This folder holds the result files behind the numbers and figures reported in the paper. Besides this README, it contains 1,084 files (69 MB). The cluster jobs in [`scripts/slurm/`](../scripts/README.md) wrote them on the Snellius cluster, except the two exact dynamic-programming files, which were computed on macOS. Their `run_environment` key records the platform. Checking the paper needs no new computation, because the two scripts in [`paper/`](../paper/README.md) read these files directly:

```bash
python paper/reproduce_paper_numbers.py                     # 112 checks in a few seconds
python paper/make_paper_figures.py --out /tmp/paper_figures
```

Run the commands from the repository root, where the full paths in this file also start, with the Python environment active. Installation is covered in the [top-level README](../README.md#installation), the cluster jobs in [scripts/README.md](../scripts/README.md), and the two paper scripts in [paper/README.md](../paper/README.md).

**Contents**

1. [Names](#names)
2. [Layout](#layout)
3. [What the files contain](#what-the-files-contain)
4. [From files to reported numbers](#from-files-to-reported-numbers)
5. [From files to figures](#from-files-to-figures)
6. [Loading the files in Python](#loading-the-files-in-python)
7. [Gaps in the stored results](#gaps-in-the-stored-results)
8. [Not included, and how to regenerate it](#not-included-and-how-to-regenerate-it)
9. [Reading the files correctly](#reading-the-files-correctly)

## Names

The files use the names of the code.

| Name in files and code | Meaning |
|---|---|
| `toy` | The synthetic 5-bus network: 2 generators, 5 lines and 3 loads, so 7 failure-prone components. Run at repair-cost scale $`s = 100`$. |
| `case14` | The IEEE 14-bus network of PGLib-OPF v23.07 (`pglib_opf_case14_ieee.m`): 5 generators, 20 branches and 11 loads, so 25 failure-prone components. Run at $`s = 45`$. |
| `case14_scale55` | The same network at $`s = 55`$. Only the folder name carries the scale. The code still calls the network `case14`. |
| `paper_unbudgeted` | Runs without a repair budget. |
| `paper_budget0.3` | Runs with a per-stage repair budget of $`\varphi = 0.3`$ times the cost of repairing every component. |
| `Pseudo_DFL` | The DFL policy. |
| `NearSightedMPC` | The Near-sighted MPC baseline, with a window of $`W = 3`$ stages. |
| `AlwaysRepair`, `NoRepair` | The two trivial heuristics. AlwaysRepair repairs every failed component except at the last stage, where a repair could not complete within the horizon. Under the budget, it repairs as many failed components as the budget allows, cheapest first. |
| `SAA`, `SDDiP` | The tree SAA and SDDiP baselines. |
| $`s`$ | Repair-cost scale. Repair costs are proportional to capacity, with mean $`550\,s`$ per generator and $`350\,s`$ per line. |
| lane | One network at one scale within one variant. Each variant has three lanes. |

Costs are in dollars, energy in MWh, power in MW and times in seconds. In the rest of this file, `<variant>` stands for `paper_unbudgeted` or `paper_budget0.3`, `<lane>` for `toy`, `case14` or `case14_scale55`, `<N>` for a training size, `<seed>` for an initialization seed (1, 2, 3, 4 or 42) and `<scale>` for a repair-cost scale.

## Layout

```text
experiments/                                       1,084 result files, 69 MB
├── paper_unbudgeted/                              526 files, no repair budget
│   ├── toy/                                       5-bus network, s = 100
│   │   ├── config_snapshot.json                   copy of a DFL snapshot (see the last section)
│   │   ├── SAA/solution.json                      tree SAA, with its out-of-sample costs
│   │   └── SDDiP/
│   │       ├── sddip_solution.json                bounds, simulated policy, convergence trace
│   │       └── sddip_convergence.png              diagnostic plot
│   ├── case14/                                    IEEE 14-bus, s = 45, same files
│   ├── case14_scale55/                            IEEE 14-bus, s = 55, same files
│   └── dfl_scaling_multiseed/
│       ├── convergence_multiseed.png              diagnostic plot, all three lanes
│       └── <lane>/
│           ├── convergence_multiseed.png          diagnostic plot
│           └── dfl_train<N>/                      15 training sizes, N = 25 to 120,000
│               ├── evaluation.json                aggregated over the five init seeds
│               └── seed_<seed>/                   init seeds 1, 2, 3, 4 and 42
│                   ├── evaluation.json            one trained model and the baselines
│                   ├── config_snapshot.json       configuration of the run
│                   └── model.pt                   only at the reported size of the lane
├── paper_budget0.3/                               504 files, budget φ = 0.3, same layout;
│                                                  case14_scale55 has 13 sizes (25 to 80,000)
├── sensitivity/                                   52 files, repair-cost sweep without budget
│   ├── scale_<scale>/{toy,case14}/                24 cells: 12 scales, s = 0.1 to 150
│   │   ├── config_snapshot.json
│   │   └── evaluation.json
│   ├── sensitivity_curve.txt                      (AR − DFL) / DFL table, read by paper/
│   └── sensitivity_curve{,_toy,_case14}.png       diagnostic plots
└── exact_dp_toy/                                  2 files, 3.1 MB
    ├── unbudgeted/exact_dp_solution.json          exact dynamic programming, 5-bus network
    └── budgeted/exact_dp_solution.json
```

| File | Count | Where |
|---|---:|---|
| `evaluation.json` | 552 | 440 per seed, 88 aggregated over seeds, 24 in the sensitivity sweep |
| `config_snapshot.json` | 470 | 440 per seed, 6 at lane level, 24 in the sensitivity sweep |
| `model.pt` | 30 | 5 init seeds × 6 lanes, at the reported size only |
| `convergence_multiseed.png` | 8 | one per lane, one per variant with all three lanes |
| `solution.json` | 6 | SAA, one per lane |
| `sddip_solution.json`, `sddip_convergence.png` | 6 + 6 | SDDiP, one each per lane |
| `exact_dp_solution.json` | 2 | exact DP, one per variant |
| `sensitivity_curve.txt`, `sensitivity_curve*.png` | 1 + 3 | sensitivity summary |

### Lanes

| Variant | Lane | Network | s | Budget per stage | Sizes | Seed runs | Reported size N |
|---|---|---|---:|---:|---:|---:|---:|
| `paper_unbudgeted` | `toy` | 5-bus | 100 | none | 15 | 75 | 15,000 |
| `paper_unbudgeted` | `case14` | IEEE 14-bus | 45 | none | 15 | 75 | 30,000 |
| `paper_unbudgeted` | `case14_scale55` | IEEE 14-bus | 55 | none | 15 | 75 | 100,000 |
| `paper_budget0.3` | `toy` | 5-bus | 100 | 85,500 | 15 | 75 | 60,000 |
| `paper_budget0.3` | `case14` | IEEE 14-bus | 45 | 131,625 | 15 | 75 | 15,000 |
| `paper_budget0.3` | `case14_scale55` | IEEE 14-bus | 55 | 160,875 | 13 | 65 | 80,000 |

The reported size N of a lane is the training size with the lowest five-seed mean DFL cost. `model.pt` is stored for that size only.

### Settings shared by all runs

| Setting | Value |
|---|---|
| Horizon and repair delay | $`T = 14`$ stages. A repair ordered at stage $`t`$ is in service at $`t + 1`$ |
| Load-shedding cost | 1,000 per MWh |
| Budget (budgeted variant) | $`\varphi = 0.3`$ of the cost of repairing every component, per stage: $`B = \varphi \sum_k \beta[k]`$, with $`\beta[k]`$ the repair cost of component $`k`$. Unspent budget is not carried over |
| Data seed | 42. It draws the failure model (vulnerability class and peak failure probability of each component), the training scenarios, the train/validation split and the batch order |
| Init seeds | 1, 2, 3, 4 and 42. They set only the initial network weights |
| Training sizes | 25, 100, 250, 500, 1,000, 2,000, 3,000, 5,000, 10,000, 15,000, 30,000, 60,000, 80,000, 100,000, 120,000 |
| Test set | 500 scenarios from eval seed 1000. All methods except SDDiP, and all lanes of the same network, use these scenarios in the same order |
| Training labels | Single-scenario MIPs solved by HiGHS through Pyomo appsi, relative gap 1e-6, 64 worker processes with 1 thread each |
| Stage LPs | SciPy's HiGHS for the DFL, AlwaysRepair, NoRepair and Near-sighted MPC rollouts; Gurobi for the SAA rollouts |
| SAA | Gurobi 10.0.1, MIPGap 0.10, time limit 419,760 s |
| SDDiP | SDDP.jl with Lagrangian cuts and HiGHS subproblems |
| Hardware | Snellius, partition rome (AMD EPYC 7H12). Each experiment job requested 1 node, 64 CPUs, 112 GB and 117:36:00. The sensitivity jobs requested 72:00:00 |

### Which job wrote which files

| Files | Written by |
|---|---|
| `experiments/<variant>/dfl_scaling_multiseed/<lane>/dfl_train<N>/seed_<seed>/` | `scripts/slurm/{unbudgeted,budgeted}/dfl_multiseed.sh`, which runs `Pseudo_DFL_main.py` once per size and init seed |
| `experiments/<variant>/dfl_scaling_multiseed/<lane>/dfl_train<N>/evaluation.json`, and the `convergence_multiseed.png` files | `dfl_aggregate.sh`, which runs `aggregate_multiseed.py --plot` |
| `experiments/<variant>/<lane>/SAA/solution.json` | `saa.sh`, which runs `main.py --mode tree` |
| `experiments/<variant>/<lane>/SDDiP/` | `sddip.sh`, which runs `SDDiP_main.py` |
| `experiments/<variant>/<lane>/config_snapshot.json` | identical to `dfl_scaling_multiseed/<lane>/dfl_train25/seed_42/config_snapshot.json` of the same variant |
| `experiments/sensitivity/` | `scripts/slurm/sensitivity/sensitivity_sweep.sh`, then `sensitivity_plot.sh`, which runs `scripts/sensitivity_plot.py` |
| `experiments/exact_dp_toy/` | `exact_dp/exact_dp_toy.py` with its default settings |

The cluster pipeline and its submission order are described in [scripts/README.md](../scripts/README.md).

## What the files contain

### DFL results per seed

`experiments/<variant>/dfl_scaling_multiseed/<lane>/dfl_train<N>/seed_<seed>/evaluation.json` holds the results of one trained DFL model (N training scenarios, one init seed) and of the three baselines, each rolled out on the same 500 test scenarios. The top-level keys are `timing`, `Pseudo_DFL`, `AlwaysRepair`, `NoRepair` and `NearSightedMPC`. All method entries have the same keys:

| Key | Content |
|---|---|
| `costs` | Total cost of each test scenario over the 14 stages (500 values). Entry i refers to test scenario i in all files and for all methods. |
| `mean_cost`, `median`, `p25`, `p75`, `min_cost`, `max_cost` | Statistics of `costs`. |
| `std_cost` | Standard deviation of `costs` across the test scenarios (population, ddof = 0). |
| `mean_opf_cost` | Mean of the summed stage DC-OPF costs: generation plus load shedding at 1,000 per MWh. |
| `mean_repair_cost` | Mean repair spend. `mean_cost` equals `mean_opf_cost` + `mean_repair_cost`. |
| `load_shed_mwhs`, `mean_load_shed_mwh` | Unserved energy of each scenario, and its mean. |
| `eval_time_s`, `mean_time_per_scenario_s` | Rollout time for all test scenarios, and per 14-stage scenario. |
| `label` | The method name. |

`timing` holds `n_train`, `n_eval` (500), `training_s` (training wall time), `evaluation_s` (the DFL, AlwaysRepair and NoRepair rollouts) and `data_generation_s`. In these runs `data_generation_s` is only the time to load the cached training labels, at most 4.43 s (see [the last section](#reading-the-files-correctly)).

AlwaysRepair, NoRepair and Near-sighted MPC do not depend on N or on the init seed, so each has a single `mean_cost` across all files of a lane. Only `Pseudo_DFL` changes from file to file.

### DFL results aggregated over seeds

`experiments/<variant>/dfl_scaling_multiseed/<lane>/dfl_train<N>/evaluation.json` is written by `aggregate_multiseed.py` from the five seed files of that size.

- `Pseudo_DFL` has no `costs` list. It holds `mean_cost` and `std_cost`, the mean and the population standard deviation of the five per-seed means. This standard deviation is a spread across seeds, not across scenarios. The entry also holds `n_seeds` (5), `seed_means` (keyed `"1"`, `"2"`, `"3"`, `"4"` and `"42"`), and the five-seed averages of `mean_opf_cost`, `mean_repair_cost`, `mean_load_shed_mwh` and `mean_time_per_scenario_s`.
- `AlwaysRepair`, `NoRepair` and `NearSightedMPC` are copied unchanged from `seed_1/evaluation.json`.
- `timing` is copied from `seed_1`, with `training_s` and `data_generation_s` replaced by their five-seed means and `n_seeds` added.

Rerunning `aggregate_multiseed.py` on the stored seed files reproduces all 88 aggregated files byte for byte. With its default `--root` the script rewrites the stored files, so check this on a copy:

```bash
rm -rf /tmp/unb && cp -r experiments/paper_unbudgeted /tmp/unb
python aggregate_multiseed.py --root /tmp/unb/dfl_scaling_multiseed
diff -r experiments/paper_unbudgeted /tmp/unb && echo identical
```

The same three lines with `paper_budget0.3` in place of `paper_unbudgeted` check the budgeted files.

### Run configuration

Each seed folder has a `config_snapshot.json` (440 files), and so does each sensitivity cell (24). The snapshot lists all upper-case constants of `config.py` at the start of the run, after `Pseudo_DFL_main.py` has applied the command-line values of the repair-cost scale and bases, the budget fraction, and the data and eval seeds. Other command-line values are not recorded. Within a lane, all seed snapshots are identical.

<details>
<summary>Snapshot keys and their values in the reported runs</summary>

| Keys | Values |
|---|---|
| `REPAIR_BASE_SCALE`, `GEN_REPAIR_BASE`, `LINE_REPAIR_BASE` | 100, 55,000, 35,000 (`toy`); 45, 24,750, 15,750 (`case14`); 55, 30,250, 19,250 (`case14_scale55`) |
| `REPAIR_BUDGET_FRACTION` | `null` or 0.3 |
| `T`, `TAU`, `LOAD_SHEDDING_COST` | 14, 1, 1000 |
| `FAILURE_K`, `FAILURE_T_STAR`, `FAILURE_CLASS_PROB` | 1.5, 1, 0.5 |
| `FAILURE_P_MAX_STD_LOW`, `_STD_HIGH`, `_ELV_LOW`, `_ELV_HIGH` | 0.55, 0.70, 0.80, 0.90 |
| `NN_HIDDEN_DIMS`, `NN_DROPOUT`, `FY_EPSILON` | [128, 256, 128], 0.3, 0.1 |
| `TRAIN_N_EPOCHS`, `TRAIN_BATCH_SIZE`, `TRAIN_LR`, `TRAIN_WEIGHT_DECAY` | 200 (an epoch budget of $`\max(1, \lfloor N/10 \rfloor)`$), 64, 0.001, 1e-5 |
| `TRAIN_VAL_SPLIT`, `TRAIN_GRAD_CLIP` | 0.2, 1.0 |
| `TRAIN_SCHED_PATIENCE`, `TRAIN_SCHED_FACTOR`, `TRAIN_EARLY_STOP_PATIENCE` | 200, 0.5, 330 |
| `DATA_SOLVER_NAME`, `LP_METHOD`, `DATA_MIP_REL_GAP`, `DATA_THREADS_PER_WORKER`, `DATA_MAXTASKS_PER_CHILD` | `appsi_highs`, `highs`, 1e-6, 1, 64 |
| `NEAR_SIGHTED_MPC_WINDOW`, `MPC_CAP_REFERENCE_COMPONENTS`, `MPC_SCHEDULE_CAP_RATIO`, `MPC_SAMPLE_SEED` | 3, 7, 1, 0 |
| `FY_MC_SAMPLES` | 5. The value is 8 in the unbudgeted `toy` lane and the sensitivity runs, where it has no effect because it is used only under a budget |
| `DEFAULT_TRAIN_SEED`, `DEFAULT_INIT_SEED`, `DEFAULT_EVAL_SEED`, `DEFAULT_N_EVAL` | 42, 42, 1000, 500 |
| `SAA_TIME_LIMIT_S`, `SAA_MIP_GAP`, `SAA_TREE_BRANCHING` | 172800, 0.05, 2. These are the `config.py` defaults. The SAA runs used other values, which `solver_metadata` records |

The 24 sensitivity snapshots lack `SAA_TREE_BRANCHING` and `DATA_MAXTASKS_PER_CHILD`. Neither key affects DFL results.

</details>

### Trained models

The 30 `model.pt` files cover the five init seeds at the reported size of each lane. Each is a `torch.save` dictionary:

- `model_state`: weights of a fully connected network with hidden widths (128, 256, 128), ReLU activations and dropout 0.3. Its input is the availability (1 = in service) of each generator, then of each line, then $`t/T`$ at stage $`t = 1, \dots, 14`$. Its output, the predicted net repair benefit $`\eta^t = \boldsymbol{\mu}^t - \beta`$ in the notation of the paper (`theta` in the code), has one entry per component: the expected reduction of future cost $`\boldsymbol{\mu}^t[k]`$ from starting the repair of component $`k`$ at stage $`t`$, minus its repair cost $`\beta[k]`$. The model has 8 inputs and 7 outputs on the 5-bus network, and 26 inputs and 25 outputs on case14. Without a budget, a failed component $`k`$ is repaired when $`\eta^t[k] > 0`$. Under the budget, a 0/1 knapsack chooses among the failed components with $`\eta^t[k] > 0`$.
- `train_losses`, `val_losses`, `val_accs`: one value per training epoch. The losses are values of the stop-gradient [training loss](../README.md#training) and can be negative. `val_accs` is a repair ratio, not an accuracy: on the validation set, the number of failed components with $`\eta^t[k] > 0`$ divided by the number of repairs in the labels.
- `epsilon` (0.1), `in_dim` and `n_comp`.

The checkpoint does not store the hidden widths, so pass `hidden_dims=(128, 256, 128)` when loading ([example](#loading-the-files-in-python)).

All linear layers start from Xavier-uniform weights and zero biases. The epoch budget is $`\max(1, \lfloor N/10 \rfloor)`$, for example 200 epochs at $`N = 2{,}000`$, and `--n-epochs` of `Pseudo_DFL_main.py` overrides it. Training stops early once the validation loss has not improved for 330 epochs, and the stored weights are those of the best validation epoch. A run with 3,000 or fewer training scenarios has a budget of at most 300 epochs, so early stopping cannot end it. All 30 models stopped early, after 332 to 926 epochs (the best epoch plus the patience of 330), well below their epoch budgets of $`N/10`$.

<details>
<summary>Epochs trained at the reported sizes (init seeds 1, 2, 3, 4 and 42)</summary>

| Variant, lane | N | Epoch budget | Epochs trained |
|---|---:|---:|---|
| unbudgeted, `toy` | 15,000 | 1,500 | 926, 787, 434, 434, 434 |
| unbudgeted, `case14` | 30,000 | 3,000 | 446, 509, 374, 397, 363 |
| unbudgeted, `case14_scale55` | 100,000 | 10,000 | 447, 332, 483, 605, 447 |
| budgeted, `toy` | 60,000 | 6,000 | 346, 359, 405, 452, 739 |
| budgeted, `case14` | 15,000 | 1,500 | 363, 706, 722, 795, 386 |
| budgeted, `case14_scale55` | 80,000 | 8,000 | 644, 744, 677, 744, 600 |

The number of epochs trained is the length of `train_losses` in each `model.pt`.

</details>

### SAA

`experiments/<variant>/<lane>/SAA/solution.json` comes from `main.py --mode tree --branching 2`, which solves an extensive-form MILP on a sampled scenario tree with 2 children per node, giving 16,384 sample paths over the 14 stages. The file has four top-level keys:

- `out_of_sample_evaluation`: the SAA result used in the paper. `test_costs` holds the cost of each of the 500 test scenarios, in the same order as the `costs` lists of the DFL files. The fields `mean_cost`, `std_cost`, `min_cost`, `max_cost`, `n_test_scenarios`, `eval_time_s` and `mean_time_per_scenario_s` summarize them. The rollout applies the tree policy in closed loop. At each stage it moves to the child whose sampled failures are nearest, in Hamming distance, to the realized ones.
- `objective_value`: the in-sample objective of the best incumbent of the tree MILP. It is not the reported SAA cost.
- `costs`: only `total` (equal to `objective_value`) is filled. The fields `generation`, `load_shedding` and `repairs` are 0.0 placeholders.
- `solver_metadata`: `solver_name`, `mode`, `branching`, `n_sample_paths`, `mip_gap_target`, `mip_gap_achieved`, `time_limit`, `solve_time` (including the time to build the tree and the model), `termination_condition`, `norepair_fallback` (false in all six files) and `n_scenarios` (2,000 or 300, not used in tree mode).

| Variant, lane | Gap reached | `solve_time` | Termination | Out-of-sample mean |
|---|---:|---:|---|---:|
| unbudgeted, `toy` | 9.75% | 3,190 s | optimal (10% target reached) | 2,028,014 |
| unbudgeted, `case14` | 60.08% | 420,291 s | maxTimeLimit | 3,193,445 |
| unbudgeted, `case14_scale55` | 59.21% | 420,288 s | maxTimeLimit | 3,255,115 |
| budgeted, `toy` | 9.97% | 8,144 s | optimal (10% target reached) | 2,069,534 |
| budgeted, `case14` | 53.01% | 420,302 s | maxTimeLimit | 3,240,817 |
| budgeted, `case14_scale55` | 42.49% | 420,305 s | maxTimeLimit | 3,214,677 |

### SDDiP

`experiments/<variant>/<lane>/SDDiP/sddip_solution.json` comes from `SDDiP_main.py`, which trains a policy with SDDP.jl and then simulates it 500 times on its own stagewise failure distribution. The file holds:

- `lower_bound`, equal to `objective_value`: the last lower bound of training.
- `upper_bound` and `gap`: the cost of the forward pass of the last training iteration, and its relative distance to the lower bound, $`\lvert \mathrm{UB} - \mathrm{LB} \rvert / \max(\lvert \mathrm{UB} \rvert, \lvert \mathrm{LB} \rvert, 1)`$. This is the cost of one sampled trajectory, not a statistical upper bound.
- `iterations`, `n_cuts` (13 per iteration, one per non-terminal stage) and `solve_time` (the whole run, including Julia setup, model build, training, writing the cuts and the 500 simulations).
- `repairs["gen"][t][k]` and `repairs["line"][t][k]`: the fraction of the 500 simulations that order a repair of component k at stage t. Stage keys run from `"1"` to `"14"`, component keys from `"0"`.
- `expected_generation[t][g]` and `expected_load_served[t][l]`: mean dispatch and mean served load in MW over the simulations.
- `convergence`: `lower_bounds`, `upper_bounds` and `gaps`, one entry per iteration. `times` (seconds since the start of training) is present in the four case14 files only.

The policy mean in the SDDiP table, the mean cost over the 500 simulations, is not stored as a number. The stage cost is affine in the stored means, so `reconstruct_saved_sddip_mean` in `exact_dp/exact_dp_toy.py` rebuilds it exactly ([example](#loading-the-files-in-python)). For the two 5-bus runs the exact-DP files also store it, as `saved_sddip_comparison.reconstructed_500_simulation_in_sample_mean`. The individual simulation costs were not saved, so no variance is available.

| Variant, lane | Iterations | Lower bound | Policy mean | `solve_time` |
|---|---:|---:|---:|---:|
| unbudgeted, `toy` | 782 | 1,024,482 | 1,584,476 | 420,611 s |
| budgeted, `toy` | 755 | 1,296,240 | 1,651,543 | 420,240 s |
| unbudgeted, `case14` | 158 | 671,289 | 2,110,204 | 398,475 s |
| budgeted, `case14` | 150 | 831,741 | 2,208,569 | 395,790 s |
| unbudgeted, `case14_scale55` | 150 | 740,410 | 2,283,357 | 396,438 s |
| budgeted, `case14_scale55` | 147 | 927,540 | 2,304,480 | 395,499 s |

The file records none of the run settings. The runs used seed 42, at most 10,000 iterations, 500 simulations and HiGHS subproblems. On the 5-bus network all 128 failure outcomes per stage are enumerated. On case14, 256 outcomes per stage are sampled. The training time limit (`--hard-time-limit`) was 419,760 s for the two 5-bus runs and 394,560 s for the four case14 runs. In all six runs this limit, not the iteration limit, ended training.

### Exact dynamic programming

`experiments/exact_dp_toy/{unbudgeted,budgeted}/exact_dp_solution.json` comes from `exact_dp/exact_dp_toy.py`, which computes the optimal expected cost of the 5-bus problem by dynamic programming over at most 2,187 states per stage. The settings are seed 42, $`T = 14`$, $`s = 100`$, and $`\varphi = 0.3`$ with a budget of 85,500 per stage in the budgeted file. The main keys of the file are:

- `solution.exact_optimal_value`: 1,382,625.16 without budget (expected OPF cost 945,380.17 plus repairs 437,245.00) and 1,493,318.65 with the budget.
- `optimal_policy`: for each of the 14 stages, the optimal repair set for each of the 128 post-failure availability masks. `value_function_by_stage` holds the optimal cost-to-go of each state, $`15 \times 2{,}187`$ values.
- `saved_sddip_comparison`: the stored 5-bus SDDiP run of the same variant compared with the optimum, including its rebuilt policy mean.

A rerun of the script takes about 2 s and gives the same optimal values, policies and value functions. [exact_dp/README.md](../exact_dp/README.md) documents all keys of the file.

### Repair-cost sensitivity sweep

`experiments/sensitivity/scale_<scale>/<network>/evaluation.json` (24 files, where `<network>` is `toy` or `case14`) has the per-seed format above, without `NearSightedMPC`. Each cell is one DFL model, trained without budget on 120,000 scenarios with data seed and init seed 42, and evaluated on the same 500 test scenarios. Here `timing.data_generation_s` is the real label-generation time with 64 workers: 1.5 to 1.8 h on the 5-bus network and 5.8 to 18.6 h on case14.

`sensitivity_curve.txt` gives $`(\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}})/\bar c_{\mathrm{DFL}}`$, with the mean AlwaysRepair cost $`\bar c_{\mathrm{AR}}`$ and the mean DFL cost $`\bar c_{\mathrm{DFL}}`$, in percent, for each scale and network.

```text
 scale         toy      case14
------------------------------
   0.1     +1.15%    -42.17%
   0.2     +1.17%    -46.10%
   0.5     +1.19%    -40.21%
   1.0     +0.36%    -41.51%
   2.0     +0.98%    -42.55%
   5.0     +1.17%    -41.08%
  10.0     +1.26%    -39.58%
  20.0     +1.48%    -31.95%
  50.0     +1.01%     -7.80%
  70.0     +2.48%    +19.31%
 100.0     +3.39%    +43.92%
 150.0     +8.02%    +76.70%
```

`scripts/sensitivity_plot.py` writes this table and the three `sensitivity_curve*.png` from the `evaluation.json` files. It reads only folders named `scale_<float>`, such as `scale_10.0` (not `scale_10`). It writes all four files into the folder of `--out`, so an `--out` outside `experiments/sensitivity/` leaves the stored files untouched. The table it writes is identical to the stored one:

```bash
python scripts/sensitivity_plot.py --base experiments/sensitivity --out /tmp/sensitivity/sensitivity_curve.png
cmp experiments/sensitivity/sensitivity_curve.txt /tmp/sensitivity/sensitivity_curve.txt && echo identical
```

### Diagnostic plots

The PNG files in this folder are the plots drawn by the cluster jobs:

- `convergence_multiseed.png`, from `aggregate_multiseed.py --plot`.
- `sddip_convergence.png`, from `SDDiP_main.py`. Its "Upper Bound" and "Optimality Gap" curves are the forward-pass cost and gap described above.
- `sensitivity_curve*.png`, from `scripts/sensitivity_plot.py`.

The figures of the paper are drawn separately, into `figures/` ([below](#from-files-to-figures)).

## From files to reported numbers

`paper/reproduce_paper_numbers.py` recomputes the reported numbers from these files and compares them with the values in the paper. Its source code is the authoritative record of which key feeds which number. The table below names the file and key behind each number, and [paper/README.md](../paper/README.md) lists which values the script asserts and which it only prints. In the table, N is the reported size of the lane, and the files are named as follows:

- aggregated file: `experiments/<variant>/dfl_scaling_multiseed/<lane>/dfl_train<N>/evaluation.json`
- seed files: the five `seed_<seed>/evaluation.json` files in the folder of an aggregated file
- SAA file: `experiments/<variant>/<lane>/SAA/solution.json`
- SDDiP file: `experiments/<variant>/<lane>/SDDiP/sddip_solution.json`
- exact-DP files: `experiments/exact_dp_toy/{unbudgeted,budgeted}/exact_dp_solution.json`

| Reported quantity | File | Keys |
|---|---|---|
| NoRepair, AlwaysRepair, Near-sighted MPC and DFL columns of the two cost tables | aggregated file | `<method>.mean_cost`; `Pseudo_DFL.mean_cost` is the five-seed mean |
| SAA column of the cost tables | SAA file | `out_of_sample_evaluation.mean_cost` |
| DFL vs AR, the saving relative to AlwaysRepair | aggregated file | $`(\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}})/\bar c_{\mathrm{AR}}`$ from the two `mean_cost` values |
| Across-seed std of DFL in both cost tables (the script checks the budgeted values) | aggregated file | `Pseudo_DFL.std_cost` |
| Choice of N | every `dfl_train*/evaluation.json` of the lane | the size with the lowest `Pseudo_DFL.mean_cost` |
| Repair spend and load shed quoted in the text | aggregated file | `mean_repair_cost` and `mean_load_shed_mwh` of `Pseudo_DFL` and `AlwaysRepair` |
| Across-scenario std | seed files at N for DFL (mean over the five seeds), aggregated file for AlwaysRepair and MPC | `std_cost` |
| Deployment time per 14-stage scenario | aggregated file, and SAA file for SAA | `mean_time_per_scenario_s` of each method, and of `out_of_sample_evaluation` for SAA |
| DFL time per scenario, range over all 440 models (not checked by the script) | seed files of all sizes | `Pseudo_DFL.mean_time_per_scenario_s` |
| SAA gaps, incumbents and solve times | SAA file | `solver_metadata.mip_gap_achieved`, `objective_value`, `solver_metadata.solve_time`, `solver_metadata.termination_condition` |
| SDDiP table: iterations, cuts, lower bound and run time | SDDiP file | `iterations`, `n_cuts`, `lower_bound`, `solve_time` |
| SDDiP table: policy mean and estimated gap | SDDiP file | rebuilt from `repairs`, `expected_generation` and `expected_load_served`; gap $`(\bar c_{\mathrm{SDDiP}} - \mathrm{LB})/\bar c_{\mathrm{SDDiP}}`$ |
| Exact optimum, and DFL and SDDiP relative to it | exact-DP files | `solution.exact_optimal_value` |
| Sample-complexity statements without budget | aggregated files of all sizes, `paper_unbudgeted`, lanes `toy` and `case14_scale55` | `Pseudo_DFL.mean_cost`, `Pseudo_DFL.std_cost`, `AlwaysRepair.mean_cost` |
| Sample-complexity comparisons with Near-sighted MPC, and the statements with the budget (not checked by the script) | aggregated files of all sizes, lanes `toy` and `case14_scale55` of both variants | `Pseudo_DFL.mean_cost`, `AlwaysRepair.mean_cost`, `NearSightedMPC.mean_cost` |
| Cost-regime curve and the case14 crossover near $`s = 55`$ | `sensitivity_curve.txt` in `experiments/sensitivity/` | toy and case14 columns |
| Longest training of a single model: 17 h without budget, 63 h with it (not checked by the script) | seed files of all sizes | maximum of `timing.training_s` |

The quantities of this table derive from the total cost $`C_i`$ of test scenario $`i`$ over the 14 stages, entry $`i`$ of `costs`. For one run of a method, `mean_cost` and `std_cost` hold

```math
\bar c = \frac{1}{500}\sum_{i=1}^{500} C_i,
\qquad
\mathrm{std} = \sqrt{\frac{1}{500}\sum_{i=1}^{500}\big(C_i - \bar c\big)^2}.
```

In an aggregated file, the `Pseudo_DFL` entry holds instead the five-seed mean and the across-seed standard deviation,

```math
\bar c_{\mathrm{DFL}} = \frac{1}{5}\sum_{j=1}^{5} \bar c_{\mathrm{DFL}}^{(j)},
\qquad
\mathrm{std}_{\mathrm{seed}} = \sqrt{\frac{1}{5}\sum_{j=1}^{5}\big(\bar c_{\mathrm{DFL}}^{(j)} - \bar c_{\mathrm{DFL}}\big)^2},
```

where $`\bar c_{\mathrm{DFL}}^{(j)}`$ is the mean cost of the model with the $`j`$-th init seed. The ratios, all reported in percent, are

```math
\begin{aligned}
\text{DFL vs AR} &= \frac{\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}}}{\bar c_{\mathrm{AR}}},
&\qquad
\text{sensitivity} &= \frac{\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}}}{\bar c_{\mathrm{DFL}}},\\[8pt]
\text{above the optimum} &= \frac{\bar c - \mathcal{V}^\star}{\mathcal{V}^\star},
&\qquad
\text{below the optimum} &= \frac{\mathcal{V}^\star - \mathrm{LB}}{\mathcal{V}^\star},\\[8pt]
\text{estimated gap} &= \frac{\bar c_{\mathrm{SDDiP}} - \mathrm{LB}}{\bar c_{\mathrm{SDDiP}}},
\end{aligned}
```

where AR stands for AlwaysRepair, $`\bar c_{\mathrm{SDDiP}}`$ is the mean cost of the 500 SDDiP simulations, $`\mathrm{LB}`$ the SDDiP lower bound and $`\mathcal{V}^\star`$ the exact optimum of the 5-bus network. In the sensitivity table, $`\bar c_{\mathrm{DFL}}`$ is the mean cost of the single DFL model with init seed 42 of a cell, and $`\bar c_{\mathrm{AR}}`$ the AlwaysRepair cost of the same cell.

Mean costs at the reported sizes, over the 500 test scenarios:

| Variant | Lane | N | NoRepair | AlwaysRepair | Near-sighted MPC | SAA | DFL | DFL vs AR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unbudgeted | `toy` | 15,000 | 2,511,488 | 1,506,639 | 1,445,096 | 2,028,014 | 1,406,419 | +6.7% |
| unbudgeted | `case14` | 30,000 | 3,614,867 | 2,098,438 | 1,949,190 | 3,193,445 | 2,271,956 | -8.3% |
| unbudgeted | `case14_scale55` | 100,000 | 3,614,867 | 2,369,039 | 2,107,215 | 3,255,115 | 2,318,269 | +2.1% |
| budgeted | `toy` | 60,000 | 2,511,488 | 1,937,457 | 2,403,624 | 2,069,534 | 1,520,472 | +21.5% |
| budgeted | `case14` | 15,000 | 3,614,867 | 2,842,476 | 2,600,141 | 3,240,817 | 2,235,973 | +21.3% |
| budgeted | `case14_scale55` | 80,000 | 3,614,867 | 3,052,613 | 3,390,760 | 3,214,677 | 2,340,615 | +23.3% |

Other percentages in the text, such as DFL relative to Near-sighted MPC, are ratios of the same means.

## From files to figures

`paper/make_paper_figures.py` draws the figures into `figures/`, or into the folder given with `--out`. [paper/README.md](../paper/README.md) describes what each figure shows. The script reads these keys:

| Figure | Keys read |
|---|---|
| `figures/failure_probability_profiles.png` | none (the figure uses the failure-model constants of `config.py`) |
| `figures/sensitivity/cost_vs_scale_{toy,case14}.png` | toy and case14 columns of `experiments/sensitivity/sensitivity_curve.txt` |
| `figures/paper/<lane>/convergence_multiseed.png` (unbudgeted) and `convergence_multiseed_budget.png` (budgeted) | for every aggregated `dfl_train<N>/evaluation.json` of the lane: `Pseudo_DFL.mean_cost` with a band of ± `Pseudo_DFL.std_cost` (across seeds), and the `mean_cost` of `AlwaysRepair`, `NoRepair` and `NearSightedMPC` as horizontal lines. The budgeted $`s = 55`$ curve ends at 80,000. The $`s = 45`$ curves are drawn but reported in the tables only |
| `figures/paper/toy/sddip_convergence.png` | `convergence.lower_bounds`, `convergence.upper_bounds` (drawn as "Forward-pass cost") and `convergence.gaps` of `experiments/paper_unbudgeted/toy/SDDiP/sddip_solution.json` |

## Loading the files in Python

Run the examples from the repository root, with the packages of `requirements.txt` installed. Each block continues the previous one.

```python
import json
from pathlib import Path

EXP = Path("experiments")

def load(path):
    return json.loads(Path(path).read_text())
```

One row of the unbudgeted cost table:

```python
ev = load(EXP / "paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train15000/evaluation.json")
saa = load(EXP / "paper_unbudgeted/toy/SAA/solution.json")["out_of_sample_evaluation"]
for method in ("NoRepair", "AlwaysRepair", "NearSightedMPC", "Pseudo_DFL"):
    print(f"{method:15s} {ev[method]['mean_cost']:12,.0f}")
print(f"{'SAA':15s} {saa['mean_cost']:12,.0f}")
ar, dfl = ev["AlwaysRepair"]["mean_cost"], ev["Pseudo_DFL"]["mean_cost"]
print(f"DFL vs AR: {(ar - dfl) / ar:+.1%}")                      # +6.7%
```

All seed runs in one table, and the sample-complexity curve recomputed from it:

```python
import pandas as pd

rows = []
for p in EXP.glob("paper_*/dfl_scaling_multiseed/*/dfl_train*/seed_*/evaluation.json"):
    variant, _, lane, size, seed = p.parts[1:6]
    ev = load(p)
    for method in ("Pseudo_DFL", "AlwaysRepair", "NoRepair", "NearSightedMPC"):
        rows.append({"variant": variant, "lane": lane,
                     "n_train": int(size.removeprefix("dfl_train")),
                     "init_seed": int(seed.removeprefix("seed_")), "method": method,
                     **{k: ev[method][k] for k in ("mean_cost", "std_cost",
                                                    "mean_repair_cost", "mean_load_shed_mwh")},
                     "training_s": ev["timing"]["training_s"]})
df = pd.DataFrame(rows)                                  # 440 runs x 4 methods = 1,760 rows
dfl = df[df.method == "Pseudo_DFL"]
sweep = dfl.groupby(["variant", "lane", "n_train"]).mean_cost.agg(
    mean="mean", across_seed_std=lambda x: x.std(ddof=0))
print(sweep.loc[("paper_unbudgeted", "toy")])            # equals the aggregated files
```

Scenario-by-scenario comparison. The `costs` lists of all methods, and SAA's `test_costs`, follow the same order of test scenarios:

```python
import numpy as np

ev = load(EXP / "paper_budget0.3/dfl_scaling_multiseed/case14/dfl_train15000/seed_42/evaluation.json")
dfl, ar = np.array(ev["Pseudo_DFL"]["costs"]), np.array(ev["AlwaysRepair"]["costs"])
print(f"DFL cheaper than AlwaysRepair on {np.mean(dfl < ar):.0%} of the test scenarios")   # 95%
```

The SDDiP policy mean, and the test scenarios themselves. Building `case14` downloads the PGLib file on first use:

```python
import sys
sys.path[:0] = [".", "exact_dp"]
from exact_dp_toy import reconstruct_saved_sddip_mean
from model_parameters import create_model_parameters
from network_data import create_network
from scenario_generator import create_scenario_generator

net = create_network("toy")                              # "case14" for both case14 lanes
params = create_model_parameters(net, seed=42, t=14, tau=1,
                                 gen_repair_base=550.0 * 100,   # 550 s and 350 s
                                 line_repair_base=350.0 * 100,
                                 repair_budget_fraction=None)   # 0.3 for paper_budget0.3
sddip = load(EXP / "paper_unbudgeted/toy/SDDiP/sddip_solution.json")
print(sddip["lower_bound"], reconstruct_saved_sddip_mean(sddip, net, params))   # 1,024,482 and 1,584,476

test = create_scenario_generator(params, seed=1000).generate_scenarios(500)
print(test[0].gen_failures.shape, test[0].line_failures.shape)   # (14, 2) (14, 5): failures per stage
```

A stored model:

```python
import torch
from Pseudo_DFL_model import load_repair_network

path = EXP / "paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train15000/seed_42/model.pt"
ckpt = torch.load(path, map_location="cpu")
print(ckpt["in_dim"], ckpt["n_comp"], len(ckpt["train_losses"]))   # 8 7 434
model, _ = load_repair_network(path, ckpt["in_dim"], ckpt["n_comp"], hidden_dims=(128, 256, 128))
x = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1 / 14]])       # G1, G2 up, line L1 failed, stage 1
with torch.no_grad():
    print(model(x))                                      # η per component (theta in the code); L1 is repaired if its η > 0
```

The exact DP policy and a sensitivity point:

```python
dp = load(EXP / "exact_dp_toy/unbudgeted/exact_dp_solution.json")
print(dp["solution"]["exact_optimal_value"])              # 1382625.16...
stage1 = dp["optimal_policy"][0]["actions"]               # 128 rows, one per availability mask
print(next(a for a in stage1 if a["availability_bits"] == [1, 1, 0, 1, 1, 1, 1]))

ev = load(EXP / "sensitivity/scale_100.0/case14/evaluation.json")
ar, dfl = ev["AlwaysRepair"]["mean_cost"], ev["Pseudo_DFL"]["mean_cost"]
print(f"{(ar - dfl) / dfl:+.2%}")                         # +43.92%, as in sensitivity_curve.txt
```

A stored model can also be checked against its stored costs. Re-evaluating it on the test set reproduces the per-scenario costs of `Pseudo_DFL`, `AlwaysRepair` and `NoRepair` exactly, in about 30 s on a laptop:

```bash
python Pseudo_DFL_main.py --mode evaluate --network toy --repair-base-scale 100 \
    --n-train 15000 --n-eval 500 --eval-seed 1000 --seed 42 \
    --load-model experiments/paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train15000/seed_42/model.pt \
    --output-dir runs/check_toy_15000_seed42
```

The following comparison of `runs/check_toy_15000_seed42/evaluation.json` with the stored `seed_42/evaluation.json` prints `True`:

```bash
python - <<'EOF'
import json
new = json.load(open("runs/check_toy_15000_seed42/evaluation.json"))
old = json.load(open("experiments/paper_unbudgeted/dfl_scaling_multiseed/toy/dfl_train15000/seed_42/evaluation.json"))
print(all(new[m]["costs"] == old[m]["costs"] for m in ("Pseudo_DFL", "AlwaysRepair", "NoRepair")))
EOF
```

For another lane, pass its network and scale (`--network case14 --repair-base-scale 45` or `55`) and, for a budgeted model, `--repair-budget-fraction 0.3`. Without `--with-mpc` the run skips Near-sighted MPC, which takes from 0.8 h (5-bus) to 13.2 h (case14) on the cluster.

## Gaps in the stored results

### Budgeted case14 at s = 55: sizes 100,000 and 120,000

`experiments/paper_budget0.3/dfl_scaling_multiseed/case14_scale55/` has 13 sizes (25 to 80,000) and 65 seed runs. Training sizes 100,000 and 120,000 did not finish within the 117:36:00 wall-clock limit of the cluster jobs. Every other lane has all 15 sizes for all five init seeds. The reported size of this lane, 80,000, is the best of its 13 sizes.

To add the two sizes on the cluster, from the repository root:

1. Extend the training labels of this lane to 120,000 scenarios. A labels-only run does this:

   ```bash
   python Pseudo_DFL_main.py --mode data --network case14 --repair-base-scale 55 \
       --repair-budget-fraction 0.3 --seed 42 --n-train 120000 \
       --output-dir runs/labels_case14_s55_phi0.3
   ```

   Run it inside a job on a node with many cores, because it starts one HiGHS worker per CPU. It writes `data_cache_scale55/case14/seed42/training_data_budgetfrac0.3.pkl` in the working directory. In the unbudgeted sensitivity runs, 120,000 case14 labels took 5.8 to 18.6 h on 64 cores. In the cluster pipeline, `scripts/slurm/budgeted/dfl_scaling.sh` fills this cache in task 2, which also trains init seed 42 at every size.
2. Train and evaluate the missing sizes for the five init seeds. Tasks 10 to 14 cover this lane, and the 13 stored sizes are skipped because their `evaluation.json` exists:

   ```bash
   sbatch --array=10-14 --export=ALL,SOLVER=appsi_highs scripts/slurm/budgeted/dfl_multiseed.sh
   ```

   Without the Near-sighted MPC cache (`mpc_W3_budgetfrac0.3.json` next to the labels), each task first recomputes MPC on the 500 test scenarios, which takes about 4 h on this lane. At 80,000 one model of this lane already trained for 38 to 49 h, so a task may need more than one 117:36:00 job. Resubmitting the same line resumes at the first missing size.
3. Aggregate the seeds again. Submit this by hand once all tasks are done, because a task that ends in TIMEOUT never satisfies an `afterok` dependency:

   ```bash
   sbatch scripts/slurm/budgeted/dfl_aggregate.sh
   ```

Budgeted training draws its Monte Carlo perturbations from NumPy's global random state, which the code does not seed, so new models will not match any earlier attempt bit for bit. `paper/reproduce_paper_numbers.py` checks that 80,000 is the best size of the lane, so the check fails if one of the new sizes has a lower five-seed mean.

## Not included, and how to regenerate it

| Not included | Regenerate with | Cost on the cluster |
|---|---|---|
| Training labels: `data_cache_scale<scale>/<network>/seed42/training_data.pkl`, or `training_data_budgetfrac0.3.pkl` with the budget (`<scale>` is 100, 45 or 55, `<network>` is `toy` or `case14`). They are gitignored and written relative to the working directory | `dfl_scaling.sh` (it grows the cache from 25 to 120,000 scenarios while it trains init seed 42 at every size), or `Pseudo_DFL_main.py --mode data` as in the [gap section](#budgeted-case14-at-s--55-sizes-100000-and-120000) | 120,000 labels: 1.5 to 1.8 h (5-bus) and 5.8 to 18.6 h (case14) on 64 cores |
| Near-sighted MPC results: `mpc_W3.json` or `mpc_W3_budgetfrac0.3.json`, next to the labels | the first run with `--with-mpc` of a lane, normally the first `dfl_scaling.sh` size | 0.8 h (5-bus) to 13.2 h (case14) per lane for 500 test scenarios |
| Single-seed sweeps `experiments/<variant>/dfl_scaling/<lane>/dfl_train<N>/` (init seed 42) | `scripts/slurm/{unbudgeted,budgeted}/dfl_scaling.sh` | similar to one seed task of `dfl_multiseed.sh` (below), plus labels and MPC |
| `model.pt` at sizes other than the reported one, and the per-run plots of `Pseudo_DFL_main.py` (`training_curves.png`, `cost_distribution.png`, repair heatmaps and others) | `Pseudo_DFL_main.py` for one lane, size and init seed (below) | see the training-time table below |
| `comparison/` folders with cross-method plots and `comparison_summary.json` | `compare.sh`, or `scripts/slurm/{unbudgeted,budgeted}/run_comparison_folders.py` | about 7 to 120 h per lane, mostly Near-sighted MPC, while `compare.sh` requests 8 h ([Comparison folders](../scripts/README.md#comparison-folders)) |
| SDDiP cuts (`cuts.json`), `sddip_checkpoint.json`, the run's `config_snapshot.json`, the other `sddip_*.png`, and the 500 individual simulation costs | `sddip.sh` (for the 5-bus limit, see below) | 110 to 117 h per run |
| SAA per-path solution, `config_snapshot.json` and plots, and the tree node policy (kept in memory only, so SAA cannot be re-evaluated without solving again) | `saa.sh` (needs a Gurobi license) | 0.9 h and 2.3 h on the 5-bus network. The case14 runs stop at the solver limit |
| SLURM logs | written to `logs/` by each job | |
| Budgeted case14 at $`s = 55`$, sizes 100,000 and 120,000 | see [Gaps](#gaps-in-the-stored-results) | |

**Retraining one model.** This command trains one lane, size and init seed and writes its output to a new folder:

```bash
python Pseudo_DFL_main.py --network toy --repair-base-scale 100 --n-train 5000 \
    --n-eval 500 --eval-seed 1000 --seed 42 --init-seeds 3 --with-mpc \
    --output-dir runs/paper_unbudgeted/toy/dfl_train5000/seed_3
```

For the budgeted variant, add `--repair-budget-fraction 0.3` and write to `runs/paper_budget0.3/`. For the case14 lanes, use `--network case14 --repair-base-scale 45` (or `55`) and replace `toy` in `--output-dir` by `case14` (or `case14_scale55`). Scenario i of the training set is drawn from its own seed, so the training set of size N consists of the first N scenarios of any larger one. Missing labels are solved first, one worker per CPU. With `--with-mpc`, the run also computes Near-sighted MPC when its cache is missing.

<details>
<summary>Mean training time per model on the cluster, in hours (five-seed mean of <code>timing.training_s</code>)</summary>

| N | unbudgeted `toy` | unbudgeted `case14` | unbudgeted `case14_scale55` | budgeted `toy` | budgeted `case14` | budgeted `case14_scale55` |
|---:|---:|---:|---:|---:|---:|---:|
| 5,000 | 0.3 | 0.3 | 0.5 | 1.0 | 1.8 | 1.3 |
| 10,000 | 0.8 | 0.8 | 1.1 | 2.1 | 4.5 | 3.5 |
| 15,000 | 1.6 | 1.1 | 1.6 | 2.9 | 7.1 | 4.7 |
| 30,000 | 2.9 | 1.9 | 3.3 | 6.1 | 12.8 | 11.2 |
| 60,000 | 7.6 | 3.5 | 5.4 | 13.9 | 21.1 | 21.6 |
| 80,000 | 7.2 | 5.6 | 7.6 | 20.6 | 39.3 | 42.6 |
| 100,000 | 8.3 | 7.1 | 9.7 | 21.2 | 37.7 | |
| 120,000 | 10.3 | 8.6 | 13.8 | 45.2 | 50.1 | |

Sizes up to 3,000 train in at most 0.7 h. One seed task of `dfl_multiseed.sh` trains all sizes of one lane for one init seed. With the labels already cached, such a task took 27 to 48 h without budget and 79 to 205 h with it, so some budgeted 5-bus tasks and all budgeted case14 tasks need more than one 117:36:00 job. Training the 440 stored models took 2,434 h in total.

</details>

**Repeating the SDDiP runs.** The stored 5-bus runs were trained with `--hard-time-limit 419760`, the job limit minus 1 h, and the stored case14 runs with 394,560 s, the job limit minus 8 h. `sddip.sh` passes 394,560 s on every lane. [SDDiP](../scripts/README.md#sddip) in scripts/README.md shows how to run the 5-bus lane with the longer limit.

**Regenerating everything.** The submission sequence is in [scripts/README.md](../scripts/README.md). The cluster scripts skip each configuration whose result file exists (`seed_<seed>/evaluation.json`, `SAA/solution.json`, `SDDiP/sddip_solution.json`, `scale_<scale>/<network>/evaluation.json`). In a checkout that contains the stored results they would skip everything stored, so move the stored files away first. Runs stopped by a time limit (SAA on case14 and all SDDiP runs) depend on the speed of the hardware and will not match the stored values exactly. Neither will budgeted DFL training (see below).

## Reading the files correctly

Statistics:

- `std_cost` has two meanings. In the `Pseudo_DFL` entry of an aggregated `dfl_train<N>/evaluation.json` it is the spread of the five per-seed means (across seeds). Everywhere else, including the baselines in the same file and the SAA file, it is the spread across the 500 test scenarios. The across-scenario std of DFL is the mean of `Pseudo_DFL.std_cost` over the five `seed_*/evaluation.json` files.
- The cost tables report DFL vs AR, $`(\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}})/\bar c_{\mathrm{AR}}`$, whereas the sensitivity table and its figures use $`(\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}})/\bar c_{\mathrm{DFL}}`$.
- The reported size N of each lane is the best size of its sweep, the one with the lowest five-seed mean cost.

Provenance:

- The init seed of a run appears only in its folder name. All 440 seed snapshots record `DEFAULT_INIT_SEED` 42, because a single-seed run writes its snapshot before it applies the seed.
- `experiments/<variant>/<lane>/config_snapshot.json` is a copy of a DFL snapshot and does not describe the SAA or SDDiP runs. It says `appsi_highs`, `SAA_MIP_GAP` 0.05 and `SAA_TIME_LIMIT_S` 172800, while SAA ran on Gurobi with a gap of 0.10 and a limit of 419,760 s (see `solver_metadata`). The SDDiP settings are not stored in any file, but the [SDDiP](#sddip) subsection lists them.
- Snapshots do not record command-line values other than the scale, cost bases, budget fraction and seeds. For example, a run with a different `--load-shedding-cost` still records 1000.
- Folder names are not network names: the $`s = 55`$ lane is `--network case14 --repair-base-scale 55`.
- The exact-DP files were produced with Python 3.13.3 on macOS. A rerun with the Python 3.11 environment of the repository gives the same results.

SAA and SDDiP fields:

- In `SAA/solution.json`, `objective_value` is the in-sample objective of the tree (1,405,197 on the unbudgeted 5-bus network), not the reported SAA cost (`out_of_sample_evaluation.mean_cost`, 2,028,014). The `costs` fields other than `total` are placeholders, and `solver_metadata.n_scenarios` is unused in tree mode. The termination `optimal` of the two 5-bus runs means that the 10% gap target was reached.
- In `sddip_solution.json`, `upper_bound` and `gap` come from a single training forward pass, and the stored `sddip_convergence.png` labels that curve "Upper Bound". The paper uses the policy mean rebuilt from the stored averages. SDDiP is simulated on its own stagewise distribution (exact on the 5-bus network, 256 sampled outcomes per stage on case14), not on the common test set.
- The two 5-bus SDDiP files lack `convergence.times` and used a longer training limit than the four case14 files (419,760 s against 394,560 s).

Timing fields:

- `timing.data_generation_s` in the paper lanes (at most 4.43 s) is the time to load the cached labels. The labels were solved earlier by the `dfl_scaling.sh` runs, which are not included. Real label-generation times appear only in the sensitivity files.
- The `NearSightedMPC` entries, including `eval_time_s` and `mean_time_per_scenario_s`, were computed once per lane and then read from the MPC cache, so they are the same in all files of a lane. `timing.evaluation_s` does not include MPC.
- SAA `solve_time` includes building the tree and the model, which is why the case14 runs exceed the 419,760 s solver limit by about 530 to 545 s.
- The SAA rollout LPs were solved with Gurobi, because the SAA jobs ran with `SOLVER=gurobi`. All other rollouts used SciPy's HiGHS. Rollout times measured on a laptop are lower than the stored cluster times.

Reproducibility:

- Re-evaluating a stored `model.pt` gives the same per-scenario costs as those stored ([command above](#loading-the-files-in-python)).
- Unbudgeted training is deterministic given the seeds. The 5-bus point at $`s = 100`$ of the sensitivity sweep and the unbudgeted 5-bus run at $`N = 120{,}000`$ with init seed 42 were trained in separate jobs and have identical per-scenario costs. Other hardware may give different results.
- Budgeted training is not repeatable, because its Monte Carlo perturbations come from NumPy's unseeded global random state.

Tools that write into this folder:

- `aggregate_multiseed.py` rewrites the aggregated `evaluation.json` files under `--root` (default `experiments/paper_unbudgeted/dfl_scaling_multiseed`) and, with `--plot`, the `convergence_multiseed.png` files. On the stored seed files, its output matches the stored files.
- `scripts/sensitivity_plot.py` rewrites `experiments/sensitivity/sensitivity_curve.txt` and its plots with its default `--out`.
- `compare_methods.py` and `run_comparison_folders.py` add `comparison/` folders, and `compare_methods.py` also writes `repair_heatmap_<method>.png` into the DFL and SDDiP folders it reads.
- `inject_mpc_baseline.py` rewrites the `evaluation.json` of each folder it is given and keeps a `.bak` copy.
- `scripts/slurm/exact_dp.sh` writes to `experiments/exact_dp_toy/run_<jobid>/`, not to the two folders that `paper/reproduce_paper_numbers.py` reads.

Point these tools at a copy of the folder unless you mean to change the stored results.
