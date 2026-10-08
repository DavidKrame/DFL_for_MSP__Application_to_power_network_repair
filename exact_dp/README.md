# Exact dynamic program for the 5-bus network

[`exact_dp_toy.py`](exact_dp_toy.py) solves the repair problem on the 5-bus network (`toy` in the code and in the result files) exactly, by backward induction over all states. It returns the optimal expected cost, an optimal repair policy for each stage and availability pattern, and the complete value function. It solves both variants of the experiments, without a repair budget and with a per-stage budget $`(\varphi = 0.3)`$, in about 2 s on a laptop.

| Variant | Exact optimal expected cost (USD) |
|---|---:|
| Unbudgeted | 1,382,625.16 |
| Budgeted, $`B = 85{,}500`$ USD per stage | 1,493,318.65 |

The script imports the network, the failure model, the repair costs and the stage DC-OPF from the repository root, so it solves the same problem that DFL, the baselines, SAA and SDDiP address. Its optimum is the reference for the other methods on this network. The DFL policy lies 1.7% (unbudgeted) and 1.8% (budgeted) above it.

**Contents**

1. [The model](#the-model)
2. [Why the recursion is exact](#why-the-recursion-is-exact)
3. [Running it](#running-it)
4. [Output](#output)
5. [Stored results](#stored-results)
6. [Comparison with the other methods](#comparison-with-the-other-methods)
7. [Scope and pitfalls](#scope-and-pitfalls)

## The model

The default settings of the script are those of the 5-bus experiments.

| Item | Value |
|---|---|
| Network | Built-in 5-bus network: 2 generators and 5 lines, the 7 failure-prone components G1, G2, L1 to L5. Three loads, 180 MW in total |
| Generators | G1 at bus 1: 100 MW, 20 USD/MWh. G2 at bus 4: 150 MW, 25 USD/MWh |
| Lines: susceptance $`b`$ and flow limit | L1, buses 1-2: 10 MW/rad, ±80 MW. L2, buses 1-3: 8 MW/rad, ±70 MW. L3, buses 3-4: 12 MW/rad, ±90 MW. L4, buses 2-4: 9 MW/rad, ±75 MW. L5, buses 4-5: 11 MW/rad, ±85 MW |
| Loads | 60 MW at bus 2, 50 MW at bus 3 and 70 MW at bus 5 |
| Horizon and repair delay | $`T = 14`$ stages. $`\tau = 1`$: a repair ordered at stage $`t`$ is in service at stage $`t + 1`$ |
| Failures | At each stage, each operational component $`k`$ fails independently with probability $`\lambda^t[k]`$, which peaks at $`t^* = 1`$ and then decays. The model seed 42 fixes each component's vulnerability class and peak probability $`p_{\max}[k]`$ |
| Peak failure probabilities $`p_{\max}[k]`$ (model seed 42) | Elevated class: G1 0.844, G2 0.870, L1 0.804, L3 0.822, L4 0.826. Standard class: L2 0.676, L5 0.618. With $`t^* = 1`$ they equal the stage-1 probabilities $`\lambda^1[k]`$, the first column of `generator_failure_probabilities` and `line_failure_probabilities` under `model` in the output |
| Decision | Which failed components to repair, after observing the failures of stage $`t`$ |
| Stage cost | DC-OPF cost of the current availability (generation cost plus 1,000 USD per MWh of load shed), plus the cost of the repairs ordered |
| Repair costs $`\beta[k]`$ at $`s = 100`$ (USD) | G1 44,000, G2 66,000; L1 to L5 35,000, 30,625, 39,375, 32,812.50, 37,187.50. They are proportional to capacity, with a mean of $`550\,s`$ per generator and $`350\,s`$ per line |
| Budget (budgeted variant) | At most $`B = \varphi \sum_{k=1}^{7} \beta[k] = 0.3 \times 285{,}000 = 85{,}500`$ USD of repairs per stage |
| Objective | Expected total cost over the 14 stages, starting with all components operational |

[`scripts/make_readme_figures.py`](../scripts/make_readme_figures.py) draws the network figure of the top-level README from the same data in [`network_data.py`](../network_data.py).

## Why the recursion is exact

1. State. With $`\tau = 1`$, each component has one of three statuses at the start of a stage: operational, failed with no repair ordered, or failed with a repair that was ordered at the previous stage and completes now. Repairs are ordered only for failed components and stay in progress for one stage, so these three statuses describe both the availability and the repair pipeline. Seven components give $`3^7 = 2{,}187`$ states per stage.
2. Outcomes. The failures of a stage hit only operational components. A component whose repair completes at this stage comes back without being exposed to them, and a failed component without a repair stays failed. After the failures, the network is described by a binary availability mask $`a \in \{0,1\}^7`$, one of $`2^7 = 128`$. For each state, the script enumerates all failure patterns of its operational components with their Bernoulli probabilities, so the expectations are computed exactly, without sampling.
3. Stage cost. The loads and generation costs do not vary across stages, so the DC-OPF cost $`Q^0(a)`$ depends only on the availability mask $`a`$. The script solves the 128 LPs once, with the same `StagewiseOPF` that the DFL and baseline rollouts use, and reuses the solutions at all stages.
4. Recombination. Failures are independent across stages and components, so the future depends on the history only through the current state. Histories that lead to the same state are merged, and the recursion keeps one value per stage and state. A scenario tree of the same problem would have $`128^{14} \approx 3.2 \times 10^{29}`$ leaves.

5\. Recursion. With $`\mathcal{V}_{15} \equiv 0`$ and $`t = 14, \dots, 1`$,

```math
\begin{aligned}
\mathcal{V}_t(x) &= \sum_{a \in \mathcal{A}(x)} P_t(a \mid x)\, \mathcal{V}^{\mathrm{post}}_t(a),\\[6pt]
\mathcal{V}^{\mathrm{post}}_t(a) &= Q^0(a) + \min_{r \in \mathcal{Y}(a)} \Big(\beta^\top r + \mathcal{V}_{t+1}\big(\mathrm{next}(a, r)\big)\Big),\\[6pt]
\mathcal{A}(x) &= \big\{\, a \in \{0,1\}^7 \,:\, a[k] = 0 \text{ for } k \in D(x),\ \ a[k] = 1 \text{ for } k \in R(x) \,\big\},\\[6pt]
\mathcal{Y}(a) &= \big\{\, r \in \{0,1\}^7 \,:\, r \le \mathbf{1} - a,\ \ \beta^\top r \le B \,\big\},\\[6pt]
P_t(a \mid x) &= \prod_{k \in U(x)} \big(\lambda^t[k]\big)^{1 - a[k]}\,\big(1 - \lambda^t[k]\big)^{a[k]},
\end{aligned}
```

where $`x`$ is the state of step 1, equivalent to the pair $`(a^{t-1}, r^{t-1})`$ of the previous availability and repairs, and $`a`$ is the availability mask after the failures. $`\mathcal{V}_t(x)`$ is the optimal expected cost from stage $`t`$ to the end in state $`x`$, before the failures of stage $`t`$, and $`\mathcal{V}^{\mathrm{post}}_t(a)`$ the optimal expected cost from the same stage on, once these failures have produced mask $`a`$. $`U(x)`$, $`D(x)`$ and $`R(x)`$ are the sets of components that are operational, failed without a repair, and failed with a pending repair in $`x`$. $`\mathcal{A}(x)`$ holds the $`2^{\lvert U(x) \rvert}`$ masks that the failures of stage $`t`$ can produce from $`x`$, and $`P_t(a \mid x)`$ is the probability of mask $`a`$. $`Q^0(a)`$ is the stage cost of step 3, $`r`$ the 0/1 vector of the failed components to repair, $`\beta^\top r`$ the cost of these repairs, and $`\mathrm{next}(a, r)`$ the next state, the pair $`(a, r)`$, in which available components are operational, repaired ones pending and the others failed. Without a budget the condition $`\beta^\top r \le B`$ is dropped, and the minimum runs over all subsets of the failed components. The optimum $`\mathcal{V}^\star`$ is $`\mathcal{V}_1`$ at the state in which all components are operational.

The [decision rule of the DFL policy](../README.md#decision-rule) writes the cost-to-go as a function of the availability after the failures, $`\mathcal{V}_{t+1}(a^{t+1})`$, which is $`\mathcal{V}^{\mathrm{post}}_{t+1}(a^{t+1})`$ here. With $`\tau = 1`$, the value $`\mathcal{V}_{t+1}`$ of the state $`(a^t, r^t)`$ reached after the repairs of stage $`t`$ is the expectation that the decision rule approximates, given the failure events $`\xi^1, \dots, \xi^t`$ observed so far and the repairs $`r^t`$:

```math
\mathcal{V}_{t+1}\big((a^t, r^t)\big) \;=\; \mathbb{E}\Big[\,\mathcal{V}^{\mathrm{post}}_{t+1}\big(a^{t+1}\big) \;\Big|\; \xi^1,\dots,\xi^t,\ r^t\Big]
```

The state and action sets are finite and each decision uses only the information available at its stage, so this backward induction gives the minimum expected cost over all non-anticipative policies. The only approximation is the numerical tolerance of the LP solver.

6. Checks. The script then propagates the state distribution forward under the optimal policy, again exactly, and recomputes the expected cost. It stops with an error if the forward and backward values differ by more than $`10^{-7} \times \max(1, \lvert \mathcal{V}^\star \rvert)`$, if a transition distribution does not sum to one, if the policy repairs an operational component, or if it exceeds the budget. In the stored results the two values differ by $`4.7 \times 10^{-10}`$ (unbudgeted) and $`2.3 \times 10^{-10}`$ (budgeted).

The computation is small: 128 LPs, 229,376 transition terms, and 30,618 evaluations of a repair set without the budget or 14,784 with it. Without the budget, each stage evaluates every subset of the failed components of each mask. A mask with $`j`$ available components has $`2^{7-j}`$ such subsets, so a stage makes $`\sum_{j=0}^{7} \binom{7}{j}\, 2^{7-j} = (2 + 1)^7 = 3^7 = 2{,}187`$ evaluations, and the 14 stages make 30,618. Each variant takes less than a second.

## Running it

### Locally

The wrapper [`run_local.sh`](run_local.sh) calls `python3` from `PATH`. Run it from the repository root, with the Python environment active:

```bash
bash exact_dp/run_local.sh
```

This solves both variants with the settings of the experiments (seed 42, $`T = 14`$, $`s = 100`$, $`\varphi = 0.3`$, SciPy's HiGHS for the stage LPs) and writes

```text
exact_dp/results/local/unbudgeted/exact_dp_solution.json
exact_dp/results/local/budgeted/exact_dp_solution.json
```

`exact_dp/results/` is ignored by git. Arguments after the script name are appended to the command and override its settings, for example:

```bash
bash exact_dp/run_local.sh --variant unbudgeted --output-dir /tmp/exact_dp
```

The banner line `Output :` of the wrapper always names `exact_dp/results/local`. The line `Output base` printed by the Python script shows the folder actually used.

The Python script can also be called directly. Without `--output-dir` it writes to `exact_dp/results/` in the repository, regardless of the working directory:

```bash
python exact_dp/exact_dp_toy.py --output-dir /tmp/exact_dp
```

### On a SLURM cluster

[`scripts/slurm/exact_dp.sh`](../scripts/slurm/exact_dp.sh) runs the same command as a batch job. Submit it from the repository root, which holds the `.venv` created by `scripts/slurm/install_env.sh` and the `logs/` folder:

```bash
sbatch scripts/slurm/exact_dp.sh
```

- The job requests 1 CPU, 4 GB and 30 minutes on the partition `rome` of Snellius. The computation itself takes seconds. On another cluster, adapt the partition and the `module load` lines (see [`scripts/README.md`](../scripts/README.md)).
- Results go to `experiments/exact_dp_toy/run_<jobid>/{unbudgeted,budgeted}/exact_dp_solution.json`, and the log to `logs/exact_dp_toy_<jobid>.out` and `.err`.
- The stage LPs run on SciPy's HiGHS. Unlike the experiment scripts, this job has no `SOLVER` switch and loads no Gurobi module.
- Arguments after the script name are passed on, as with `run_local.sh`, for example `sbatch scripts/slurm/exact_dp.sh --variant budgeted`.
- The script reads `SLURM_JOB_ID` and stops under plain `bash`. Use `run_local.sh` outside SLURM.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--variant` | `both` | `unbudgeted`, `budgeted` or `both` |
| `--seed` | 42 | Model seed. It draws each component's vulnerability class and $`p_{\max}[k]`$, and so fixes the failure probabilities. The DP itself samples nothing |
| `--horizon` | 14 | Number of stages $`T`$ |
| `--repair-base-scale` | 100 | Repair-cost scale $`s`$: mean repair cost $`550\,s`$ per generator and $`350\,s`$ per line |
| `--budget-fraction` | 0.3 | $`\varphi`$ of the budgeted variant, in $`(0, 1]`$, with $`B = \varphi \sum_{k=1}^{7} \beta[k]`$ |
| `--lp-solver` | `highs` | Backend of the 128 stage LPs: `highs` (SciPy) or `gurobi` (needs gurobipy, which is not in the requirements) |
| `--output-dir` | `exact_dp/results` | Results go to `<output-dir>/<variant>/exact_dp_solution.json` |
| `--sddip-unbudgeted`, `--sddip-budgeted` | the stored 5-bus SDDiP results (below) | SDDiP result to compare with |
| `--no-sddip-comparison` | off | Skip the SDDiP comparison |

The two default SDDiP results are `experiments/paper_unbudgeted/toy/SDDiP/sddip_solution.json` and `experiments/paper_budget0.3/toy/SDDiP/sddip_solution.json`. `--help` does not show these defaults, nor those of `--seed`, `--horizon`, `--budget-fraction` and `--output-dir`, which are the values in the table above.

## Output

Each variant writes `<output-dir>/<variant>/exact_dp_solution.json`, about 1.6 MB:

| Key | Content |
|---|---|
| `schema`, `exactness_scope` | Format tag `exact_dp_toy_v1`, and the scope of the exactness claim |
| `model` | Variant, seed, horizon, repair delay, scale $`s`$, $`\varphi`$ and $`B`$, component order (G1, G2, L1 to L5), repair costs, the failure probabilities $`\lambda^t[k]`$ (one row per component, one column per stage) and the LP solver |
| `state_space` | Status codes (0 operational, 1 failed with no repair pending, 2 failed with a repair pending), the base-3 state index $`\sum_{k} 3^k x[k]`$ and binary mask $`\sum_{k} 2^k a[k]`$ encodings, and the counts of 2,187 states and 128 masks per stage. Component 0 is the least significant digit or bit |
| `solution` | `exact_optimal_value`, its split into `expected_opf_cost` (generation and load shed) and `expected_repair_cost`, the value recomputed by the forward pass, and the difference between the two |
| `work` | Numbers of LPs, repair sets evaluated and transition terms, and the run times |
| `validation` | Four flags, all `true`, because the script stops with an error before writing if a check fails |
| `forward_evaluation` | Expected costs under the optimal policy and, in `stage_metrics`, for each stage: the number of reachable states, expected OPF and repair cost, expected generation, load served, availability and repairs |
| `optimal_policy` | The optimal repair set (`repair_mask`, `repair_bits`) and its cost for each of the 14 stages and each of the 128 availability masks |
| `opf_by_availability` | The 128 stage DC-OPF solutions, with the cost, generation and load served for each availability mask |
| `value_function_by_stage` | $`15 \times 2{,}187`$ values. Row $`t`$ (0-based) is the optimal expected cost $`\mathcal{V}_{t+1}`$ from stage $`t + 1`$ to the end for each state. The last row is zero, and row 0 at state 0 (all operational) is the optimum |
| `saved_sddip_comparison` | The SDDiP lower bound, the reconstructed policy mean, their distances from the optimum, the last forward-pass cost, iterations and cuts. Absent with `--no-sddip-comparison`, and `"available": false` if the file does not exist |
| `run_environment` | Python version, platform and output path of the run |

For each variant, the script also prints the settings, the optimum and its split, the work counts, a table of the optimal policy's expected costs, load served and number of repairs per stage, and the SDDiP comparison. When it solves both variants, it ends with the effect of the budget on the optimum: +110,693.48 USD, or +8.0%.

<details>
<summary>Terminal output for the unbudgeted variant</summary>

```text
====================================================================================================
EXACT DYNAMIC PROGRAM — TOY NETWORK
====================================================================================================
Variant                    : unbudgeted
Seed                       : 42
Horizon / repair delay     : 14 / 1
Repair-base scale          : 100
Repair budget per stage    : none
Maximum DP states/stage    : 2,187
Failure outcomes/stage     : 128
Exact optimal value        : $1,382,625.164865
  Expected OPF cost        : $945,380.165026
  Expected repair cost     : $437,244.999840
Backward/forward check     : 4.657e-10
OPFs solved                : 128
Repair actions evaluated   : 30,618
Transition contributions   : 229,376
OPF precomputation time    : 0.148s
DP recursion time          : 0.057s
Total time                 : 0.653s

Exact optimal-policy expectations by stage
----------------------------------------------------------------------------------------------------
 stage | pre-states | post-masks | next-states |      OPF cost | repair cost |   total cost | served MW | E[# repairs]
----------------------------------------------------------------------------------------------------
     1 |          1 |        128 |         128 |    172,335.26 |    8,621.39 |   180,956.65 |      7.85 |       0.1902
     2 |        128 |        128 |         128 |    171,517.01 |    2,199.47 |   173,716.48 |      8.70 |       0.0485
     3 |        128 |        128 |         128 |    177,135.44 |  165,420.53 |   342,555.97 |      2.94 |       3.7569
     4 |        128 |        128 |         128 |     40,610.83 |    8,414.22 |    49,025.05 |    142.96 |       0.2085
     5 |        128 |        128 |         128 |    119,222.48 |  139,999.12 |   259,221.59 |     62.33 |       3.3857
   ...
    14 |        128 |        128 |         128 |     17,328.75 |        0.00 |    17,328.75 |    166.74 |       0.0000

Comparison with saved SDDiP artifact
----------------------------------------------------------------------------------------------------
Saved artifact             : experiments/paper_unbudgeted/toy/SDDiP/sddip_solution.json
SDDiP training LB           : $1,024,482.105123
Exact optimum - LB          : $358,143.059742
Relative LB shortfall       : 25.9031%
LB validity check           : PASS (LB <= exact optimum)
Reconstructed 500-sim mean  : $1,584,475.716522
Mean - exact optimum        : $201,850.551657
Relative mean excess        : 14.5991%
Legacy 'upper_bound' field  : $1,383,222.156971 (single training forward pass)
Iterations / cuts           : 782 / 10,166
Training forward costs read : 782
Caution                    : no variance/CI for the reconstructed mean
```

The columns `pre-states` and `next-states` count the states that the optimal policy reaches. The value function covers all 2,187. The "Legacy 'upper_bound' field" is the cost of the single forward pass of the last SDDiP training iteration, printed for reference only.

</details>

<details>
<summary>Looking up the optimal policy in Python</summary>

```python
import json

sol = json.load(open("experiments/exact_dp_toy/unbudgeted/exact_dp_solution.json"))
names = sol["model"]["component_order"]  # G1, G2, L1, ..., L5; component k is bit k of a mask

def optimal_repairs(stage, failed):
    """Optimal repair set and its cost at a 1-based stage, given the failed components."""
    mask = sum(1 << k for k, name in enumerate(names) if name not in failed)
    action = sol["optimal_policy"][stage - 1]["actions"][mask]
    return [n for n, bit in zip(names, action["repair_bits"]) if bit], action["repair_cost"]

print(sol["solution"]["exact_optimal_value"])  # 1382625.16...
print(optimal_repairs(3, {"G2", "L4"}))         # (['G2', 'L4'], 98812.5)
```

</details>

## Stored results

The results used in the paper are stored in

```text
experiments/exact_dp_toy/unbudgeted/exact_dp_solution.json
experiments/exact_dp_toy/budgeted/exact_dp_solution.json
```

They were written by `exact_dp_toy.py` with its default settings, the same as in `run_local.sh` and `exact_dp.sh`, and include the comparison with the stored toy SDDiP runs.

| | Unbudgeted | Budgeted |
|---|---:|---:|
| Exact optimal expected cost (USD) | 1,382,625.16 | 1,493,318.65 |
| Expected OPF cost: generation and load shed (USD) | 945,380.17 | 1,081,932.42 |
| Expected repair cost (USD) | 437,245.00 | 411,386.23 |
| Repair budget per stage (USD) | none | 85,500 |
| Repair sets evaluated | 30,618 | 14,784 |
| Backward/forward difference (USD) | $`4.7 \times 10^{-10}`$ | $`2.3 \times 10^{-10}`$ |
| Stored run time (s) | 0.88 | 0.96 |

The budget raises the optimal expected cost by 110,693.48 USD (8.0%). The stored files were written with Python 3.13 on macOS (see `run_environment`). A rerun with the Python 3.11 environment of `requirements-lock.txt` reproduces their optima, value functions and policies exactly.

Two scripts read these files:

- [`paper/reproduce_paper_numbers.py`](../paper/reproduce_paper_numbers.py) checks the exact optima and how far the DFL and SDDiP results lie from them (see [`paper/README.md`](../paper/README.md));
- [`scripts/quick_check.sh`](../scripts/quick_check.sh) runs the DP again and compares the new optima with the stored ones (relative tolerance $`10^{-6}`$).

Neither `run_local.sh` nor `exact_dp.sh` writes into these two folders. To make the paper scripts use a new run, copy its two `exact_dp_solution.json` files over them.

## Comparison with the other methods

The table puts the exact optimum next to the results of the other methods on the 5-bus network, all taken from the files in `experiments/`. Costs are rounded to the dollar. The percentages give $`(\bar c - \mathcal{V}^\star)/\mathcal{V}^\star`$ for the cost $`\bar c`$ of each row and are computed from the unrounded values.

| Method | Unbudgeted (USD) | vs. optimum | Budgeted (USD) | vs. optimum | Evaluated on |
|---|---:|---:|---:|---:|---|
| Exact DP | 1,382,625 | 0.0% | 1,493,319 | 0.0% | exact expectation |
| DFL, five-seed mean (n_train 15,000 / 60,000) | 1,406,419 | +1.7% | 1,520,472 | +1.8% | 500 test scenarios |
| Near-sighted MPC ($`W = 3`$) | 1,445,096 | +4.5% | 2,403,624 | +61.0% | 500 test scenarios |
| AlwaysRepair | 1,509,596 | +9.2% | 1,940,415 | +29.9% | 500 test scenarios |
| SDDiP, final policy | 1,584,476 | +14.6% | 1,651,543 | +10.6% | 500 SDDiP simulations |
| SAA, nearest-child rollout | 2,028,014 | +46.7% | 2,069,534 | +38.6% | 500 test scenarios |
| NoRepair | 2,511,488 | +81.6% | 2,511,488 | +68.2% | 500 test scenarios |
| SDDiP lower bound | 1,024,482 | -25.9% | 1,296,240 | -13.2% | bound |
| SAA tree objective | 1,405,197 | +1.6% | 1,515,653 | +1.5% | 16,384 sampled tree paths, in-sample |

`paper/reproduce_paper_numbers.py` checks the exact optima and the distances of DFL and SDDiP from them. The other percentages follow from the same files.

Notes on the table:

- The exact value is an expectation over the complete failure distribution. The other policy costs are sample means, so the differences are descriptive, not certified optimality gaps. DFL, Near-sighted MPC, AlwaysRepair, SAA and NoRepair are evaluated on the same 500 test scenarios, generated with seed 1000.
- The SDDiP policy mean comes from 500 simulations that SDDiP draws from its own stagewise distribution. On this network, that distribution enumerates all 128 outcomes per stage and is the true failure law. Only stagewise averages of these simulations were kept. Because the stage cost is affine in these averages, `reconstruct_saved_sddip_mean` in `exact_dp_toy.py` rebuilds the exact mean from them, but not a variance. The SDDiP runs stopped at their time limit after 782 (unbudgeted) and 755 (budgeted) iterations. Their lower bounds lie below the exact optimum, as they must, but far from it.
- The `upper_bound` field of `sddip_solution.json` (1,383,222 unbudgeted, 1,727,002 budgeted) is the cost of one sampled trajectory, the forward pass of the last training iteration. It is not a bound and is not used above.
- The SAA tree objective is the in-sample objective of the MILP incumbent over the sampled tree. It is close to the exact optimum but is not the cost of a deployable policy. The nearest-child rollout applies the same solution to the test scenarios, following at each stage the child of the tree closest to the realized failures. This rollout costs 46.7% (unbudgeted) and 38.6% (budgeted) more than the exact optimum.
- With the budget, AlwaysRepair repairs the cheapest failed components that fit within $`B`$.

## Scope and pitfalls

- Only the built-in 5-bus network and $`\tau = 1`$ are supported. The script always builds `toy`, and its state encoding assumes seven components and one-stage repairs. On case14, with 25 failure-prone components, the same construction would need $`3^{25} \approx 8.5 \times 10^{11}`$ states and $`2^{25} \approx 3.4 \times 10^{7}`$ outcomes per stage. The exact DP is a reference for the small network, not a method for larger ones.
- The model is the DC-OPF model of the experiments, not an AC power flow.
- The default SDDiP files correspond to the settings of the experiments and do not follow `--seed`, `--horizon`, `--repair-base-scale` or `--budget-fraction`. The SDDiP result files do not record these settings, so the script cannot detect a mismatch. After changing any of them, pass `--no-sddip-comparison`, or SDDiP results of a run with the same settings.
- The repair-cost bases 550 and 350 are written into `exact_dp_toy.py` itself, as in the other entry points, so changing them in `config.py` alone does not change the DP. Use `--repair-base-scale` to change the repair costs.
- `run_local.sh` uses the first `python3` on `PATH`. Activate the environment of `requirements-lock.txt` first, or another interpreter is used.

See the [top-level README](../README.md) for the installation and the other methods, and [`experiments/README.md`](../experiments/README.md) for the other stored results.
