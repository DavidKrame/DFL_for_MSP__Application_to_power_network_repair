# Tests

The unit tests check the parts of the code that all methods in this repository share: the failure-probability model, scenario sampling, the availability dynamics with repair delay, the stage DC-OPF with line outages, the derated DC-OPF of the Near-sighted MPC lookahead, and the scenario tree of the multistage SAA. Apart from one test on a hand-built 3-bus network, they run on the built-in 5-bus network (`toy` in the code), so they need no internet connection, no Julia and no Gurobi. The suite has 118 tests and takes about 15 s on a laptop.

The tests do not rerun experiments and do not read the stored results in `experiments/`. The scripts that do are listed under [Other checks](#other-checks).

**Contents**

1. [Running the tests](#running-the-tests)
2. [Expected result and run time](#expected-result-and-run-time)
3. [Optional packages and solvers](#optional-packages-and-solvers)
4. [What each file checks](#what-each-file-checks)
5. [What a pass guarantees, and what it does not](#what-a-pass-guarantees-and-what-it-does-not)
6. [Extending the tests](#extending-the-tests)
7. [Other checks](#other-checks)

## Running the tests

1. Create a Python 3.11 environment with the pinned package versions. This needs a Python 3.11 interpreter available as `python3.11`. The [top-level README](../README.md#python-environment) covers installation in more detail, including what to do when there is none:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-lock.txt
   ```

2. Unset the solver variables, which the tests read like every other script (see [Solver environment variables](#solver-environment-variables)):

   ```bash
   unset ED_OPF_SOLVER ED_OPF_MIP_SOLVER ED_OPF_LP_SOLVER ED_OPF_SDDIP_SOLVER
   ```

3. Run the suite from the repository root:

   ```bash
   python -m pytest tests
   ```

   The run ends with `118 passed`, or with `117 passed, 1 skipped` when GLPK is not installed.

To run part of the suite, again from the repository root:

| To run | Command |
|---|---|
| all tests, one line per test | `python -m pytest tests -v` |
| the 112 fast tests (about 2 s) | `python -m pytest tests -m "not slow"` |
| the 6 slow tests, which solve small MIPs | `python -m pytest tests -m slow` |
| one file | `python -m pytest tests/test_state_dynamics.py` |
| one test class | `python -m pytest tests/test_state_dynamics.py::TestRepairDelay` |
| one test | `python -m pytest tests/test_bigM_validity.py::test_bigM_matches_active_topology` |
| one case of a parametrized test | `python -m pytest "tests/test_opf_constraints.py::TestOPFConstraints::test_f_line_flow_bounds[line0_down]"` |
| the tests whose names match an expression | `python -m pytest tests -k "bigM or derated"` |
| everything up to the first failure | `python -m pytest tests -x` |
| nothing, only list the test IDs | `python -m pytest tests --collect-only -q` |

Test IDs have the form `file::Class::test[case]`, as printed by `-v` and `--collect-only -q`. Quote an ID that contains square brackets, so the shell does not read the brackets as a file pattern.

`tests/conftest.py` puts the repository root on the import path, so the suite also runs from another folder when given its path (`python -m pytest <repo>/tests`). The tests write nothing into the checkout except the `.pytest_cache/` and `__pycache__/` folders that pytest and Python create, which git ignores. `scripts/quick_check.sh` adds `-p no:cacheprovider` and sets `PYTHONDONTWRITEBYTECODE=1`, so it does not create these folders either.

## Expected result and run time

On an Apple Silicon laptop with Python 3.11.16 and `requirements-lock.txt`, pytest reports `118 passed` after 13 to 15 s. The six slow tests account for about 11 s of that, the other 112 tests for about 2 s. The first run after an installation can take longer while Python loads the newly installed packages.

| File | Tests | Marked slow | Time |
|---|---:|---:|---:|
| `test_failure_model.py` | 8 | 0 | < 0.1 s |
| `test_scenario_generator.py` | 8 | 0 | 0.3 s |
| `test_state_dynamics.py` | 12 | 0 | < 0.1 s |
| `test_opf_constraints.py` | 65 | 0 | < 0.1 s |
| `test_integration.py` | 9 | 1 | 4.4 s |
| `test_bigM_validity.py` | 2 | 0 | 1.0 s |
| `test_derated_mpc_opf.py` | 2 | 0 | 0.7 s |
| `test_saa_tree.py` | 12 | 5 | 6.5 s |
| Total | 118 | 6 | about 13 s |

The times are sums of the per-test durations. Python start-up and test collection add one to two seconds. The slow test in `test_integration.py` (the GLPK solve, 3.8 s) is the longest single test.

`test_bigM_validity.py::test_bigM_is_valid_and_reports_usage` prints a table of the big-M constants even without `-s`, so a normal run looks like this:

<details>
<summary>Output of <code>python -m pytest tests</code></summary>

```text
collected 118 items

tests/test_bigM_validity.py .
--- big-M diagnostics (toy network, 300 random outage states) ---
line       b_e      M_e = 2*pi*b_e     f_max
  0      10.000        62.8319      80.00
  1       8.000        50.2655      70.00
  2      12.000        75.3982      90.00
  3       9.000        56.5487      75.00
  4      11.000        69.1150      85.00

failed-line samples                : 729
max |b*dtheta| / M_e on failed lines : 1.0000  (1.0 = angle box saturated; big-M valid iff <= 1)
failed-line big-M ever binding       : 97 / 729
max |f| on failed lines              : 0.00e+00  (should be ~0)
max |f - b*dtheta| on live lines     : 1.42e-14  (should be ~0: DC law binds)
.                                           [  1%]
tests/test_derated_mpc_opf.py ..                                         [  3%]
tests/test_failure_model.py ........                                     [ 10%]
tests/test_integration.py .........                                      [ 17%]
tests/test_opf_constraints.py .......................................... [ 53%]
.......................                                                  [ 72%]
tests/test_saa_tree.py ............                                      [ 83%]
tests/test_scenario_generator.py ........                                [ 89%]
tests/test_state_dynamics.py ............                                [100%]

============================= 118 passed in 13.20s =============================
```

</details>

The table is informational. A ratio of 1.0 on failed lines means that some solutions put the endpoint angles of a failed line at opposite ends of the range $`[-\pi, \pi]`$ and so use the full constant $`M_j = 2\pi b_j`$ of line $`j`$ (column `M_e` of the printed table). The test fails if any ratio exceeds 1.

## Optional packages and solvers

| Package or solver | Needed by | If it is missing |
|---|---|---|
| NumPy, SciPy and pytest | every test | nothing runs |
| Pyomo with HiGHS (`highspy`, Pyomo solver `appsi_highs`) | the 5 slow tests in `test_saa_tree.py` | without Pyomo, these tests are skipped |
| PyTorch | `test_saa_tree.py::TestTreeSAASolve::test_closed_loop_evaluation_runs`, because the SAA rollout reuses the state update of `Pseudo_DFL_solver.py`, which imports PyTorch | the test fails with `NameError: name 'nn' is not defined` |
| Pyomo and GLPK (`glpsol` on the `PATH`) | `test_integration.py::TestSAAOneSampleValid::test_saa_n1_returns_non_negative_objective` | the test is skipped (`GLPK (glpsol) is not installed`) |
| Gurobi | no test | the tests use Gurobi only when a solver variable selects it |

All Python packages in this table are in `requirements.txt` and `requirements-lock.txt`. GLPK is a separate program, used only by the one test above and by none of the experiments:

```bash
brew install glpk              # macOS with Homebrew
sudo apt install glpk-utils    # Debian and Ubuntu
```

Measured outcomes of `python -m pytest tests`:

| Environment | Result |
|---|---|
| `requirements-lock.txt` and GLPK | `118 passed` |
| `requirements-lock.txt` without GLPK | `117 passed, 1 skipped` |
| NumPy, SciPy and pytest only | `112 passed, 6 skipped` |
| any of the above, with `-m "not slow"` | `112 passed, 6 deselected` |
| Python 3.13 with other package versions, and GLPK | `118 passed` |

Only the Python 3.11 environment of `requirements-lock.txt` is validated for the whole workflow. The Python 3.13 row is given for information.

### Solver environment variables

`solver_config.py` reads `ED_OPF_SOLVER`, `ED_OPF_MIP_SOLVER`, `ED_OPF_LP_SOLVER` and `ED_OPF_SDDIP_SOLVER` once, at import, and the tests use the resolved solvers:

- The stage LP fixture `opf` in `conftest.py` uses the LP backend (SciPy's HiGHS by default). It serves the 64 availability-pattern tests in `test_opf_constraints.py` and four tests in `test_integration.py`. The closed-loop SAA rollout in `test_saa_tree.py` uses the same backend.
- The five slow tests in `test_saa_tree.py` solve their MIPs with the MIP solver (`appsi_highs` by default).
- `test_bigM_validity.py`, `test_derated_mpc_opf.py` and the 3-bus test in `test_opf_constraints.py` always use SciPy's HiGHS. The slow test in `test_integration.py` always uses GLPK.

The lock-file environment has no gurobipy. With GLPK installed, `ED_OPF_SOLVER=gurobi` ends a run in this environment with `5 failed, 45 passed, 68 errors`, reporting `No module named 'gurobipy'` and `No executable found for solver 'gurobi'`. The LP backend must be a method name of `scipy.optimize.linprog` or exactly `gurobi`. For any other name, the stage LP falls back to SciPy's `revised simplex` method. All 118 tests then still pass, with deprecation warnings, but the intended backend is never exercised.

## What each file checks

The fixtures in `tests/conftest.py` build:

- the 5-bus network, with 2 generators, 5 lines and 3 loads, so 7 components that can fail;
- its model parameters, with seed 42 and the `config.py` defaults, among them $`T = 14`$ stages, repair delay $`\tau = 1`$ and load-shedding penalty $`\alpha = 1{,}000`$ USD/MWh;
- the availability dynamics (`dynamics`);
- the stage DC-OPF LP (`opf`);
- a sample of 10,000 failure scenarios drawn with seed 42 (`large_sample`).

Tolerances are absolute unless stated otherwise.

### `test_failure_model.py` (8 tests)

The failure probability $`\lambda^t[k]`$ of component $`k`$ at stage $`t`$ follows the log-linear logit model

```math
\begin{aligned}
\operatorname{logit}\lambda^t[k] &= A[k]\ln(t + K) - B[k]\,t,
\qquad\text{that is}\qquad
\lambda^t[k] = \frac{1}{1 + (t + K)^{-A[k]}\, e^{B[k]\,t}},\\[6pt]
B[k] &= \frac{\operatorname{logit}(p_{\max}[k])}{\delta},
\qquad
A[k] = (t^* + K)\, B[k],
\qquad
\delta = (t^* + K)\ln(t^* + K) - t^*,
\end{aligned}
```

with shift $`K = 1.5`$, peak stage $`t^* = 1`$ and peak probability $`p_{\max}[k]`$ of each component (`FAILURE_K`, `FAILURE_T_STAR` and the `FAILURE_P_MAX_*` ranges in `config.py`). $`A[k]`$ and $`B[k]`$ are named as in `model_parameters.py`, and $`B[k]`$ is unrelated to the per-stage repair budget $`B`$.

| Tests | Check |
|---|---|
| `test_generators_at_stage_1`, `test_lines_at_stage_1` | the probability of each generator and line at stage 1 equals its $`p_{\max}[k]`$ to within 1e-10 |
| `test_generator_probabilities_strictly_positive`, `test_generator_probabilities_strictly_below_one` and the two line versions | all probabilities lie strictly between 0 and 1 |
| `test_generators_strictly_decreasing`, `test_lines_strictly_decreasing` | each probability decreases strictly over the 14 stages |

The monotonicity tests assume the peak at stage 1. With a later `FAILURE_T_STAR` they fail by design.

### `test_scenario_generator.py` (8 tests)

| Tests | Check |
|---|---|
| `test_gen_failures_binary`, `test_line_failures_binary` | every sampled failure indicator is 0 or 1 (50 scenarios, seed 0) |
| `test_generator_empirical_frequencies`, `test_line_empirical_frequencies` | over the 10,000 scenarios of `large_sample`, the failure frequency of each component at each stage is within 0.03 of its probability. One standard deviation is at most 0.005, so the tolerance is six standard deviations |
| `test_zero_lambda_never_produces_gen_failure`, `test_zero_lambda_never_produces_line_failure` | with all probabilities set to 0, no failure is drawn (200 scenarios) |
| `test_one_lambda_always_produces_gen_failure`, `test_one_lambda_always_produces_line_failure` | with all probabilities set to 1, every component fails at every stage (200 scenarios) |

The last four tests use a stand-in parameter object, because the model itself only produces probabilities strictly between 0 and 1.

### `test_state_dynamics.py` (12 tests)

Availability follows the repair pipeline. A component that fails stays down until a repair is ordered, and a repair ordered at stage $`t`$ puts it back in service at stage $`t + \tau`$, with $`\tau = 1`$.

| Class (tests) | Check |
|---|---|
| `TestInitialAvailability` (4) | availability at stage 1 is one minus the stage-1 failure indicator. With no failure everything is up, a generator or line that fails at stage 1 is down while the other generators stay up, and when everything fails everything is down |
| `TestRepairDelay` (5) | a failed component is still down at the stage its repair is ordered and up $`\tau`$ stages later, for generators repaired at stages 1 and 3, and for a line repaired at stage 2. Without a repair, it stays down for all 14 stages |
| `TestPersistence` (2) | without failures, every component stays up for all 14 stages. A repaired component stays up when nothing fails afterwards |
| `TestAvailabilityBinary` (1) | availability is 0 or 1 at every stage of 30 sampled scenarios (seed 17) |

### `test_opf_constraints.py` (65 tests)

At each stage the dispatch is a DC-OPF LP with load shedding (`StagewiseOPF` in `Pseudo_DFL_stage_opf.py`). Its variables are the generator outputs $`p_g`$, the served loads $`d`$, the voltage angles $`\theta`$ of all buses except the slack (reference) bus, and one explicit flow $`f[j]`$ per line $`j`$. An available line obeys the DC law $`f[j] = b_j\big(\theta[\mathrm{from}(j)] - \theta[\mathrm{to}(j)]\big)`$ within its thermal limits $`\underline{f}[j] \le f[j] \le \overline{f}[j]`$. A failed line carries zero flow and leaves its endpoint angles free, through the big-M constraint

```math
\Big\lvert\, f[j] - b_j\big(\theta[\mathrm{from}(j)] - \theta[\mathrm{to}(j)]\big) \Big\rvert \;\le\; M_j\,\big(1 - a_e[j]\big),
\qquad
M_j = 2\,\theta^{\max} b_j = 2\pi b_j,
```

where $`a_e[j]`$ is 1 for an available line and 0 for a failed one, and $`\theta^{\max} = \pi`$ bounds the voltage angles.

Eight checks run on each of eight availability patterns of the 5-bus network, which gives 64 tests. The patterns are `all_available`, `gen0_down`, `gen1_down`, `both_gens_down`, `line0_down`, `line4_down`, `gen0_and_line1_down` and `all_lines_down`. The tolerance is 1e-6 unless stated otherwise.

| Check | Asserts |
|---|---|
| `test_a_feasibility` | the LP is solved. Load shedding keeps it feasible |
| `test_b_system_power_balance` | total generation equals total served load |
| `test_c_per_bus_power_balance` | the nodal balance holds at every bus, the slack bus included |
| `test_d_generation_bounds` | $`0 \le p_g[i] \le a_g[i]\,\overline{p}_g[i]`$ for every generator $`i`$, with capacity $`\overline{p}_g[i]`$ |
| `test_e_load_shedding_bounds` | $`0 \le d[\ell] \le p_l[\ell]`$: the served load stays within the demand $`p_l[\ell]`$ |
| `test_f_line_flow_bounds` | $`a_e[j]\,\underline{f}[j] \le f[j] \le a_e[j]\,\overline{f}[j]`$. Failed lines carry zero flow, and available lines satisfy the DC law to within 1e-4 |
| `test_g_slack_bus_angle_is_zero` | the slack bus has no angle variable, so its angle is 0 by construction |
| `test_h_cost_formula_consistency` | the returned cost equals $`c^\top p_g + \alpha\,\mathbf{1}^\top (p_l - d)`$, with generation costs $`c`$ and load-shedding penalty $`\alpha = 1{,}000`$ |

The 65th test, `test_failed_cycle_line_reroutes_without_shorting_angles`, uses its own 3-bus network: a generator bus, a load bus and a transit bus joined in a cycle by three lines with ample capacity. With the direct line between generator and load failed, the full load must still be served over the other two lines, the failed line must carry no flow, and its endpoint angles must differ. A model that imposed the zero flow on $`b_j\big(\theta[\mathrm{from}(j)] - \theta[\mathrm{to}(j)]\big)`$ instead of on an explicit flow variable would force equal angles and shed the whole load.

Checks (c), (f) and (g) and the 3-bus test re-solve the LP with SciPy's HiGHS to read the full solution vector. The other checks call `StagewiseOPF.solve` on the configured LP backend.

### `test_integration.py` (9 tests, 1 slow)

| Tests | Check |
|---|---|
| `test_all_available_cost_is_non_negative` | with every component available, the stage cost is non-negative |
| `test_all_available_cost_is_deterministic` | two identical calls return the same cost, dispatch and served load (to within 1e-8) |
| `test_cost_identical_across_all_stages_with_all_up` | 14 identical calls return the same cost, so the stage LP keeps no state between calls |
| `test_saa_n1_returns_non_negative_objective` (slow) | the wait-and-see (perfect-information) SAA with one scenario, a 14-stage MIP solved with GLPK, returns a finite, non-negative objective |
| `TestDimensionConsistency` (5 tests) | the network has 2 generators and 5 lines, and the failure probabilities, sampled scenarios, availability trajectories and stage LP all use these sizes |

### `test_bigM_validity.py` (2 tests)

With voltage angles in $`[-\pi, \pi]`$, the angle difference across a line can reach $`2\pi`$. The constant $`M_j = 2\pi b_j`$ is therefore large enough never to cut off a dispatch when the line has failed, and for a line between two non-slack buses no smaller constant has this property. For a line at the slack bus, whose angle is fixed at 0, $`\pi b_j`$ would suffice. Both tests draw 300 random availability patterns of the 7 components, which cover 117 and 116 of the $`2^7 = 128`$ possible patterns.

| Test | Check |
|---|---|
| `test_bigM_matches_active_topology` | the stage cost of the big-M LP equals the cost of an independent DC-OPF on the surviving network (failed lines deleted, no big-M constraint and no flow variables), with a relative gap below 1e-6 |
| `test_bigM_is_valid_and_reports_usage` | $`M_j`$ equals $`2\pi b_j`$. Failed lines carry zero flow, and their angle term $`b_j\big(\theta[\mathrm{from}(j)] - \theta[\mathrm{to}(j)]\big)`$ never exceeds $`M_j`$ in magnitude. Available lines satisfy the DC law to within 1e-4. The test also prints the table shown under [Expected result and run time](#expected-result-and-run-time) |

### `test_derated_mpc_opf.py` (2 tests)

The Near-sighted MPC scores a repair schedule by rolling the expected failures forward, so the availability in its lookahead steps is fractional. Those steps are priced with `StagewiseOPF.solve_derated`, in which the susceptance and the thermal limit of a line both scale with its availability $`a_e[j]`$. The flow is then $`f[j] = a_e[j]\,b_j\big(\theta[\mathrm{from}(j)] - \theta[\mathrm{to}(j)]\big)`$, bounded in magnitude by $`a_e[j]\,\overline{f}[j]`$.

| Test | Check |
|---|---|
| `test_derated_matches_bigM_at_binary` | on 300 random binary patterns (114 of the 128), the derated LP and the big-M LP agree on feasibility, and on cost and total served load to within 1e-4 |
| `test_derated_flow_scales_with_availability` | on three fractional patterns, the derated flows close the nodal balance at every bus (to within 1e-4) and stay within the availability-scaled thermal limits |

### `test_saa_tree.py` (12 tests, 5 slow)

The multistage SAA solves an extensive-form MILP on a sampled scenario tree. Every node has $`q`$ children (the branching factor, argument `branching` in the code), each with conditional probability $`1/q`$, and each node carries one repair decision, which makes the policy non-anticipative.

| Test | Check |
|---|---|
| `test_node_and_leaf_counts` (3 cases) | for $`(q, T) = (2, 3)`$, $`(3, 2)`$ and $`(2, 1)`$, the tree has $`q + q^2 + \dots + q^T`$ decision nodes and $`q^T`$ leaves |
| `test_path_probabilities_sum_to_one` | the path probabilities of the leaves, and those of the nodes at each stage, sum to 1 $`(q = 2,\ T = 4)`$ |
| `test_every_decision_node_has_valid_parent` | the parent of every node exists and belongs to the previous stage |
| `test_tree_is_reproducible` | two trees built with the same seed are identical |
| `test_invalid_branching_rejected` | a branching factor of 0 raises `ValueError` |
| `test_objective_finite_and_non_negative` (slow) | the tree SAA with $`q = 2`$ and $`T = 3`$ (14 decision nodes, 8 leaves) returns a finite, non-negative objective |
| `test_non_anticipativity_shared_node_decision` (slow) | each stage-1 node stores one generator repair vector of the right length and has at least one leaf below it. Because decisions are stored per node, all paths through the node share it |
| `test_branches_can_differ` (slow) | each stage-1 node has its own repair decision |
| `test_closed_loop_evaluation_runs` (slow) | the closed-loop rollout of the tree policy on 20 test scenarios returns the mean, standard deviation, minimum, maximum and the 20 costs |
| `test_ws_mode_still_available` (slow) | the wait-and-see mode (3 scenarios) returns a finite objective and no tree policy |

The slow tests in this file use a horizon of 3 stages instead of 14, so that the tree stays small. They solve with the configured MIP solver (`appsi_highs` by default).

## What a pass guarantees, and what it does not

For the 5-bus network and the `config.py` defaults, a passing suite guarantees that:

- the failure probabilities peak at $`p_{\max}[k]`$ in stage 1, stay strictly between 0 and 1, and decrease strictly afterwards;
- sampled failures are binary, and their frequencies match the model probabilities to within 0.03 over 10,000 scenarios;
- availability follows the repair pipeline: a repair ordered at stage $`t`$ takes effect at stage $`t + 1`$ and not earlier, and a component changes state only through a failure or a completed repair;
- the stage DC-OPF satisfies every constraint listed above in eight availability patterns, a failed line carries no flow and does not tie its endpoint angles, and on 300 random patterns the big-M model gives the same cost as a DC-OPF on the surviving network;
- the derated LP of the Near-sighted MPC lookahead reduces to the big-M LP at binary availability and follows the derated flow law on three fractional patterns;
- the scenario tree has the stated size, valid parent links and probabilities that sum to 1;
- small tree and wait-and-see SAA problems solve, and the closed-loop rollout of a tree policy runs;
- the modules agree on the sizes of their arrays.

A pass says nothing about:

- Other networks. No test loads a PGLib-OPF case such as case14, so the download and the conversion of PGLib data are not tested.
- The DFL pipeline: label generation and its cache, the Fenchel-Young loss, training, the knapsack used under a repair budget, and the DFL, AlwaysRepair and NoRepair rollouts.
- The Near-sighted MPC controller beyond its derated LP: schedule enumeration and sampling, the budget filter and the result cache.
- Repair budgets. No test sets a budget fraction.
- SDDiP. No test imports the Julia bridge or SDDP.jl.
- The exact dynamic program in `exact_dp/` ([its README](../exact_dp/README.md)). `scripts/quick_check.sh` reruns it and compares the optima with the stored ones.
- Solution quality. The MIP tests check that small instances solve to a finite, non-negative objective. They do not compare any MIP solution with an independent optimum, and they never build the full tree of the experiments, with $`q = 2`$, $`T = 14`$ and $`2^{14} = 16{,}384`$ paths. The tree tests check the structure of the stored policy, not its decisions.
- The stored results and the reported numbers. `paper/reproduce_paper_numbers.py` checks those (see [Other checks](#other-checks)).
- Gurobi, unless it is selected through the [solver variables](#solver-environment-variables).
- Identical numbers across machines or solver versions. The tests compare repeated calls within one run and use tolerances everywhere else.

## Extending the tests

- Name a new file `tests/test_<topic>.py`. pytest collects functions named `test_*` and classes named `Test*`.
- Reuse the fixtures of `conftest.py`: `network`, `params`, `dynamics`, `opf` and `large_sample`.
- Mark a test that solves a MIP with `@pytest.mark.slow`. Skip it when an optional dependency is missing, as the existing slow tests do with `pytest.importorskip("pyomo")` or `shutil.which("glpsol")`.
- To make a result independent of the solver variables, pass the solver explicitly, for example `StagewiseOPF(network, params, lp_method="highs")`.

To run the checks on another network, change `create_network()` in the `network` fixture of `conftest.py` to that case, for example `create_network("case14")`. The case14 file is downloaded from PGLib-OPF into `~/.cache/pglib_opf/` on first use. Three parts of the suite hard-code the sizes of the 5-bus network and fail on a network of another size: the 64 availability-pattern tests of `test_opf_constraints.py`, `test_network_dimensions_are_correct` in `test_integration.py`, and the fractional patterns of `test_derated_flow_scales_with_availability`. With the fixture switched to case14, all other fast tests pass. `test_bigM_validity.py` and `test_derated_mpc_opf.py` build their own 5-bus network with `create_network()`. With that call changed to case14, their three tests on random binary patterns pass as well.

## Other checks

| Check | Command, from the repository root | What it covers | Time on a laptop |
|---|---|---|---|
| Installation | `python verify_installation.py` | imports the core modules, builds the 5-bus model and a 3-scenario MIP without solving it, and lists the available solvers | about 1 s |
| End to end | `bash scripts/quick_check.sh` | the installation check, this test suite, a short DFL run, a 4-scenario wait-and-see SAA solve, the exact DP compared with the stored optimum, `paper/reproduce_paper_numbers.py`, and a redraw of the figures compared with `figures/`. The option `--with-sddip` adds a 5-iteration SDDiP run, which installs Julia and SDDP.jl on first use | about a minute without `--with-sddip` |
| Reported numbers | `python paper/reproduce_paper_numbers.py` | recomputes the reported numbers from the files in `experiments/` and ends with `All checks agree with the paper.` after 112 `[ok]` checks (see [`paper/README.md`](../paper/README.md)) | about 3 s |

`scripts/quick_check.sh` runs `python3` from the `PATH`. With the environment active, this is the right interpreter. Otherwise, name the interpreter explicitly, as in `PYTHON=.venv/bin/python bash scripts/quick_check.sh`. `paper/reproduce_paper_numbers.py` and the quick check need the PGLib case14 file, which is downloaded into `~/.cache/pglib_opf/` on first use. On the cluster, the jobs in [`scripts/slurm/smoke/`](../scripts/slurm/smoke/README.md) check the installation and the solvers before the long runs.
