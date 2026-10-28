# Tests

The unit tests check the parts of the code that all methods share: the failure-probability model, scenario sampling, the availability dynamics, the stage DC-OPF with line outages, the derated DC-OPF of Near-sighted MPC and the scenario tree of SAA. Apart from one test on a hand-built 3-bus network, they run on the synthetic 5-bus network of the paper (`toy` in the code), so they need no internet connection, Julia or Gurobi. The [Tests](../README.md#tests) section of the top-level README lists what each file checks, and the name of each test states what it asserts (`python -m pytest tests -v` lists them).

## Running the tests

With the environment of the [top-level README](../README.md#python-environment) active, from the repository root:

```bash
unset ED_OPF_SOLVER ED_OPF_MIP_SOLVER ED_OPF_LP_SOLVER ED_OPF_SDDIP_SOLVER
python -m pytest tests
```

The run ends with `118 passed` after about 15 s, or with `117 passed, 1 skipped` when GLPK is not installed.

| To run | Command |
|---|---|
| the 112 fast tests (about 2 s) | `python -m pytest tests -m "not slow"` |
| the 6 slow tests, which solve small MIPs | `python -m pytest tests -m slow` |
| one file | `python -m pytest tests/test_state_dynamics.py` |
| the tests whose names match an expression | `python -m pytest tests -k "bigM or derated"` |

## Optional packages and solvers

| Package or solver | Needed by | If it is missing |
|---|---|---|
| Pyomo with HiGHS (`appsi_highs`) | the 5 slow tests in `test_saa_tree.py` | without Pyomo, these tests are skipped |
| PyTorch | `test_saa_tree.py::TestTreeSAASolve::test_closed_loop_evaluation_runs`, because the SAA rollout reuses the state update of `Pseudo_DFL_solver.py` | the test fails with `NameError: name 'nn' is not defined` |
| GLPK (`glpsol` on the `PATH`) | `test_integration.py::TestSAAOneSampleValid::test_saa_n1_returns_non_negative_objective` | the test is skipped |

GLPK is used by this one test only (`brew install glpk` or `sudo apt install glpk-utils`).

The `ED_OPF_*` variables select the solvers of the tests as they do for every other script. In the lock-file environment, which has no gurobipy, `ED_OPF_SOLVER=gurobi` makes most tests fail or error. An LP backend that is neither a `scipy.optimize.linprog` method nor `gurobi` falls back silently to SciPy's `revised simplex`, so the tests still pass without exercising the intended backend.

## What the tests do not cover

No test loads a PGLib-OPF case or sets a repair budget, and none covers the DFL pipeline (labels, loss, training, knapsack, rollouts), the MPC schedule search and cache, SDDiP or the exact DP. The MIP tests check that small instances solve, not that their solutions are optimal. `scripts/quick_check.sh` runs a short DFL run, a small SAA solve and the exact DP, and `paper/reproduce_paper_numbers.py` checks the stored results.

## Extending the tests

- Name a new file `tests/test_<topic>.py`, and reuse the fixtures of `conftest.py`: `network`, `params`, `dynamics`, `opf` and `large_sample`.
- Mark a test that solves a MIP with `@pytest.mark.slow`, and skip it when an optional dependency is missing, as the existing slow tests do with `pytest.importorskip("pyomo")` or `shutil.which("glpsol")`.
- To make a result independent of the solver variables, pass the solver explicitly, for example `StagewiseOPF(network, params, lp_method="highs")`.

To run the checks on another network, change `create_network()` in the `network` fixture of `conftest.py`, for example to `create_network("case14")`. Three parts of the suite hard-code the sizes of the 5-bus network and fail on another network: the 64 availability-pattern tests of `test_opf_constraints.py`, `test_network_dimensions_are_correct` in `test_integration.py`, and the fractional patterns of `test_derated_flow_scales_with_availability`. With the fixture switched to case14, all other fast tests pass. `test_bigM_validity.py` and `test_derated_mpc_opf.py` build their own 5-bus network with `create_network()`; with that call changed to case14, their three tests on random binary patterns pass as well.
