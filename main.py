"""
main.py: SAA entry point for the multistage stochastic ED-OPF with component failures.

Builds the network and model parameters, solves the SAA extensive form (tree or
Wait-and-See mode, see saa_solver.py) and writes config_snapshot.json,
solution.json and, unless --no-plots, the result plots to --output-dir (default:
a time-stamped folder Experiment_<DD>_<Mon>_<YYYY>__<HH>:<MM> in the working
directory). --validate prints feasibility checks; --evaluate runs an
out-of-sample rollout and stores its results in solution.json.

Usage:
    python main.py [options]
    python main.py --test      # smoke test on the toy network (run_quick_test)
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path

import config as _config
from config import save_snapshot, REPAIR_BUDGET_FRACTION
import solver_config
from network_data import create_network
from model_parameters import create_model_parameters
from scenario_generator import create_scenario_generator
from state_dynamics import create_state_dynamics
from saa_solver import create_saa_solver
from utils import (
    save_solution, compute_solution_statistics,
    validate_solution, print_validation_report, print_statistics_report
)
from visualization import create_results_report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Multistage Stochastic ED-OPF with Component Failures'
    )
    parser.add_argument(
        '--n-scenarios',
        type=int,
        default=50,
        help='Number of scenarios for SAA in ws mode (default: 50). In tree '
             'mode the tree size is set by --branching and T, and this value '
             'only sets the default of --n-test.'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='tree',
        choices=['tree', 'ws'],
        help="SAA formulation: 'tree' (default) is the multistage SAA on a "
             "sampled scenario tree with non-anticipativity; 'ws' is the "
             "Wait-and-See (perfect-information) problem, one MIP over "
             "independent scenarios."
    )
    parser.add_argument(
        '--branching',
        type=int,
        default=_config.SAA_TREE_BRANCHING,
        help=f'Scenario-tree branching factor B for tree mode '
             f'(default: {_config.SAA_TREE_BRANCHING}). Tree has B^T sample '
             f'paths; keep small (B=2) since node count grows as B^T.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--solver',
        type=str,
        default=_config.DATA_SOLVER_NAME,
        choices=sorted(solver_config.MIP_SOLVERS),
        help='MIP solver for the SAA extensive form. Default follows '
             'solver_config / ED_OPF_SOLVER (appsi_highs unless overridden); '
             'appsi_highs/highs recommended, glpk works but is slow.'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=172800,
        help='Solver time limit in seconds (default: 172800 = 48 h)'
    )
    parser.add_argument(
        '--mip-gap',
        type=float,
        default=0.05,
        help='MIP gap tolerance: solver stops when gap <= this (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: '
             'Experiment_<DD>_<Mon>_<YYYY>__<HH>:<MM> in the working directory)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate solution feasibility'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate solution on test scenarios'
    )
    parser.add_argument(
        '--n-test',
        type=int,
        default=None,
        help='Number of test scenarios for out-of-sample evaluation '
             '(default: max(5, n_scenarios // 3), about a third of --n-scenarios)'
    )
    parser.add_argument(
        '--eval-seed',
        type=int,
        default=None,
        help='Seed for test scenarios (default: seed+1000)'
    )
    parser.add_argument(
        '--network',
        type=str,
        default='toy',
        help='Network to solve: "toy" (built-in 5-bus) or an IEEE case name '
             'such as case14, case30, case57, case118 (requires pandapower).'
    )
    parser.add_argument(
        '--repair-base-scale',
        type=float,
        default=1.0,
        help='Uniform multiplier for both repair cost bases (default: 1.0)'
    )
    parser.add_argument(
        '--repair-budget-fraction',
        type=float,
        default=REPAIR_BUDGET_FRACTION,
        help='Per-stage repair budget as a fraction phi of total fleet repair '
             'cost: B = phi * sum(repair costs). Adds the '
             'per-stage budget constraint to the SAA MIP. Omit the flag for no budget.'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=600,
        help='Seconds between checkpoint saves during the SAA solve (default: '
             '600). Values <= 0 disable checkpointing and run one uninterrupted '
             'solve, which large MIPs (case14) need: restarting '
             'the solver every interval re-runs presolve and prevents any '
             'branch-and-bound progress. The solver still returns its best '
             'incumbent at the time limit.'
    )

    args = parser.parse_args()

    # Default test-set size: n_scenarios // 3, at least 5.
    if args.evaluate and args.n_test is None:
        args.n_test = max(5, args.n_scenarios // 3)

    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = datetime.now().strftime("Experiment_%d_%b_%Y__%H:%M")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Copy the repair-cost scale s, both repair cost bases and the budget fraction
    # phi into config so config_snapshot.json records them. ModelParameters binds
    # its config defaults at import time, so the bases and phi are also passed to
    # create_model_parameters explicitly. Other flags (e.g. --mip-gap,
    # --time-limit, --branching) are not in the snapshot; the solver settings are
    # recorded in solution['solver_metadata'] instead.
    _config.REPAIR_BASE_SCALE = args.repair_base_scale
    _config.GEN_REPAIR_BASE   = 550.0 * args.repair_base_scale
    _config.LINE_REPAIR_BASE  = 350.0 * args.repair_base_scale
    _config.REPAIR_BUDGET_FRACTION = args.repair_budget_fraction
    save_snapshot(output_dir)

    print()
    print("=" * 70)
    print("MULTISTAGE STOCHASTIC ED-OPF WITH COMPONENT FAILURES")
    print("=" * 70)
    print()

    # -- step 1: network and parameters --
    print("Step 1: Initializing network and parameters...")
    print("-" * 70)

    network = create_network(args.network)
    network.print_network_info()

    params = create_model_parameters(
        network, seed=args.seed,
        gen_repair_base=_config.GEN_REPAIR_BASE,
        line_repair_base=_config.LINE_REPAIR_BASE,
        repair_budget_fraction=args.repair_budget_fraction,
    )
    params.print_parameters()
    if params.repair_budget_per_stage is not None:
        print(f"  Repair budget    : ${params.repair_budget_per_stage:,.0f} / stage"
              f"  (phi={args.repair_budget_fraction} of fleet repair cost)")

    # -- step 2: SAA solve --
    print()
    print("Step 2: Solving with Sample Average Approximation...")
    print("-" * 70)

    saa = create_saa_solver(network, params, seed=args.seed)

    # Checkpoint callback: saves intermediate results while the solver runs.
    skip_plots = args.no_plots

    def checkpoint_callback(solution, gap, elapsed):
        """Write solution.json and, unless --no-plots, the result plots.

        The chunked solve calls this whenever the relative MIP gap drops by at
        least 0.001. On a solver without chunking support (glpk) it is called
        once after the solve.
        """
        try:
            # Save solution JSON
            sol_file = output_dir / 'solution.json'
            _save_solution_json(solution, sol_file)

            # Generate images (unless --no-plots)
            if not skip_plots:
                create_results_report(
                    solution, network, params,
                    output_dir=str(output_dir)
                )
            gap_str = f"{gap*100:.2f}%" if gap is not None else "N/A"
            print(f"  Checkpoint saved (gap: {gap_str}, time: {elapsed:.0f}s)")
        except Exception as e:
            print(f"  Warning: checkpoint save failed: {e}")

    # --checkpoint-interval <= 0 disables chunked checkpointing: one uninterrupted
    # solve with no callback. Large MIPs need this to make branch-and-bound
    # progress, since every chunk restarts the solver and repeats presolve. The
    # final solution is still saved in step 3.
    _use_checkpoints = args.checkpoint_interval > 0

    solution = saa.solve(
        n_scenarios=args.n_scenarios,
        mode=args.mode,
        branching=args.branching,
        solver_name=args.solver,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        verbose=True,
        checkpoint_callback=checkpoint_callback if _use_checkpoints else None,
        checkpoint_interval=args.checkpoint_interval,
    )

    if solution is None:
        print("\nERROR: Failed to solve problem. Exiting.")
        sys.exit(1)

    # -- step 3: save --
    print()
    print("Step 3: Saving solution...")
    print("-" * 70)

    solution_file = output_dir / 'solution.json'
    save_solution(solution, solution_file)

    # -- step 4: statistics --
    print()
    print("Step 4: Computing solution statistics...")
    print("-" * 70)

    stats = compute_solution_statistics(solution, network, params)
    print_statistics_report(stats, params)

    # -- step 5: validation (optional) --
    if args.validate:
        print()
        print("Step 5: Validating solution...")
        print("-" * 70)

        validation = validate_solution(
            solution, network, params, saa.scenarios
        )
        print_validation_report(validation)

    # -- step 6: out-of-sample evaluation (optional) --
    if args.evaluate:
        print()
        print("Step 6: Evaluating solution on test scenarios...")
        print("-" * 70)

        evaluation = saa.evaluate_solution(
            n_test_scenarios=args.n_test,
            verbose=True,
            seed=args.eval_seed
        )
        solution['out_of_sample_evaluation'] = evaluation
        _save_solution_json(solution, solution_file)

    # -- step 7: final plots --
    if not args.no_plots:
        print()
        print("Step 7: Generating final visualizations...")
        print("-" * 70)

        try:
            create_results_report(
                solution, network, params, saa.scenarios,
                output_dir=str(output_dir)
            )
        except Exception as e:
            print(f"Warning: Could not generate all plots: {e}")
            print("Continuing without plots...")

    print()
    print("=" * 70)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Summary:")
    if args.mode == 'tree':
        _obj_label = "Objective value:     " + \
            f"${solution['objective_value']:.2f}  (biased SAA point estimate)"
        _sample_label = f"  Sample paths (B^T):  {args.branching ** params.T}"
    else:
        _obj_label = "Objective value:     " + \
            f"${solution['objective_value']:.2f}  (Wait-and-See lower bound)"
        _sample_label = f"  Number of scenarios: {args.n_scenarios}"
    print(f"  {_obj_label}")
    print(f"  Total solve time:    {saa.solve_time:.2f}s")
    print(_sample_label)
    print(f"  MIP gap tolerance:   {args.mip_gap*100:.1f}%")
    print(f"  Results saved to:    {output_dir}/")
    print()
    print("=" * 70)


def _save_solution_json(solution, filepath):
    """Write solution to JSON: numpy arrays become lists, numpy scalars become
    Python numbers and dict keys become str.

    Used by the checkpoint callback and to rewrite solution.json with the
    out-of-sample evaluation in step 6.
    """
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(solution), f, indent=2)


def run_quick_test():
    """Smoke test on the toy (5-bus) network with a 60 s solver time limit.

    Generates 5 scenarios and prints the availability trajectory of the first
    one, then runs saa.solve in its default tree mode. n_scenarios=5 has no
    effect there: the tree has SAA_TREE_BRANCHING**T sample paths.
    """
    print("=" * 70)
    print("QUICK TEST MODE")
    print("=" * 70)
    print()

    network = create_network()
    params = create_model_parameters(network, seed=42)

    gen = create_scenario_generator(params, seed=42)
    scenarios = gen.generate_scenarios(n_scenarios=5)

    print(f"Generated {len(scenarios)} scenarios")

    state_dyn = create_state_dynamics(network, params)
    availability = state_dyn.compute_availability_trajectories(scenarios[0])

    print("Computed availability trajectories")
    state_dyn.print_availability_trajectory(availability, scenario_id=0)

    saa = create_saa_solver(network, params, seed=42)
    solution = saa.solve(
        n_scenarios=5,
        solver_name=_config.DATA_SOLVER_NAME,
        time_limit=60,
        verbose=True
    )

    if solution is not None:
        print("\nQuick test completed successfully!")
        print(f"Objective: ${solution['objective_value']:.2f}")
    else:
        print("\nQuick test failed!")


if __name__ == "__main__":
    # `python main.py --test` runs the smoke test instead of the CLI.
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_quick_test()
    else:
        main()
