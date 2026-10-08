"""
SDDiP Main Execution Script
===========================
Solves the multistage stochastic repair-scheduling problem (DC-OPF dispatch
under stochastic component failures, with repair decisions) by SDDiP, using
SDDP.jl (Julia) through juliacall.

Usage:
    python SDDiP_main.py [options]
    python SDDiP_main.py --network case14 --n-samples 256 --repair-base-scale 45

Output, written to --output-dir (default: a timestamped folder
Experiment_<dd>_<Mon>_<YYYY>__<HH:MM> in the working directory):
    config_snapshot.json             config.py constants, with the repair-cost
                                     and budget settings of this run
    cuts.json                        SDDP.jl cuts, written after training
    sddip_solution.json              bounds, gap and convergence history, written
                                     before the policy simulation and then
                                     overwritten with the full solution
    sddip_checkpoint.json            copy of the final solution
    sddip_*.png                      figures (the checkpoint callback writes them
                                     even with --no-plots)
    sddip_solution_interrupted.json  partial solution after SIGTERM/SIGINT or an
                                     abnormal exit

SDDP.jl also appends its training log to SDDP.log in the working directory.
"""

import argparse
import sys
import json
import signal
import atexit
from pathlib import Path
import numpy as np

# Shared project modules
import config as _config
from config import save_snapshot, REPAIR_BUDGET_FRACTION
import solver_config
from network_data import create_network
from model_parameters import create_model_parameters

# SDDiP-specific modules
from SDDiP_solver import create_sddip_solver
from SDDiP_visualization import create_sddip_results_report


# Module-level state for the emergency save on SIGTERM/SIGINT or exit
_sddip_solver = None
_network = None
_params = None
_output_dir = None
_shutdown_triggered = False


def save_emergency_results():
    """
    Save the partial solution and the figures after an interrupt or an abnormal exit.

    Called by the SIGTERM/SIGINT handler and registered with atexit. Runs at
    most once; after a normal finish main() sets _shutdown_triggered, so the
    atexit call does nothing. Writes sddip_solution_interrupted.json only when
    at least one SDDiP iteration has completed.
    """
    global _sddip_solver, _network, _params, _output_dir, _shutdown_triggered

    if _shutdown_triggered:
        return
    _shutdown_triggered = True

    if _sddip_solver is None or _output_dir is None:
        return

    print("\n" + "=" * 70)
    print("EMERGENCY SHUTDOWN - SAVING INTERMEDIATE RESULTS")
    print("=" * 70)

    try:
        solution = _sddip_solver.get_emergency_solution()

        if solution is not None and solution.get('iterations', 0) > 0:
            solution_file = _output_dir / 'sddip_solution_interrupted.json'
            save_sddip_solution(solution, solution_file)

            if _network is not None and _params is not None:
                print("\nGenerating visualizations from interrupted run...")
                try:
                    create_sddip_visualization(solution, _output_dir, _network, _params)
                except Exception as viz_error:
                    print(f"  Warning: Could not create some visualizations: {viz_error}")

            print("\n" + "=" * 70)
            print("EMERGENCY SAVE COMPLETED")
            print("=" * 70)
            print(f"  Iterations completed: {solution.get('iterations', 0)}")
            print(f"  Lower bound: ${solution.get('lower_bound', 0):.2f}")
            print(f"  Upper bound: ${solution.get('upper_bound', 0):.2f}")
            print(f"  Gap: {solution.get('gap', 0) * 100:.2f}%")
            print(f"  Results saved to: {_output_dir}/")
            print("=" * 70)
        else:
            print("  No completed iterations to save.")
    except Exception as e:
        print(f"  Error during emergency save: {e}")


def signal_handler(signum, frame):
    """Save intermediate results on SIGTERM or SIGINT, then exit with status 1."""
    signal_name = signal.Signals(signum).name
    print(f"\n\nReceived {signal_name} signal - initiating graceful shutdown...")
    save_emergency_results()
    sys.exit(1)


def setup_signal_handlers():
    """Install the SIGTERM/SIGINT handlers and the atexit emergency save."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_emergency_results)


def save_sddip_solution(solution: dict, filepath: Path) -> None:
    """Write the solution dict to JSON, converting numpy types to Python types."""
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_solution = convert_to_serializable(solution)

    with open(filepath, 'w') as f:
        json.dump(serializable_solution, f, indent=2)

    print(f"  Solution saved to: {filepath}")


def print_sddip_statistics(solution: dict, params) -> None:
    """Print bounds, gap, repair counts and convergence of an SDDiP solution."""
    print("=" * 70)
    print("SDDiP SOLUTION STATISTICS")
    print("=" * 70)

    print("\nOptimization Results:")
    print("-" * 70)
    print(f"  Lower bound:    ${solution['lower_bound']:.2f}")
    print(f"  Upper bound:    ${solution['upper_bound']:.2f}")
    print(f"  Optimality gap: {solution['gap'] * 100:.4f}%")
    print(f"  Iterations:     {solution['iterations']}")
    print(f"  Total cuts:     {solution['n_cuts']}")
    print(f"  Solve time:     {solution['solve_time']:.2f}s")

    print("\nRepair Decisions Summary:")
    print("-" * 70)

    total_gen_repairs = 0
    total_line_repairs = 0

    # repairs[...][t][k] is the fraction of simulated trajectories that order a
    # repair of component k at stage t; values above 0.3 count as a repair.
    for t in sorted(solution['repairs']['gen'].keys()):
        gen_repairs = solution['repairs']['gen'][t]
        line_repairs = solution['repairs']['line'][t]

        gen_count = sum(1 for r in gen_repairs.values() if r > 0.3)
        line_count = sum(1 for r in line_repairs.values() if r > 0.3)

        total_gen_repairs += gen_count
        total_line_repairs += line_count

    print(f"  Total generator repairs (across all stages): {total_gen_repairs}")
    print(f"  Total line repairs (across all stages):      {total_line_repairs}")

    print("\nConvergence History:")
    print("-" * 70)

    conv = solution.get('convergence', {})
    if 'gaps' in conv and len(conv['gaps']) > 0:
        print(f"  Initial gap: {conv['gaps'][0] * 100:.2f}%")
        if len(conv['gaps']) > 1:
            print(f"  After 5 iterations: {conv['gaps'][min(4, len(conv['gaps'])-1)] * 100:.2f}%")
        print(f"  Final gap: {conv['gaps'][-1] * 100:.4f}%")

    print("=" * 70)


def create_sddip_visualization(solution: dict, output_dir: Path,
                                network_data, model_params) -> None:
    """Write the SDDiP figures to output_dir; errors are printed, not raised."""
    try:
        create_sddip_results_report(
            solution, network_data, model_params,
            output_dir=str(output_dir)
        )
    except ImportError as e:
        print(f"  Warning: Could not import visualization module: {e}")
        print("  Skipping plots...")
    except Exception as e:
        print(f"  Warning: Could not create plots: {e}")


def main():
    """Parse the CLI, build the network and parameters, run SDDiP and save the results."""
    global _sddip_solver, _network, _params, _output_dir, _shutdown_triggered

    setup_signal_handlers()

    parser = argparse.ArgumentParser(
        description='Multistage Stochastic ED-OPF with SDDiP (SDDP.jl backend)'
    )
    parser.add_argument(
        '--iterations', type=int, default=50,
        help='Number of SDDiP iterations (default: 50)'
    )
    parser.add_argument(
        '--duality-handler', type=str, default='lagrangian',
        choices=['lagrangian', 'conic', 'strengthened', 'bandit'],
        help='Duality handler for cut generation (default: lagrangian). '
             'lagrangian = LagrangianDuality (tightest cuts, default). '
             'conic = ContinuousConicDuality (LP relaxation of binary subproblems). '
             'strengthened = StrengthenedConicDuality (tighter conic cuts). '
             'bandit = BanditDuality (adaptive selection between handlers).'
    )
    parser.add_argument(
        '--n-simulations', type=int, default=500,
        help='Number of policy simulations after training (default: 500); '
             'their averages are saved in sddip_solution.json'
    )
    parser.add_argument(
        '--n-samples', type=int, default=1024,
        help='Sampled failure realisations per stage for networks with >= 10 '
             'components (default: 1024). Ignored for small networks such as '
             'toy, which enumerate all outcomes exactly. Lower it to make each '
             'SDDiP iteration tractable on larger networks (e.g. case14).'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--hard-time-limit', type=int, default=None,
        help='Time limit in seconds for SDDP.jl training (default: none); the '
             'policy simulation that follows is not limited'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: Experiment_<DD>_<Mon>_<YYYY>__<HH>:<MM> '
             'in the working directory)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run quick test with minimal settings'
    )
    parser.add_argument(
        '--network', type=str, default='toy',
        help='Network to solve: "toy" (built-in 5-bus) or an IEEE case name '
             'such as case14, case30, case57, case118 (requires pandapower).'
    )
    parser.add_argument(
        '--solver', type=str, default=solver_config.SDDIP_SOLVER,
        choices=['highs', 'gurobi'],
        help="SDDP.jl optimizer (Layer B): 'highs' (default) or 'gurobi' "
             '(requires Gurobi.jl in the Julia env; see README). Default '
             'follows solver_config / ED_OPF_SOLVER.'
    )
    parser.add_argument(
        '--repair-base-scale', type=float, default=1.0,
        help='Uniform multiplier for both repair cost bases (default: 1.0)'
    )
    parser.add_argument(
        '--repair-budget-fraction', type=float, default=REPAIR_BUDGET_FRACTION,
        help='Per-stage repair budget as a fraction phi of total fleet repair '
             'cost: B = phi * sum(repair costs). Adds the '
             'per-stage budget constraint to the SDDiP subproblems. Omit the flag for no budget.'
    )

    args = parser.parse_args()

    # --test: quick run with few iterations and simulations
    if args.test:
        args.iterations = 5
        args.duality_handler = 'lagrangian'
        args.n_simulations = 100
        args.hard_time_limit = 120

    # Create output directory
    if args.output_dir is None:
        from datetime import datetime
        args.output_dir = datetime.now().strftime("Experiment_%d_%b_%Y__%H:%M")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _output_dir = output_dir
    # Copy the CLI repair-cost and budget settings into config, so that
    # config_snapshot.json records them; the scaled repair bases are also passed
    # to create_model_parameters below. 550 and 350 are the scale-1 values of
    # GEN_REPAIR_BASE and LINE_REPAIR_BASE in config.py.
    _config.REPAIR_BASE_SCALE = args.repair_base_scale
    _config.GEN_REPAIR_BASE   = 550.0 * args.repair_base_scale
    _config.LINE_REPAIR_BASE  = 350.0 * args.repair_base_scale
    _config.REPAIR_BUDGET_FRACTION = args.repair_budget_fraction
    save_snapshot(output_dir)

    print()
    print("=" * 70)
    print("MULTISTAGE STOCHASTIC ED-OPF WITH SDDiP")
    print("  Powered by SDDP.jl (Julia) via juliacall")
    print("=" * 70)
    print()

    # ========================================================================
    # STEP 1: Initialize Network and Parameters
    # ========================================================================
    print("Step 1: Initializing network and parameters...")
    print("-" * 70)

    network = create_network(args.network)
    network.print_network_info()
    _network = network

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
    _params = params

    # ========================================================================
    # STEP 2: Solve using SDDiP (SDDP.jl)
    # ========================================================================
    print()
    print("Step 2: Solving with SDDiP (SDDP.jl)...")
    print("-" * 70)

    sddip = create_sddip_solver(network, params, seed=args.seed)
    _sddip_solver = sddip

    # SDDiPSolver.solve() calls this once, with the final solution after the
    # policy simulation; there is no per-iteration checkpoint. It writes
    # sddip_checkpoint.json, sddip_solution.json and the figures (also with
    # --no-plots).
    def checkpoint_callback(partial_solution):
        try:
            serializable = _make_serializable(partial_solution)

            checkpoint_file = output_dir / 'sddip_checkpoint.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(serializable, f, indent=2)

            solution_file = output_dir / 'sddip_solution.json'
            with open(solution_file, 'w') as f:
                json.dump(serializable, f, indent=2)

            if partial_solution.get('iterations', 0) >= 1:
                try:
                    create_sddip_visualization(partial_solution, output_dir, network, params)
                except Exception:
                    pass
        except Exception:
            pass

    sddip.set_checkpoint_callback(checkpoint_callback, interval=1)

    time_limit = args.hard_time_limit if args.hard_time_limit is not None else float('inf')

    solution = sddip.solve(
        max_iterations=args.iterations,
        time_limit=time_limit,
        verbose=True,
        solver_name=args.solver,
        duality_handler=args.duality_handler,
        n_simulations=args.n_simulations,
        n_samples=args.n_samples,
        cuts_path=str(output_dir / "cuts.json"),
        lb_solution_path=str(output_dir / "sddip_solution.json"),
    )

    if solution is None:
        print("\nERROR: SDDiP failed to find a solution. Exiting.")
        sys.exit(1)

    # ========================================================================
    # STEP 3: Save Solution
    # ========================================================================
    print()
    print("Step 3: Saving solution...")
    print("-" * 70)

    solution_file = output_dir / 'sddip_solution.json'
    save_sddip_solution(solution, solution_file)

    # ========================================================================
    # STEP 4: Print Statistics
    # ========================================================================
    print()
    print("Step 4: Solution statistics...")
    print("-" * 70)

    print_sddip_statistics(solution, params)

    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    if not args.no_plots:
        print()
        print("Step 5: Generating visualizations...")
        print("-" * 70)

        create_sddip_visualization(solution, output_dir, network, params)

    # ========================================================================
    # Final Summary
    # ========================================================================
    # Normal finish: the atexit emergency save does nothing from here on.
    _shutdown_triggered = True

    print()
    print("=" * 70)
    print("SDDiP EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  Lower bound:         ${solution['lower_bound']:.2f}")
    print(f"  Upper bound:         ${solution['upper_bound']:.2f}")
    print(f"  Optimality gap:      {solution['gap'] * 100:.4f}%")
    print(f"  Total solve time:    {solution['solve_time']:.2f}s")
    print(f"  SDDiP iterations:    {solution['iterations']}")
    print(f"  Cuts generated:      {solution['n_cuts']}")
    print(f"  Results saved to:    {output_dir}/")
    print()
    print("=" * 70)


def _make_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    return obj


if __name__ == "__main__":
    main()
