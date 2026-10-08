"""
SDDiP Solver Module
===================
Thin wrapper around SDDP.jl (through juliacall, see SDDiP_julia_bridge.py) for
the multistage stochastic ED-OPF problem with component failures and repairs.

SDDiPSolver.solve() builds the SDDP.jl model, trains it, writes the cuts and a
pre-simulation solution file, simulates the trained policy and returns the
solution dict. SDDP.jl provides:
- the forward and backward passes
- Benders and Lagrangian cut generation (the duality handlers)
- convergence guarantees for binary state variables when Lagrangian cuts are used
"""

import time
import json
import numpy as np
from typing import Dict, Optional

from network_data import NetworkData
from model_parameters import ModelParameters
from SDDiP_julia_bridge import SDDPJuliaBridge
import solver_config


class SDDiPSolver:
    """
    SDDiP solver wrapping SDDP.jl through SDDPJuliaBridge.

    SDDiP_main.py uses solve(), set_checkpoint_callback() and
    get_emergency_solution(). The solution dict is the input of
    SDDiP_visualization.py and compare_methods.py.
    """

    def __init__(self, network: NetworkData, params: ModelParameters, seed: int = None):
        self.network = network
        self.params = params
        self.seed = seed

        self.bridge: Optional[SDDPJuliaBridge] = None
        self.solution: Optional[Dict] = None
        self.solve_time: Optional[float] = None
        self.iterations: int = 0

        self.upper_bounds = []
        self.lower_bounds = []
        self.gaps = []

        self._checkpoint_callback = None
        self._checkpoint_interval: int = 1

    def solve(self, n_branches: int = 10, max_iterations: int = 50,
              gap_tolerance: float = 0.01, solver_name: str = None,
              time_limit: float = 600, verbose: bool = True,
              max_nodes: int = 2000,
              duality_handler: str = "lagrangian",
              n_simulations: int = 500,
              n_samples: int = 1024,
              cuts_path: str = None,
              lb_solution_path: str = None) -> Optional[Dict]:
        """
        Train an SDDP.jl policy, simulate it and return the solution dict.

        Parameters
        ----------
        n_branches : int
            Unused.
        max_iterations : int
            Iteration limit for SDDP.train (forward-backward iterations).
        gap_tolerance : float
            Unused; training stops only at the iteration or time limit.
        solver_name : str or None
            SDDP.jl optimizer: 'highs' or 'gurobi'. None resolves to
            solver_config.SDDIP_SOLVER, which is 'highs' unless
            ED_OPF_SDDIP_SOLVER or a Gurobi-family ED_OPF_SOLVER is set.
            Gurobi requires Gurobi.jl in the Julia environment, which
            juliapkg.json does not include (see README).
        time_limit : float
            Time limit in seconds for SDDP.train. The policy simulation after
            training is not limited. inf means no limit (1e8 s is passed to
            SDDP.jl).
        verbose : bool
            Print the run settings and the solution summary. The bridge and
            SDDP.jl print their progress regardless.
        max_nodes : int
            Unused.
        duality_handler : str
            One of: 'lagrangian', 'conic', 'strengthened', 'bandit'.
        n_simulations : int
            Number of Monte Carlo replications of the trained policy. They give
            the expected repairs, generation and load served per stage; the
            upper bound comes from the training log, not from these runs.
        n_samples : int
            Sampled failure realizations per stage, used only for networks with
            10 or more components (e.g. case14), where enumerating all 2^n
            outcomes is impractical. Small networks such as the toy (5-bus)
            network (7 components) enumerate all outcomes and ignore this value.
            The backward pass solves one subproblem per realization at every
            stage, so the cost of an iteration grows linearly with this count.
            It is the per-network setting for tractability: the Lagrangian
            duality handler stays fixed (finite convergence is guaranteed only
            for Lagrangian cuts) and n_samples is reduced instead. Default 1024.
        cuts_path : str or None
            If given, the cuts are written to this JSON file after training.
        lb_solution_path : str or None
            If given, a pre-simulation solution (bounds, gap and convergence
            history) is written to this file before the policy simulation.

        Returns
        -------
        dict
            Solution dictionary from SDDPJuliaBridge.extract_results(), with
            solve_time set to the wall-clock time of the whole call.
        """
        start_time = time.time()

        # None falls back to the Layer B default of solver_config.py.
        solver_name = solver_name or solver_config.SDDIP_SOLVER

        if verbose:
            print("=" * 70)
            print("STOCHASTIC DUAL DYNAMIC INTEGER PROGRAMMING (SDDiP)")
            print("  Powered by SDDP.jl (Julia)")
            print("=" * 70)
            print(f"  Target iterations:  {max_iterations}")
            print(f"  Duality handler:    {duality_handler}")
            print(f"  Optimizer:          {solver_name}")
            print(f"  Simulations:        {n_simulations}")
            if time_limit != float('inf'):
                print(f"  Hard time limit:    {time_limit}s")
            else:
                print(f"  Hard time limit:    None")
            print()

        # Julia bridge: solver_name selects the SDDP.jl optimizer; a missing
        # seed falls back to 42.
        self.bridge = SDDPJuliaBridge(self.network, self.params,
                                      seed=self.seed if self.seed is not None else 42,
                                      solver=solver_name)
        self.bridge.setup_julia()

        # Enumerate or sample the per-stage failure outcomes.
        # Fewer than 10 components (toy: 2 generators + 5 lines = 7): enumerate
        #   all 2^n outcomes exactly (128 for toy); n_samples is not used.
        # 10 or more (case14: 25): draw n_samples outcomes per stage.
        #   A lower per-network value (CLI --n-samples) keeps each SDDiP
        #   iteration tractable on the larger networks.
        n_components = self.network.n_generators + self.network.n_lines
        enumerate_all = (n_components < 10)

        self.bridge.build_model(enumerate_all=enumerate_all, n_samples=n_samples)

        # Train
        effective_time_limit = time_limit if time_limit != float('inf') else 1e8
        self.bridge.train(
            iteration_limit=max_iterations,
            time_limit=effective_time_limit,
            duality_handler=duality_handler,
            log_frequency=1,
        )

        # Save the cuts before the simulation (SDDiP_main.py has created the output
        # directory)
        if cuts_path is not None:
            self.bridge.save_cuts(cuts_path)

        # Write the gap and the full bound history at this point, before the
        # policy simulation, which a wall-clock kill can cut off. Before simulate(),
        # extract_results() reads the per-iteration lower bounds and forward-pass
        # costs (the upper bounds) from the training log, computes the gaps and
        # leaves the repair and dispatch fields empty, so the gap reaches the disk
        # even for a run that stopped at the time limit without converging. If the
        # simulation finishes, SDDiP_main.py overwrites this file with the full
        # solution, which adds the repair and dispatch statistics; if it is
        # killed, this file remains.
        if lb_solution_path is not None:
            try:
                _presol = self.bridge.extract_results()
                _presol['solve_time'] = time.time() - start_time
                _presol['note'] = ('Pre-simulation: lower/upper bounds, gap and the '
                                    'per-iteration convergence trajectory from the '
                                    'training log. Repair and dispatch detail are '
                                    'filled in only if the policy simulation '
                                    'finishes.')
                with open(lb_solution_path, 'w') as _f:
                    json.dump(_presol, _f, indent=2)
                if verbose:
                    print(f"  Pre-simulation solution persisted "
                          f"(LB=${_presol['lower_bound']:,.2f}, "
                          f"UB=${_presol['upper_bound']:,.2f}, "
                          f"gap={_presol['gap'] * 100:.4f}%) -> {lb_solution_path}")
            except Exception as _e:
                if verbose:
                    print(f"  Warning: could not persist pre-simulation solution ({_e})")

        # Simulate the trained policy for the expected repairs, generation and
        # load served (the upper bound comes from the training log)
        self.bridge.simulate(n_replications=n_simulations)

        # Extract results
        self.solution = self.bridge.extract_results()

        self.solve_time = time.time() - start_time
        self.solution['solve_time'] = self.solve_time
        self.iterations = self.solution['iterations']

        conv = self.solution.get('convergence', {})
        self.upper_bounds = conv.get('upper_bounds', [])
        self.lower_bounds = conv.get('lower_bounds', [])
        self.gaps = conv.get('gaps', [])

        if verbose:
            print()
            print("=" * 70)
            print("SDDiP SOLUTION SUMMARY")
            print("=" * 70)
            print(f"  Total solve time:    {self.solve_time:.2f}s")
            print(f"  Iterations:          {self.iterations}")
            print(f"  Lower bound:         ${self.solution['lower_bound']:.2f}")
            print(f"  Upper bound:         ${self.solution['upper_bound']:.2f}")
            print(f"  Gap:                 {self.solution['gap'] * 100:.4f}%")
            # Smoothed gap from the means of the last 3% (at least one) of the
            # per-iteration bounds
            if self.upper_bounds and self.lower_bounds:
                n_tail = max(1, int(np.ceil(len(self.upper_bounds) * 0.03)))
                sm_upper = np.mean(self.upper_bounds[-n_tail:])
                sm_lower = np.mean(self.lower_bounds[-n_tail:])
                sm_gap = (sm_upper - sm_lower) / sm_upper if sm_upper > 0 else 0
                print(f"  Smoothed gap (3%):   {sm_gap * 100:.4f}%  (avg last {n_tail} iters)")
            print(f"  Cuts generated:      {self.solution['n_cuts']}")
            print("=" * 70)

        # The checkpoint callback receives the final solution, once
        if self._checkpoint_callback:
            self._checkpoint_callback(self.solution)

        return self.solution

    def set_checkpoint_callback(self, callback, interval: int = 1):
        """
        Register a callback that solve() calls once with the final solution.

        Parameters
        ----------
        callback : callable
            Called as callback(solution_dict) at the end of solve().
        interval : int
            Stored but unused; there is no per-iteration checkpoint.
        """
        self._checkpoint_callback = callback
        self._checkpoint_interval = interval

    def get_emergency_solution(self) -> Optional[Dict]:
        """
        Return the best available solution for an emergency save.

        In order: the final solution if solve() has finished; otherwise the
        results the bridge can extract from the training log; otherwise a
        placeholder built from the recorded bounds, or None if there are none.
        """
        if self.solution is not None:
            return self.solution

        if self.bridge is not None:
            try:
                return self.bridge.extract_results()
            except Exception:
                pass

        # Placeholder marked 'interrupted', or None when no bounds are recorded
        if not self.upper_bounds and not self.lower_bounds:
            return None

        return {
            'objective_value': 0.0,
            'upper_bound': 0.0,
            'lower_bound': 0.0,
            'gap': 1.0,
            'iterations': self.iterations,
            'solve_time': self.solve_time or 0.0,
            'n_cuts': 0,
            'repairs': {'gen': {}, 'line': {}},
            'expected_generation': {},
            'expected_load_served': {},
            'convergence': {
                'upper_bounds': self.upper_bounds,
                'lower_bounds': self.lower_bounds,
                'gaps': self.gaps,
            },
            'interrupted': True,
        }

    def get_solution_summary(self) -> Optional[Dict]:
        """Return the headline numbers of the last solve, or None before solve()."""
        if self.solution is None:
            return None
        return {
            'objective_value': self.solution['objective_value'],
            'upper_bound': self.solution['upper_bound'],
            'lower_bound': self.solution['lower_bound'],
            'gap': self.solution['gap'],
            'iterations': self.iterations,
            'solve_time': self.solve_time,
            'n_cuts': self.solution['n_cuts'],
        }


def create_sddip_solver(network: NetworkData, params: ModelParameters,
                         seed: int = None) -> SDDiPSolver:
    """Factory function for SDDiPSolver."""
    return SDDiPSolver(network, params, seed=seed)


if __name__ == "__main__":
    from network_data import create_network
    from model_parameters import create_model_parameters

    print("Testing SDDiP Solver (SDDP.jl backend)")
    print()

    network = create_network()
    params = create_model_parameters(network, seed=42)

    sddip = create_sddip_solver(network, params, seed=42)
    solution = sddip.solve(
        max_iterations=5,
        time_limit=120,
        verbose=True,
        duality_handler="conic",
        n_simulations=100,
    )

    if solution is not None:
        print()
        print("SDDiP test completed successfully!")
        print(f"  Lower bound: ${solution['lower_bound']:.2f}")
        print(f"  Upper bound: ${solution['upper_bound']:.2f}")
        print(f"  Gap: {solution['gap'] * 100:.4f}%")
    else:
        print("SDDiP test failed!")
