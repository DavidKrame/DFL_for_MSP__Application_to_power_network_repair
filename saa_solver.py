"""
saa_solver.py: Sample Average Approximation (SAA) for the multistage stochastic ED-OPF.

Two modes
---------
mode='tree' (default): multistage SAA on a sampled scenario tree that branches by
  a fixed factor B at every stage (B^T sample paths). Variables live on tree
  nodes, so all sample paths through a node share that node's decision. This
  enforces non-anticipativity, and the objective approximates the optimal value
  v* of the multistage problem: it is a biased point estimate of v*, not a bound.
  Each node represents the next stage's failure distribution by only B samples,
  so with a small B the result is best read as a cross-check against SDDiP.

mode='ws': Wait-and-See (perfect-information) relaxation. Every repair r^t is
  pure recourse and each of the N sampled scenarios keeps its own trajectory;
  no constraint couples the scenarios, so the extensive form decomposes into N
  independent single-scenario MIPs. The objective estimates E[WS], and
  E[WS] <= v* <= expected cost of any implementable policy. The estimate is a
  sample mean solved to the MIP gap, so it is not a guaranteed bound itself.

Usage in experiments
--------------------
  * solve(): builds and solves the tree SAA (or WS) MIP.
  * evaluate_solution(): rolls out an implementable policy on out-of-sample
    scenarios, with dispatch from StagewiseOPF. In tree mode the policy is
    closed-loop: at each stage the rollout moves to the child node whose sampled
    failure vector is nearest (Hamming distance) to the realized one and applies
    that node's repair decision. In ws mode it applies a fixed schedule obtained
    by rounding the mean repair decisions.
"""

import numpy as np
import time
from typing import Optional

from config import DATA_SOLVER_NAME, SAA_TIME_LIMIT_S, SAA_MIP_GAP, SAA_TREE_BRANCHING


class SAASolver:
    """Sample Average Approximation solver for stochastic ED-OPF."""
    
    def __init__(self, network_data, model_params, seed=None):
        self.network = network_data
        self.params = model_params
        self.seed = seed
        
        self.solution = None
        self.solve_time = None
        self.scenarios = None
        self.tree = None
        self._tree_policy = None   # node-indexed policy, kept in memory for evaluation

    def _scenarios_from_tree(self, tree):
        """Rebuild one Scenario per leaf from its root->leaf failure history.

        Leaves are enumerated in tree order, the same order as the leaf-path view
        built by EDOPFTreeProblem.extract_solution, so scenario i lines up with
        solution['generation'][i] etc. Each Scenario carries its path
        probability 1/B^T.
        """
        from scenario_generator import Scenario

        nG = self.network.n_generators
        nE = self.network.n_lines
        T = tree.T
        leaves = tree.leaves
        scenarios = []
        for s_idx, leaf in enumerate(leaves):
            gen_fail  = np.zeros((T, nG), dtype=int)
            line_fail = np.zeros((T, nE), dtype=int)
            cur = leaf
            while cur is not None and tree.node(cur).stage >= 1:
                nd = tree.node(cur)
                gen_fail[nd.stage - 1, :]  = nd.xi_gen
                line_fail[nd.stage - 1, :] = nd.xi_line
                cur = nd.parent
            scenarios.append(Scenario(
                scenario_id=s_idx,
                gen_failures=gen_fail,
                line_failures=line_fail,
                probability=tree.node(leaf).path_prob,
            ))
        return scenarios

    def solve(self, n_scenarios, mode='tree', branching=None,
              solver_name=DATA_SOLVER_NAME,
              time_limit=SAA_TIME_LIMIT_S, mip_gap=SAA_MIP_GAP,
              verbose=True, checkpoint_callback=None, checkpoint_interval=600):
        """Build and solve the SAA extensive-form MIP.

        mode='tree' (default) samples a scenario tree with branching factor
        `branching` (default config.SAA_TREE_BRANCHING) and solves the
        node-indexed, non-anticipative form; `n_scenarios` is then only recorded
        in solver_metadata. mode='ws' samples `n_scenarios` independent scenarios
        and solves the scenario-flat form, whose objective estimates the
        Wait-and-See bound.

        With a checkpoint_callback and a solver that supports chunked solves
        (solver_config.supports_chunking: every registered solver except glpk),
        the solver is restarted every checkpoint_interval seconds and
        checkpoint_callback(solution, gap, elapsed) is called whenever the
        relative MIP gap drops by at least 0.001. Otherwise the MIP is solved in
        one uninterrupted run (no repeated presolve) and a callback, if given, is
        called once at the end.

        Returns the solution dict with 'solver_metadata' added. If the solver
        finds no solution, tree mode falls back to the NoRepair solution and
        ws mode returns None.
        """
        from scenario_generator import create_scenario_generator
        from state_dynamics import create_state_dynamics
        from problem_formulation import create_problem, create_tree_problem

        if mode not in ('tree', 'ws'):
            raise ValueError(f"mode must be 'tree' or 'ws', got {mode!r}")
        if branching is None:
            branching = SAA_TREE_BRANCHING

        start_time = time.time()
        gen = create_scenario_generator(self.params, seed=self.seed)

        if mode == 'tree':
            if verbose:
                print("=" * 70)
                print("SAMPLE AVERAGE APPROXIMATION (SAA) — SAMPLED SCENARIO TREE")
                print("=" * 70)
                print(f"Branching factor B: {branching}")
                print(f"Horizon T:          {self.params.T}")
                print(f"Sample paths (B^T): {branching ** self.params.T}")
                print(f"Random seed:        {self.seed}")
                print()
                print("Step 1: Building sampled scenario tree...")

            self.tree = gen.generate_scenario_tree(branching, self.params.T)
            # One Scenario per leaf path, in the order of the solution's leaf-path
            # view. main.py passes the list on to validation and plotting, which
            # read the solution dict instead; get_solution_summary reports its length.
            self.scenarios = self._scenarios_from_tree(self.tree)

            if verbose:
                print(f"  Tree built: {len(self.tree.decision_nodes)} decision nodes, "
                      f"{len(self.tree.leaves)} leaves")
                print()
                print("Step 2: Formulating node-indexed optimization problem...")

            problem = create_tree_problem(self.network, self.params, self.tree)
        else:
            if verbose:
                print("=" * 70)
                print("SAMPLE AVERAGE APPROXIMATION (SAA) — WAIT-AND-SEE LOWER BOUND")
                print("=" * 70)
                print(f"Number of scenarios: {n_scenarios}")
                print(f"Random seed:         {self.seed}")
                print()
                print("Step 1: Generating failure scenarios...")

            self.tree = None
            self.scenarios = gen.generate_scenarios(n_scenarios)
            if verbose:
                stats = gen.get_scenario_statistics(self.scenarios)
                print(f"  Generated {len(self.scenarios)} scenarios")
                print(f"  Mean total failures per scenario: {stats['mean_total_failures']:.2f}")
                print()
                print("Step 2: Formulating optimization problem...")

            state_dyn = create_state_dynamics(self.network, self.params)
            problem = create_problem(self.network, self.params, self.scenarios, state_dyn)

        problem.build_model()

        if mode == 'tree':
            # Set every variable to the NoRepair point (no repairs, zero dispatch,
            # all load shed), which is always feasible. An uninterrupted solve
            # passes it as a MIP start to solvers with warm-start support (Gurobi,
            # CPLEX, CBC; not HiGHS). The same point is the fallback solution below.
            problem.set_norepair_start()

        if verbose:
            print("  Problem formulated")
            print(f"  Number of variables: {problem.model.nvariables()}")
            print(f"  Number of constraints: {problem.model.nconstraints()}")
            print()
            print("Step 3: Solving optimization problem...")
            print()

        solution = problem.solve(
            solver_name=solver_name,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=verbose,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval=checkpoint_interval,
            warmstart=(mode == 'tree'),
        )

        # Guarantee a result in tree mode: if the solver returned no incumbent,
        # fall back to the NoRepair solution (always feasible) so the run still
        # produces a policy to evaluate and plot. The NoRepair values are set
        # again in case the solve changed them.
        used_norepair_fallback = False
        if mode == 'tree' and solution is None:
            if verbose:
                print("\nNo incumbent returned by the solver; falling back to the "
                      "NoRepair solution so a result is always produced.")
            problem.set_norepair_start()
            solution = problem.extract_solution(force=True)
            used_norepair_fallback = True

        self.solution = solution
        self.solve_time = time.time() - start_time

        # The node-indexed policy is bulky and only needed in-process by
        # evaluate_solution; keep it on the solver so the serialized solution
        # stays the compact scenario-flat shape the downstream code expects.
        if solution is not None:
            self._tree_policy = solution.pop('tree_policy', None)

        if solution is not None:
            tc = str(problem.results.solver.termination_condition) if problem.results else 'unknown'
            solution['solver_metadata'] = {
                'solver_name': solver_name,
                'mode': mode,
                'branching': branching if mode == 'tree' else None,
                'n_sample_paths': (branching ** self.params.T) if mode == 'tree' else n_scenarios,
                'mip_gap_target': mip_gap,
                'mip_gap_achieved': problem._extract_mip_gap(),
                'norepair_fallback': used_norepair_fallback,
                'time_limit': time_limit,
                'solve_time': self.solve_time,
                'n_scenarios': n_scenarios,
                'termination_condition': tc,
            }

        if verbose and solution is not None:
            print()
            print("=" * 70)
            print("SAA SOLUTION SUMMARY")
            print("=" * 70)
            print(f"Total solve time: {self.solve_time:.2f}s")
            print(f"Objective value: ${solution['objective_value']:.2f}")
            print()
            
            self._print_solution_summary(solution)
        
        return solution
    
    def _print_solution_summary(self, solution):
        """Print summary of the solution."""
        print("Expected Repair Decisions (Recourse):")
        print("-" * 70)

        for t in sorted(solution['repairs']['gen'].keys()):
            gen_repairs = solution['repairs']['gen'][t]
            line_repairs = solution['repairs']['line'][t]

            gen_active = [(g, r) for g, r in gen_repairs.items() if r > 0.01]
            line_active = [(e, r) for e, r in line_repairs.items() if r > 0.01]

            if gen_active or line_active:
                print(f"  Stage {t}:")
                for g, r in gen_active:
                    print(f"    Generator {g+1}: repair rate {r:.0%}")
                for e, r in line_active:
                    print(f"    Line {e+1}: repair rate {r:.0%}")
        
        print()
        print("Expected Operational Statistics:")
        print("-" * 70)
        
        # Average over scenarios and stages. Scenarios (leaf paths in tree mode)
        # are equally likely, so this is the expected value.
        total_gen = 0.0
        total_shedding = 0.0
        n_scenarios = len(solution['generation'])
        
        for s in solution['generation'].keys():
            for t in solution['generation'][s].keys():
                # Generation
                for g, pg in solution['generation'][s][t].items():
                    total_gen += pg / (n_scenarios * self.params.T)
                
                # Load shedding
                for l in solution['load_served'][s][t].keys():
                    demand = self.network.load_nominal[l]
                    served = solution['load_served'][s][t][l]
                    total_shedding += (demand - served) / (n_scenarios * self.params.T)
        
        print(f"  Average generation per stage: {total_gen:.2f} MW")
        print(f"  Average load shedding per stage: {total_shedding:.2f} MW")
        print()
    
    def evaluate_solution(self, n_test_scenarios, verbose=True, seed=None):
        """Out-of-sample evaluation of the SAA policy.

        tree mode (closed-loop): the rollout starts at the root. At each stage it
        moves to the child of the current node whose incoming failure shock
        (xi_gen, xi_line) has the smallest Hamming distance to the realized shock
        of that stage (the first child wins ties) and applies that child's repair
        decision. This is an implementable, history-dependent policy.

        ws mode (open-loop): a fixed schedule from rounding the mean repair rates
        r_bar^t[k] = round(E_s[r^{s,t,k}]).

        Both rollouts follow the DFL evaluation so the costs are comparable:
          - r^t is masked by current availability (repair charged only for a
            component that is actually failed).
          - Dispatch solved by the same DC-OPF LP (StagewiseOPF) used by DFL.
          - State transitions use the same availability update
            (Pseudo_DFL_solver._next_availability).

        Test scenarios are drawn with `seed`, or self.seed + 1000 when seed is
        None (unseeded if self.seed is also None). Returns the mean, std, min
        and max of the per-scenario total cost (dispatch, load shedding and
        repairs over the horizon), the list of costs, and the evaluation time.
        """
        if self.solution is None:
            raise ValueError("No solution available. Call solve() first.")

        from scenario_generator import create_scenario_generator
        from Pseudo_DFL_stage_opf import StagewiseOPF
        from Pseudo_DFL_solver import _next_availability

        tree_pol = self._tree_policy
        use_tree = tree_pol is not None
        policy_name = "tree closed-loop policy" if use_tree else "mean-repair policy"

        if verbose:
            print("=" * 70)
            print(f"OUT-OF-SAMPLE EVALUATION (SAA {policy_name})")
            print("=" * 70)
            print(f"Generating {n_test_scenarios} test scenarios...")

        test_seed = seed if seed is not None else (self.seed + 1000 if self.seed is not None else None)
        gen = create_scenario_generator(self.params, seed=test_seed)
        test_scenarios = gen.generate_scenarios(n_test_scenarios)

        if use_tree:
            def stage_decision(state):
                """Move to the child nearest the realized shock; return its repairs.

                Returns (r_gen, r_line, node). A node without children (tree
                shorter than params.T) gives no repairs and stays where it is.
                """
                cur, xi_g, xi_e = state['node'], state['xi_g'], state['xi_e']
                children = tree_pol['children'].get(cur, [])
                if not children:
                    return (np.zeros(self.network.n_generators),
                            np.zeros(self.network.n_lines), cur)
                best, best_d = None, None
                for c in children:
                    d = (int(np.sum(np.abs(tree_pol['xi_gen'][c] - xi_g)))
                         + int(np.sum(np.abs(tree_pol['xi_line'][c] - xi_e))))
                    if best_d is None or d < best_d:
                        best_d, best = d, c
                node = tree_pol['nodes'][best]
                return (node['r_gen'].astype(float), node['r_line'].astype(float), best)
        else:
            # Fixed schedule from rounded mean repair rates.
            repair_policy = {}
            for t_key in self.solution['repairs']['gen'].keys():
                t_idx = int(t_key) - 1
                gen_keys  = sorted(self.solution['repairs']['gen'][t_key].keys(),  key=int)
                line_keys = sorted(self.solution['repairs']['line'][t_key].keys(), key=int)
                repair_policy[t_idx] = {
                    'gen':  np.array([round(self.solution['repairs']['gen'][t_key][g])
                                      for g in gen_keys],  dtype=float),
                    'line': np.array([round(self.solution['repairs']['line'][t_key][e])
                                      for e in line_keys], dtype=float),
                }

        opf = StagewiseOPF(self.network, self.params)
        tau = self.params.tau

        if verbose:
            print(f"  Rolling out {policy_name} on {n_test_scenarios} scenarios "
                  f"(DC-OPF dispatch)...")

        test_costs = []
        t_eval_start = time.time()

        for scenario in test_scenarios:
            # Stage-1 availability: a^1[k] = 1 - xi^1[k]
            a_gen  = (1 - scenario.gen_failures[0, :]).astype(float)
            a_line = (1 - scenario.line_failures[0, :]).astype(float)
            r_gen_hist  = []
            r_line_hist = []
            total = 0.0
            cur_node = tree_pol['root'] if use_tree else None

            for t in range(self.params.T):
                # Raw repair decision from the policy, then mask by availability
                # (1 - a^t[k]): cost is charged only for actually failed components.
                if use_tree:
                    r_gen_raw, r_line_raw, cur_node = stage_decision({
                        'node': cur_node,
                        'xi_g': scenario.gen_failures[t, :],
                        'xi_e': scenario.line_failures[t, :],
                    })
                elif t in repair_policy:
                    r_gen_raw  = repair_policy[t]['gen']
                    r_line_raw = repair_policy[t]['line']
                else:
                    r_gen_raw  = np.zeros(self.network.n_generators)
                    r_line_raw = np.zeros(self.network.n_lines)

                r_gen  = r_gen_raw  * (1 - a_gen)
                r_line = r_line_raw * (1 - a_line)

                r_gen_hist.append(r_gen.copy())
                r_line_hist.append(r_line.copy())

                opf_cost, _, _, _ = opf.solve(a_gen, a_line)
                repair_cost = opf.compute_repair_cost(r_gen, r_line)
                total += opf_cost + repair_cost

                if t < self.params.T - 1:
                    xi_gen_next  = scenario.gen_failures[t + 1, :].astype(float)
                    xi_line_next = scenario.line_failures[t + 1, :].astype(float)
                    a_gen, a_line = _next_availability(
                        a_gen, a_line,
                        xi_gen_next, xi_line_next,
                        r_gen_hist, r_line_hist,
                        tau, t,
                    )

            test_costs.append(total)

        eval_time_s = time.time() - t_eval_start
        evaluation = {
            'n_test_scenarios': n_test_scenarios,
            'mean_cost': np.mean(test_costs),
            'std_cost':  np.std(test_costs),
            'min_cost':  np.min(test_costs),
            'max_cost':  np.max(test_costs),
            'test_costs': test_costs,
            'eval_time_s': round(eval_time_s, 3),
            'mean_time_per_scenario_s': round(eval_time_s / n_test_scenarios, 6) if n_test_scenarios > 0 else 0.0,
        }

        if verbose:
            print()
            print("Evaluation Results:")
            print("-" * 70)
            print(f"  Mean cost: ${evaluation['mean_cost']:.2f}")
            print(f"  Std dev:   ${evaluation['std_cost']:.2f}")
            print(f"  Min cost:  ${evaluation['min_cost']:.2f}")
            print(f"  Max cost:  ${evaluation['max_cost']:.2f}")
            print("=" * 70)

        return evaluation
    
    def get_solution_summary(self):
        """Return a concise dict with objective, solve time, and repair decisions."""
        if self.solution is None:
            return None
        
        return {
            'objective_value': self.solution['objective_value'],
            'solve_time': self.solve_time,
            'n_scenarios': len(self.scenarios),
            'repairs': self.solution['repairs']
        }


def create_saa_solver(network_data, model_params, seed=None):
    """Instantiate and return an SAASolver."""
    return SAASolver(network_data, model_params, seed=seed)


if __name__ == "__main__":
    from network_data import create_network
    from model_parameters import create_model_parameters

    print("Testing SAA Solver")
    print()

    network = create_network()
    params = create_model_parameters(network, seed=42)

    # A full-horizon B=2 tree has 2^T sample paths (2^14 = 16,384 at config.T = 14),
    # too many for a quick self-test, so this demo truncates the horizon to T = 3.
    # main.py runs use the full params.T.
    params.T = 3

    saa = create_saa_solver(network, params, seed=42)
    solution = saa.solve(
        n_scenarios=20,                 # not used in tree mode
        mode='tree',
        branching=2,
        solver_name=DATA_SOLVER_NAME,   # solver_config default (ED_OPF_SOLVER overrides)
        time_limit=300,
        verbose=True
    )

    if solution is not None:
        evaluation = saa.evaluate_solution(n_test_scenarios=50, verbose=True)
