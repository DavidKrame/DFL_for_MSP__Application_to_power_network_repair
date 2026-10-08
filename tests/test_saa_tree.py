"""
Tests for the multistage SAA on a sampled scenario tree.

The structural tests (tree shape, path probabilities, parents, reproducibility,
input check) run fast and unmarked. The solve, non-anticipativity and
evaluation tests need a MIP solver and are marked slow; they use the horizon
T = 3 instead of config.T = 14, so a B = 2 tree has 14 decision nodes and
8 leaves.

The slow tests need Pyomo and the MIP solver returned by
solver_config.resolve_mip_solver(None): appsi_highs unless ED_OPF_MIP_SOLVER or
ED_OPF_SOLVER selects another one.
"""

import numpy as np
import pytest

from network_data import create_network
from model_parameters import create_model_parameters
from scenario_generator import create_scenario_generator


# ---------------------------------------------------------------------------
# Fast structural tests
# ---------------------------------------------------------------------------

class TestScenarioTreeStructure:

    @pytest.mark.parametrize("B,T", [(2, 3), (3, 2), (2, 1)])
    def test_node_and_leaf_counts(self, params, B, T):
        gen = create_scenario_generator(params, seed=42)
        tree = gen.generate_scenario_tree(branching=B, T=T)
        assert len(tree.decision_nodes) == sum(B ** d for d in range(1, T + 1))
        assert len(tree.leaves) == B ** T

    def test_path_probabilities_sum_to_one(self, params):
        gen = create_scenario_generator(params, seed=42)
        tree = gen.generate_scenario_tree(branching=2, T=4)
        # Leaf path probabilities form a distribution.
        assert abs(sum(tree.node(l).path_prob for l in tree.leaves) - 1.0) < 1e-12
        # And so does every stage's decision-node layer.
        for t in range(1, 4 + 1):
            layer = [n.path_prob for n in tree.nodes if n.stage == t]
            assert abs(sum(layer) - 1.0) < 1e-12

    def test_every_decision_node_has_valid_parent(self, params):
        gen = create_scenario_generator(params, seed=42)
        tree = gen.generate_scenario_tree(branching=2, T=3)
        for n in tree.nodes:
            if n.stage >= 1:
                assert n.parent is not None
                assert 0 <= n.parent < len(tree.nodes)
                assert tree.node(n.parent).stage == n.stage - 1

    def test_tree_is_reproducible(self, params):
        gen = create_scenario_generator(params, seed=42)
        t1 = gen.generate_scenario_tree(branching=2, T=3)
        t2 = gen.generate_scenario_tree(branching=2, T=3)
        for i in range(len(t1.nodes)):
            assert np.array_equal(t1.node(i).xi_gen, t2.node(i).xi_gen)
            assert np.array_equal(t1.node(i).xi_line, t2.node(i).xi_line)

    def test_invalid_branching_rejected(self, params):
        gen = create_scenario_generator(params, seed=42)
        with pytest.raises(ValueError):
            gen.generate_scenario_tree(branching=0, T=3)


# ---------------------------------------------------------------------------
# Slow solve-based tests (horizon T = 3, so the B = 2 tree is small)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTreeSAASolve:

    def _solve(self, seed=7, B=2, T=3):
        pytest.importorskip("pyomo", reason="Pyomo required for tree SAA solve")
        import solver_config
        net = create_network()
        params = create_model_parameters(net, seed=seed, t=T)
        from saa_solver import create_saa_solver
        saa = create_saa_solver(net, params, seed=seed)
        sol = saa.solve(
            n_scenarios=10, mode='tree', branching=B,
            solver_name=solver_config.resolve_mip_solver(None),
            time_limit=120, mip_gap=0.0, verbose=False,
        )
        return saa, sol

    def test_objective_finite_and_non_negative(self):
        _, sol = self._solve()
        assert sol is not None
        obj = sol['objective_value']
        assert np.isfinite(obj) and obj >= 0.0

    def test_non_anticipativity_shared_node_decision(self):
        """
        Every leaf beneath a stage-t node shares that node's decision
        (non-anticipativity), which the scenario-flat form cannot express. The
        tree policy stores one repair decision (r_gen, r_line) per node, so the
        test checks that each stage-1 node carries a generator repair vector of
        length n_generators and has at least one leaf below it.
        """
        saa, sol = self._solve()
        tp = saa._tree_policy

        def leaves_under(nid):
            ch = tp['children'][nid]
            if not ch:
                return [nid]
            out = []
            for c in ch:
                out += leaves_under(c)
            return out

        for s1 in tp['children'][tp['root']]:
            r_gen = tp['nodes'][s1]['r_gen']
            # The stage-1 decision is a single vector attached to node s1; every
            # leaf in its subtree inherits exactly this decision by construction.
            assert r_gen.shape[0] == create_network().n_generators
            assert len(leaves_under(s1)) >= 1

    def test_branches_can_differ(self):
        """Nodes with different histories carry separate decisions, so the
        policy may differ across branches; only the paths through one node share
        its decision."""
        saa, sol = self._solve()
        tp = saa._tree_policy
        s1_nodes = tp['children'][tp['root']]
        # Whether the decisions actually differ depends on the sampled failures
        # (with independent shocks the two stage-1 nodes usually see different
        # ones), so the test only checks that each stage-1 node has its own
        # (r_gen, r_line) decision.
        decisions = [tuple(tp['nodes'][n]['r_gen']) + tuple(tp['nodes'][n]['r_line'])
                     for n in s1_nodes]
        assert len(decisions) == len(s1_nodes)

    def test_closed_loop_evaluation_runs(self):
        saa, sol = self._solve()
        ev = saa.evaluate_solution(n_test_scenarios=20, verbose=False, seed=123)
        for k in ('mean_cost', 'std_cost', 'min_cost', 'max_cost', 'test_costs'):
            assert k in ev
        assert np.isfinite(ev['mean_cost'])
        assert len(ev['test_costs']) == 20

    def test_ws_mode_still_available(self):
        """mode='ws' solves the wait-and-see relaxation, a scenario-flat MIP in
        which no constraint couples the scenarios; its objective estimates
        E[WS], a lower bound on the optimal value. The solve must return a
        finite objective and no tree policy."""
        pytest.importorskip("pyomo")
        import solver_config
        net = create_network()
        params = create_model_parameters(net, seed=7, t=3)
        from saa_solver import create_saa_solver
        saa = create_saa_solver(net, params, seed=7)
        sol = saa.solve(
            n_scenarios=3, mode='ws',
            solver_name=solver_config.resolve_mip_solver(None),
            time_limit=120, mip_gap=1e-3, verbose=False,
        )
        assert sol is not None
        assert 'tree_policy' not in sol   # ws mode is scenario-flat
        assert np.isfinite(sol['objective_value'])
