"""
Group 5: Integration Tests
==========================
Cross-module consistency checks that are independent of any single
module's internal implementation.

Tests
-----
17. No-failure scenario: the stage OPF cost is deterministic and non-negative
    (same network and same availability give the same cost on every call)
18. SAA in Wait-and-See mode with n=1 scenario returns a finite, non-negative
    objective [marked slow; requires Pyomo and GLPK, skipped without them]
19. Dimension consistency: the toy (5-bus) network's 2 generators and 5 lines
    (n_gen + n_lines == 7) give consistent array shapes in network_data,
    model_parameters, scenario_generator, state_dynamics and the OPF solver
"""

import shutil

import numpy as np
import pytest

from scenario_generator import create_scenario_generator


# ===========================================================================
# Test 17: no-failure scenario, deterministic baseline cost
# ===========================================================================

class TestNoFailureScenario:
    """
    With all components available (a_gen = 1, a_line = 1), the single-stage
    OPF has no randomness. The same call must return the same cost every time,
    and that cost must be non-negative.
    """

    def test_all_available_cost_is_non_negative(self, opf, params):
        a_gen  = np.ones(params.n_generators)
        a_line = np.ones(params.n_lines)
        cost, _, _, feasible = opf.solve(a_gen, a_line)
        assert feasible
        assert cost >= 0.0, (
            f"Cost with all components available is negative: {cost:.4f}"
        )

    def test_all_available_cost_is_deterministic(self, opf, params):
        """Identical inputs must produce identical outputs (LP is deterministic)."""
        a_gen  = np.ones(params.n_generators)
        a_line = np.ones(params.n_lines)
        cost1, pg1, dl1, _ = opf.solve(a_gen, a_line)
        cost2, pg2, dl2, _ = opf.solve(a_gen, a_line)

        assert abs(cost1 - cost2) < 1e-8, (
            f"OPF cost is not reproducible: {cost1:.8f} vs {cost2:.8f}"
        )
        assert np.allclose(pg1, pg2, atol=1e-8), (
            "Generation dispatch differs across identical calls"
        )
        assert np.allclose(dl1, dl2, atol=1e-8), (
            "Load dispatch differs across identical calls"
        )

    def test_cost_identical_across_all_stages_with_all_up(self, opf, params):
        """
        Calling the LP T times with the same all-up availability must return
        the same cost each time (the LP is stateless).
        """
        a_gen  = np.ones(params.n_generators)
        a_line = np.ones(params.n_lines)
        costs  = [opf.solve(a_gen, a_line)[0] for _ in range(params.T)]
        for t, c in enumerate(costs):
            assert abs(c - costs[0]) < 1e-8, (
                f"Stage {t+1}: cost {c:.8f} differs from stage 1 cost {costs[0]:.8f} "
                f"despite identical all-up availability"
            )


# ===========================================================================
# Test 18: SAA with n=1 returns a valid solution  [slow]
# ===========================================================================

@pytest.mark.slow
class TestSAAOneSampleValid:
    """
    SAA with a single scenario must produce a non-negative, finite objective.
    Only basic solution validity is checked; comparing the value with a direct
    MIP solve would require matching seeds and internal state.

    Requires: Pyomo  (`pip install pyomo`)  and  GLPK  (`brew install glpk`).
    The test is skipped when Pyomo or the GLPK executable is missing.
    """

    def test_saa_n1_returns_non_negative_objective(self, network, params):
        pytest.importorskip("pyomo", reason="Pyomo is required for SAA tests")
        if shutil.which("glpsol") is None:
            pytest.skip("GLPK (glpsol) is not installed")

        from saa_solver import create_saa_solver

        saa      = create_saa_solver(network, params, seed=99)
        solution = saa.solve(
            n_scenarios=1,
            mode="ws",   # single-scenario Wait-and-See solve (not the default tree SAA)
            solver_name="glpk",
            time_limit=120,
            mip_gap=1e-4,
            verbose=False,
        )
        assert solution is not None, "SAA solver returned None — solver may have failed"
        obj = solution["objective_value"]
        assert np.isfinite(obj), f"Objective value is not finite: {obj}"
        assert obj >= 0.0,       f"Objective value is negative: {obj:.4f}"


# ===========================================================================
# Test 19: dimension consistency across modules
# ===========================================================================

class TestDimensionConsistency:
    """
    Toy network: n_generators = 2,  n_lines = 5,  n_generators + n_lines = 7.
    Every shared module must use these dimensions.
    """

    def test_network_dimensions_are_correct(self, network):
        assert network.n_generators == 2, (
            f"Expected n_generators = 2, got {network.n_generators}"
        )
        assert network.n_lines == 5, (
            f"Expected n_lines = 5, got {network.n_lines}"
        )
        assert network.n_generators + network.n_lines == 7

    def test_failure_probability_matrix_shapes(self, network, params):
        assert params.lambda_g.shape == (network.n_generators, params.T), (
            f"lambda_g shape {params.lambda_g.shape} != "
            f"({network.n_generators}, {params.T})"
        )
        assert params.lambda_e.shape == (network.n_lines, params.T), (
            f"lambda_e shape {params.lambda_e.shape} != "
            f"({network.n_lines}, {params.T})"
        )

    def test_scenario_array_shapes(self, network, params):
        gen = create_scenario_generator(params, seed=0)
        s   = gen.generate_single_scenario(scenario_id=0)
        assert s.gen_failures.shape  == (params.T, network.n_generators), (
            f"gen_failures shape {s.gen_failures.shape} != "
            f"({params.T}, {network.n_generators})"
        )
        assert s.line_failures.shape == (params.T, network.n_lines), (
            f"line_failures shape {s.line_failures.shape} != "
            f"({params.T}, {network.n_lines})"
        )

    def test_availability_array_shapes(self, dynamics, network, params):
        gen   = create_scenario_generator(params, seed=0)
        s     = gen.generate_single_scenario(scenario_id=0)
        avail = dynamics.compute_availability_trajectories(s)
        assert avail["gen"].shape  == (params.T, network.n_generators), (
            f"avail['gen'] shape {avail['gen'].shape} != "
            f"({params.T}, {network.n_generators})"
        )
        assert avail["line"].shape == (params.T, network.n_lines), (
            f"avail['line'] shape {avail['line'].shape} != "
            f"({params.T}, {network.n_lines})"
        )

    def test_opf_internal_dimensions(self, opf, network):
        assert opf.n_gen  == network.n_generators, (
            f"OPF n_gen = {opf.n_gen} != network.n_generators = {network.n_generators}"
        )
        assert opf.n_load == network.n_loads, (
            f"OPF n_load = {opf.n_load} != network.n_loads = {network.n_loads}"
        )
        assert opf.n_line == network.n_lines, (
            f"OPF n_line = {opf.n_line} != network.n_lines = {network.n_lines}"
        )
        assert opf.n_bus  == network.n_buses, (
            f"OPF n_bus = {opf.n_bus} != network.n_buses = {network.n_buses}"
        )
