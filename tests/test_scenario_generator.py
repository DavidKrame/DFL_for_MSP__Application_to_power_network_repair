"""
Group 2: Scenario Generator
===========================
Checks that the sampled failure scenarios follow the failure probabilities
lambda^t[k] of the model.

Tests
-----
4. All sampled values are binary {0, 1}
5. The empirical failure frequency over a large sample is close to lambda^t[k]
   (law of large numbers; N = 10 000, fixed seed, absolute tolerance 0.03)
6. When lambda = 0 for all (t, k): no failure is ever generated
7. When lambda = 1 for all (t, k): a failure is always generated
"""

import numpy as np
import pytest

from scenario_generator import create_scenario_generator


# ---------------------------------------------------------------------------
# Minimal stand-in for ModelParameters used in edge-case tests 6 and 7.
# ScenarioGenerator only needs: .T, .n_generators, .n_lines,
#                                .lambda_g[g, t], .lambda_e[e, t]
# ---------------------------------------------------------------------------

class _ForcedParams:
    """
    Duck-typed replacement for ModelParameters with constant lambda arrays.
    Used only for the lambda = 0 and lambda = 1 edge cases, which the real
    failure-probability formula never produces (its values lie in (0, 1)).
    """
    T = 14
    n_generators = 2
    n_lines = 5

    def __init__(self, lam_value: float):
        self.lambda_g = np.full((self.n_generators, self.T), lam_value)
        self.lambda_e = np.full((self.n_lines,      self.T), lam_value)


# ===========================================================================
# Test 4: Binary values
# ===========================================================================

class TestBinaryValues:
    """Every entry in gen_failures and line_failures must be 0 or 1."""

    N_SCENARIOS = 50

    def test_gen_failures_binary(self, params):
        gen = create_scenario_generator(params, seed=0)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            unique = np.unique(s.gen_failures)
            assert set(unique).issubset({0, 1}), (
                f"Non-binary values in gen_failures: {unique}"
            )

    def test_line_failures_binary(self, params):
        gen = create_scenario_generator(params, seed=0)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            unique = np.unique(s.line_failures)
            assert set(unique).issubset({0, 1}), (
                f"Non-binary values in line_failures: {unique}"
            )


# ===========================================================================
# Test 5: Law of large numbers
# ===========================================================================

class TestLawOfLargeNumbers:
    """
    With N = 10 000 scenarios and a fixed seed, the empirical failure
    frequency for every (component, stage) pair must be within 0.03 (absolute)
    of the theoretical lambda value.

    With the config.py defaults, p = lambda^t[k] lies in (0, 0.90): it peaks at
    p_max[k] < 0.90 at t = 1 and decreases afterwards. The standard deviation
    of an empirical frequency is sqrt(p(1-p)/N) <= sqrt(0.25 / 10000) = 0.005
    for any p, so the tolerance 0.03 is 6 standard deviations (conservative).
    """

    TOL = 0.03  # absolute tolerance (3 percentage points)

    def test_generator_empirical_frequencies(self, large_sample, params):
        # large_sample: list of N Scenario objects (N = 10 000, seed = 42)
        all_gen = np.array([s.gen_failures for s in large_sample])  # (N, T, n_gen)
        empirical = all_gen.mean(axis=0).T                           # (n_gen, T)

        for g in range(params.n_generators):
            for t in range(params.T):
                emp  = empirical[g, t]
                theo = params.lambda_g[g, t]
                err  = abs(emp - theo)
                assert err < self.TOL, (
                    f"Generator {g}, stage {t+1}: "
                    f"empirical = {emp:.4f}, theoretical = {theo:.4f}, "
                    f"error = {err:.4f} > tol = {self.TOL}"
                )

    def test_line_empirical_frequencies(self, large_sample, params):
        all_line = np.array([s.line_failures for s in large_sample])  # (N, T, n_line)
        empirical = all_line.mean(axis=0).T                            # (n_line, T)

        for e in range(params.n_lines):
            for t in range(params.T):
                emp  = empirical[e, t]
                theo = params.lambda_e[e, t]
                err  = abs(emp - theo)
                assert err < self.TOL, (
                    f"Line {e}, stage {t+1}: "
                    f"empirical = {emp:.4f}, theoretical = {theo:.4f}, "
                    f"error = {err:.4f} > tol = {self.TOL}"
                )


# ===========================================================================
# Test 6: Edge case lambda = 0
# ===========================================================================

class TestZeroFailureProbability:
    """When lambda = 0 for all components and stages, no failure is generated."""

    N_SCENARIOS = 200

    def test_zero_lambda_never_produces_gen_failure(self):
        mock = _ForcedParams(lam_value=0.0)
        gen  = create_scenario_generator(mock, seed=7)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            assert np.all(s.gen_failures == 0), (
                "A generator failure was generated despite lambda = 0"
            )

    def test_zero_lambda_never_produces_line_failure(self):
        mock = _ForcedParams(lam_value=0.0)
        gen  = create_scenario_generator(mock, seed=7)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            assert np.all(s.line_failures == 0), (
                "A line failure was generated despite lambda = 0"
            )


# ===========================================================================
# Test 7: Edge case lambda = 1
# ===========================================================================

class TestCertainFailureProbability:
    """When lambda = 1 for all components and stages, a failure is always generated."""

    N_SCENARIOS = 200

    def test_one_lambda_always_produces_gen_failure(self):
        mock = _ForcedParams(lam_value=1.0)
        gen  = create_scenario_generator(mock, seed=7)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            assert np.all(s.gen_failures == 1), (
                "A generator failure was missing despite lambda = 1"
            )

    def test_one_lambda_always_produces_line_failure(self):
        mock = _ForcedParams(lam_value=1.0)
        gen  = create_scenario_generator(mock, seed=7)
        for s in gen.generate_scenarios(self.N_SCENARIOS):
            assert np.all(s.line_failures == 1), (
                "A line failure was missing despite lambda = 1"
            )
