"""
Group 1: Failure Probability Model
==================================
Verifies mathematical properties of the log-linear logit formula

    lambda^t[k] = 1 / (1 + (t + K)^{-A[k]} * exp(B[k] * t))

with:  B[k] = logit(p_max[k]) / delta,   A[k] = (t_star + K) * B[k]
       delta = (t_star + K) * ln(t_star + K) - t_star

Here K is the failure-model shift FAILURE_K (not the load-shedding cost). The
tests assume the config.py default t_star = FAILURE_T_STAR = 1.

Tests
-----
1. lambda^1[k] == p_max[k]  to within 1e-10 (peak value at t_star = 1)
2. lambda^t[k] in (0, 1)    for all stages t and all components k
3. lambda^t[k] is strictly  decreasing in t for all k
   (burn-in model with its peak at t_star = 1: highest risk at the first stage,
    decaying roughly exponentially afterwards)
"""

import numpy as np
import pytest

# Absolute tolerance used for the peak-value identity check
ATOL_PEAK = 1e-10


# ===========================================================================
# Test 1: lambda^{t_star}[k] == p_max[k], with t_star = 1
# ===========================================================================

class TestPeakValueIdentity:
    """lambda^1[k] must equal p_max[k] to within ATOL_PEAK (1e-10)."""

    def test_generators_at_stage_1(self, params):
        for g in range(params.n_generators):
            lam   = params.lambda_g[g, 0]   # index 0 <-> stage t=1
            p_max = params.p_max_gen[g]
            assert abs(lam - p_max) < ATOL_PEAK, (
                f"Generator {g}: lambda^1 = {lam:.15f}  !=  p_max = {p_max}"
            )

    def test_lines_at_stage_1(self, params):
        for e in range(params.n_lines):
            lam   = params.lambda_e[e, 0]
            p_max = params.p_max_line[e]
            assert abs(lam - p_max) < ATOL_PEAK, (
                f"Line {e}: lambda^1 = {lam:.15f}  !=  p_max = {p_max}"
            )


# ===========================================================================
# Test 2: lambda^t[k] in (0, 1)
# ===========================================================================

class TestProbabilityRange:
    """All failure probabilities must lie in the open interval (0, 1)."""

    def test_generator_probabilities_strictly_positive(self, params):
        assert np.all(params.lambda_g > 0), (
            f"Generator lambda has non-positive entry. Min = {params.lambda_g.min()}"
        )

    def test_generator_probabilities_strictly_below_one(self, params):
        assert np.all(params.lambda_g < 1), (
            f"Generator lambda has entry >= 1. Max = {params.lambda_g.max()}"
        )

    def test_line_probabilities_strictly_positive(self, params):
        assert np.all(params.lambda_e > 0), (
            f"Line lambda has non-positive entry. Min = {params.lambda_e.min()}"
        )

    def test_line_probabilities_strictly_below_one(self, params):
        assert np.all(params.lambda_e < 1), (
            f"Line lambda has entry >= 1. Max = {params.lambda_e.max()}"
        )


# ===========================================================================
# Test 3: strictly decreasing in t
# ===========================================================================

class TestMonotonicity:
    """
    lambda^t[k] is strictly decreasing for t = 1, 2, ..., T.

    In the burn-in model the failure risk peaks at t_star = 1 and decreases
    monotonically after that.
    """

    def test_generators_strictly_decreasing(self, params):
        for g in range(params.n_generators):
            curve = params.lambda_g[g, :]            # shape (T,); index i -> stage i+1
            diffs = np.diff(curve)                   # diff[i] = lambda^{i+2} - lambda^{i+1}
            bad   = np.where(diffs >= 0)[0]
            assert len(bad) == 0, (
                f"Generator {g}: lambda is not strictly decreasing. "
                f"Non-negative differences at (0-indexed) positions: {bad}. "
                f"Values: {curve}"
            )

    def test_lines_strictly_decreasing(self, params):
        for e in range(params.n_lines):
            curve = params.lambda_e[e, :]
            diffs = np.diff(curve)
            bad   = np.where(diffs >= 0)[0]
            assert len(bad) == 0, (
                f"Line {e}: lambda is not strictly decreasing. "
                f"Non-negative differences at (0-indexed) positions: {bad}. "
                f"Values: {curve}"
            )
