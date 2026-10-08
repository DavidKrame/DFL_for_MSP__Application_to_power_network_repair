"""
Group 4: OPF Constraint Satisfaction
====================================
Checks that the stage DC-OPF LP solution satisfies every physical constraint
for a set of availability configurations of the toy (5-bus) network.

The 8 constraint checks below run for each of the 8 availability
configurations through pytest parametrization (64 test cases). One more test,
on a 3-bus cycle, checks that a failed line lets power reroute over the rest
of the network.

Constraints verified
--------------------
  (a) LP always returns a feasible solution (load shedding ensures this)
  (b) System-wide power balance:  sum(p_gen) == sum(d_load)
  (c) Per-bus power balance:      A_eq · x == 0  (all buses, including slack)
  (d) Generation bounds:          0 <= p_gen[g] <= Pmax_g * a_gen[g]
  (e) Load shedding bounds:       0 <= d_load[l] <= D_l
  (f) Line flows:                 Fmin_e * a_line[e] <= f_e <= Fmax_e * a_line[e];
                                  f_e = b_e * (theta_from - theta_to) on an
                                  available line, f_e = 0 on a failed line
  (g) Slack bus angle == 0        (eliminated from LP by construction)
  (h) Cost formula consistency:   cost == sum(c_g*p_g) + K*sum(D_l - d_l)

Helper
------
_full_solve(opf, a_gen, a_line) re-solves the LP from the OPF's public
matrices and returns the scipy result, whose x is the full variable vector
[p_gen | d_load | theta_ns | f]. Checks (c), (f) and (g) and the 3-bus test
use it.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from Pseudo_DFL_stage_opf import StagewiseOPF

# Numerical tolerance for constraint checks
ATOL = 1e-6


# ===========================================================================
# Availability configurations under test
# Format: (a_gen, a_line, description_string)
# ===========================================================================

CONFIGS = [
    ([1, 1], [1, 1, 1, 1, 1], "all_available"),
    ([0, 1], [1, 1, 1, 1, 1], "gen0_down"),
    ([1, 0], [1, 1, 1, 1, 1], "gen1_down"),
    ([0, 0], [1, 1, 1, 1, 1], "both_gens_down"),
    ([1, 1], [0, 1, 1, 1, 1], "line0_down"),
    ([1, 1], [1, 1, 1, 1, 0], "line4_down"),
    ([0, 1], [1, 0, 1, 1, 1], "gen0_and_line1_down"),
    ([1, 1], [0, 0, 0, 0, 0], "all_lines_down"),
]


def _config_ids():
    return [c[2] for c in CONFIGS]


# ===========================================================================
# Helper: re-solve the LP and return the scipy result (full vector in .x)
# ===========================================================================

def _full_solve(opf, a_gen, a_line):
    """
    Re-solve the LP from the OPF's public matrices and return the scipy result,
    with the full variable vector result.x = [p_gen, d_load, theta_ns, f].

    Same LP as the scipy path of StagewiseOPF.solve() with method 'highs' (the
    default backend). solve() returns only (cost, p_gen, d_load, feasible); the
    balance, angle and flow checks need the complete solution vector.
    """
    a_gen  = np.asarray(a_gen,  dtype=float)
    a_line = np.asarray(a_line, dtype=float)

    lb = opf.lb_static.copy()
    ub = opf.ub_static.copy()
    ub[: opf.n_gen] = opf.ub_pmax * a_gen
    if opf.n_line > 0:
        ub[opf.f_off:] = opf.f_max * a_line   # thermal box on flows
        lb[opf.f_off:] = opf.f_min * a_line

    if opf.n_line > 0:
        rhs = opf.big_M * (1.0 - a_line)      # big-M relaxes failed-line DC law
        A_ub = opf.A_bigM
        b_ub = np.concatenate([rhs, rhs])
    else:
        A_ub = b_ub = None

    bounds = list(zip(lb, ub))
    return linprog(
        opf.c,
        A_ub=A_ub, b_ub=b_ub,
        A_eq=opf.A_eq, b_eq=opf.b_eq,
        bounds=bounds,
        method="highs",
    )


# ===========================================================================
# Parametrized test class
# ===========================================================================

@pytest.mark.parametrize("a_gen,a_line,desc", CONFIGS, ids=_config_ids())
class TestOPFConstraints:
    """
    Each method runs once per availability configuration in CONFIGS, giving one
    pytest node per (method, configuration) pair.
    """

    # ------------------------------------------------------------------
    # (a) Feasibility
    # ------------------------------------------------------------------

    def test_a_feasibility(self, opf, a_gen, a_line, desc):
        """LP must always return feasible=True (load shedding ensures this)."""
        _, _, _, feasible = opf.solve(a_gen, a_line)
        assert feasible, (
            f"[{desc}] LP returned infeasible — should always be feasible "
            f"because load shedding covers any imbalance"
        )

    # ------------------------------------------------------------------
    # (b) System-wide power balance
    # ------------------------------------------------------------------

    def test_b_system_power_balance(self, opf, a_gen, a_line, desc):
        """sum(p_gen) == sum(d_load): generation equals served load in a lossless DC network."""
        _, p_gen, d_load, feasible = opf.solve(a_gen, a_line)
        if not feasible:
            pytest.skip("Infeasible LP — covered by test_a_feasibility")

        imbalance = abs(np.sum(p_gen) - np.sum(d_load))
        assert imbalance < ATOL, (
            f"[{desc}] System-wide power imbalance = {imbalance:.2e} MW  "
            f"(sum p_gen = {np.sum(p_gen):.4f}, sum d_load = {np.sum(d_load):.4f})"
        )

    # ------------------------------------------------------------------
    # (c) Per-bus power balance  (A_eq · x = 0)
    # ------------------------------------------------------------------

    def test_c_per_bus_power_balance(self, opf, a_gen, a_line, desc):
        """Power balance must hold at each individual bus, including the slack bus."""
        result = _full_solve(opf, a_gen, a_line)
        if not result.success:
            pytest.skip("LP infeasible — covered by test_a_feasibility")

        residuals = np.abs(opf.A_eq @ result.x - opf.b_eq)
        worst = residuals.max()
        assert worst < ATOL, (
            f"[{desc}] Per-bus power balance violated. "
            f"Max residual = {worst:.2e} at bus(es) {np.where(residuals >= ATOL)[0]}"
        )

    # ------------------------------------------------------------------
    # (d) Generation bounds
    # ------------------------------------------------------------------

    def test_d_generation_bounds(self, opf, network, a_gen, a_line, desc):
        """0 <= p_gen[g] <= Pmax_g * a_gen[g] for every generator."""
        _, p_gen, _, feasible = opf.solve(a_gen, a_line)
        if not feasible:
            pytest.skip("Infeasible LP")

        a_gen_arr = np.asarray(a_gen, dtype=float)
        for g in range(network.n_generators):
            ub = network.gen_pmax[g] * a_gen_arr[g]
            assert p_gen[g] >= -ATOL, (
                f"[{desc}] p_gen[{g}] = {p_gen[g]:.6f} < 0"
            )
            assert p_gen[g] <= ub + ATOL, (
                f"[{desc}] p_gen[{g}] = {p_gen[g]:.6f} > Pmax*a = {ub:.4f}"
            )

    # ------------------------------------------------------------------
    # (e) Load shedding bounds
    # ------------------------------------------------------------------

    def test_e_load_shedding_bounds(self, opf, network, a_gen, a_line, desc):
        """0 <= d_load[l] <= D_l for every load (served load within demand)."""
        _, _, d_load, feasible = opf.solve(a_gen, a_line)
        if not feasible:
            pytest.skip("Infeasible LP")

        for l in range(network.n_loads):
            assert d_load[l] >= -ATOL, (
                f"[{desc}] d_load[{l}] = {d_load[l]:.6f} < 0"
            )
            assert d_load[l] <= network.load_nominal[l] + ATOL, (
                f"[{desc}] d_load[{l}] = {d_load[l]:.6f} > D_l = {network.load_nominal[l]:.4f}"
            )

    # ------------------------------------------------------------------
    # (f) Line flow limits, DC law, and outage behavior
    # ------------------------------------------------------------------

    def test_f_line_flow_bounds(self, opf, network, a_gen, a_line, desc):
        """Each explicit flow f_e lies in the availability-scaled thermal box
        [Fmin_e * a_line[e], Fmax_e * a_line[e]]. On an available line f_e
        equals the DC law b_e * (theta_from - theta_to); a failed line carries
        zero flow. That a failed line leaves its endpoint angles free is checked
        by test_failed_cycle_line_reroutes_without_shorting_angles."""
        result = _full_solve(opf, a_gen, a_line)
        if not result.success:
            pytest.skip("Infeasible LP")

        x          = result.x
        f          = x[opf.f_off:]           # explicit line flows
        dc_flow    = opf.A_line @ x          # b_e * (theta_from - theta_to)
        a_line_arr = np.asarray(a_line, dtype=float)
        fmin       = np.array(network.line_flow_min, dtype=float) * a_line_arr
        fmax       = np.array(network.line_flow_max, dtype=float) * a_line_arr

        for e in range(network.n_lines):
            assert fmin[e] - ATOL <= f[e] <= fmax[e] + ATOL, (
                f"[{desc}] flow[{e}] = {f[e]:.4f} outside "
                f"[{fmin[e]:.4f}, {fmax[e]:.4f}]"
            )
            if a_line_arr[e] > 0.5:
                # Available: flow equals the DC power-flow law.
                assert abs(f[e] - dc_flow[e]) < 1e-4, (
                    f"[{desc}] available line {e}: flow {f[e]:.4f} "
                    f"!= b*dtheta {dc_flow[e]:.4f}"
                )
            else:
                # Failed: zero flow (the big-M term relaxes the DC law, so the
                # line does not tie its endpoint angles).
                assert abs(f[e]) < ATOL, (
                    f"[{desc}] failed line {e} carries nonzero flow {f[e]:.4f}"
                )

    # ------------------------------------------------------------------
    # (g) Slack bus angle == 0
    # ------------------------------------------------------------------

    def test_g_slack_bus_angle_is_zero(self, opf, a_gen, a_line, desc):
        """
        The slack bus angle is eliminated from the LP variable vector by
        construction (theta_slack = 0 is enforced structurally, not as a
        bound): the angle block theta_ns has one column per bus in
        opf.non_slack. The test checks that the slack bus is not in that list.
        The power balance at the slack bus is checked by test_c.
        """
        result = _full_solve(opf, a_gen, a_line)
        if not result.success:
            pytest.skip("Infeasible LP")

        # The slack bus must have no angle variable.
        slack = opf.slack
        assert slack not in opf.non_slack, (
            f"Slack bus {slack} appears in the non-slack bus list — "
            f"it should have been eliminated from the LP."
        )

    # ------------------------------------------------------------------
    # (h) Cost formula consistency
    # ------------------------------------------------------------------

    def test_h_cost_formula_consistency(self, opf, network, params, a_gen, a_line, desc):
        """
        Returned cost == sum(c_g * p_g) + K * sum(D_l - d_l)
        where K = load_shedding_cost.
        """
        cost, p_gen, d_load, feasible = opf.solve(a_gen, a_line)
        if not feasible:
            pytest.skip("Infeasible LP")

        expected = (
            float(np.dot(network.gen_cost, p_gen))
            + float(params.load_shedding_cost * np.sum(network.load_nominal - d_load))
        )
        assert abs(cost - expected) < ATOL, (
            f"[{desc}] returned cost = {cost:.6f}, recomputed = {expected:.6f}, "
            f"diff = {abs(cost - expected):.2e}"
        )


# ===========================================================================
# Line outage on a cycle: power reroutes, endpoint angles stay free
# ===========================================================================

def _three_bus_opf():
    """Minimal 3-bus network: generator at bus 0, load at bus 1, transit bus 2,
    and three unit-susceptance lines forming a single cycle 0-1-2. Capacities and
    generation are ample, so the only thing that can prevent serving the load is
    an incorrect outage model."""
    net = SimpleNamespace(
        n_buses=3, n_lines=3, n_generators=1, n_loads=1, slack_bus=0,
        gen_buses=[0],
        gen_cost=np.array([1.0]),
        gen_pmin=np.array([0.0]),
        gen_pmax=np.array([10.0]),
        load_buses=[1],
        load_nominal=np.array([1.0]),
        line_connections=[(0, 1), (0, 2), (1, 2)],
        line_susceptances=np.array([1.0, 1.0, 1.0]),
        line_flow_min=np.array([-5.0, -5.0, -5.0]),
        line_flow_max=np.array([5.0, 5.0, 5.0]),
    )
    params = SimpleNamespace(load_shedding_cost=1000.0)
    return StagewiseOPF(net, params, lp_method="highs")


def test_failed_cycle_line_reroutes_without_shorting_angles():
    """Failing the direct line (0,1) must still serve the full load by rerouting
    over 0-2-1, and must not force the failed line's endpoint angles to be equal.

    If the zero-flow limit of a failed line were imposed on the angle expression
    b_e * (theta_0 - theta_1) instead of on an explicit flow variable, it would
    force theta_0 = theta_1. Flow over 0-2-1 needs theta_0 > theta_2 > theta_1,
    so the whole load would then be shed."""
    opf = _three_bus_opf()

    # With every line available the load is fully served.
    _, _, d_up, ok_up = opf.solve([1.0], [1.0, 1.0, 1.0])
    assert ok_up and abs(d_up.sum() - 1.0) < ATOL

    # Fail the direct line (0,1); the surviving path 0-2-1 must carry the load.
    cost, p_gen, d_load, feasible = opf.solve([1.0], [0.0, 1.0, 1.0])
    assert feasible
    assert abs(d_load.sum() - 1.0) < 1e-6, (
        f"failed cycle line caused artificial shedding: served {d_load.sum():.4f}"
    )

    # Inspect the full solution vector: failed line has zero flow, and its
    # endpoint angles are free to differ (here they must, to reroute the power).
    a_line = np.array([0.0, 1.0, 1.0])
    result = _full_solve(opf, np.array([1.0]), a_line)
    assert result.success
    x = result.x
    f = x[opf.f_off:]
    theta = np.concatenate([[0.0], x[opf.theta_off:opf.f_off]])  # slack bus 0 has angle 0

    assert abs(f[0]) < ATOL, f"failed line carries flow {f[0]:.4f}"
    assert abs(theta[0] - theta[1]) > 1e-3, (
        "failed line still forces theta_from == theta_to "
        f"(|dtheta| = {abs(theta[0] - theta[1]):.2e})"
    )
