"""
Derated DC-OPF for the Near-sighted MPC lookahead
=================================================
The Near-sighted MPC scores repair schedules by rolling the failure process
forward at its mean (expected failures lambda), so the lookahead steps after
step 0 have fractional availability. Those steps are priced on the derated
network (StagewiseOPF.solve_derated): a line's susceptance and capacity both
scale with its availability,

    f_e = a_line[e] * b_e * (theta_from - theta_to),   |f_e| <= a_line[e] * f_max[e],

so a partly available line is a weaker conductor that still obeys the DC
flow-angle law. Passing a fractional a_line to the big-M model in solve() would
instead relax the DC law by M_e * (1 - a_line[e]).

Two properties must hold:
  (1) at binary availability the derated OPF reproduces the big-M OPF of solve()
      (cost and total served load within 1e-4);
  (2) at fractional availability the dispatch satisfies the derated balance and
      the availability-scaled thermal limit with f_e = a_line[e]*b_e*dtheta.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from network_data import create_network
from model_parameters import create_model_parameters
from Pseudo_DFL_stage_opf import StagewiseOPF

ATOL = 1e-6


@pytest.fixture(scope="module")
def opf():
    """StagewiseOPF with the scipy HiGHS backend; in this module it replaces the
    session fixture of the same name from conftest.py."""
    net = create_network()                       # toy (5-bus) network
    params = create_model_parameters(net, seed=0)
    return StagewiseOPF(net, params, lp_method="highs")


def _derated_full_solve(opf, a_gen, a_line):
    """Solve the derated LP and return the linprog result.

    Builds the same LP as StagewiseOPF.solve_derated, which does not return
    the angles. Here result.x = [p_gen, d_load, theta_ns], so the angle
    solution can be inspected.
    """
    a_gen  = np.asarray(a_gen,  dtype=float)
    a_line = np.asarray(a_line, dtype=float)
    n = opf.network
    n_gen, n_load, n_bus, n_line = opf.n_gen, opf.n_load, opf.n_bus, opf.n_line
    slack, non_slack = opf.slack, opf.non_slack
    theta_off = n_gen + n_load
    n_var = n_gen + n_load + opf.n_theta

    c = np.zeros(n_var)
    c[:n_gen]             = n.gen_cost
    c[n_gen:n_gen+n_load] = -opf.K

    b_eff = np.array(n.line_susceptances, dtype=float) * a_line
    B_full = np.zeros((n_bus, n_bus))
    for e in range(n_line):
        fb, tb = n.line_connections[e]
        be = b_eff[e]
        B_full[fb, fb] += be; B_full[tb, tb] += be
        B_full[fb, tb] -= be; B_full[tb, fb] -= be

    A_eq = np.zeros((n_bus, n_var))
    for n_idx in range(n_bus):
        for g in range(n_gen):
            if n.gen_buses[g] == n_idx:
                A_eq[n_idx, g] = 1.0
        for l in range(n_load):
            if n.load_buses[l] == n_idx:
                A_eq[n_idx, n_gen + l] = -1.0
        for j, ns_bus in enumerate(non_slack):
            A_eq[n_idx, theta_off + j] = -B_full[n_idx, ns_bus]

    rows, ub = [], []
    for e in range(n_line):
        fb, tb = n.line_connections[e]
        be = b_eff[e]
        row = np.zeros(n_var)
        if fb != slack:
            row[theta_off + non_slack.index(fb)] = be
        if tb != slack:
            row[theta_off + non_slack.index(tb)] = -be
        rows.append(row);  ub.append(a_line[e] * opf.f_max[e])
        rows.append(-row); ub.append(-a_line[e] * opf.f_min[e])
    A_ub = np.vstack(rows) if rows else None
    b_ub = np.array(ub) if rows else None

    lb = np.zeros(n_var); lb[theta_off:] = -np.pi
    hb = np.empty(n_var)
    hb[:n_gen]             = opf.ub_pmax * a_gen
    hb[n_gen:n_gen+n_load] = opf.ub_dmax
    hb[theta_off:]         = np.pi

    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n_bus),
                   bounds=list(zip(lb, hb)), method="highs")


def test_derated_matches_bigM_at_binary(opf):
    """At binary availability, solve_derated must reproduce the big-M solve()
    on 300 random patterns: same feasibility flag, and cost and total served
    load within 1e-4. How the served load splits across loads can differ
    between optimal solutions (every load has the same shedding cost K), so it
    is compared only in aggregate."""
    net = opf.network
    rng = np.random.default_rng(2024)
    worst_cost = worst_served = 0.0
    for _ in range(300):
        a_gen  = rng.integers(0, 2, net.n_generators).astype(float)
        a_line = rng.integers(0, 2, net.n_lines).astype(float)
        c1, pg1, dl1, ok1 = opf.solve(a_gen, a_line)
        c2, pg2, dl2, ok2 = opf.solve_derated(a_gen, a_line)
        assert ok1 == ok2
        worst_cost   = max(worst_cost, abs(c1 - c2))
        worst_served = max(worst_served, abs(float(dl1.sum() - dl2.sum())))
    assert worst_cost   < 1e-4, f"cost gap {worst_cost:.2e}"
    assert worst_served < 1e-4, f"total-served gap {worst_served:.2e}"


def test_derated_flow_scales_with_availability(opf):
    """At fractional availability the solution obeys the derated network:
    f_e = a_line[e]*b_e*dtheta closes the nodal balance and stays within the
    availability-scaled thermal limit."""
    net = opf.network
    b = np.asarray(net.line_susceptances, dtype=float)
    configs = [
        (np.array([1.0, 0.8]), np.array([0.7, 1.0, 0.4, 1.0, 0.9])),
        (np.array([0.5, 1.0]), np.array([0.3, 0.6, 1.0, 0.2, 0.5])),
        (np.array([1.0, 1.0]), np.array([0.0, 0.5, 0.5, 1.0, 0.0])),
    ]
    for a_gen, a_line in configs:
        res = _derated_full_solve(opf, a_gen, a_line)
        assert res.success
        x = res.x
        p_gen  = x[:opf.n_gen]
        d_load = x[opf.n_gen:opf.n_gen + opf.n_load]
        theta = np.zeros(net.n_buses)
        theta[opf.non_slack] = x[opf.n_gen + opf.n_load:]

        # Derated flows: f_e = a_line[e] * b_e * (theta_from - theta_to).
        f = np.array([
            a_line[e] * b[e] * (theta[fb] - theta[tb])
            for e, (fb, tb) in enumerate(net.line_connections)
        ])

        # (a) Thermal limit scaled by availability.
        for e in range(net.n_lines):
            assert abs(f[e]) <= a_line[e] * net.line_flow_max[e] + ATOL, (
                f"line {e}: |f|={abs(f[e]):.4f} > a*f_max={a_line[e]*net.line_flow_max[e]:.4f}"
            )

        # (b) Nodal balance closes with the derated flows.
        for n_idx in range(net.n_buses):
            inj = sum(p_gen[g] for g in range(opf.n_gen) if net.gen_buses[g] == n_idx)
            inj -= sum(d_load[l] for l in range(opf.n_load) if net.load_buses[l] == n_idx)
            out = sum(f[e] for e, (fb, tb) in enumerate(net.line_connections) if fb == n_idx)
            out -= sum(f[e] for e, (fb, tb) in enumerate(net.line_connections) if tb == n_idx)
            assert abs(inj - out) < 1e-4, f"bus {n_idx} balance residual {abs(inj-out):.2e}"
