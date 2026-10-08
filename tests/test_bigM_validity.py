"""
Big-M validity and tightness for the conditional DC power-flow law
==================================================================
The stage DC-OPF enforces the DC law conditionally on line availability:

    |f_e - b_e*(theta_from - theta_to)| <= M_e * (1 - a_line[e]),   M_e = 2*pi*b_e

with voltage angles bounded to [-pi, pi] and theta_slack = 0. The thermal bounds
fix the flow of a failed line to f_e = 0, so for that line the constraint reads
|b_e*(theta_from - theta_to)| <= M_e. Two properties matter for the choice of M_e:

  (1) Validity: the big-M must never cut off a physically feasible dispatch.
      A failed line must allow any angle difference the angle box allows, so
      M_e must be at least max |b_e*(theta_from - theta_to)| with both angles
      in [-pi, pi], which for b_e > 0 equals b_e * (pi - (-pi)) = 2*pi*b_e. A
      smaller constant would constrain the endpoint angles of a failed line.

  (2) Tightness: for a line between two non-slack buses, no smaller constant is
      valid over the angle box, so M_e = 2*pi*b_e gives the strongest LP
      relaxation this formulation allows. This matters in the Pyomo MIPs
      (problem_formulation.py), which use the same constant with a binary
      a_line; in the stage LP a_line is fixed. A line at the slack bus
      (theta_slack = 0) would need only pi*b_e.

The main validity check compares behavior: for a fixed availability the big-M
model must give the same dispatch cost as an independent DC-OPF solved on the
surviving subnetwork (failed lines deleted, no big-M). If the two agree over
many random outage patterns, the big-M model reproduces the post-contingency
physics without over- or under-constraining the network.

test_bigM_is_valid_and_reports_usage also prints each M_e and how much of it the
solutions use. It prints with output capture disabled, so the table appears
without -s.
"""

import numpy as np
import pytest
from scipy.optimize import linprog

from network_data import create_network
from model_parameters import create_model_parameters
from Pseudo_DFL_stage_opf import StagewiseOPF

ATOL = 1e-6


# ---------------------------------------------------------------------------
# Full solve of the big-M stage LP (exposes the whole variable vector x)
# ---------------------------------------------------------------------------

def _bigM_full_solve(opf, a_gen, a_line):
    """Solve the big-M stage LP of StagewiseOPF.solve with scipy HiGHS.

    Returns (result, theta, f): the linprog result, the angles of all buses
    (theta_slack = 0) and the line flows. theta and f are None if the solve fails.
    """
    a_gen  = np.asarray(a_gen,  dtype=float)
    a_line = np.asarray(a_line, dtype=float)

    lb = opf.lb_static.copy()
    ub = opf.ub_static.copy()
    ub[: opf.n_gen] = opf.ub_pmax * a_gen
    if opf.n_line > 0:
        ub[opf.f_off:] = opf.f_max * a_line
        lb[opf.f_off:] = opf.f_min * a_line

    if opf.n_line > 0:
        rhs = opf.big_M * (1.0 - a_line)
        A_ub, b_ub = opf.A_bigM, np.concatenate([rhs, rhs])
    else:
        A_ub = b_ub = None

    result = linprog(opf.c, A_ub=A_ub, b_ub=b_ub,
                     A_eq=opf.A_eq, b_eq=opf.b_eq,
                     bounds=list(zip(lb, ub)), method="highs")
    if not result.success:
        return result, None, None
    x = result.x
    theta = np.zeros(opf.n_bus)
    theta[opf.non_slack] = x[opf.theta_off:opf.f_off]
    f = x[opf.f_off:]
    return result, theta, f


# ---------------------------------------------------------------------------
# Independent reference: DC-OPF on the surviving subnetwork (no big-M)
# ---------------------------------------------------------------------------

def _active_topology_cost(net, params, a_gen, a_line):
    """Exact DC-OPF cost with failed lines removed from the network.

    Angles enter the nodal balance only through available lines, each available
    line obeys the ordinary DC law within its thermal limit, and failed lines are
    absent. The LP has no big-M and no flow variables, so it is an independent
    formulation of the physics the big-M model should reproduce. Returns None if
    the LP is not solved.
    """
    a_gen  = np.asarray(a_gen,  dtype=float)
    a_line = np.asarray(a_line, dtype=float)

    n_gen, n_load, n_bus = net.n_generators, net.n_loads, net.n_buses
    slack = net.slack_bus
    non_slack = sorted(i for i in range(n_bus) if i != slack)
    theta_off = n_gen + n_load
    n_var = n_gen + n_load + len(non_slack)
    K = params.load_shedding_cost

    c = np.zeros(n_var)
    c[:n_gen] = net.gen_cost
    c[n_gen:n_gen + n_load] = -K

    # DC Laplacian over available lines only (the slack column is dropped in A_eq).
    B_full = np.zeros((n_bus, n_bus))
    for e in range(net.n_lines):
        if a_line[e] < 0.5:
            continue
        fb, tb = net.line_connections[e]
        b_e = net.line_susceptances[e]
        B_full[fb, fb] += b_e; B_full[tb, tb] += b_e
        B_full[fb, tb] -= b_e; B_full[tb, fb] -= b_e

    A_eq = np.zeros((n_bus, n_var))
    for n_idx in range(n_bus):
        for g in range(n_gen):
            if net.gen_buses[g] == n_idx:
                A_eq[n_idx, g] = 1.0
        for l in range(n_load):
            if net.load_buses[l] == n_idx:
                A_eq[n_idx, n_gen + l] = -1.0
        for j, ns_bus in enumerate(non_slack):
            A_eq[n_idx, theta_off + j] = -B_full[n_idx, ns_bus]

    # Thermal limits on available lines only.
    rows, ub = [], []
    for e in range(net.n_lines):
        if a_line[e] < 0.5:
            continue
        fb, tb = net.line_connections[e]
        b_e = net.line_susceptances[e]
        row = np.zeros(n_var)
        if fb != slack:
            row[theta_off + non_slack.index(fb)] = b_e
        if tb != slack:
            row[theta_off + non_slack.index(tb)] = -b_e
        rows.append(row);  ub.append(net.line_flow_max[e])   #  flow <= fmax
        rows.append(-row); ub.append(-net.line_flow_min[e])  # -flow <= -fmin
    A_ub = np.vstack(rows) if rows else None
    b_ub = np.array(ub) if rows else None

    lb = np.zeros(n_var); lb[theta_off:] = -np.pi
    hb = np.empty(n_var)
    hb[:n_gen] = net.gen_pmax * a_gen
    hb[n_gen:n_gen + n_load] = net.load_nominal
    hb[theta_off:] = np.pi

    r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n_bus),
                bounds=list(zip(lb, hb)), method="highs")
    if not r.success:
        return None
    p_gen  = np.clip(r.x[:n_gen], 0.0, None)
    d_load = np.clip(r.x[n_gen:n_gen + n_load], 0.0, net.load_nominal)
    return (float(np.dot(net.gen_cost, p_gen))
            + float(K * np.sum(net.load_nominal - d_load)))


def _random_states(net, rng, n):
    for _ in range(n):
        yield (rng.integers(0, 2, net.n_generators).astype(float),
               rng.integers(0, 2, net.n_lines).astype(float))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bigM_matches_active_topology():
    """The big-M model must reproduce the exact removed-line DC-OPF cost on 300
    random availability patterns (relative gap below 1e-6). Agreement shows that
    the big-M neither cuts off feasible dispatch (validity) nor lets flow onto
    failed lines."""
    net = create_network()
    params = create_model_parameters(net, seed=0)
    opf = StagewiseOPF(net, params, lp_method="highs")

    rng = np.random.default_rng(12345)
    worst = 0.0
    for a_gen, a_line in _random_states(net, rng, 300):
        cost_bigM, _, _, feasible = opf.solve(a_gen, a_line)
        cost_ref = _active_topology_cost(net, params, a_gen, a_line)
        assert feasible and cost_ref is not None
        # Relative gap (costs can be large when load is shed at penalty K).
        denom = max(1.0, abs(cost_ref))
        worst = max(worst, abs(cost_bigM - cost_ref) / denom)
    assert worst < 1e-6, f"big-M vs active-topology relative cost gap = {worst:.2e}"


def test_bigM_is_valid_and_reports_usage(capsys):
    """Checks M_e = 2*pi*b_e, that a failed line's DC-law residual stays within
    its big-M (validity) and its flow is zero, and that an available line
    satisfies the DC law (residual below 1e-4). Prints the big-M values and how
    much of each the solutions use, so the constant can be inspected."""
    net = create_network()
    params = create_model_parameters(net, seed=0)
    opf = StagewiseOPF(net, params, lp_method="highs")

    b = np.asarray(net.line_susceptances, dtype=float)
    M = opf.big_M
    assert np.allclose(M, 2.0 * np.pi * b), "M_e must equal 2*pi*b_e"

    rng = np.random.default_rng(7)
    max_used_ratio = 0.0            # max |b*dtheta| / M_e on failed lines
    max_live_residual = 0.0         # max |f - b*dtheta| on available lines
    max_failed_flow = 0.0           # max |f| on failed lines
    n_failed_binding = 0            # failed-line samples with |b*dtheta| within 1e-6 of M_e
    n_failed_total = 0

    for a_gen, a_line in _random_states(net, rng, 300):
        result, theta, f = _bigM_full_solve(opf, a_gen, a_line)
        assert result.success
        for e in range(net.n_lines):
            fb, tb = net.line_connections[e]
            dc = b[e] * (theta[fb] - theta[tb])
            residual = abs(f[e] - dc)
            if a_line[e] > 0.5:
                max_live_residual = max(max_live_residual, residual)
            else:
                n_failed_total += 1
                max_failed_flow = max(max_failed_flow, abs(f[e]))
                # Validity: residual (= |b*dtheta| since f=0) must not exceed M_e.
                assert residual <= M[e] + 1e-6, (
                    f"big-M VIOLATED on failed line {e}: "
                    f"|b*dtheta| = {residual:.4f} > M_e = {M[e]:.4f}"
                )
                max_used_ratio = max(max_used_ratio, residual / M[e])
                if residual > M[e] - 1e-6:
                    n_failed_binding += 1

    with capsys.disabled():
        print("\n--- big-M diagnostics (toy network, 300 random outage states) ---")
        print("line       b_e      M_e = 2*pi*b_e     f_max")
        for e in range(net.n_lines):
            print(f"  {e:<3d}   {b[e]:7.3f}   {M[e]:12.4f}   {net.line_flow_max[e]:8.2f}")
        print(f"\nfailed-line samples                : {n_failed_total}")
        print(f"max |b*dtheta| / M_e on failed lines : {max_used_ratio:.4f}  "
              f"(1.0 = angle box saturated; big-M valid iff <= 1)")
        print(f"failed-line big-M ever binding       : "
              f"{n_failed_binding} / {n_failed_total}")
        print(f"max |f| on failed lines              : {max_failed_flow:.2e}  "
              f"(should be ~0)")
        print(f"max |f - b*dtheta| on live lines     : {max_live_residual:.2e}  "
              f"(should be ~0: DC law binds)")

    # Failed lines carry no flow, available lines obey the DC law, and no failed
    # line uses more than its M_e.
    assert max_failed_flow < ATOL
    assert max_live_residual < 1e-4
    assert max_used_ratio <= 1.0 + 1e-9
