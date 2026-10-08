"""
Pseudo_DFL Stage OPF Module
===========================
Single-stage DC-OPF LP that prices the dispatch at each stage of the policy
rollouts (DFL, NoRepair, AlwaysRepair, Near-sighted MPC, SAA out-of-sample)
and the stage costs of exact DP.

Formulation (DC power flow, load shedding allowed). An explicit flow variable
f_e per line carries the line flow, so a failed line is forced to zero flow
while its endpoint angles stay free (an outaged line is removed from the
network, not turned into an equal-angle short):
    min  Σ_g c_g * p_g + K * Σ_l (D_l - d_l)
    s.t. Nodal power balance in flow form  (C f)[n] = p_g[n] - d[n]  for all n
         (the slack-bus row is kept: it is the only row that contains the
         slack bus's generators and loads)
         0 ≤ p_g ≤ Pmax_g * a_gen[g]
         0 ≤ d_l ≤ D_l
         conditional DC law:  |f_e - B_e*(θ_from - θ_to)| ≤ M_e * (1 - a_line[e])
         thermal limit:       Fmin_e * a_line[e] ≤ f_e ≤ Fmax_e * a_line[e]
         -π ≤ θ_n ≤ π,  θ_slack = 0  (eliminated by construction)
with big-M constant M_e = 2π * B_e. When a_line[e] = 1 the DC law binds
(f_e = B_e*Δθ); when a_line[e] = 0 the thermal limit forces f_e = 0 and the DC
law relaxes to a bound implied by the angle box, decoupling the endpoints.
Here c_g is the generation cost ($/MWh), K the load-shedding cost (value of
lost load, $/MWh), d_l the served load and D_l the nominal demand (MW), B_e
the line susceptance (MW/rad), and C the bus-line incidence matrix (+1 where a
line leaves a bus, -1 where it enters).

Solver: scipy.optimize.linprog (method 'highs' by default) or, when the LP
backend is 'gurobi', a persistent gurobipy model of the same LP. The default
backend is solver_config.LP_SOLVER. The LP matrices are built once from the
network topology; between solve() calls only the variable bounds (generation
and line-flow limits, which depend on availability) and the big-M right-hand
side change. solve_derated() rebuilds its matrices on every call.

Variable ordering in solve(): x = [p_gen (n_gen), d_load (n_load),
θ_ns (n_bus-1), f (n_line)] (non-slack voltage angles; θ_slack = 0 eliminated;
f = explicit line flows). solve_derated() uses x = [p_gen, d_load, θ_ns].
"""

import numpy as np
from scipy.optimize import linprog

import solver_config


# One shared Gurobi environment for all StagewiseOPF instances in a process,
# built on first use with OutputFlag = 0. This silences the license banner and
# avoids starting a separate environment for every model. With the scipy backend
# gurobipy is never imported.
_GUROBI_ENV = None


def _get_gurobi_env():
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        _GUROBI_ENV = env
    return _GUROBI_ENV


class StagewiseOPF:
    """
    Single-stage DC-OPF LP solver.

    Build once with network topology; call solve(a_gen, a_line) repeatedly.
    """

    def __init__(self, network, params, lp_method=None):
        """
        Parameters
        ----------
        network : NetworkData
        params  : ModelParameters
        lp_method : str or None
            Per-stage DC-OPF backend. A scipy linprog method ('highs' needs
            SciPy >= 1.6; if the linprog call raises, it is retried with
            'revised simplex'), or the literal 'gurobi' to solve the same LP
            through gurobipy. None resolves to solver_config.LP_SOLVER.
        """
        self.network = network
        self.params = params
        self.lp_method = lp_method if lp_method is not None else solver_config.LP_SOLVER
        self._use_gurobi = (self.lp_method == 'gurobi')
        self._build_lp_structure()
        if self._use_gurobi:
            self._build_gurobi_model()

    # ------------------------------------------------------------------
    # Internal LP construction
    # ------------------------------------------------------------------

    def _build_lp_structure(self):
        """Pre-compute all LP matrices that do not depend on availability."""
        n = self.network
        p = self.params

        # Trailing numbers: sizes of the toy (5-bus) network.
        n_gen   = n.n_generators        # 2
        n_load  = n.n_loads             # 3
        n_bus   = n.n_buses             # 5
        n_line  = n.n_lines             # 5
        slack   = n.slack_bus           # 0
        n_theta = n_bus - 1             # 4  (θ for non-slack buses)
        n_var   = n_gen + n_load + n_theta + n_line  # + explicit line flows

        self.n_gen   = n_gen
        self.n_load  = n_load
        self.n_bus   = n_bus
        self.n_line  = n_line
        self.n_theta = n_theta
        self.n_var   = n_var
        self.slack   = slack

        # Column offsets for the four variable blocks.
        theta_off = n_gen + n_load
        f_off     = n_gen + n_load + n_theta
        self.theta_off = theta_off
        self.f_off     = f_off

        # Non-slack buses (sorted)
        non_slack = sorted([i for i in range(n_bus) if i != slack])
        self.non_slack = non_slack

        # ---- Objective: min c·x + constant ----------------------------
        # c_g * p_g - K * d_l; θ and f have zero cost. The cost that solve()
        # reports, gen cost + K*Σ(D_l - d_l), adds the constant K*ΣD_l.
        K = p.load_shedding_cost
        c = np.zeros(n_var)
        c[:n_gen]             = n.gen_cost
        c[n_gen:n_gen+n_load] = -K
        self.c = c
        self.K = K
        self.D_total = float(np.sum(n.load_nominal))

        # ---- Power-balance equality constraints in flow form (all buses) ----
        # At each bus n_idx:
        #   Σ_g p_g·1{gen_bus=n_idx} - Σ_l d_l·1{load_bus=n_idx}
        #       - (net flow out of n_idx) = 0
        # with (net flow out) = Σ_e (+f_e if e leaves n_idx, -f_e if e enters).
        # Angles do not appear here, so a failed line (f_e = 0) drops out of the
        # balance. Restricted to the flow columns the rows sum to zero (each f_e
        # appears once with +1 and once with -1), but the full rows are
        # independent: the slack row is the only row with the slack bus's
        # generators and loads, so it must be kept or they would be unconstrained.
        A_eq = np.zeros((n_bus, n_var))
        for n_idx in range(n_bus):
            for g in range(n_gen):
                if n.gen_buses[g] == n_idx:
                    A_eq[n_idx, g] = 1.0
            for l in range(n_load):
                if n.load_buses[l] == n_idx:
                    A_eq[n_idx, n_gen + l] = -1.0
            for e in range(n_line):
                fb, tb = n.line_connections[e]
                if fb == n_idx:
                    A_eq[n_idx, f_off + e] -= 1.0
                elif tb == n_idx:
                    A_eq[n_idx, f_off + e] += 1.0

        self.A_eq = A_eq
        self.b_eq = np.zeros(n_bus)

        # ---- DC-law coefficient matrix (angle side) --------------------
        # A_line[e,:]·x = B_e * (θ_from - θ_to)   (θ_slack = 0 eliminated)
        A_line = np.zeros((n_line, n_var))
        for e in range(n_line):
            fb, tb = n.line_connections[e]
            b_e = n.line_susceptances[e]
            if fb != slack:
                fi = non_slack.index(fb)
                A_line[e, theta_off + fi] = b_e
            if tb != slack:
                ti = non_slack.index(tb)
                A_line[e, theta_off + ti] = -b_e
        self.A_line = A_line

        # ---- Conditional DC-law (big-M) inequality block ---------------
        # Encodes  |f_e - B_e·Δθ_e| ≤ M_e·(1 - a_line[e])  as two rows per line
        # (all upper rows first, then all lower rows):
        #   upper:  f_e - B_e·Δθ_e ≤  M_e·(1 - a_line[e])
        #   lower:  B_e·Δθ_e - f_e ≤  M_e·(1 - a_line[e])
        # The matrix is fixed; only the RHS (through a_line) changes per solve.
        # M_e = 2π·B_e bounds |B_e·Δθ_e| over θ ∈ [-π, π], so a failed line's
        # DC law is slack and the line does not couple its endpoint angles.
        E_f = np.zeros((n_line, n_var))
        for e in range(n_line):
            E_f[e, f_off + e] = 1.0
        self.A_bigM = np.vstack([E_f - A_line, A_line - E_f]) if n_line > 0 \
            else np.zeros((0, n_var))
        self.big_M = 2.0 * np.pi * np.array(n.line_susceptances, dtype=float)

        # ---- Static variable bounds (availability-independent parts) ---
        # p ≥ 0 (≤ Pmax·a_gen per solve); d ∈ [0, D]; θ ∈ [-π, π];
        # f ∈ [Fmin·a_line, Fmax·a_line] per solve (static = the a_line=1 box).
        self.f_min = np.array(n.line_flow_min, dtype=float)
        self.f_max = np.array(n.line_flow_max, dtype=float)

        self.lb_static = np.zeros(n_var)
        self.lb_static[theta_off:f_off] = -np.pi
        self.lb_static[f_off:]          = self.f_min

        self.ub_pmax   = np.array(n.gen_pmax, dtype=float)   # scaled by a_gen
        self.ub_dmax   = np.array(n.load_nominal, dtype=float)
        self.ub_static = np.empty(n_var)
        self.ub_static[:n_gen]             = self.ub_pmax       # set to Pmax·a_gen per solve
        self.ub_static[n_gen:n_gen+n_load] = self.ub_dmax
        self.ub_static[theta_off:f_off]    = np.pi
        self.ub_static[f_off:]             = self.f_max

    # ------------------------------------------------------------------
    # Gurobi backend (built once, reused; only bounds/RHS change per solve)
    # ------------------------------------------------------------------

    def _build_gurobi_model(self):
        """Build a persistent gurobipy model for the same LP scipy solves.

        Uses the same dense matrices as _build_lp_structure. The objective
        and equality (power-balance) constraints are fixed; per solve only the
        generator upper bounds (a_gen), the line-flow bounds (a_line), and the
        big-M right-hand side (a_line) change, updated in place in _solve_gurobi.
        """
        import gurobipy as gp
        from gurobipy import GRB

        m = gp.Model(env=_get_gurobi_env())
        m.Params.Threads = 1   # tiny LP: avoid oversubscription in parallel workers

        x = m.addMVar(self.n_var, lb=self.lb_static, ub=self.ub_static, name='x')
        m.setObjective(self.c @ x, GRB.MINIMIZE)

        # Power balance in flow form, one row per bus (slack row included).
        m.addConstr(self.A_eq @ x == self.b_eq, name='balance')

        # Conditional DC law (big-M); RHS filled per solve from a_line.
        if self.n_line > 0:
            self._g_con_bigM = m.addConstr(
                self.A_bigM @ x <= np.zeros(self.A_bigM.shape[0]), name='dc_law')
        else:
            self._g_con_bigM = None

        m.update()
        self._g_model = m
        self._g_x = x

    def _solve_gurobi(self, a_gen, a_line):
        """Solve the stage LP with the persistent Gurobi model."""
        from gurobipy import GRB
        n = self.network
        m, x = self._g_model, self._g_x

        # Availability-dependent bounds: generator UB scaled by a_gen, and the
        # line-flow box scaled by a_line (a failed line is pinned to f_e = 0).
        new_ub = self.ub_static.copy()
        new_lb = self.lb_static.copy()
        new_ub[:self.n_gen] = self.ub_pmax * a_gen
        if self.n_line > 0:
            new_ub[self.f_off:] = self.f_max * a_line
            new_lb[self.f_off:] = self.f_min * a_line
        x.setAttr(GRB.Attr.UB, new_ub)
        x.setAttr(GRB.Attr.LB, new_lb)

        # Big-M RHS relaxes the DC law only on failed lines (a_line = 0).
        if self._g_con_bigM is not None:
            rhs = self.big_M * (1.0 - a_line)
            self._g_con_bigM.setAttr(GRB.Attr.RHS, np.concatenate([rhs, rhs]))

        m.optimize()

        if m.Status != GRB.OPTIMAL:
            # Not solved to optimality (infeasibility should not occur, since all
            # load can be shed): report the full-shedding cost K*ΣD with zero
            # dispatch and feasible=False, as the scipy path does.
            return (self.K * self.D_total,
                    np.zeros(self.n_gen),
                    np.zeros(self.n_load),
                    False)

        xval = x.X
        p_gen  = np.clip(xval[:self.n_gen], 0.0, None)
        d_load = np.clip(xval[self.n_gen:self.n_gen + self.n_load], 0.0,
                         self.ub_dmax)
        gen_cost      = float(np.dot(n.gen_cost, p_gen))
        shedding_cost = float(self.K * np.sum(
            np.array(n.load_nominal) - d_load))
        return gen_cost + shedding_cost, p_gen, d_load, True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, a_gen, a_line):
        """
        Solve the single-stage DC-OPF LP.

        Parameters
        ----------
        a_gen  : array-like, shape (n_gen,),  binary availability (0/1)
        a_line : array-like, shape (n_line,), binary availability (0/1);
                 fractional availability goes through solve_derated()

        Returns
        -------
        cost     : float   stage cost, generation cost + K * shed load
        p_gen    : ndarray shape (n_gen,), generator output (MW)
        d_load   : ndarray shape (n_load,), served load (MW)
        feasible : bool    False if the LP was not solved; cost is then K * total
                           demand and p_gen, d_load are zero
        """
        a_gen  = np.asarray(a_gen,  dtype=float)
        a_line = np.asarray(a_line, dtype=float)
        n = self.network

        if self._use_gurobi:
            return self._solve_gurobi(a_gen, a_line)

        # ---- Update availability-dependent bounds ---------------------
        lb = self.lb_static.copy()
        ub = self.ub_static.copy()
        ub[:self.n_gen] = self.ub_pmax * a_gen   # generation limit
        if self.n_line > 0:
            # Thermal limit as a box on the flow variables; a_line = 0 -> f_e = 0.
            ub[self.f_off:] = self.f_max * a_line
            lb[self.f_off:] = self.f_min * a_line

        # ---- Conditional DC law (big-M) as inequalities ---------------
        # A_bigM · x ≤ M·(1 - a_line) (stacked upper/lower); only the RHS
        # depends on availability, relaxing the law on failed lines.
        if self.n_line > 0:
            rhs = self.big_M * (1.0 - a_line)
            A_ub = self.A_bigM
            b_ub = np.concatenate([rhs, rhs])
        else:
            A_ub = None
            b_ub = None

        bounds = list(zip(lb, ub))

        # ---- Solve LP ------------------------------------------------
        try:
            result = linprog(
                self.c,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=self.A_eq, b_eq=self.b_eq,
                bounds=bounds,
                method=self.lp_method
            )
        except Exception:
            # Fallback for SciPy < 1.6, which has no HiGHS (any exception
            # triggers the retry)
            result = linprog(
                self.c,
                A_ub=A_ub, b_ub=b_ub,
                A_eq=self.A_eq, b_eq=self.b_eq,
                bounds=bounds,
                method='revised simplex'
            )

        if not result.success:
            # Not solved (infeasibility should not occur, since all load can be
            # shed): report the full-shedding cost with zero dispatch
            return (self.K * self.D_total,
                    np.zeros(self.n_gen),
                    np.zeros(self.n_load),
                    False)

        x = result.x
        p_gen  = np.clip(x[:self.n_gen], 0.0, None)
        d_load = np.clip(x[self.n_gen:self.n_gen + self.n_load], 0.0,
                         self.ub_dmax)

        gen_cost      = float(np.dot(n.gen_cost, p_gen))
        shedding_cost = float(self.K * np.sum(
            np.array(n.load_nominal) - d_load
        ))
        total_cost = gen_cost + shedding_cost

        return total_cost, p_gen, d_load, True

    def solve_derated(self, a_gen, a_line):
        """
        Single-stage DC-OPF on the derated network, for fractional availability.

        A fractional line availability scales both the line's susceptance in the
        DC law and its thermal limits, so the flow is
        f_e = a_line[e] * b_e * (θ_from - θ_to) with
        a_line[e] * f_min[e] <= f_e <= a_line[e] * f_max[e];
        a fractional generator availability scales its capacity. At binary
        availability this coincides exactly with solve(): a fully available line
        obeys the DC law, a fully failed line carries zero flow and leaves its
        endpoint angles free.

        Used to score the expected-failure lookahead steps of the Near-sighted
        MPC, where a partly available line acts as a weaker conductor. Passing
        a fractional a_line to solve() would instead relax the big-M DC law by
        M_e * (1 - a_line[e]). Unlike solve(), the balance and flow-limit
        coefficients depend on a_line through the scaled susceptance, so the
        matrices are rebuilt on each call. The flows are not separate variables:
        x = [p_gen, d_load, θ_ns].

        Returns (cost, p_gen, d_load, feasible), as solve() does.
        """
        a_gen  = np.asarray(a_gen,  dtype=float)
        a_line = np.asarray(a_line, dtype=float)
        n = self.network

        n_gen, n_load = self.n_gen, self.n_load
        n_bus, n_line = self.n_bus, self.n_line
        slack, non_slack = self.slack, self.non_slack
        theta_off = n_gen + n_load
        n_var = n_gen + n_load + self.n_theta

        # Objective: c_g p_g - K d (θ has zero cost); the reported cost below
        # adds the constant K*ΣD back.
        c = np.zeros(n_var)
        c[:n_gen]             = n.gen_cost
        c[n_gen:n_gen+n_load] = -self.K

        # Derated (effective) susceptance per line.
        b_eff = np.array(n.line_susceptances, dtype=float) * a_line

        # Power balance in angle form via the derated Laplacian
        # B = C diag(b_eff) C^T (all buses, slack row included): at bus n_idx,
        #   Σ_g p_g - Σ_l d = Σ_e C[n_idx, e] * b_eff[e] * (θ_from(e) - θ_to(e)),
        # with C[n, e] = +1 if line e leaves bus n and -1 if it enters it.
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
        b_eq = np.zeros(n_bus)

        # Thermal limits on the derated flow, scaled by availability:
        #   a_line*f_min <= b_eff*(θ_from - θ_to) <= a_line*f_max.
        # An a_line=0 line yields the trivial 0 <= 0 and drops out.
        rows, ub = [], []
        for e in range(n_line):
            fb, tb = n.line_connections[e]
            be = b_eff[e]
            row = np.zeros(n_var)
            if fb != slack:
                row[theta_off + non_slack.index(fb)] = be
            if tb != slack:
                row[theta_off + non_slack.index(tb)] = -be
            rows.append(row);  ub.append(a_line[e] * self.f_max[e])   #  f <=  a*f_max
            rows.append(-row); ub.append(-a_line[e] * self.f_min[e])  # -f <= -a*f_min
        A_ub = np.vstack(rows) if rows else None
        b_ub = np.array(ub) if rows else None

        # Bounds: p ∈ [0, a_g*pmax], d ∈ [0, D], θ ∈ [-π, π].
        lb = np.zeros(n_var); lb[theta_off:] = -np.pi
        hb = np.empty(n_var)
        hb[:n_gen]             = self.ub_pmax * a_gen
        hb[n_gen:n_gen+n_load] = self.ub_dmax
        hb[theta_off:]         = np.pi
        bounds = list(zip(lb, hb))

        # Always solved with scipy linprog (the LP is tiny). Under the Gurobi
        # backend this uses 'highs', since 'gurobi' is not a linprog method.
        method = 'highs' if self._use_gurobi else self.lp_method
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method=method)
        except Exception:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='revised simplex')

        if not result.success:
            # Not solved: report the full-shedding cost, as solve() does.
            return (self.K * self.D_total,
                    np.zeros(n_gen), np.zeros(n_load), False)

        x = result.x
        p_gen  = np.clip(x[:n_gen], 0.0, None)
        d_load = np.clip(x[n_gen:n_gen + n_load], 0.0, self.ub_dmax)
        gen_cost      = float(np.dot(n.gen_cost, p_gen))
        shedding_cost = float(self.K * np.sum(np.array(n.load_nominal) - d_load))
        return gen_cost + shedding_cost, p_gen, d_load, True

    def compute_repair_cost(self, r_gen, r_line):
        """Repair cost Σ_k β_k r[k] of the given generator and line repair decisions."""
        p = self.params
        return (float(np.dot(r_gen, p.gen_repair_cost)) +
                float(np.dot(r_line, p.line_repair_cost)))


def create_stagewise_opf(network, params, lp_method=None):
    """Factory function (lp_method=None -> solver_config.LP_SOLVER)."""
    return StagewiseOPF(network, params, lp_method=lp_method)


if __name__ == "__main__":
    from network_data import create_network
    from model_parameters import create_model_parameters

    net = create_network()
    prm = create_model_parameters(net)
    opf = create_stagewise_opf(net, prm)

    # Test: all operational
    cost, pg, dl, ok = opf.solve(np.ones(2), np.ones(5))
    print(f"All available  : cost={cost:.2f}, p_gen={pg}, feasible={ok}")

    # Test: both generators failed
    cost, pg, dl, ok = opf.solve(np.zeros(2), np.ones(5))
    print(f"No generators  : cost={cost:.2f}, shedding={np.sum(net.load_nominal)-np.sum(dl):.2f} MW")

    # Test: generator 2 and line 2 failed
    cost, pg, dl, ok = opf.solve([1, 0], [1, 0, 1, 1, 1])
    print(f"Gen2+Line2 fail: cost={cost:.2f}, p_gen={pg}")
