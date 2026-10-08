"""
Problem Formulation Module
==========================
Pyomo MILP formulations of the multistage economic dispatch problem with
stochastic component failures and repair decisions. Dispatch at every stage is
subject to DC-OPF constraints; unserved load is penalized at load_shedding_cost.

Two extensive forms are provided:
- EDOPFProblem: scenario-flat form indexed by (scenario, stage). Each scenario
  has its own repair decisions and there is no non-anticipativity coupling.
  With one scenario it is the perfect-information MIP that labels DFL training
  data; with N sampled scenarios its optimum is the sample-average wait-and-see
  value, an estimate of the lower bound E[WS] on the optimal expected cost
  (mode='ws' in saa_solver.py).
- EDOPFTreeProblem: node-indexed form on a sampled scenario tree, the
  multistage SAA (mode='tree' in saa_solver.py). Non-anticipativity holds by
  construction.

Both forms contain the same repair dynamics:
- Binary availability states with stage-to-stage transitions
- Repair pipeline of length tau: a repair ordered at stage t takes effect at
  stage t + tau
- McCormick linearization of the repair-activation product
- Repair feasibility (only failed components can be repaired)
- Optional per-stage repair budget (params.repair_budget_per_stage)
- Explicit line-flow variables with a big-M DC power-flow law, so a failed line
  carries zero flow and leaves its endpoint angles free
"""

import os
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from typing import List, Dict
import warnings

import solver_config


class EDOPFProblem:
    """Scenario-flat extensive form of the ED-OPF problem with failures and repairs."""

    def __init__(self, network_data, model_params, scenarios, state_dynamics=None):
        """
        Initialize the problem formulation.

        Parameters:
        -----------
        network_data : NetworkData
            Network topology and parameters
        model_params : ModelParameters
            Horizon T, repair delay tau, cost data and optional
            repair_budget_per_stage
        scenarios : List[Scenario]
            Failure scenarios. gen_failures has shape (T, n_generators) and
            line_failures (T, n_lines); probability weights the scenario in
            the objective.
        state_dynamics : StateDynamics, optional
            Unused. Accepted because callers pass it; availability is a state
            variable of the model.
        """
        self.network = network_data
        self.params = model_params
        self.scenarios = scenarios

        self.model = None
        self.results = None

    def build_model(self):
        """Build the scenario-flat Pyomo model, store it in self.model and return it."""
        model = pyo.ConcreteModel(name="ED_OPF_with_Failures")

        # ============================================================
        # SETS
        # ============================================================
        model.T = pyo.RangeSet(1, self.params.T)  # Stages
        model.S = pyo.RangeSet(0, len(self.scenarios) - 1)  # Scenarios
        model.G = pyo.RangeSet(0, self.network.n_generators - 1)  # Generators
        model.L = pyo.RangeSet(0, self.network.n_loads - 1)  # Loads
        model.N = pyo.RangeSet(0, self.network.n_buses - 1)  # Buses
        model.E = pyo.RangeSet(0, self.network.n_lines - 1)  # Lines
        model.K = pyo.RangeSet(1, self.params.tau)  # Pipeline positions

        # ============================================================
        # PARAMETERS
        # ============================================================
        # Scenario probabilities
        model.prob_s = pyo.Param(
            model.S,
            initialize={s: self.scenarios[s].probability for s in model.S},
            within=pyo.NonNegativeReals
        )

        # Generator parameters
        model.gen_cost = pyo.Param(
            model.G,
            initialize={g: self.network.gen_cost[g] for g in model.G}
        )
        model.gen_pmin = pyo.Param(
            model.G,
            initialize={g: self.network.gen_pmin[g] for g in model.G}
        )
        model.gen_pmax = pyo.Param(
            model.G,
            initialize={g: self.network.gen_pmax[g] for g in model.G}
        )

        # Load parameters
        model.load_demand = pyo.Param(
            model.L,
            initialize={l: self.network.load_nominal[l] for l in model.L}
        )
        model.load_shedding_cost = pyo.Param(
            initialize=self.params.load_shedding_cost
        )

        # Repair costs
        model.gen_repair_cost = pyo.Param(
            model.G,
            initialize={g: self.params.gen_repair_cost[g] for g in model.G}
        )
        model.line_repair_cost = pyo.Param(
            model.E,
            initialize={e: self.params.line_repair_cost[e] for e in model.E}
        )

        # Network parameters
        model.line_flow_min = pyo.Param(
            model.E,
            initialize={e: self.network.line_flow_min[e] for e in model.E}
        )
        model.line_flow_max = pyo.Param(
            model.E,
            initialize={e: self.network.line_flow_max[e] for e in model.E}
        )

        # Failure events xi, fixed data taken from the scenarios: xi = 1 if a
        # failure event hits the component at stage t. The event takes the
        # component down only if it was available at stage t - 1.
        def init_xi_gen(model, s, t, g):
            return int(self.scenarios[s].gen_failures[t - 1, g])

        def init_xi_line(model, s, t, e):
            return int(self.scenarios[s].line_failures[t - 1, e])

        model.xi_gen = pyo.Param(
            model.S, model.T, model.G,
            initialize=init_xi_gen,
            within=pyo.Binary
        )
        model.xi_line = pyo.Param(
            model.S, model.T, model.E,
            initialize=init_xi_line,
            within=pyo.Binary
        )

        # ============================================================
        # DECISION VARIABLES
        # ============================================================
        # Generator dispatch (continuous)
        model.p_gen = pyo.Var(
            model.S, model.T, model.G,
            within=pyo.NonNegativeReals,
            bounds=(0, max(self.network.gen_pmax))
        )

        # Voltage angles in radians (continuous)
        model.theta = pyo.Var(
            model.S, model.T, model.N,
            within=pyo.Reals,
            bounds=(-np.pi, np.pi)
        )

        # Load served (continuous, allows load shedding)
        model.d_load = pyo.Var(
            model.S, model.T, model.L,
            within=pyo.NonNegativeReals,
            bounds=(0, max(self.network.load_nominal))
        )

        # Line flows (continuous). Each line carries an explicit flow variable so
        # that a failed line can be forced to zero flow while leaving its endpoint
        # voltage angles free. The symmetric box is a safety bound; the thermal
        # limits below tighten it to the (availability-scaled) capacity.
        f_bound = float(max(
            np.max(np.abs(self.network.line_flow_min)),
            np.max(np.abs(self.network.line_flow_max))
        ))
        model.f_line = pyo.Var(
            model.S, model.T, model.E,
            within=pyo.Reals,
            bounds=(-f_bound, f_bound)
        )

        # Repair decisions (binary, scenario-dependent recourse)
        model.r_gen = pyo.Var(
            model.S, model.T, model.G,
            within=pyo.Binary
        )
        model.r_line = pyo.Var(
            model.S, model.T, model.E,
            within=pyo.Binary
        )

        # ============================================================
        # STATE VARIABLES
        # ============================================================
        # Component availability (binary)
        model.a_gen = pyo.Var(
            model.S, model.T, model.G,
            within=pyo.Binary
        )
        model.a_line = pyo.Var(
            model.S, model.T, model.E,
            within=pyo.Binary
        )

        # Repair pipeline (binary): q[s,t,g,k] = 1 if a repair of g is in
        # progress at stage t and takes effect at stage t + k
        model.q_gen = pyo.Var(
            model.S, model.T, model.G, model.K,
            within=pyo.Binary
        )
        model.q_line = pyo.Var(
            model.S, model.T, model.E, model.K,
            within=pyo.Binary
        )

        # Repair completion signal (binary): R[s,t,g] = 1 if a repair of g
        # takes effect at stage t
        model.R_gen = pyo.Var(
            model.S, model.T, model.G,
            within=pyo.Binary
        )
        model.R_line = pyo.Var(
            model.S, model.T, model.E,
            within=pyo.Binary
        )

        # McCormick auxiliary: w = (1 - a_prev) * R. The linearization is exact
        # because both factors are binary.
        model.w_gen = pyo.Var(
            model.S, model.T, model.G,
            within=pyo.NonNegativeReals,
            bounds=(0, 1)
        )
        model.w_line = pyo.Var(
            model.S, model.T, model.E,
            within=pyo.NonNegativeReals,
            bounds=(0, 1)
        )

        # ============================================================
        # OBJECTIVE FUNCTION
        # ============================================================
        def objective_rule(model):
            total_cost = 0.0
            for s in model.S:
                scenario_cost = 0.0
                for t in model.T:
                    # Generation cost
                    gen_cost = sum(
                        model.gen_cost[g] * model.p_gen[s, t, g]
                        for g in model.G
                    )
                    # Load shedding penalty
                    shedding_cost = model.load_shedding_cost * sum(
                        model.load_demand[l] - model.d_load[s, t, l]
                        for l in model.L
                    )
                    # Repair costs (per-scenario recourse)
                    repair_cost = sum(
                        model.gen_repair_cost[g] * model.r_gen[s, t, g]
                        for g in model.G
                    ) + sum(
                        model.line_repair_cost[e] * model.r_line[s, t, e]
                        for e in model.E
                    )
                    scenario_cost += gen_cost + shedding_cost + repair_cost
                total_cost += model.prob_s[s] * scenario_cost
            return total_cost

        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # ============================================================
        # INITIAL CONDITIONS (t = 1)
        # ============================================================
        tau = self.params.tau

        # At t=1: no repair has completed (pipeline was empty)
        def R_gen_init_rule(model, s, g):
            return model.R_gen[s, 1, g] == 0
        model.R_gen_init = pyo.Constraint(model.S, model.G, rule=R_gen_init_rule)

        def R_line_init_rule(model, s, e):
            return model.R_line[s, 1, e] == 0
        model.R_line_init = pyo.Constraint(model.S, model.E, rule=R_line_init_rule)

        # w = 0 at t=1 (no repair can activate)
        def w_gen_init_rule(model, s, g):
            return model.w_gen[s, 1, g] == 0
        model.w_gen_init = pyo.Constraint(model.S, model.G, rule=w_gen_init_rule)

        def w_line_init_rule(model, s, e):
            return model.w_line[s, 1, e] == 0
        model.w_line_init = pyo.Constraint(model.S, model.E, rule=w_line_init_rule)

        # Availability at t=1: all components are available before stage 1 and
        # w = 0, so a = (1 - xi) * 1 + 0 = 1 - xi
        def a_gen_init_rule(model, s, g):
            return model.a_gen[s, 1, g] == 1 - model.xi_gen[s, 1, g]
        model.a_gen_init = pyo.Constraint(model.S, model.G, rule=a_gen_init_rule)

        def a_line_init_rule(model, s, e):
            return model.a_line[s, 1, e] == 1 - model.xi_line[s, 1, e]
        model.a_line_init = pyo.Constraint(model.S, model.E, rule=a_line_init_rule)

        # Pipeline at t=1: positions k < tau are 0 (shifted from empty pipeline)
        if tau > 1:
            def q_gen_init_shift_rule(model, s, g, k):
                return model.q_gen[s, 1, g, k] == 0
            model.q_gen_init_shift = pyo.Constraint(
                model.S, model.G, pyo.RangeSet(1, tau - 1),
                rule=q_gen_init_shift_rule
            )

            def q_line_init_shift_rule(model, s, e, k):
                return model.q_line[s, 1, e, k] == 0
            model.q_line_init_shift = pyo.Constraint(
                model.S, model.E, pyo.RangeSet(1, tau - 1),
                rule=q_line_init_shift_rule
            )

        # Pipeline entry at t=1: position tau = repair decision
        def q_gen_init_entry_rule(model, s, g):
            return model.q_gen[s, 1, g, tau] == model.r_gen[s, 1, g]
        model.q_gen_init_entry = pyo.Constraint(model.S, model.G, rule=q_gen_init_entry_rule)

        def q_line_init_entry_rule(model, s, e):
            return model.q_line[s, 1, e, tau] == model.r_line[s, 1, e]
        model.q_line_init_entry = pyo.Constraint(model.S, model.E, rule=q_line_init_entry_rule)

        # ============================================================
        # DYNAMIC CONSTRAINTS (t >= 2)
        # ============================================================
        stages_ge2 = pyo.RangeSet(2, self.params.T)

        # Repair completion signal: R[s,t,g] = q[s,t-1,g,1]
        def R_gen_rule(model, s, t, g):
            return model.R_gen[s, t, g] == model.q_gen[s, t - 1, g, 1]
        model.R_gen_dynamic = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=R_gen_rule
        )

        def R_line_rule(model, s, t, e):
            return model.R_line[s, t, e] == model.q_line[s, t - 1, e, 1]
        model.R_line_dynamic = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=R_line_rule
        )

        # McCormick linearization: w = (1 - a_prev) * R
        # w <= 1 - a_prev
        def w_gen_ub1_rule(model, s, t, g):
            return model.w_gen[s, t, g] <= 1 - model.a_gen[s, t - 1, g]
        model.w_gen_ub1 = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=w_gen_ub1_rule
        )

        def w_line_ub1_rule(model, s, t, e):
            return model.w_line[s, t, e] <= 1 - model.a_line[s, t - 1, e]
        model.w_line_ub1 = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=w_line_ub1_rule
        )

        # w <= R
        def w_gen_ub2_rule(model, s, t, g):
            return model.w_gen[s, t, g] <= model.R_gen[s, t, g]
        model.w_gen_ub2 = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=w_gen_ub2_rule
        )

        def w_line_ub2_rule(model, s, t, e):
            return model.w_line[s, t, e] <= model.R_line[s, t, e]
        model.w_line_ub2 = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=w_line_ub2_rule
        )

        # w >= (1 - a_prev) + R - 1
        def w_gen_lb_rule(model, s, t, g):
            return model.w_gen[s, t, g] >= (
                (1 - model.a_gen[s, t - 1, g]) + model.R_gen[s, t, g] - 1
            )
        model.w_gen_lb = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=w_gen_lb_rule
        )

        def w_line_lb_rule(model, s, t, e):
            return model.w_line[s, t, e] >= (
                (1 - model.a_line[s, t - 1, e]) + model.R_line[s, t, e] - 1
            )
        model.w_line_lb = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=w_line_lb_rule
        )

        # Availability transition: a[s,t,g] = (1 - xi[s,t,g]) * a[s,t-1,g] + w[s,t,g]
        # xi is evaluated at build time, so the coefficient (1 - xi) is a numeric
        # constant and the constraint matrix of the MILP is fixed.
        def a_gen_trans_rule(model, s, t, g):
            coeff = 1 - pyo.value(model.xi_gen[s, t, g])
            return model.a_gen[s, t, g] == (
                coeff * model.a_gen[s, t - 1, g] + model.w_gen[s, t, g]
            )
        model.a_gen_transition = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=a_gen_trans_rule
        )

        def a_line_trans_rule(model, s, t, e):
            coeff = 1 - pyo.value(model.xi_line[s, t, e])
            return model.a_line[s, t, e] == (
                coeff * model.a_line[s, t - 1, e] + model.w_line[s, t, e]
            )
        model.a_line_transition = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=a_line_trans_rule
        )

        # Pipeline shift: q[s,t,g,k] = q[s,t-1,g,k+1] for k=1..tau-1
        if tau > 1:
            def q_gen_shift_rule(model, s, t, g, k):
                return model.q_gen[s, t, g, k] == model.q_gen[s, t - 1, g, k + 1]
            model.q_gen_shift = pyo.Constraint(
                model.S, stages_ge2, model.G, pyo.RangeSet(1, tau - 1),
                rule=q_gen_shift_rule
            )

            def q_line_shift_rule(model, s, t, e, k):
                return model.q_line[s, t, e, k] == model.q_line[s, t - 1, e, k + 1]
            model.q_line_shift = pyo.Constraint(
                model.S, stages_ge2, model.E, pyo.RangeSet(1, tau - 1),
                rule=q_line_shift_rule
            )

        # Pipeline entry: q[s,t,g,tau] = r[s,t,g]
        def q_gen_entry_rule(model, s, t, g):
            return model.q_gen[s, t, g, tau] == model.r_gen[s, t, g]
        model.q_gen_entry = pyo.Constraint(
            model.S, stages_ge2, model.G, rule=q_gen_entry_rule
        )

        def q_line_entry_rule(model, s, t, e):
            return model.q_line[s, t, e, tau] == model.r_line[s, t, e]
        model.q_line_entry = pyo.Constraint(
            model.S, stages_ge2, model.E, rule=q_line_entry_rule
        )

        # ============================================================
        # REPAIR FEASIBILITY (all stages)
        # ============================================================
        def repair_gen_feasibility_rule(model, s, t, g):
            return model.r_gen[s, t, g] <= 1 - model.a_gen[s, t, g]
        model.repair_gen_feasibility = pyo.Constraint(
            model.S, model.T, model.G, rule=repair_gen_feasibility_rule
        )

        def repair_line_feasibility_rule(model, s, t, e):
            return model.r_line[s, t, e] <= 1 - model.a_line[s, t, e]
        model.repair_line_feasibility = pyo.Constraint(
            model.S, model.T, model.E, rule=repair_line_feasibility_rule
        )

        # Per-stage monetary repair budget:
        #   sum_g beta_g r_gen[s,t,g] + sum_e beta_e r_line[s,t,e] <= B,  for all s,t,
        # with B = params.repair_budget_per_stage = phi * (sum of all component
        # repair costs), the same value at every stage. The constraint is built
        # only when a budget fraction phi is set; with phi = None the attribute
        # is None and the model has no budget.
        budget = getattr(self.params, 'repair_budget_per_stage', None)
        if budget is not None:
            def repair_budget_rule(model, s, t):
                return (sum(model.gen_repair_cost[g]  * model.r_gen[s, t, g]
                            for g in model.G)
                        + sum(model.line_repair_cost[e] * model.r_line[s, t, e]
                              for e in model.E)
                        <= budget)
            model.repair_budget = pyo.Constraint(
                model.S, model.T, rule=repair_budget_rule
            )

        # ============================================================
        # OPF CONSTRAINTS
        # ============================================================

        # Slack bus constraint (reference angle)
        def slack_bus_rule(model, s, t):
            slack_bus = self.network.slack_bus
            return model.theta[s, t, slack_bus] == 0

        model.slack_bus_constraint = pyo.Constraint(
            model.S, model.T,
            rule=slack_bus_rule
        )

        # Power balance constraint (Kirchhoff's current law, flow form).
        # Balance is written on the explicit flow variables, so a failed line
        # (with f_line = 0) drops out of the nodal balance automatically.
        # p_gen enters directly (not multiplied by a_gen); the generation limits
        # already enforce p_gen = 0 when a_gen = 0.
        def power_balance_rule(model, s, t, n):
            injection = 0.0
            for g in model.G:
                if self.network.gen_buses[g] == n:
                    injection += model.p_gen[s, t, g]

            for l in model.L:
                if self.network.load_buses[l] == n:
                    injection -= model.d_load[s, t, l]

            # Net flow out of bus n: +f on lines leaving n, -f on lines entering.
            flow_sum = 0.0
            for e in model.E:
                from_bus, to_bus = self.network.line_connections[e]
                if from_bus == n:
                    flow_sum += model.f_line[s, t, e]
                elif to_bus == n:
                    flow_sum -= model.f_line[s, t, e]

            return injection == flow_sum

        model.power_balance = pyo.Constraint(
            model.S, model.T, model.N,
            rule=power_balance_rule
        )

        # Conditional DC power-flow law (big-M). When line e is available
        # (a_line = 1) the flow equals the DC law b_e * (theta_from - theta_to);
        # when it is failed (a_line = 0) the law relaxes to |b_e * dtheta| <= M_e,
        # which is implied by the angle bounds and therefore imposes no coupling
        # between the endpoint angles. For b_e > 0, M_e = 2 * theta_max * b_e is
        # the largest value of |b_e * dtheta| over theta in [-theta_max, theta_max].
        theta_max = np.pi
        big_M = {
            e: 2.0 * theta_max * self.network.line_susceptances[e]
            for e in range(self.network.n_lines)
        }

        def dc_law_upper_rule(model, s, t, e):
            from_bus, to_bus = self.network.line_connections[e]
            susceptance = self.network.line_susceptances[e]
            dc_flow = susceptance * (
                model.theta[s, t, from_bus] - model.theta[s, t, to_bus]
            )
            return (model.f_line[s, t, e] - dc_flow
                    <= big_M[e] * (1 - model.a_line[s, t, e]))

        def dc_law_lower_rule(model, s, t, e):
            from_bus, to_bus = self.network.line_connections[e]
            susceptance = self.network.line_susceptances[e]
            dc_flow = susceptance * (
                model.theta[s, t, from_bus] - model.theta[s, t, to_bus]
            )
            return (model.f_line[s, t, e] - dc_flow
                    >= -big_M[e] * (1 - model.a_line[s, t, e]))

        model.dc_law_upper = pyo.Constraint(
            model.S, model.T, model.E, rule=dc_law_upper_rule
        )
        model.dc_law_lower = pyo.Constraint(
            model.S, model.T, model.E, rule=dc_law_lower_rule
        )

        # Generation limits: bounded by availability variable
        def gen_min_rule(model, s, t, g):
            return model.p_gen[s, t, g] >= model.a_gen[s, t, g] * model.gen_pmin[g]

        def gen_max_rule(model, s, t, g):
            return model.p_gen[s, t, g] <= model.a_gen[s, t, g] * model.gen_pmax[g]

        model.gen_min_constraint = pyo.Constraint(
            model.S, model.T, model.G,
            rule=gen_min_rule
        )
        model.gen_max_constraint = pyo.Constraint(
            model.S, model.T, model.G,
            rule=gen_max_rule
        )

        # Load shedding bounds: served load is at most the demand, so the shed
        # amount (demand - d_load) is nonnegative
        def load_shedding_rule(model, s, t, l):
            return model.d_load[s, t, l] <= model.load_demand[l]

        model.load_shedding_constraint = pyo.Constraint(
            model.S, model.T, model.L,
            rule=load_shedding_rule
        )

        # Line flow (thermal) limits on the explicit flow variable, scaled by
        # availability. A failed line (a_line = 0) is forced to zero flow.
        def line_flow_min_rule(model, s, t, e):
            return model.f_line[s, t, e] >= model.a_line[s, t, e] * model.line_flow_min[e]

        def line_flow_max_rule(model, s, t, e):
            return model.f_line[s, t, e] <= model.a_line[s, t, e] * model.line_flow_max[e]

        model.line_flow_min_constraint = pyo.Constraint(
            model.S, model.T, model.E,
            rule=line_flow_min_rule
        )
        model.line_flow_max_constraint = pyo.Constraint(
            model.S, model.T, model.E,
            rule=line_flow_max_rule
        )

        self.model = model
        return model

    def solve(self, solver_name=None, time_limit=172800, mip_gap=0.05,
              verbose=True, checkpoint_callback=None, checkpoint_interval=600,
              warmstart=False):
        """
        Solve the model, optionally in chunks with periodic checkpoints.

        Without a checkpoint_callback, or when the solver's solver_config entry
        has supports_chunking=False (GLPK and unregistered solvers), the model
        is solved once with the full time limit and the callback, if given,
        fires once with the final solution. An unregistered solver has no
        time-limit or gap option in solver_config, so it receives neither
        time_limit nor mip_gap and runs with its own defaults.

        Otherwise the solve runs in chunks of `checkpoint_interval` seconds.
        Each chunk is a separate solver call, so the solver restarts (and
        repeats presolve) at every chunk. Once the variables on the model hold
        values after a chunk (the solver's incumbent, or a start set before the
        solve), later chunks pass these values as a MIP start to solvers whose
        solver_config entry has supports_warmstart=True. After each chunk the
        incumbent is read, and `checkpoint_callback` is called when the MIP gap
        has dropped by at least 0.001 (0.1 percentage points) since the last
        checkpoint; the first computable gap always triggers one. The chunked
        solve stops when:
          - the MIP gap is at or below `mip_gap`,
          - the total elapsed time reaches `time_limit`, or
          - the solver reports optimality.

        Parameters:
        -----------
        solver_name : str or None
            Pyomo MIP solver to use ('appsi_highs', 'gurobi', 'glpk', 'cbc',
            'cplex', ...). None (default) resolves to solver_config.MIP_SOLVER
            (appsi_highs unless ED_OPF_MIP_SOLVER or ED_OPF_SOLVER is set).
        time_limit : float
            Total time limit in seconds (default: 172800 = 48 h)
        mip_gap : float or None
            Relative MIP gap tolerance; the solve stops once the gap is at or
            below it (default: 0.05 = 5%). None keeps the solver's default.
        verbose : bool
            Print progress and solver output
        checkpoint_callback : callable or None
            Called as checkpoint_callback(solution_dict, gap, elapsed_seconds).
            main.py uses it to save solution.json and plots during the solve.
        checkpoint_interval : float
            Length of one solve chunk in seconds, i.e. the time between solver
            restarts (default: 600 = 10 min)
        warmstart : bool
            Pass the variable values already on the model (e.g. from
            set_norepair_start) as a MIP start. Applies to the single-solve
            path only, and only for solvers whose solver_config entry has
            supports_warmstart=True.

        Returns:
        --------
        results : dict or None
            Solution dictionary, or None if infeasible / no feasible solution found
        """
        import time as time_module

        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        solver_name = solver_config.resolve_mip_solver(solver_name)
        solver = SolverFactory(solver_name)

        # Solvers with a threads option in solver_config (the HiGHS interfaces)
        # get SLURM_CPUS_PER_TASK threads, or os.cpu_count() outside SLURM.
        # Setting it explicitly keeps the solver from sizing its thread pool by
        # the node's full core count when only part of the node is allocated.
        # All other solvers (threads_option=None) are left at their own
        # defaults; Gurobi's core detection respects the SLURM allocation.
        _threads_option = solver_config.get_spec(solver_name).threads_option
        if _threads_option is not None:
            _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
            _n_threads  = int(_slurm_cpus) if _slurm_cpus else (os.cpu_count() or 1)
            solver.options[_threads_option] = _n_threads

        # Set MIP gap tolerance
        if mip_gap is not None:
            self._set_mip_gap(solver, solver_name, mip_gap)

        if verbose:
            print("=" * 60)
            print("SOLVING OPTIMIZATION PROBLEM")
            print("=" * 60)
            print(f"Solver: {solver_name}")
            print(f"Number of scenarios: {len(self.scenarios)}")
            print(f"Number of stages: {self.params.T}")
            gap_str = f"{mip_gap*100:.1f}%" if mip_gap else "solver default"
            print(f"MIP gap tolerance: {gap_str}")
            print(f"Time limit: {time_limit}s ({time_limit/3600:.1f}h)")
            if checkpoint_callback:
                print(f"Checkpoint interval: {checkpoint_interval}s")
            print()

        # Chunked checkpointing runs only for solvers whose solver_config entry
        # sets supports_chunking (every registered solver except GLPK). GLPK
        # has no MIP start and restarts from scratch on each call, so every
        # chunk would re-solve the LP relaxation.
        supports_chunking = solver_config.supports_chunking(solver_name)

        if checkpoint_callback is None or not supports_chunking:
            # Single solve with the full time limit (an unregistered solver has
            # no time-limit option and keeps its own default)
            self._set_time_limit(solver, solver_name, time_limit)

            start = time_module.time()
            _solve_kwargs = dict(tee=verbose)
            if warmstart and solver_config.supports_warmstart(solver_name):
                # Use the variable values already set on the model (e.g. the
                # NoRepair start) as a MIP warm start, so the solver begins from
                # a feasible incumbent. Not passed to solvers whose solver_config
                # entry has supports_warmstart=False (the HiGHS interfaces, GLPK).
                _solve_kwargs['warmstart'] = True
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    self.results = solver.solve(self.model, **_solve_kwargs)
            except RuntimeError as _e:
                # Some solver interfaces (e.g. appsi_highs) raise rather than
                # return a status when no feasible solution is found within the
                # time limit. Treat that as "no solution" and return None
                # instead of crashing the run.
                if 'feasible solution was not found' in str(_e).lower():
                    if verbose:
                        print("\nNo feasible solution found within the time limit.")
                    return None
                raise
            elapsed = time_module.time() - start

            result = self._process_solve_results(verbose)

            # Fire checkpoint once after solve completes (if callback provided)
            if result and checkpoint_callback:
                gap = self._extract_mip_gap()
                try:
                    checkpoint_callback(result, gap, elapsed)
                except Exception:
                    pass

            return result

        # ---- Chunked solve with periodic checkpointing ----
        total_elapsed = 0
        best_gap = float('inf')
        found_feasible = False
        chunk_num = 0

        while total_elapsed < time_limit:
            remaining = time_limit - total_elapsed
            chunk_time = min(checkpoint_interval, remaining)
            if chunk_time <= 0:
                break

            self._set_time_limit(solver, solver_name, int(chunk_time))

            chunk_num += 1
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Solve chunk {chunk_num}  "
                      f"(budget: {chunk_time:.0f}s | elapsed: {total_elapsed:.0f}s / {time_limit}s)")
                print(f"{'='*60}")

            start = time_module.time()

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    solve_kwargs = dict(tee=verbose)
                    if found_feasible and solver_config.supports_warmstart(solver_name):
                        solve_kwargs['warmstart'] = True
                    self.results = solver.solve(self.model, **solve_kwargs)
            except RuntimeError as _e:
                if 'feasible solution was not found' in str(_e).lower():
                    total_elapsed += time_module.time() - start
                    if verbose:
                        print(f"  No feasible solution in this chunk. Continuing...")
                    continue
                raise

            elapsed = time_module.time() - start
            total_elapsed += elapsed

            tc = self.results.solver.termination_condition

            if tc == pyo.TerminationCondition.infeasible:
                print("\nWARNING: Problem is infeasible!")
                return None

            # Objective at the current variable values; fails while they are unset
            try:
                obj_val = pyo.value(self.model.objective)
                found_feasible = True
            except Exception:
                if verbose:
                    print(f"  No feasible solution yet. Continuing...")
                continue

            # Extract gap
            current_gap = self._extract_mip_gap()

            if verbose:
                gap_str = f"{current_gap*100:.2f}%" if current_gap is not None else "N/A"
                print(f"\n  >> Best objective: ${obj_val:.2f} | "
                      f"Gap: {gap_str} | Elapsed: {total_elapsed:.0f}s")

            # Checkpoint when the gap is at least 0.001 (0.1 percentage points)
            # below the gap of the last checkpoint. best_gap starts at inf, so
            # the first computable gap always triggers one.
            if current_gap is not None:
                improvement = best_gap - current_gap
                if improvement >= 0.001:
                    best_gap = current_gap
                    if verbose:
                        print(f"  >> Gap improved by {improvement*100:.2f}% — saving checkpoint...")
                    try:
                        solution = self.extract_solution()
                        if solution:
                            checkpoint_callback(solution, current_gap, total_elapsed)
                    except Exception as e:
                        if verbose:
                            print(f"  Warning: checkpoint save failed: {e}")

            # Stop conditions
            if tc == pyo.TerminationCondition.optimal:
                if verbose:
                    print(f"\nOptimal solution found!")
                break

            if current_gap is not None and mip_gap is not None and current_gap <= mip_gap:
                if verbose:
                    print(f"\nTarget MIP gap reached: "
                          f"{current_gap*100:.4f}% <= {mip_gap*100:.2f}%")
                break

        # Final status
        if not found_feasible:
            print(f"\nNo feasible solution found within {time_limit}s time limit.")
            return None

        if total_elapsed >= time_limit and verbose:
            print(f"\nTime limit ({time_limit}s = {time_limit/3600:.1f}h) reached.")
            print(f"Returning best solution found (gap: {best_gap*100:.2f}%).")

        return self.extract_solution()

    def _set_mip_gap(self, solver, solver_name, mip_gap):
        """Set the MIP gap tolerance (option key from the solver_config registry)."""
        solver_config.set_mip_gap(solver, solver_name, mip_gap)

    def _set_time_limit(self, solver, solver_name, time_limit):
        """Set the time limit (option key and cast from the solver_config registry)."""
        solver_config.set_time_limit(solver, solver_name, time_limit)

    def _extract_mip_gap(self):
        """Return the relative gap (upper - lower) / |upper| of the last solve, or None."""
        try:
            lower = self.results.problem[0].lower_bound
            upper = self.results.problem[0].upper_bound
            # Fall back to objective value for the upper bound if missing
            if upper is None and self.model is not None:
                try:
                    upper = pyo.value(self.model.objective)
                except Exception:
                    pass
            if upper is not None and lower is not None and abs(upper) > 1e-10:
                gap = (upper - lower) / abs(upper)
                if np.isfinite(gap):
                    return gap
        except (IndexError, AttributeError):
            pass
        return None

    def _process_solve_results(self, verbose):
        """Report the status of a single (non-chunked) solve and return the
        extracted solution, or None if no feasible solution is available."""
        tc = self.results.solver.termination_condition
        if tc == pyo.TerminationCondition.optimal:
            if verbose:
                print(f"\nOptimal solution found!")
                print(f"Objective value: ${pyo.value(self.model.objective):.2f}")
        elif tc == pyo.TerminationCondition.infeasible:
            print("\nWARNING: Problem is infeasible!")
            return None
        else:
            print(f"\nSolver terminated with condition: {tc}")
            try:
                obj_val = pyo.value(self.model.objective)
                if verbose:
                    print(f"Best feasible solution: ${obj_val:.2f}")
            except Exception:
                print("No feasible solution available.")
                return None

        return self.extract_solution()

    def extract_solution(self):
        """
        Read the solution from the solved model.

        generation, angles, load_served and line_flows are keyed as
        [scenario][stage][index]; availability[scenario] holds 'gen' and 'line'
        int arrays of shapes (T, n_generators) and (T, n_lines); repairs['gen']
        and repairs['line'] are keyed [stage][index] and hold the mean repair
        decision over scenarios. Only costs['total'] is filled; the other cost
        entries stay 0.0. Returns None if the model has not been built or solved.
        """
        if self.model is None or self.results is None:
            return None

        model = self.model
        n_scenarios = len(self.scenarios)
        T = self.params.T

        solution = {
            'objective_value': pyo.value(model.objective),
            'generation': {},
            'angles': {},
            'load_served': {},
            'line_flows': {},
            'repairs': {
                'gen': {},
                'line': {}
            },
            'availability': {},
            'costs': {
                'generation': 0.0,
                'load_shedding': 0.0,
                'repairs': 0.0,
                'total': pyo.value(model.objective)
            }
        }

        # Extract generation, angles, load, and line flows per scenario and stage
        for s in model.S:
            solution['generation'][s] = {}
            solution['angles'][s] = {}
            solution['load_served'][s] = {}
            solution['line_flows'][s] = {}

            for t in model.T:
                solution['generation'][s][t] = {
                    g: pyo.value(model.p_gen[s, t, g])
                    for g in model.G
                }
                solution['angles'][s][t] = {
                    n: pyo.value(model.theta[s, t, n])
                    for n in model.N
                }
                solution['load_served'][s][t] = {
                    l: pyo.value(model.d_load[s, t, l])
                    for l in model.L
                }
                solution['line_flows'][s][t] = {
                    e: pyo.value(model.f_line[s, t, e])
                    for e in model.E
                }

        # Extract availability (per scenario)
        for s in model.S:
            a_gen = np.zeros((T, self.network.n_generators), dtype=int)
            a_line = np.zeros((T, self.network.n_lines), dtype=int)
            for t in model.T:
                for g in model.G:
                    a_gen[t - 1, g] = int(round(pyo.value(model.a_gen[s, t, g])))
                for e in model.E:
                    a_line[t - 1, e] = int(round(pyo.value(model.a_line[s, t, e])))
            solution['availability'][s] = {'gen': a_gen, 'line': a_line}

        # Mean repair decision per stage over scenarios. The mean is unweighted,
        # so it equals the expectation when all scenarios have probability 1/N
        # (as produced by generate_scenarios).
        for t in model.T:
            solution['repairs']['gen'][t] = {}
            solution['repairs']['line'][t] = {}
            for g in model.G:
                total = sum(pyo.value(model.r_gen[s, t, g]) for s in model.S)
                solution['repairs']['gen'][t][g] = total / n_scenarios
            for e in model.E:
                total = sum(pyo.value(model.r_line[s, t, e]) for s in model.S)
                solution['repairs']['line'][t][e] = total / n_scenarios

        return solution


class EDOPFTreeProblem(EDOPFProblem):
    """
    Node-indexed extensive form on a sampled scenario tree (multistage SAA).

    Every variable is indexed by a tree node instead of a (scenario, stage)
    pair, so one decision is shared by all sample paths through that node.
    This enforces non-anticipativity: the stage-t decision depends only on the
    observed history xi^1..xi^t, not on branches that are not yet revealed.
    The scenario-flat form cannot express this coupling.

    The stage model (availability transition, repair pipeline, McCormick
    linearization, big-M DC power flow, thermal and generation limits, optional
    repair budget) is the same as in EDOPFProblem; the link from stage t - 1 to
    stage t runs from parent node to child node instead of from (s, t - 1) to
    (s, t). This subclass overrides build_model and extract_solution and
    inherits solve(); EDOPFProblem keeps the scenario-flat form, which also
    serves as the single-scenario perfect-information MIP for DFL training
    labels and as the wait-and-see form (mode='ws' in saa_solver.py).
    """

    def __init__(self, network_data, model_params, scenario_tree):
        self.network = network_data
        self.params = model_params
        self.tree = scenario_tree
        # The leaves (one per sample path) stand in for self.scenarios, whose
        # length the inherited solve() prints; extract_solution is overridden.
        self.scenarios = scenario_tree.leaves
        self.model = None
        self.results = None

    def build_model(self):
        """Build the node-indexed Pyomo model on the scenario tree."""
        model = pyo.ConcreteModel(name="ED_OPF_Tree_SAA")
        tree = self.tree

        # Only decision nodes (stages 1..T) carry variables and constraints. The
        # stage-0 root has none: the stage-1 initial conditions encode its
        # all-available state and empty repair pipeline directly.
        dnodes  = tree.decision_nodes
        stage1  = [n.node_id for n in tree.nodes if n.stage == 1]
        deep    = [n.node_id for n in tree.nodes if n.stage >= 2]
        parent  = {n.node_id: n.parent for n in tree.nodes}
        pathp   = {n.node_id: n.path_prob for n in tree.nodes}

        def xi_g(node, g):
            return int(tree.node(node).xi_gen[g])

        def xi_e(node, e):
            return int(tree.node(node).xi_line[e])

        # ============================================================
        # SETS
        # ============================================================
        model.ND  = pyo.Set(initialize=dnodes, ordered=True)   # decision nodes
        model.ND1 = pyo.Set(initialize=stage1, ordered=True)   # stage-1 (init) nodes
        model.NDdeep = pyo.Set(initialize=deep, ordered=True)  # stage>=2 nodes
        model.G = pyo.RangeSet(0, self.network.n_generators - 1)
        model.L = pyo.RangeSet(0, self.network.n_loads - 1)
        model.N = pyo.RangeSet(0, self.network.n_buses - 1)
        model.E = pyo.RangeSet(0, self.network.n_lines - 1)
        model.K = pyo.RangeSet(1, self.params.tau)

        # ============================================================
        # PARAMETERS (same physical data as the scenario-flat form)
        # ============================================================
        model.gen_cost = pyo.Param(model.G, initialize={g: self.network.gen_cost[g] for g in model.G})
        model.gen_pmin = pyo.Param(model.G, initialize={g: self.network.gen_pmin[g] for g in model.G})
        model.gen_pmax = pyo.Param(model.G, initialize={g: self.network.gen_pmax[g] for g in model.G})
        model.load_demand = pyo.Param(model.L, initialize={l: self.network.load_nominal[l] for l in model.L})
        model.load_shedding_cost = pyo.Param(initialize=self.params.load_shedding_cost)
        model.gen_repair_cost = pyo.Param(model.G, initialize={g: self.params.gen_repair_cost[g] for g in model.G})
        model.line_repair_cost = pyo.Param(model.E, initialize={e: self.params.line_repair_cost[e] for e in model.E})
        model.line_flow_min = pyo.Param(model.E, initialize={e: self.network.line_flow_min[e] for e in model.E})
        model.line_flow_max = pyo.Param(model.E, initialize={e: self.network.line_flow_max[e] for e in model.E})

        # ============================================================
        # DECISION / STATE VARIABLES (indexed by decision node)
        # ============================================================
        model.p_gen  = pyo.Var(model.ND, model.G, within=pyo.NonNegativeReals,
                               bounds=(0, max(self.network.gen_pmax)))
        model.theta  = pyo.Var(model.ND, model.N, within=pyo.Reals, bounds=(-np.pi, np.pi))
        model.d_load = pyo.Var(model.ND, model.L, within=pyo.NonNegativeReals,
                               bounds=(0, max(self.network.load_nominal)))
        f_bound = float(max(np.max(np.abs(self.network.line_flow_min)),
                            np.max(np.abs(self.network.line_flow_max))))
        model.f_line = pyo.Var(model.ND, model.E, within=pyo.Reals, bounds=(-f_bound, f_bound))

        model.r_gen  = pyo.Var(model.ND, model.G, within=pyo.Binary)
        model.r_line = pyo.Var(model.ND, model.E, within=pyo.Binary)
        model.a_gen  = pyo.Var(model.ND, model.G, within=pyo.Binary)
        model.a_line = pyo.Var(model.ND, model.E, within=pyo.Binary)
        model.q_gen  = pyo.Var(model.ND, model.G, model.K, within=pyo.Binary)
        model.q_line = pyo.Var(model.ND, model.E, model.K, within=pyo.Binary)
        model.R_gen  = pyo.Var(model.ND, model.G, within=pyo.Binary)
        model.R_line = pyo.Var(model.ND, model.E, within=pyo.Binary)
        model.w_gen  = pyo.Var(model.ND, model.G, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.w_line = pyo.Var(model.ND, model.E, within=pyo.NonNegativeReals, bounds=(0, 1))

        # ============================================================
        # OBJECTIVE: path-probability-weighted expected cost over nodes
        # ============================================================
        def objective_rule(model):
            total = 0.0
            for nd in model.ND:
                gen_cost = sum(model.gen_cost[g] * model.p_gen[nd, g] for g in model.G)
                shedding_cost = model.load_shedding_cost * sum(
                    model.load_demand[l] - model.d_load[nd, l] for l in model.L)
                repair_cost = (sum(model.gen_repair_cost[g] * model.r_gen[nd, g] for g in model.G)
                               + sum(model.line_repair_cost[e] * model.r_line[nd, e] for e in model.E))
                total += pathp[nd] * (gen_cost + shedding_cost + repair_cost)
            return total
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        tau = self.params.tau

        # ============================================================
        # STAGE-1 INITIAL CONDITIONS (root children)
        # ============================================================
        model.R_gen_init  = pyo.Constraint(model.ND1, model.G, rule=lambda m, nd, g: m.R_gen[nd, g] == 0)
        model.R_line_init = pyo.Constraint(model.ND1, model.E, rule=lambda m, nd, e: m.R_line[nd, e] == 0)
        model.w_gen_init  = pyo.Constraint(model.ND1, model.G, rule=lambda m, nd, g: m.w_gen[nd, g] == 0)
        model.w_line_init = pyo.Constraint(model.ND1, model.E, rule=lambda m, nd, e: m.w_line[nd, e] == 0)
        model.a_gen_init  = pyo.Constraint(model.ND1, model.G,
                                           rule=lambda m, nd, g: m.a_gen[nd, g] == 1 - xi_g(nd, g))
        model.a_line_init = pyo.Constraint(model.ND1, model.E,
                                           rule=lambda m, nd, e: m.a_line[nd, e] == 1 - xi_e(nd, e))

        if tau > 1:
            model.q_gen_init_shift = pyo.Constraint(
                model.ND1, model.G, pyo.RangeSet(1, tau - 1),
                rule=lambda m, nd, g, k: m.q_gen[nd, g, k] == 0)
            model.q_line_init_shift = pyo.Constraint(
                model.ND1, model.E, pyo.RangeSet(1, tau - 1),
                rule=lambda m, nd, e, k: m.q_line[nd, e, k] == 0)

        model.q_gen_init_entry  = pyo.Constraint(model.ND1, model.G,
                                                 rule=lambda m, nd, g: m.q_gen[nd, g, tau] == m.r_gen[nd, g])
        model.q_line_init_entry = pyo.Constraint(model.ND1, model.E,
                                                 rule=lambda m, nd, e: m.q_line[nd, e, tau] == m.r_line[nd, e])

        # ============================================================
        # DYNAMIC CONSTRAINTS (stage>=2 nodes, coupled to their parent node)
        # ============================================================
        model.R_gen_dynamic  = pyo.Constraint(model.NDdeep, model.G,
                                              rule=lambda m, nd, g: m.R_gen[nd, g] == m.q_gen[parent[nd], g, 1])
        model.R_line_dynamic = pyo.Constraint(model.NDdeep, model.E,
                                              rule=lambda m, nd, e: m.R_line[nd, e] == m.q_line[parent[nd], e, 1])

        # McCormick: w = (1 - a_parent) * R
        model.w_gen_ub1 = pyo.Constraint(model.NDdeep, model.G,
                                         rule=lambda m, nd, g: m.w_gen[nd, g] <= 1 - m.a_gen[parent[nd], g])
        model.w_line_ub1 = pyo.Constraint(model.NDdeep, model.E,
                                          rule=lambda m, nd, e: m.w_line[nd, e] <= 1 - m.a_line[parent[nd], e])
        model.w_gen_ub2 = pyo.Constraint(model.NDdeep, model.G,
                                         rule=lambda m, nd, g: m.w_gen[nd, g] <= m.R_gen[nd, g])
        model.w_line_ub2 = pyo.Constraint(model.NDdeep, model.E,
                                          rule=lambda m, nd, e: m.w_line[nd, e] <= m.R_line[nd, e])
        model.w_gen_lb = pyo.Constraint(
            model.NDdeep, model.G,
            rule=lambda m, nd, g: m.w_gen[nd, g] >= (1 - m.a_gen[parent[nd], g]) + m.R_gen[nd, g] - 1)
        model.w_line_lb = pyo.Constraint(
            model.NDdeep, model.E,
            rule=lambda m, nd, e: m.w_line[nd, e] >= (1 - m.a_line[parent[nd], e]) + m.R_line[nd, e] - 1)

        # Availability transition: a_node = (1 - xi_node) * a_parent + w_node
        model.a_gen_transition = pyo.Constraint(
            model.NDdeep, model.G,
            rule=lambda m, nd, g: m.a_gen[nd, g] == (1 - xi_g(nd, g)) * m.a_gen[parent[nd], g] + m.w_gen[nd, g])
        model.a_line_transition = pyo.Constraint(
            model.NDdeep, model.E,
            rule=lambda m, nd, e: m.a_line[nd, e] == (1 - xi_e(nd, e)) * m.a_line[parent[nd], e] + m.w_line[nd, e])

        # Pipeline shift and entry
        if tau > 1:
            model.q_gen_shift = pyo.Constraint(
                model.NDdeep, model.G, pyo.RangeSet(1, tau - 1),
                rule=lambda m, nd, g, k: m.q_gen[nd, g, k] == m.q_gen[parent[nd], g, k + 1])
            model.q_line_shift = pyo.Constraint(
                model.NDdeep, model.E, pyo.RangeSet(1, tau - 1),
                rule=lambda m, nd, e, k: m.q_line[nd, e, k] == m.q_line[parent[nd], e, k + 1])

        model.q_gen_entry  = pyo.Constraint(model.NDdeep, model.G,
                                            rule=lambda m, nd, g: m.q_gen[nd, g, tau] == m.r_gen[nd, g])
        model.q_line_entry = pyo.Constraint(model.NDdeep, model.E,
                                            rule=lambda m, nd, e: m.q_line[nd, e, tau] == m.r_line[nd, e])

        # ============================================================
        # REPAIR FEASIBILITY (all decision nodes)
        # ============================================================
        model.repair_gen_feasibility = pyo.Constraint(
            model.ND, model.G, rule=lambda m, nd, g: m.r_gen[nd, g] <= 1 - m.a_gen[nd, g])
        model.repair_line_feasibility = pyo.Constraint(
            model.ND, model.E, rule=lambda m, nd, e: m.r_line[nd, e] <= 1 - m.a_line[nd, e])

        budget = getattr(self.params, 'repair_budget_per_stage', None)
        if budget is not None:
            model.repair_budget = pyo.Constraint(
                model.ND,
                rule=lambda m, nd: (sum(m.gen_repair_cost[g] * m.r_gen[nd, g] for g in m.G)
                                    + sum(m.line_repair_cost[e] * m.r_line[nd, e] for e in m.E)) <= budget)

        # ============================================================
        # OPF CONSTRAINTS (all decision nodes)
        # ============================================================
        model.slack_bus_constraint = pyo.Constraint(
            model.ND, rule=lambda m, nd: m.theta[nd, self.network.slack_bus] == 0)

        def power_balance_rule(m, nd, n):
            injection = 0.0
            for g in m.G:
                if self.network.gen_buses[g] == n:
                    injection += m.p_gen[nd, g]
            for l in m.L:
                if self.network.load_buses[l] == n:
                    injection -= m.d_load[nd, l]
            flow_sum = 0.0
            for e in m.E:
                from_bus, to_bus = self.network.line_connections[e]
                if from_bus == n:
                    flow_sum += m.f_line[nd, e]
                elif to_bus == n:
                    flow_sum -= m.f_line[nd, e]
            return injection == flow_sum
        model.power_balance = pyo.Constraint(model.ND, model.N, rule=power_balance_rule)

        theta_max = np.pi
        big_M = {e: 2.0 * theta_max * self.network.line_susceptances[e]
                 for e in range(self.network.n_lines)}

        def dc_law_upper_rule(m, nd, e):
            from_bus, to_bus = self.network.line_connections[e]
            b = self.network.line_susceptances[e]
            dc_flow = b * (m.theta[nd, from_bus] - m.theta[nd, to_bus])
            return m.f_line[nd, e] - dc_flow <= big_M[e] * (1 - m.a_line[nd, e])

        def dc_law_lower_rule(m, nd, e):
            from_bus, to_bus = self.network.line_connections[e]
            b = self.network.line_susceptances[e]
            dc_flow = b * (m.theta[nd, from_bus] - m.theta[nd, to_bus])
            return m.f_line[nd, e] - dc_flow >= -big_M[e] * (1 - m.a_line[nd, e])

        model.dc_law_upper = pyo.Constraint(model.ND, model.E, rule=dc_law_upper_rule)
        model.dc_law_lower = pyo.Constraint(model.ND, model.E, rule=dc_law_lower_rule)

        model.gen_min_constraint = pyo.Constraint(
            model.ND, model.G, rule=lambda m, nd, g: m.p_gen[nd, g] >= m.a_gen[nd, g] * m.gen_pmin[g])
        model.gen_max_constraint = pyo.Constraint(
            model.ND, model.G, rule=lambda m, nd, g: m.p_gen[nd, g] <= m.a_gen[nd, g] * m.gen_pmax[g])

        model.load_shedding_constraint = pyo.Constraint(
            model.ND, model.L, rule=lambda m, nd, l: m.d_load[nd, l] <= m.load_demand[l])

        model.line_flow_min_constraint = pyo.Constraint(
            model.ND, model.E, rule=lambda m, nd, e: m.f_line[nd, e] >= m.a_line[nd, e] * m.line_flow_min[e])
        model.line_flow_max_constraint = pyo.Constraint(
            model.ND, model.E, rule=lambda m, nd, e: m.f_line[nd, e] <= m.a_line[nd, e] * m.line_flow_max[e])

        self.model = model
        return model

    def set_norepair_start(self):
        """Set every variable to the NoRepair point: no repairs (r = 0),
        availability propagated from the sampled failures with an empty
        pipeline, and an all-off dispatch (p = d = f = theta = 0, all load shed).
        The point is feasible because NetworkData sets gen_pmin = 0 for every
        generator (toy and PGLib networks alike).
        saa_solver uses it as a MIP warm start (passed only to solvers that
        support one, and only on the single-solve path of solve()) and as the
        fallback solution if the solver returns none.
        model.ND is in breadth-first order, so a parent's availability is
        computed before its children read it.
        """
        model = self.model
        tree = self.tree
        parent = {n.node_id: n.parent for n in tree.nodes}
        a_gen, a_line = {}, {}
        for nd in model.ND:
            n = tree.node(nd)
            if n.stage == 1:
                ag = (1 - n.xi_gen).astype(int)
                al = (1 - n.xi_line).astype(int)
            else:
                p = parent[nd]
                ag = ((1 - n.xi_gen) * a_gen[p]).astype(int)
                al = ((1 - n.xi_line) * a_line[p]).astype(int)
            a_gen[nd], a_line[nd] = ag, al
            for g in model.G:
                model.a_gen[nd, g].value = int(ag[g])
                model.r_gen[nd, g].value = 0
                model.R_gen[nd, g].value = 0
                model.w_gen[nd, g].value = 0.0
                model.p_gen[nd, g].value = 0.0
                for k in model.K:
                    model.q_gen[nd, g, k].value = 0
            for e in model.E:
                model.a_line[nd, e].value = int(al[e])
                model.r_line[nd, e].value = 0
                model.R_line[nd, e].value = 0
                model.w_line[nd, e].value = 0.0
                model.f_line[nd, e].value = 0.0
                for k in model.K:
                    model.q_line[nd, e, k].value = 0
            for b in model.N:
                model.theta[nd, b].value = 0.0
            for l in model.L:
                model.d_load[nd, l].value = 0.0

    def extract_solution(self, force=False):
        """
        Extract the tree solution.

        Returns a node-indexed policy ('tree_policy') for the closed-loop
        out-of-sample evaluation in saa_solver, plus a leaf-path view in the
        layout of EDOPFProblem.extract_solution (generation / angles /
        load_served / line_flows keyed by path index, availability by path,
        repairs as the path-probability-weighted mean per stage), which the
        statistics, plotting and validation code reads.

        With force=True the current variable values are read even without solver
        results attached (used by the NoRepair fallback, where the values are set
        by set_norepair_start rather than produced by a solve).
        """
        if self.model is None or (self.results is None and not force):
            return None

        model = self.model
        tree = self.tree
        T = tree.T   # the tree's own horizon, which can be shorter than params.T
        nG, nL, nN, nE = (self.network.n_generators, self.network.n_loads,
                          self.network.n_buses, self.network.n_lines)

        def rd(x):
            return int(round(pyo.value(x)))

        # ---- node-indexed policy ----
        node_pol = {}
        for nd in model.ND:
            node_pol[nd] = {
                'stage':  tree.node(nd).stage,
                'parent': tree.node(nd).parent,
                'r_gen':  np.array([rd(model.r_gen[nd, g]) for g in model.G], dtype=int),
                'r_line': np.array([rd(model.r_line[nd, e]) for e in model.E], dtype=int),
                'a_gen':  np.array([rd(model.a_gen[nd, g]) for g in model.G], dtype=int),
                'a_line': np.array([rd(model.a_line[nd, e]) for e in model.E], dtype=int),
            }

        tree_policy = {
            'branching': tree.branching,
            'T': T,
            'root': 0,
            'nodes': node_pol,
            'children': {n.node_id: list(n.children) for n in tree.nodes},
            'xi_gen':  {n.node_id: n.xi_gen  for n in tree.nodes},
            'xi_line': {n.node_id: n.xi_line for n in tree.nodes},
        }

        # ---- leaf-path view in the scenario-flat layout ----
        # Map each leaf to the chain of decision nodes on its root->leaf path.
        def path_nodes(leaf_id):
            chain = []
            cur = leaf_id
            while cur is not None and tree.node(cur).stage >= 1:
                chain.append(cur)
                cur = tree.node(cur).parent
            chain.reverse()  # stage 1 .. T
            return chain

        leaves = tree.leaves
        solution = {
            'objective_value': pyo.value(model.objective),
            'generation': {}, 'angles': {}, 'load_served': {}, 'line_flows': {},
            'repairs': {'gen': {}, 'line': {}},
            'availability': {},
            'costs': {'generation': 0.0, 'load_shedding': 0.0, 'repairs': 0.0,
                      'total': pyo.value(model.objective)},
            'tree_policy': tree_policy,
        }

        for s_idx, leaf in enumerate(leaves):
            chain = path_nodes(leaf)
            solution['generation'][s_idx] = {}
            solution['angles'][s_idx] = {}
            solution['load_served'][s_idx] = {}
            solution['line_flows'][s_idx] = {}
            a_gen  = np.zeros((T, nG), dtype=int)
            a_line = np.zeros((T, nE), dtype=int)
            for t, nd in enumerate(chain, start=1):
                solution['generation'][s_idx][t] = {g: pyo.value(model.p_gen[nd, g]) for g in model.G}
                solution['angles'][s_idx][t]     = {n: pyo.value(model.theta[nd, n]) for n in model.N}
                solution['load_served'][s_idx][t] = {l: pyo.value(model.d_load[nd, l]) for l in model.L}
                solution['line_flows'][s_idx][t]  = {e: pyo.value(model.f_line[nd, e]) for e in model.E}
                for g in model.G:
                    a_gen[t - 1, g] = rd(model.a_gen[nd, g])
                for e in model.E:
                    a_line[t - 1, e] = rd(model.a_line[nd, e])
            solution['availability'][s_idx] = {'gen': a_gen, 'line': a_line}

        # Repairs as the path-probability-weighted mean per stage, the tree
        # counterpart of the mean over scenarios in EDOPFProblem.extract_solution.
        for t in range(1, T + 1):
            solution['repairs']['gen'][t] = {g: 0.0 for g in model.G}
            solution['repairs']['line'][t] = {e: 0.0 for e in model.E}
        for nd in model.ND:
            t = tree.node(nd).stage
            w = tree.node(nd).path_prob
            for g in model.G:
                solution['repairs']['gen'][t][g] += w * pyo.value(model.r_gen[nd, g])
            for e in model.E:
                solution['repairs']['line'][t][e] += w * pyo.value(model.r_line[nd, e])

        return solution


def create_problem(network_data, model_params, scenarios, state_dynamics=None):
    """Return the scenario-flat EDOPFProblem (state_dynamics is unused)."""
    return EDOPFProblem(network_data, model_params, scenarios, state_dynamics)


def create_tree_problem(network_data, model_params, scenario_tree):
    """Return the node-indexed tree SAA formulation (EDOPFTreeProblem)."""
    return EDOPFTreeProblem(network_data, model_params, scenario_tree)


if __name__ == "__main__":
    # Smoke test: build (not solve) the scenario-flat model for 10 scenarios
    # on the toy (5-bus) network and print its size
    from network_data import create_network
    from model_parameters import create_model_parameters
    from scenario_generator import create_scenario_generator
    from state_dynamics import create_state_dynamics

    print("Testing Problem Formulation")
    print()

    # Create components
    network = create_network()
    params = create_model_parameters(network, seed=42)
    gen = create_scenario_generator(params, seed=42)
    scenarios = gen.generate_scenarios(n_scenarios=10)
    state_dyn = create_state_dynamics(network, params)

    # Create and build problem
    problem = create_problem(network, params, scenarios, state_dyn)
    problem.build_model()

    print("Model built successfully!")
    print(f"Number of variables: {len(list(problem.model.component_data_objects(pyo.Var)))}")
    print(f"Number of constraints: {len(list(problem.model.component_data_objects(pyo.Constraint)))}")
