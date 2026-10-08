"""
Pseudo_DFL Solver Module
========================
Rolls out the trained DFL repair policy on fresh scenarios and evaluates the
resulting total cost. Also provides three baselines for comparison:

  - AlwaysRepair     : repair every failed component at every stage (under a
                       repair budget, the cheapest affordable subset)
  - NoRepair         : never repair anything
  - Near-sighted MPC : certainty-equivalent lookahead over a short window,
                       replanned every stage (NearSightedMPCRollout; cached
                       on disk by evaluate_mpc_cached)

At each stage t (0-indexed) the DFL inference pipeline is:
    1. Observe state  x^in = (a_gen, a_line, (t+1)/T)
    2. NN forward     θ  = RepairNetwork(x^in)
    3. Hard decision  r[k] = 1 iff θ[k] > 0 and mask[k] = 1 (component failed);
                      with a per-stage repair budget, the 0/1 knapsack argmax
                      of θ·r instead (knapsack.budgeted_argmax)
    4. Solve DC-OPF LP (StagewiseOPF) with current a_gen, a_line → stage cost
    5. Add repair costs
    6. Transition state to stage t+1 using the failure realizations of stage
       t+1 (row t+1 of the scenario's failure arrays) and the repairs ordered
       at stage t+1-tau (a repair takes effect tau stages after it is ordered;
       TAU = 1 in config.py)

Dispatch is solved exactly at each stage by the StagewiseOPF LP (scipy linprog,
or gurobipy when lp_method is 'gurobi').
"""

import time
import math
import itertools
import numpy as np
from pathlib import Path

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from Pseudo_DFL_stage_opf import StagewiseOPF
from Pseudo_DFL_model import hard_repair
from knapsack import budgeted_argmax


# ---------------------------------------------------------------------------
# State transition helper (works for any tau)
# ---------------------------------------------------------------------------

def _next_availability(a_gen, a_line, xi_gen_next, xi_line_next,
                        r_gen_history, r_line_history, tau, t):
    """
    Compute availability at stage t+1 given current state and repair history.

    A repair ordered at stage t completes at stage t + tau, so the repairs
    that complete at stage t+1 are the ones ordered at stage t+1-tau.

    Parameters
    ----------
    a_gen, a_line      : ndarray  current availability (stage t)
    xi_gen_next, xi_line_next : ndarray  failure realizations for stage t+1
                         (the MPC lookahead passes failure probabilities
                         instead and gets the expected availability back)
    r_gen_history, r_line_history : list of ndarrays  repair decisions,
                         indexed by stage
    tau : int
    t   : int  current 0-indexed stage

    Returns
    -------
    new_a_gen, new_a_line : ndarray
    """
    t_complete = t + 1 - tau   # index in history that completes at t+1
    R_gen  = r_gen_history [t_complete] if t_complete >= 0 else np.zeros_like(a_gen)
    R_line = r_line_history[t_complete] if t_complete >= 0 else np.zeros_like(a_line)

    # Vectorized availability transition:
    #   a=1, xi=1 → 0 (new failure);  a=1, xi=0 → 1 (stays up)
    #   a=0, R=1  → 1 (repair done);  a=0, R=0  → 0 (stays down)
    new_a_gen  = a_gen  * (1 - xi_gen_next)  + (1 - a_gen)  * R_gen
    new_a_line = a_line * (1 - xi_line_next) + (1 - a_line) * R_line
    return new_a_gen, new_a_line


# ---------------------------------------------------------------------------
# DFL rollout
# ---------------------------------------------------------------------------

class PseudoDFLRollout:
    """
    Rollout evaluator for the trained DFL repair policy.

    During rollout the policy is:
        θ = RepairNetwork(state_features)
        r[k] = 1 iff θ[k] > 0 and component k is currently failed
    With a per-stage repair budget, r is instead the 0/1 knapsack argmax of
    θ·r over the failed components (knapsack.budgeted_argmax).
    Dispatch is solved exactly (not predicted) by the DC-OPF LP.
    """

    def __init__(self, network, params, model, epsilon=0.1,
                 lp_method=None):
        """
        Parameters
        ----------
        network   : NetworkData
        params    : ModelParameters
        model     : RepairNetwork (trained; put in eval mode here)
        epsilon   : float   stored only; inference uses the unperturbed
                    decision (knapsack.budgeted_argmax)
        lp_method : str or None  per-stage DC-OPF backend: scipy linprog method
                    or 'gurobi'. None -> solver_config.LP_SOLVER.
        """
        self.network  = network
        self.params   = params
        self.model    = model
        self.epsilon  = epsilon
        self.opf      = StagewiseOPF(network, params, lp_method=lp_method)
        # Repair costs in network-output order [gen, line] and the per-stage
        # budget (None = no budget, which gives the threshold rule).
        self._beta    = np.concatenate([np.asarray(params.gen_repair_cost,  float),
                                        np.asarray(params.line_repair_cost, float)])
        self._budget  = getattr(params, 'repair_budget_per_stage', None)

        if _TORCH_AVAILABLE:
            self.model.eval()

    # ------------------------------------------------------------------
    # Single-scenario rollout
    # ------------------------------------------------------------------

    def rollout_scenario(self, scenario):
        """
        Evaluate the DFL policy on one scenario.

        Returns
        -------
        dict with keys:
          'total_cost'       : float  opf_cost + repair_cost
          'opf_cost'         : float  generation + load-shedding cost
          'repair_cost'      : float
          'load_shed_mwh'    : float  unserved load, summed over loads and stages
          'availability_gen' : ndarray (T, n_gen)
          'availability_line': ndarray (T, n_lines)
          'repairs_gen'      : ndarray (T, n_gen)
          'repairs_line'     : ndarray (T, n_lines)
          'stage_costs'      : list of floats length T
        """
        n = self.network
        p = self.params
        T       = p.T
        tau     = p.tau
        n_gen   = n.n_generators
        n_lines = n.n_lines

        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Pseudo_DFL rollout.")

        self.model.eval()

        # ----- Initialize state ----------------------------------------
        # Stage 1 (t=0): a = 1 - ξ¹ (pre-horizon state is all-operational)
        a_gen  = (1 - scenario.gen_failures [0, :]).astype(float)
        a_line = (1 - scenario.line_failures[0, :]).astype(float)

        # Repair history (to handle tau > 1)
        r_gen_history  = []
        r_line_history = []

        # Recording arrays
        avail_gen  = np.zeros((T, n_gen))
        avail_line = np.zeros((T, n_lines))
        rep_gen    = np.zeros((T, n_gen))
        rep_line   = np.zeros((T, n_lines))
        stage_costs = []

        total_opf       = 0.0
        total_repair    = 0.0
        total_load_shed = 0.0
        load_nom = np.asarray(n.load_nominal)

        for t in range(T):
            # ---- Record current availability --------------------------
            avail_gen [t, :] = a_gen
            avail_line[t, :] = a_line

            # ---- Build feature vector ---------------------------------
            stage_frac = float(t + 1) / float(T)
            features   = np.concatenate([a_gen, a_line, [stage_frac]])
            x          = torch.FloatTensor(features).unsqueeze(0)

            # ---- Predict θ and determine repairs ----------------------
            with torch.no_grad():
                theta = self.model(x).squeeze(0).numpy()

            mask_gen  = 1.0 - a_gen
            mask_line = 1.0 - a_line
            mask = np.concatenate([mask_gen, mask_line])

            # Budget-aware decision: with no budget this is exactly the threshold
            # rule (hard_repair); with a budget it is the per-stage knapsack
            # argmax over theta subject to repair spend <= B.
            r = budgeted_argmax(theta, mask, self._beta, self._budget)
            r_gen  = r[:n_gen]
            r_line = r[n_gen:]

            r_gen_history.append(r_gen.copy())
            r_line_history.append(r_line.copy())
            rep_gen [t, :] = r_gen
            rep_line[t, :] = r_line

            # ---- Solve DC-OPF LP (dispatch given current a) -----------
            opf_cost, _, d_load, feasible = self.opf.solve(a_gen, a_line)
            repair_cost = self.opf.compute_repair_cost(r_gen, r_line)

            stage_cost = opf_cost + repair_cost
            stage_costs.append(stage_cost)
            total_opf       += opf_cost
            total_repair    += repair_cost
            total_load_shed += float(np.sum(load_nom - np.clip(d_load, 0.0, None)))

            # ---- Transition to next stage -----------------------------
            if t < T - 1:
                xi_gen_next  = scenario.gen_failures [t + 1, :].astype(float)
                xi_line_next = scenario.line_failures[t + 1, :].astype(float)

                a_gen, a_line = _next_availability(
                    a_gen, a_line,
                    xi_gen_next, xi_line_next,
                    r_gen_history, r_line_history,
                    tau, t
                )

        return {
            'total_cost':        total_opf + total_repair,
            'opf_cost':          total_opf,
            'repair_cost':       total_repair,
            'load_shed_mwh':     total_load_shed,
            'availability_gen':  avail_gen,
            'availability_line': avail_line,
            'repairs_gen':       rep_gen,
            'repairs_line':      rep_line,
            'stage_costs':       stage_costs,
        }

    # ------------------------------------------------------------------
    # Multi-scenario evaluation
    # ------------------------------------------------------------------

    def evaluate(self, scenarios, verbose=True):
        """
        Evaluate the DFL policy on a list of scenarios.

        Returns
        -------
        _cost_stats dict (label 'Pseudo_DFL'): summary statistics plus
        per-scenario costs and load shed
        """
        n_sc = len(scenarios)
        if verbose:
            print(f"  Evaluating Pseudo_DFL policy on {n_sc} scenarios …")

        t0 = time.time()
        costs, opf_costs, repair_costs, load_shed_mwhs = [], [], [], []
        for s_idx, sc in enumerate(scenarios):
            res = self.rollout_scenario(sc)
            costs.append(res['total_cost'])
            opf_costs.append(res['opf_cost'])
            repair_costs.append(res['repair_cost'])
            load_shed_mwhs.append(res['load_shed_mwh'])
            if verbose and (s_idx + 1) % max(1, n_sc // 5) == 0:
                print(f"    {s_idx+1}/{n_sc}  mean so far: ${np.mean(costs):.2f}")

        return _cost_stats(np.array(costs), 'Pseudo_DFL',
                           eval_time_s=time.time() - t0,
                           opf_costs=np.array(opf_costs),
                           repair_costs=np.array(repair_costs),
                           load_shed_mwhs=np.array(load_shed_mwhs))


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class AlwaysRepairRollout:
    """Greedy baseline: repair every failed component at every stage.

    Under a per-stage repair budget, repairing everything may be unaffordable, so
    the baseline repairs as many failed components as the budget allows,
    cheapest first. With no budget it repairs every failed component.
    """

    def __init__(self, network, params, lp_method=None):
        self.network = network
        self.params  = params
        self.opf     = StagewiseOPF(network, params, lp_method=lp_method)
        self._budget = getattr(params, 'repair_budget_per_stage', None)

    def _affordable_repairs(self, a_gen, a_line):
        """Repairs to order this stage: all failed if unbudgeted, else the
        cheapest affordable subset of the failed components."""
        r_g = 1.0 - a_gen
        r_l = 1.0 - a_line
        if self._budget is None:
            return r_g, r_l
        items = ([(float(self.params.gen_repair_cost[g]),  'g', g)
                  for g in np.where(a_gen  == 0)[0]]
               + [(float(self.params.line_repair_cost[e]), 'l', e)
                  for e in np.where(a_line == 0)[0]])
        items.sort(key=lambda x: x[0])          # cheapest first
        rg = np.zeros_like(a_gen)
        rl = np.zeros_like(a_line)
        spent = 0.0
        for cost, kind, idx in items:
            if spent + cost <= self._budget + 1e-9:
                spent += cost
                if kind == 'g':
                    rg[idx] = 1.0
                else:
                    rl[idx] = 1.0
        return rg, rl

    def rollout_scenario(self, scenario):
        """Single-scenario rollout (always repair). Returns same dict as PseudoDFLRollout."""
        n = self.network
        p = self.params
        T       = p.T
        n_gen   = n.n_generators
        n_lines = n.n_lines
        load_nom = np.asarray(n.load_nominal)

        a_gen  = (1 - scenario.gen_failures [0, :]).astype(float)
        a_line = (1 - scenario.line_failures[0, :]).astype(float)
        r_gen_hist, r_line_hist = [], []

        avail_gen  = np.zeros((T, n_gen))
        avail_line = np.zeros((T, n_lines))
        rep_gen    = np.zeros((T, n_gen))
        rep_line   = np.zeros((T, n_lines))
        stage_costs = []
        total_opf = 0.0
        total_repair = 0.0
        total_load_shed = 0.0

        for t in range(T):
            avail_gen [t, :] = a_gen
            avail_line[t, :] = a_line
            r_g, r_l = self._affordable_repairs(a_gen, a_line)
            r_gen_hist.append(r_g.copy())
            r_line_hist.append(r_l.copy())
            rep_gen [t, :] = r_g
            rep_line[t, :] = r_l
            opf_cost, _, d_load, _ = self.opf.solve(a_gen, a_line)
            rc = self.opf.compute_repair_cost(r_g, r_l)
            stage_costs.append(opf_cost + rc)
            total_opf       += opf_cost
            total_repair    += rc
            total_load_shed += float(np.sum(load_nom - np.clip(d_load, 0.0, None)))
            if t < T - 1:
                a_gen, a_line = _next_availability(
                    a_gen, a_line,
                    scenario.gen_failures [t+1, :].astype(float),
                    scenario.line_failures[t+1, :].astype(float),
                    r_gen_hist, r_line_hist, p.tau, t
                )

        return {
            'total_cost':        total_opf + total_repair,
            'opf_cost':          total_opf,
            'repair_cost':       total_repair,
            'load_shed_mwh':     total_load_shed,
            'availability_gen':  avail_gen,
            'availability_line': avail_line,
            'repairs_gen':       rep_gen,
            'repairs_line':      rep_line,
            'stage_costs':       stage_costs,
        }

    def evaluate(self, scenarios, verbose=True):
        n_sc = len(scenarios)
        if verbose:
            print(f"  Evaluating 'always-repair' baseline on {n_sc} scenarios …")

        t0 = time.time()
        costs, opf_costs, repair_costs, load_shed_mwhs = [], [], [], []
        for sc in scenarios:
            res = self.rollout_scenario(sc)
            costs.append(res['total_cost'])
            opf_costs.append(res['opf_cost'])
            repair_costs.append(res['repair_cost'])
            load_shed_mwhs.append(res['load_shed_mwh'])

        return _cost_stats(np.array(costs), 'AlwaysRepair',
                           eval_time_s=time.time() - t0,
                           opf_costs=np.array(opf_costs),
                           repair_costs=np.array(repair_costs),
                           load_shed_mwhs=np.array(load_shed_mwhs))


class NoRepairRollout:
    """Do-nothing baseline: never repair anything."""

    def __init__(self, network, params, lp_method=None):
        self.network = network
        self.params  = params
        self.opf     = StagewiseOPF(network, params, lp_method=lp_method)

    def rollout_scenario(self, scenario):
        """Single-scenario rollout (no repairs). Returns same dict as PseudoDFLRollout."""
        n = self.network
        p = self.params
        T = p.T
        n_gen   = n.n_generators
        n_lines = n.n_lines
        load_nom = np.asarray(n.load_nominal)

        a_gen  = (1 - scenario.gen_failures [0, :]).astype(float)
        a_line = (1 - scenario.line_failures[0, :]).astype(float)
        r_gen_hist  = [np.zeros(n_gen)   for _ in range(T)]
        r_line_hist = [np.zeros(n_lines) for _ in range(T)]

        avail_gen  = np.zeros((T, n_gen))
        avail_line = np.zeros((T, n_lines))
        stage_costs = []
        total = 0.0
        total_load_shed = 0.0

        for t in range(T):
            avail_gen [t, :] = a_gen
            avail_line[t, :] = a_line
            opf_cost, _, d_load, _ = self.opf.solve(a_gen, a_line)
            stage_costs.append(opf_cost)
            total += opf_cost
            total_load_shed += float(np.sum(load_nom - np.clip(d_load, 0.0, None)))
            if t < T - 1:
                xi_gn = scenario.gen_failures [t+1, :].astype(float)
                xi_ln = scenario.line_failures[t+1, :].astype(float)
                a_gen, a_line = _next_availability(
                    a_gen, a_line, xi_gn, xi_ln, r_gen_hist, r_line_hist, p.tau, t
                )

        return {
            'total_cost':        total,
            'opf_cost':          total,
            'repair_cost':       0.0,
            'load_shed_mwh':     total_load_shed,
            'availability_gen':  avail_gen,
            'availability_line': avail_line,
            'repairs_gen':       np.zeros((T, n_gen)),
            'repairs_line':      np.zeros((T, n_lines)),
            'stage_costs':       stage_costs,
        }

    def evaluate(self, scenarios, verbose=True):
        n_sc = len(scenarios)
        if verbose:
            print(f"  Evaluating 'no-repair' baseline on {n_sc} scenarios …")

        t0 = time.time()
        costs, load_shed_mwhs = [], []
        for sc in scenarios:
            res = self.rollout_scenario(sc)
            costs.append(res['total_cost'])
            load_shed_mwhs.append(res['load_shed_mwh'])

        return _cost_stats(np.array(costs), 'NoRepair',
                           eval_time_s=time.time() - t0,
                           opf_costs=np.array(costs),
                           repair_costs=np.zeros(len(costs)),
                           load_shed_mwhs=np.array(load_shed_mwhs))


_USE_CONFIG = object()   # sentinel: argument not given, read the value from config.py


class NearSightedMPCRollout:
    """Certainty-equivalent MPC baseline with a zero terminal cost.

    At each true stage t (0-indexed) the controller decides which
    currently-failed components to repair by a lookahead over
    L = min(W, T - t) stages:

      1. Enumerate every repair schedule over the window: each currently-failed
         component is assigned to a repair step (one of the actionable steps
         0 .. L-1-tau) or "never". W = tau + 1 has a single actionable step, so
         a schedule reduces to a subset of the failed components; larger W
         allows repairs to be deferred to later steps inside the window.
      2. For each schedule, roll the state forward over the window using
         expected failures (params.lambda_*) instead of realized ones. Step 0
         is scored with the StagewiseOPF LP on the true binary state; later
         steps have fractional availability and are scored with the derated
         DC-OPF (StagewiseOPF.solve_derated). A repair ordered at step k
         completes tau steps later (state transition shared with the real
         rollout via _next_availability), so it reduces cost from step k+tau on.
      3. Pick the schedule with the smallest lookahead cost. The all-"never"
         schedule is always a candidate, so "repair nothing" wins when no
         repair pays off inside the window. Under a per-stage repair budget,
         schedules that overspend at any step are skipped.
      4. Execute only the current-stage decision (the step-0 repairs); advance
         the true state with the scenario's realized failures and replan
         (rolling horizon).

    The window W is the ``window`` argument and must satisfy tau + 1 <= W <= T;
    Pseudo_DFL_main.py passes --mpc-window, whose default is
    NEAR_SIGHTED_MPC_WINDOW in config.py. Larger W captures more stages of
    repair benefit but enumerates (L - tau + 1)^|F_t| schedules per stage,
    where F_t is the set of failed components. The last tau stages never
    repair, because a repair ordered there would complete after the last stage.

    Schedule-search cap (MPC_SCHEDULE_CAP_RATIO in config.py). The exact
    enumeration above is intractable when |F_t| is large (e.g. case14). When
    a cap is set, each stage whose exact schedule count exceeds the cap instead
    evaluates a uniform on-the-fly sample of `cap` schedules (the no-repair
    schedule always included), so MPC is an approximate randomized lookahead
    there and exact wherever the count is within the cap. The cap is
    ceil(ratio * (W - tau + 1)^MPC_CAP_REFERENCE_COMPONENTS). With the default
    ratio 1 and 7 reference components it equals the largest schedule count on
    the 7-component toy network (3^7 = 2187 for W = 3, tau = 1), so toy stays
    exact while the larger networks become tractable. Ratio None disables the
    cap (exact everywhere, intractable beyond toy).
    """

    def __init__(self, network, params, window=2, lp_method=None,
                 cap_ratio=_USE_CONFIG, cap_reference_components=_USE_CONFIG,
                 sample_seed=_USE_CONFIG):
        self.network = network
        self.params  = params
        self.window  = self._validate_window(window, params)
        self.opf     = StagewiseOPF(network, params, lp_method=lp_method)

        # Schedule-search cap. Defaults come from config.py; explicit arguments
        # override them. ratio None means exact enumeration.
        from config import (MPC_SCHEDULE_CAP_RATIO,
                            MPC_CAP_REFERENCE_COMPONENTS, MPC_SAMPLE_SEED)
        self.cap_ratio = (MPC_SCHEDULE_CAP_RATIO if cap_ratio is _USE_CONFIG
                          else cap_ratio)
        self.cap_reference_components = (
            MPC_CAP_REFERENCE_COMPONENTS if cap_reference_components is _USE_CONFIG
            else cap_reference_components)
        self.sample_seed = (MPC_SAMPLE_SEED if sample_seed is _USE_CONFIG
                            else sample_seed)

        # Absolute per-stage schedule cap. The reference is a fixed component
        # count from config, not taken from the network being solved:
        #     reference = (W - tau + 1) ** MPC_CAP_REFERENCE_COMPONENTS
        #     cap       = ceil(ratio * reference)            # when ratio is not None
        if self.cap_ratio is None:
            self.schedule_cap = None                          # exact, no cap
        else:
            base      = self.window - params.tau + 1
            reference = base ** int(self.cap_reference_components)
            self.schedule_cap = int(math.ceil(float(self.cap_ratio) * reference))
        self._rng = np.random.default_rng(self.sample_seed)

    @staticmethod
    def _validate_window(window, params):
        """Coerce window to int and check tau + 1 <= W <= T (else ValueError)."""
        try:
            w = int(window)
        except (TypeError, ValueError):
            raise ValueError(
                f"NearSightedMPC window must be an integer, got {window!r}."
            )
        if w != window:
            raise ValueError(
                f"NearSightedMPC window must be a whole number, got {window!r}."
            )
        w_min = params.tau + 1
        w_max = params.T
        if not (w_min <= w <= w_max):
            raise ValueError(
                f"NearSightedMPC window W={w} is out of range. With tau="
                f"{params.tau} and T={params.T}, W must satisfy "
                f"{w_min} <= W <= {w_max} (a repair ordered now completes tau "
                f"stages later, so W < tau+1 can never benefit; W > T looks "
                f"past the horizon)."
            )
        return w

    # ------------------------------------------------------------------
    # Lookahead decision
    # ------------------------------------------------------------------

    def _decide(self, a_gen, a_line, t):
        """Return (rep_gen, rep_line) binary masks for the current stage t.

        Enumerates all repair schedules of the currently-failed components over
        the effective window, skips those that exceed the per-stage budget, and
        returns the step-0 repairs of the cheapest one. If a schedule cap is
        configured and the exact count exceeds it, a uniform on-the-fly sample
        of `cap` schedules is searched instead (see the class docstring).
        """
        n        = self.network
        p        = self.params
        n_gen    = n.n_generators
        n_lines  = n.n_lines
        T        = p.T
        tau      = p.tau

        zero_g = np.zeros(n_gen)
        zero_l = np.zeros(n_lines)

        # Effective window (truncates near the horizon end). A repair ordered at
        # step k completes at k+tau; it benefits the window only if k+tau <= L-1,
        # so the actionable steps are 0 .. L-1-tau. If none exist (the last tau
        # stages), never repair: a repair would complete after the last stage.
        L = min(self.window, T - t)
        n_actionable = L - tau
        if n_actionable <= 0:
            return zero_g, zero_l

        failed = ([('g', g) for g in np.where(a_gen  == 0)[0]] +
                  [('l', l) for l in np.where(a_line == 0)[0]])
        if not failed:
            return zero_g, zero_l

        # OPF at step 0 uses the true binary state and is identical for every
        # schedule (a repair only helps from step tau on). Compute it once.
        c0, _, _, _ = self.opf.solve(a_gen, a_line)

        # Each failed component is assigned to an actionable step or "never".
        choices = list(range(n_actionable)) + [None]

        # Per-stage repair budget: every step's repair spend must be affordable.
        budget = getattr(p, 'repair_budget_per_stage', None)

        # Exact enumeration when the schedule space is within the cap (always the
        # case on the toy network when the reference component count is at least
        # 7, the default, and the ratio is at least 1); otherwise a uniform
        # on-the-fly sample of `schedule_cap` schedules (never materializing the
        # full product).
        n_failed    = len(failed)
        exact_count = len(choices) ** n_failed
        if self.schedule_cap is None or exact_count <= self.schedule_cap:
            schedules = itertools.product(choices, repeat=n_failed)
        else:
            schedules = self._sample_assignments(choices, n_failed, self.schedule_cap)

        best_key  = np.inf
        best_gen  = zero_g
        best_line = zero_l
        for assign in schedules:
            sched_gen  = [np.zeros(n_gen)   for _ in range(L)]
            sched_line = [np.zeros(n_lines) for _ in range(L)]
            for (kind, idx), step in zip(failed, assign):
                if step is None:
                    continue
                if kind == 'g':
                    sched_gen[step][idx]  = 1.0
                else:
                    sched_line[step][idx] = 1.0

            # Drop schedules whose per-step spend exceeds the budget (the
            # all-"never" schedule always survives, so a feasible minimizer exists).
            if budget is not None and any(
                    self.opf.compute_repair_cost(sched_gen[k], sched_line[k])
                    > budget + 1e-9 for k in range(L)):
                continue

            key = self._window_cost(a_gen, a_line, t, L, c0, sched_gen, sched_line)
            if key < best_key:
                best_key  = key
                best_gen  = sched_gen[0]
                best_line = sched_line[0]

        return best_gen, best_line

    def _sample_assignments(self, choices, n_failed, cap):
        """Yield `cap` repair-schedule assignments generated on the fly.

        The all-"never" (no-repair) schedule is always yielded first as a quality
        floor; each of the remaining cap-1 draws picks a choice for every failed
        component independently and uniformly, so a schedule can be drawn more
        than once and the full product is never materialized. One RNG, seeded
        with self.sample_seed in __init__, is shared by all stages and
        scenarios, so the draws are reproducible for a fixed seed and
        evaluation order.
        """
        yield (None,) * n_failed                     # no-repair schedule (floor)
        n_choices = len(choices)
        for _ in range(max(cap - 1, 0)):
            idx = self._rng.integers(0, n_choices, size=n_failed)
            yield tuple(choices[i] for i in idx)

    def _window_cost(self, a_gen, a_line, t, L, c0, sched_gen, sched_line):
        """Certainty-equivalent cost of one repair schedule over the L-step window.

        The state is propagated with expected failures (lambda) through
        _next_availability, the same transition as the real rollout, so the
        repair delay tau is handled identically. Step 0 uses the binary-state
        OPF cost c0; later steps have fractional availability and are scored
        with StagewiseOPF.solve_derated. The repair cost of every step is added.
        The repair history starts at step 0, so repairs ordered before stage t
        that are still in progress (possible only when tau > 1) are not credited.
        """
        p = self.params
        total = c0 + self.opf.compute_repair_cost(sched_gen[0], sched_line[0])

        a_g = np.asarray(a_gen,  dtype=float).copy()
        a_l = np.asarray(a_line, dtype=float).copy()
        r_g_hist = [sched_gen[0]]
        r_l_hist = [sched_line[0]]
        for k in range(L - 1):
            lam_g = p.lambda_g[:, t + k + 1]
            lam_e = p.lambda_e[:, t + k + 1]
            a_g, a_l = _next_availability(a_g, a_l, lam_g, lam_e,
                                          r_g_hist, r_l_hist, tau=p.tau, t=k)
            # Mean-failure rollout makes availability fractional; score it on the
            # derated network (a weaker conductor), not the binary big-M OPF.
            opf_k, _, _, _ = self.opf.solve_derated(a_g, a_l)
            total += opf_k + self.opf.compute_repair_cost(sched_gen[k+1],
                                                          sched_line[k+1])
            r_g_hist.append(sched_gen[k+1])
            r_l_hist.append(sched_line[k+1])
        return total

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout_scenario(self, scenario):
        """Single-scenario rollout. Returns same dict as PseudoDFLRollout."""
        n = self.network
        p = self.params
        T       = p.T
        n_gen   = n.n_generators
        n_lines = n.n_lines
        load_nom = np.asarray(n.load_nominal)

        a_gen  = (1 - scenario.gen_failures [0, :]).astype(float)
        a_line = (1 - scenario.line_failures[0, :]).astype(float)
        r_gen_hist, r_line_hist = [], []

        avail_gen  = np.zeros((T, n_gen))
        avail_line = np.zeros((T, n_lines))
        rep_gen    = np.zeros((T, n_gen))
        rep_line   = np.zeros((T, n_lines))
        stage_costs = []
        total_opf = 0.0
        total_repair = 0.0
        total_load_shed = 0.0

        for t in range(T):
            avail_gen [t, :] = a_gen
            avail_line[t, :] = a_line

            r_g, r_l = self._decide(a_gen, a_line, t)
            r_gen_hist.append(r_g.copy())
            r_line_hist.append(r_l.copy())
            rep_gen [t, :] = r_g
            rep_line[t, :] = r_l

            opf_cost, _, d_load, _ = self.opf.solve(a_gen, a_line)
            rc = self.opf.compute_repair_cost(r_g, r_l)
            stage_costs.append(opf_cost + rc)
            total_opf       += opf_cost
            total_repair    += rc
            total_load_shed += float(np.sum(load_nom - np.clip(d_load, 0.0, None)))

            if t < T - 1:
                a_gen, a_line = _next_availability(
                    a_gen, a_line,
                    scenario.gen_failures [t+1, :].astype(float),
                    scenario.line_failures[t+1, :].astype(float),
                    r_gen_hist, r_line_hist, p.tau, t
                )

        return {
            'total_cost':        total_opf + total_repair,
            'opf_cost':          total_opf,
            'repair_cost':       total_repair,
            'load_shed_mwh':     total_load_shed,
            'availability_gen':  avail_gen,
            'availability_line': avail_line,
            'repairs_gen':       rep_gen,
            'repairs_line':      rep_line,
            'stage_costs':       stage_costs,
        }

    def evaluate(self, scenarios, verbose=True):
        n_sc = len(scenarios)
        if verbose:
            print(f"  Evaluating 'near-sighted MPC' baseline (W={self.window}) "
                  f"on {n_sc} scenarios …")

        t0 = time.time()
        costs, opf_costs, repair_costs, load_shed_mwhs = [], [], [], []
        for sc in scenarios:
            res = self.rollout_scenario(sc)
            costs.append(res['total_cost'])
            opf_costs.append(res['opf_cost'])
            repair_costs.append(res['repair_cost'])
            load_shed_mwhs.append(res['load_shed_mwh'])

        return _cost_stats(np.array(costs), 'NearSightedMPC',
                           eval_time_s=time.time() - t0,
                           opf_costs=np.array(opf_costs),
                           repair_costs=np.array(repair_costs),
                           load_shed_mwhs=np.array(load_shed_mwhs))


def evaluate_mpc_cached(network, params, scenarios, window, cache_path,
                        eval_seed, network_name='', lp_method=None,
                        verbose=True):
    """Evaluate the Near-sighted MPC baseline, caching the result on disk.

    The MPC result does not depend on the DFL training-set size, so in a sweep
    over n_train with a fixed n_eval it is the same for every output folder
    (without --n-eval, Pseudo_DFL_main.py sets n_eval = max(5, n_train // 3)).
    The first call writes ``cache_path``; a later call loads the stored result
    when the signature matches (network name and size, mean generator and line
    repair costs, repair budget, model and eval seeds, n_eval, window, T, tau
    and the schedule-cap settings) and recomputes otherwise. Settings outside
    the signature are not checked. They include the load-shedding cost and the
    failure-model parameters, whether set in config.py (LOAD_SHEDDING_COST,
    FAILURE_*) or with the --load-shedding-cost and --failure-* flags of
    Pseudo_DFL_main.py; delete the cache file after changing any of them.
    The AlwaysRepair and NoRepair baselines are not cached; they are recomputed
    in every run.

    Returns the same ``_cost_stats`` dict that ``NearSightedMPCRollout.evaluate``
    produces, so the caller stores it in ``results_dict`` like the other methods.
    """
    import json
    import os
    # The cap settings are part of the signature: exact vs sampled search (and
    # the ratio, reference and seed) change the result, so a cached result is
    # reused only under the same settings.
    from config import (MPC_SCHEDULE_CAP_RATIO,
                        MPC_CAP_REFERENCE_COMPONENTS, MPC_SAMPLE_SEED)

    n_eval = len(scenarios)
    signature = {
        'network':    network_name,
        'n_gen':      int(network.n_generators),
        'n_line':     int(network.n_lines),
        'gen_base':   round(float(np.mean(params.gen_repair_cost)), 6),
        'line_base':  round(float(np.mean(params.line_repair_cost)), 6),
        'model_seed': int(params.seed),
        'eval_seed':  int(eval_seed),
        'n_eval':     int(n_eval),
        'window':     int(window),
        'T':          int(params.T),
        'tau':        int(params.tau),
        'repair_budget': (round(float(params.repair_budget_per_stage), 6)
                          if getattr(params, 'repair_budget_per_stage', None) is not None
                          else None),
        'cap_ratio':   MPC_SCHEDULE_CAP_RATIO,
        'cap_ref':     int(MPC_CAP_REFERENCE_COMPONENTS),
        'sample_seed': int(MPC_SAMPLE_SEED),
    }
    cache_path = Path(cache_path)

    # ---- Try the cache ------------------------------------------------------
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                blob = json.load(f)
            if blob.get('signature') == signature and 'result' in blob:
                if verbose:
                    print(f"  NearSightedMPC (W={window}): loaded from cache "
                          f"→ {cache_path}")
                return blob['result']
            if verbose:
                print("  NearSightedMPC: cache signature mismatch — recomputing")
        except Exception:
            if verbose:
                print("  NearSightedMPC: unreadable cache — recomputing")

    # ---- Compute (cache miss) ----------------------------------------------
    rollout = NearSightedMPCRollout(network, params, window=window,
                                    lp_method=lp_method)
    result = rollout.evaluate(scenarios, verbose=verbose)

    # ---- Write the cache atomically ----------------------------------------
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + '.tmp')
        with open(tmp, 'w') as f:
            json.dump({'signature': signature, 'result': result}, f, indent=2)
        os.replace(tmp, cache_path)
        if verbose:
            print(f"  NearSightedMPC (W={window}): cached → {cache_path}")
    except Exception as exc:
        if verbose:
            print(f"  NearSightedMPC: could not write cache ({exc})")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cost_stats(costs, label, eval_time_s=None,
                opf_costs=None, repair_costs=None, load_shed_mwhs=None):
    """Summary statistics of per-scenario costs, in the result format shared by all methods."""
    n = len(costs)
    result = {
        'label':     label,
        'costs':     costs.tolist(),
        'mean_cost': float(np.mean(costs)),
        'std_cost':  float(np.std(costs)),
        'min_cost':  float(np.min(costs)),
        'max_cost':  float(np.max(costs)),
        'median':    float(np.median(costs)),
        'p25':       float(np.percentile(costs, 25)),
        'p75':       float(np.percentile(costs, 75)),
    }
    if eval_time_s is not None:
        result['eval_time_s']              = round(eval_time_s, 3)
        result['mean_time_per_scenario_s'] = round(eval_time_s / n, 6) if n > 0 else 0.0
    if opf_costs is not None:
        result['mean_opf_cost']    = float(np.mean(opf_costs))
        result['mean_repair_cost'] = float(np.mean(repair_costs)) if repair_costs is not None else 0.0
    if load_shed_mwhs is not None:
        result['load_shed_mwhs']      = [float(x) for x in load_shed_mwhs]
        result['mean_load_shed_mwh']  = float(np.mean(load_shed_mwhs))
    return result


def print_evaluation_summary(result):
    """Pretty-print evaluation result dict."""
    lbl = result.get('label', 'Policy')
    print(f"  [{lbl}]")
    print(f"    Mean   : ${result['mean_cost']:>12,.2f}")
    print(f"    Std    : ${result['std_cost']:>12,.2f}")
    print(f"    Median : ${result['median']:>12,.2f}")
    print(f"    Min    : ${result['min_cost']:>12,.2f}")
    print(f"    Max    : ${result['max_cost']:>12,.2f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from network_data import create_network
    from model_parameters import create_model_parameters
    from scenario_generator import create_scenario_generator
    from Pseudo_DFL_model import RepairNetwork

    net    = create_network()
    params = create_model_parameters(net, seed=42)

    gen = create_scenario_generator(params, seed=99)
    test_scenarios = gen.generate_scenarios(20)

    # Untrained (random) model on the toy network: 7 components (2 generators,
    # 5 lines), so 7 availabilities + 1 stage fraction as input
    model = RepairNetwork(input_dim=8, n_components=7)

    dfl = PseudoDFLRollout(net, params, model, lp_method=None)
    res = dfl.evaluate(test_scenarios, verbose=True)
    print("\nDFL (untrained):")
    print_evaluation_summary(res)

    nr  = NoRepairRollout(net, params)
    res_nr = nr.evaluate(test_scenarios, verbose=True)
    print("\nNo-repair baseline:")
    print_evaluation_summary(res_nr)

    ar  = AlwaysRepairRollout(net, params)
    res_ar = ar.evaluate(test_scenarios, verbose=True)
    print("\nAlways-repair baseline:")
    print_evaluation_summary(res_ar)
