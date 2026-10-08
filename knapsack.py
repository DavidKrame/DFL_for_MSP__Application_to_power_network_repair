"""
knapsack.py
===========
Exact 0/1 knapsack oracle for the per-stage repair budget.

At each stage the budgeted repair decision is

    max_r  theta . r
    s.t.   weights . r <= capacity      (per-stage repair budget B^t)
           r <= mask                    (repair only failed components)
           r in {0,1}^n

where ``theta`` are the repair-value coefficients predicted by the network
(theta[k] is read as the value of repairing component k net of its repair
cost), ``weights`` are the per-component repair costs beta, and ``capacity`` is
the budget B^t. The oracle is used by:

  * DFL inference (PseudoDFLRollout in Pseudo_DFL_solver.py),
  * the Monte Carlo Fenchel-Young surrogate in training (Gaussian
    perturb-and-MAP, perturbed_repair_budgeted in Pseudo_DFL_model.py).

``capacity is None`` disables the budget and gives the separable threshold
rule ``r[k] = 1 iff theta[k] > 0 and mask[k] = 1`` exactly.

The solver is a depth-first branch-and-bound, pruned with the fractional
(LP relaxation) bound obtained by filling items in decreasing value/weight
order. Repair costs are real-valued, so the integer-weight DP does not apply.
The networks here have 7 (toy, 5-bus) or 25 (case14) components,
and only failed components with theta > 0 enter the search. At these sizes the
search is fast even in the early stages, when most components are failed. When
all candidates fit in the budget, the threshold set is returned in O(n) without
branching.
"""

import numpy as np

_TOL = 1e-12


def budget_disabled(capacity):
    """True when no budget is in force (callers use the closed-form path)."""
    return capacity is None


def threshold_repair(theta, mask):
    """Unconstrained optimum: r[k] = 1 iff theta[k] > 0 and mask[k] = 1."""
    theta = np.asarray(theta, dtype=float)
    mask  = np.asarray(mask,  dtype=float)
    return ((theta > 0.0) & (mask > 0.5)).astype(float)


def budgeted_argmax(theta, mask, weights, capacity):
    """Exact argmax of theta . r subject to the budget, mask, and {0,1}.

    Parameters
    ----------
    theta    : array (n,)  repair-value coefficients
    mask     : array (n,)  1 = component failed (eligible), 0 = operational
    weights  : array (n,)  per-component repair costs beta (> 0)
    capacity : float or None  per-stage budget B^t; None disables the budget

    Returns
    -------
    r : ndarray (n,) of 0.0/1.0
    """
    theta   = np.asarray(theta,   dtype=float)
    mask    = np.asarray(mask,    dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = theta.shape[0]
    r = np.zeros(n)

    if capacity is None:
        return threshold_repair(theta, mask)

    # Candidates: failed components with strictly positive value. Items with
    # theta <= 0 cannot improve the objective and only use budget, so they are
    # left out.
    cand = [k for k in range(n) if mask[k] > 0.5 and theta[k] > 0.0]
    if not cand:
        return r

    total_w = float(np.sum(weights[cand]))
    if total_w <= capacity + _TOL:        # slack: everything fits → threshold set
        r[cand] = 1.0
        return r

    v = theta[cand].astype(float)
    w = weights[cand].astype(float)
    chosen = _knapsack_bnb(v, w, float(capacity))
    for j, k in enumerate(cand):
        if chosen[j]:
            r[k] = 1.0
    return r


def _knapsack_bnb(values, weights, capacity):
    """Exact 0/1 knapsack with real weights: maximize value s.t. weight <= capacity.

    Depth-first branch-and-bound over the items sorted by value/weight ratio,
    pruned with the fractional (greedy LP) bound. All values and weights are
    assumed > 0. Returns a boolean array of the chosen items, in input order.
    """
    n = len(values)
    order = np.argsort(-(values / np.maximum(weights, _TOL)))
    v = values[order]
    w = weights[order]

    best_val = [-1.0]
    best_set = [np.zeros(n, dtype=bool)]
    cur      = np.zeros(n, dtype=bool)

    def upper_bound(i, cap, val):
        # Greedy fractional fill of the remaining (sorted) items.
        b, c = val, cap
        while i < n and w[i] <= c + _TOL:
            c -= w[i]
            b += v[i]
            i += 1
        if i < n and c > _TOL:
            b += v[i] * (c / w[i])
        return b

    def dfs(i, cap, val):
        if val > best_val[0]:
            best_val[0] = val
            best_set[0] = cur.copy()
        if i >= n:
            return
        if upper_bound(i, cap, val) <= best_val[0] + _TOL:   # prune
            return
        if w[i] <= cap + _TOL:                                # include item i
            cur[i] = True
            dfs(i + 1, cap - w[i], val + v[i])
            cur[i] = False
        dfs(i + 1, cap, val)                                  # exclude item i

    dfs(0, capacity, 0.0)

    chosen = np.zeros(n, dtype=bool)
    chosen[order] = best_set[0]
    return chosen


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import itertools

    def brute_force(theta, mask, weights, capacity):
        n = len(theta)
        best_v, best_r = -np.inf, np.zeros(n)
        for bits in itertools.product((0, 1), repeat=n):
            r = np.array(bits, dtype=float)
            if np.any(r > mask):                      # respect mask
                continue
            if capacity is not None and np.dot(weights, r) > capacity + _TOL:
                continue
            val = float(np.dot(theta, r))
            if val > best_v:
                best_v, best_r = val, r
        return best_v, best_r

    rng = np.random.default_rng(0)
    n_fail = 0
    for trial in range(2000):
        n = int(rng.integers(1, 13))
        theta   = rng.normal(size=n)                  # mixed-sign values
        mask    = (rng.random(n) < 0.7).astype(float)
        weights = rng.uniform(0.5, 5.0, size=n)
        cap     = float(rng.uniform(0.0, weights.sum()))
        r = budgeted_argmax(theta, mask, weights, cap)
        # feasibility
        assert np.all(r <= mask + _TOL)
        assert np.dot(weights, r) <= cap + 1e-9, (np.dot(weights, r), cap)
        # optimality (compare objective value to brute force)
        bf_val, _ = brute_force(theta, mask, weights, cap)
        if abs(float(np.dot(theta, r)) - bf_val) > 1e-7:
            n_fail += 1
            print(f"  MISMATCH trial {trial}: bnb={np.dot(theta,r):.6f} bf={bf_val:.6f}")
    print(f"optimality check: {2000 - n_fail}/2000 instances matched brute force")

    # slack budget and capacity=None both reproduce the threshold rule
    theta = np.array([2.0, -1.0, 0.5, 3.0])
    mask  = np.array([1.0,  1.0, 0.0, 1.0])
    w     = np.array([1.0,  1.0, 1.0, 1.0])
    thr   = threshold_repair(theta, mask)
    assert np.array_equal(budgeted_argmax(theta, mask, w, None), thr)
    assert np.array_equal(budgeted_argmax(theta, mask, w, 1e6),  thr)
    assert budget_disabled(None) and not budget_disabled(10.0)
    print("slack/None → threshold rule: OK")
    print(f"threshold example: r = {thr}  (expected [1,0,0,1])")
