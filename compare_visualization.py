"""
compare_visualization.py
========================
Cross-method comparison plots for the ED-OPF repair experiments. Methods are
keyed by their results_dict labels (Pseudo_DFL is the DFL method).

Functions
---------
plot_cost_bars_with_bounds        mean ± std cost bars per method; SDDiP LB/UB lines
plot_cost_distributions_all       violin + box plots of per-scenario total cost
plot_load_shedding_distributions  violin + box plots of per-scenario load shed (MWh)
plot_cost_breakdown_stacked       stacked mean OPF cost + mean repair cost bars
plot_repair_heatmap_comparison    availability (top) and repair-rate (bottom) heatmaps,
                                  one column per method
plot_sddip_convergence            SDDiP lower bound and upper-bound estimate per iteration
plot_repair_rate_by_stage         fraction of failed components repaired at each stage
plot_stage_cost_profile           mean cost per stage
plot_cost_share_pct               generation / load-shedding / repair shares of mean cost
plot_cost_share_pct_normalized    the same shares after per-component normalization
plot_computation_time             log-scale bars of computation times
plot_single_repair_heatmap        availability and repair-rate heatmap of one method
create_comparison_folder          writes a subset of these plots and comparison_summary.json
                                  to out_dir/comparison/ (listed in its docstring)

Low-level helpers:
  _rollout_policy_on_scenarios    simulate a fixed (stage, component) repair schedule
  _avg_heatmap_data               average a rollout object's availability, repairs and
                                  stage costs over scenarios
  _extract_policy_from_solution   rounded repair schedule from an SAA or SDDiP solution
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Method colors keyed by results_dict label; other labels are drawn in
# _DEFAULT_COLOR. Pseudo_DFL, SDDiP and NearSightedMPC match the colors in
# Pseudo_DFL_visualization.py; AlwaysRepair and NoRepair have other colors there.
# ---------------------------------------------------------------------------
COLORS = {
    'Pseudo_DFL':  'steelblue',
    'SDDiP':       '#2ecc71',
    'SAA':         '#9b59b6',
    'AlwaysRepair': 'chocolate',
    'NoRepair':    '#7f8c8d',
    'NearSightedMPC': '#16a085',
}
_DEFAULT_COLOR = '#7f7f7f'

METHOD_ORDER = ['NoRepair', 'AlwaysRepair', 'NearSightedMPC', 'SAA', 'SDDiP', 'Pseudo_DFL']


def _color(label):
    return COLORS.get(label, _DEFAULT_COLOR)


def _save(fig, path, default_name):
    path = Path(path)
    if path.is_dir():
        path = path / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _rollout_policy_on_scenarios(policy_gen_by_stage, policy_line_by_stage,
                                  scenarios, network, params):
    """
    Simulate a fixed stage-by-stage repair schedule on a list of scenarios.

    At stage t the schedule's 0/1 vector is applied to the components that are
    unavailable at t (r = policy[t] * (1 - a)); a stage missing from the dicts
    means no repair. Apart from this mask the schedule does not react to the
    realized failures, and no repair budget is checked. Dispatch uses
    StagewiseOPF with the default LP backend (solver_config.LP_SOLVER) and the
    state update is Pseudo_DFL_solver._next_availability, as in the rollouts
    of Pseudo_DFL_solver.py.

    Parameters
    ----------
    policy_gen_by_stage  : dict {0-indexed stage -> np.array(n_gen)}
    policy_line_by_stage : dict {0-indexed stage -> np.array(n_lines)}
    scenarios            : list of Scenario objects
    network              : NetworkData
    params               : ModelParameters

    Returns
    -------
    dict with keys, all averaged over the scenarios:
      avl_gen, avl_line, rep_gen, rep_line  shape (T, n_comp), availability and
                                            repair rate per stage and component
      stage_costs                           shape (T,), OPF + repair cost per stage
      mean_opf_cost, mean_repair_cost       per-scenario totals over the horizon
      mean_load_shed_mwh                    unserved load (MWh), summed over loads
                                            and stages
    """
    from Pseudo_DFL_stage_opf import StagewiseOPF
    from Pseudo_DFL_solver import _next_availability

    opf      = StagewiseOPF(network, params)
    T        = params.T
    n_gen    = network.n_generators
    n_lines  = network.n_lines
    n_sc     = len(scenarios)

    load_nominal   = np.array(network.load_nominal, dtype=float)

    avl_gen_sum    = np.zeros((T, n_gen))
    avl_line_sum   = np.zeros((T, n_lines))
    rep_gen_sum    = np.zeros((T, n_gen))
    rep_line_sum   = np.zeros((T, n_lines))
    stage_cost_sum = np.zeros(T)
    opf_cost_sum   = 0.0
    repair_cost_sum = 0.0
    load_shed_sum  = 0.0

    for sc in scenarios:
        # Stage 1: a = 1 - xi^1 (every component is up before the horizon)
        a_gen  = (1 - sc.gen_failures [0, :]).astype(float)
        a_line = (1 - sc.line_failures[0, :]).astype(float)
        r_gen_hist, r_line_hist = [], []

        for t in range(T):
            avl_gen_sum [t] += a_gen
            avl_line_sum[t] += a_line

            r_g = policy_gen_by_stage.get(t, np.zeros(n_gen))   * (1 - a_gen)
            r_l = policy_line_by_stage.get(t, np.zeros(n_lines)) * (1 - a_line)
            rep_gen_sum [t] += r_g
            rep_line_sum[t] += r_l
            r_gen_hist.append(r_g.copy())
            r_line_hist.append(r_l.copy())

            opf_cost, _, d_load, _ = opf.solve(a_gen, a_line)
            repair_cost = opf.compute_repair_cost(r_g, r_l)
            stage_cost_sum[t] += opf_cost + repair_cost
            opf_cost_sum    += opf_cost
            repair_cost_sum += repair_cost
            load_shed_sum   += float(np.sum(load_nominal - d_load))

            if t < T - 1:
                a_gen, a_line = _next_availability(
                    a_gen, a_line,
                    sc.gen_failures [t + 1, :].astype(float),
                    sc.line_failures[t + 1, :].astype(float),
                    r_gen_hist, r_line_hist, params.tau, t
                )

    return {
        'avl_gen':          avl_gen_sum    / n_sc,
        'avl_line':         avl_line_sum   / n_sc,
        'rep_gen':          rep_gen_sum    / n_sc,
        'rep_line':         rep_line_sum   / n_sc,
        'stage_costs':      stage_cost_sum / n_sc,
        'mean_opf_cost':    opf_cost_sum   / n_sc,
        'mean_repair_cost': repair_cost_sum / n_sc,
        'mean_load_shed_mwh': load_shed_sum / n_sc,
    }


def _avg_heatmap_data(rollout_obj, scenarios, n_eval=None):
    """Average availability, repair and stage-cost arrays over the first n_eval
    scenarios (all if n_eval is None), using rollout_obj.rollout_scenario."""
    n_sc = min(n_eval, len(scenarios)) if n_eval else len(scenarios)
    T       = rollout_obj.params.T
    n_gen   = rollout_obj.network.n_generators
    n_lines = rollout_obj.network.n_lines

    avl_gen_sum    = np.zeros((T, n_gen))
    avl_line_sum   = np.zeros((T, n_lines))
    rep_gen_sum    = np.zeros((T, n_gen))
    rep_line_sum   = np.zeros((T, n_lines))
    stage_cost_sum = np.zeros(T)

    for sc in scenarios[:n_sc]:
        res = rollout_obj.rollout_scenario(sc)
        avl_gen_sum    += res['availability_gen']
        avl_line_sum   += res['availability_line']
        rep_gen_sum    += res['repairs_gen']
        rep_line_sum   += res['repairs_line']
        stage_cost_sum += np.array(res['stage_costs'])

    return {
        'avl_gen':     avl_gen_sum    / n_sc,
        'avl_line':    avl_line_sum   / n_sc,
        'rep_gen':     rep_gen_sum    / n_sc,
        'rep_line':    rep_line_sum   / n_sc,
        'stage_costs': stage_cost_sum / n_sc,
    }


def _extract_policy_from_solution(solution):
    """
    Extract a (policy_gen, policy_line) pair from an SAA or SDDiP solution dict.

    solution['repairs'][kind][t][k], kind in {'gen', 'line'}, is the mean repair
    decision for component k at 1-indexed stage t: over the scenarios (SAA, ws
    mode), over the tree nodes of stage t weighted by path probability (SAA,
    tree mode), or over the simulated trajectories (SDDiP). Rounding gives a
    fixed schedule that repairs k at t when this mean exceeds 0.5. For ws-mode
    SAA this is the policy behind its out-of-sample costs. Tree-mode SAA
    (nearest child node) and SDDiP (the trained SDDP.jl policy) have closed-loop
    policies that react to the realized failures; the schedule only
    approximates them.

    Returns two dicts {0-indexed stage -> np.array}, suitable for
    _rollout_policy_on_scenarios(). An empty 'repairs' entry (e.g. an SDDiP
    run without simulation) gives empty dicts, i.e. no repairs.
    """
    repairs = solution.get('repairs', {})
    policy_gen, policy_line = {}, {}

    for t_key in repairs.get('gen', {}).keys():
        t_idx     = int(t_key) - 1   # 1-indexed → 0-indexed
        gen_keys  = sorted(repairs['gen'] [t_key].keys(), key=int)
        line_keys = sorted(repairs['line'][t_key].keys(), key=int)
        policy_gen [t_idx] = np.array(
            [round(float(repairs['gen'] [t_key][g])) for g in gen_keys],  dtype=float)
        policy_line[t_idx] = np.array(
            [round(float(repairs['line'][t_key][e])) for e in line_keys], dtype=float)

    return policy_gen, policy_line


# ---------------------------------------------------------------------------
# 1. Cost bars with SDDiP lower / upper bound annotation
# ---------------------------------------------------------------------------

def plot_cost_bars_with_bounds(results_dict, sddip_solution=None, save_path=None,
                               selection_labels=None):
    """
    Mean ± 1 std cost bars for every method in results_dict.

    Methods in METHOD_ORDER come first, then any other entry that is a dict
    with 'mean_cost'. If sddip_solution is supplied, a dashed line marks the
    SDDiP lower bound (LB) and a dotted line its upper bound (UB). LB is the
    SDDP.jl bound; UB is the cost of the last training forward pass (one
    sampled trajectory), a noisy estimate rather than a guaranteed bound.

    selection_labels : dict {method_key: str}
        Optional short label appended below each bar tick, e.g.
        {'Pseudo_DFL': 'n=8,000', 'SDDiP': '100 iter'}.
        Shown as a second line on the x-axis tick.
    """
    ordered = [k for k in METHOD_ORDER if k in results_dict]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict) and 'mean_cost' in results_dict[k]]
    labels  = ordered + extras

    means  = [results_dict[k]['mean_cost'] for k in labels]
    stds   = [results_dict[k].get('std_cost', 0.0) for k in labels]
    colors = [_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(labels)), 6))
    x    = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    for bar, mean in zip(bars, means):
        ax.annotate(f'${mean:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # SDDiP LB / UB lines
    if sddip_solution is not None:
        lb = sddip_solution.get('lower_bound')
        ub = sddip_solution.get('upper_bound')
        if lb is not None:
            ax.axhline(lb, color='#2ecc71', linestyle='--', linewidth=3.6,
                       label=f'SDDiP LB  ${lb:,.0f}')
        if ub is not None:
            ax.axhline(ub, color='#27ae60', linestyle=':', linewidth=3.6,
                       label=f'SDDiP UB  ${ub:,.0f}')
        if lb is not None or ub is not None:
            ax.legend(fontsize=10, loc='upper right')

    # Build tick labels: add selection sub-label when available
    sl = selection_labels or {}
    tick_labels = [
        f'{k}\n({sl[k]})' if sl.get(k) else k
        for k in labels
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel('Mean total cost ($)', fontsize=12)
    ax.set_title('Cost Comparison — All Methods  (mean ± 1 std, test set)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'cost_bars_with_bounds.png')


# ---------------------------------------------------------------------------
# 2. Cost distributions (violin + box) for all methods
# ---------------------------------------------------------------------------

def plot_cost_distributions_all(results_dict, save_path=None):
    """Violin + box plots of per-scenario total cost for every method with a
    'costs' list; a black diamond marks the mean."""
    ordered = [k for k in METHOD_ORDER if k in results_dict
               and isinstance(results_dict[k], dict) and 'costs' in results_dict[k]]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict) and 'costs' in results_dict[k]]
    labels  = ordered + extras
    data    = [np.array(results_dict[k]['costs']) for k in labels]
    colors  = [_color(k) for k in labels]

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(labels)), 6))
    positions = np.arange(1, len(labels) + 1)

    vp = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.4)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        vp[part].set_color('black')
        vp[part].set_linewidth(1.5)

    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

    for pos, d in zip(positions, data):
        ax.scatter(pos, np.mean(d), marker='D', color='black', s=50, zorder=5)
        ax.text(pos + 0.18, np.mean(d), f'${np.mean(d):,.0f}',
                va='center', fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Total scenario cost ($)', fontsize=12)
    ax.set_title('Cost Distribution — All Methods  (test set)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'cost_distributions_all.png')


# ---------------------------------------------------------------------------
# 3. Load-shedding distribution comparison
# ---------------------------------------------------------------------------

def plot_load_shedding_distributions(results_dict, save_path=None):
    """
    Violin + box of per-scenario load shedding (MWh) for methods that track it.
    Methods without 'load_shed_mwhs' are skipped silently.
    """
    ordered = [k for k in METHOD_ORDER if k in results_dict
               and isinstance(results_dict[k], dict) and 'load_shed_mwhs' in results_dict[k]]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict) and 'load_shed_mwhs' in results_dict[k]]
    labels  = ordered + extras

    if not labels:
        return   # nothing to plot

    data   = [np.array(results_dict[k]['load_shed_mwhs']) for k in labels]
    colors = [_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(labels)), 6))
    positions = np.arange(1, len(labels) + 1)

    vp = ax.violinplot(data, positions=positions, showmedians=True, showextrema=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.4)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        vp[part].set_color('black')
        vp[part].set_linewidth(1.5)

    bp = ax.boxplot(data, positions=positions, widths=0.25, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

    for pos, d in zip(positions, data):
        ax.scatter(pos, np.mean(d), marker='D', color='black', s=50, zorder=5)
        ax.text(pos + 0.18, np.mean(d), f'{np.mean(d):.1f} MWh',
                va='center', fontsize=9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Total load shed (MWh, summed over stages)', fontsize=12)
    ax.set_title('Load Shedding Distribution — All Methods  (test set)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'load_shedding_distributions.png')


# ---------------------------------------------------------------------------
# 4. OPF cost vs repair cost stacked bar
# ---------------------------------------------------------------------------

def plot_cost_breakdown_stacked(results_dict, save_path=None):
    """
    Stacked bar: mean OPF cost (bottom) + mean repair cost (top) per method.
    The OPF cost includes the load-shedding penalty. Only methods with
    'mean_opf_cost' are shown; they must also have 'mean_repair_cost'.
    """
    ordered = [k for k in METHOD_ORDER if k in results_dict
               and isinstance(results_dict[k], dict) and 'mean_opf_cost' in results_dict[k]]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict) and 'mean_opf_cost' in results_dict[k]]
    labels  = ordered + extras

    if not labels:
        return

    opf_vals    = [results_dict[k]['mean_opf_cost']    for k in labels]
    repair_vals = [results_dict[k]['mean_repair_cost'] for k in labels]
    colors      = [_color(k) for k in labels]

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(labels)), 6))
    x = np.arange(len(labels))

    bars_opf    = ax.bar(x, opf_vals,    color=colors, edgecolor='black',
                         linewidth=0.8, alpha=0.85, label='OPF / shedding cost')
    bars_repair = ax.bar(x, repair_vals, bottom=opf_vals, color=colors,
                         edgecolor='black', linewidth=0.8, alpha=0.5,
                         hatch='//', label='Repair cost')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Mean cost ($)', fontsize=12)
    ax.set_title('Cost Breakdown — OPF vs Repair  (mean, test set)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'cost_breakdown_stacked.png')


# ---------------------------------------------------------------------------
# 5. Repair heatmap comparison (side-by-side for all methods)
# ---------------------------------------------------------------------------

def plot_repair_heatmap_comparison(methods_data, network, params,
                                    n_scenarios=None, save_path=None):
    """
    Side-by-side 4-row heatmap comparing multiple methods.

    Layout: 4 rows × N_methods columns
      Row 0: Generator availability rate  (Greens)
      Row 1: Line availability rate       (Greens)
      Row 2: Generator repair rate        (Reds)
      Row 3: Line repair rate             (Blues)

    Parameters
    ----------
    methods_data : list of (name, data_dict) where data_dict has keys:
                   avl_gen (T, n_gen), avl_line (T, n_lines),
                   rep_gen (T, n_gen), rep_line (T, n_lines)
    n_scenarios  : int or None, displayed in the figure title
    """
    if not methods_data:
        return

    T       = params.T
    n_gen   = network.n_generators
    n_lines = network.n_lines
    N       = len(methods_data)
    stages  = list(range(1, T + 1))

    h_gen   = max(1.5, n_gen   * 0.8)
    h_lines = max(2.0, n_lines * 0.8)
    height_ratios = [h_gen, h_lines, h_gen, h_lines]
    total_h = sum(height_ratios) + 2.0

    fig, axes = plt.subplots(4, N,
                             figsize=(max(6 * N, 14), total_h),
                             gridspec_kw={'height_ratios': height_ratios})

    # Ensure axes is always 2-D (4, N)
    if N == 1:
        axes = axes[:, np.newaxis]

    row_titles = [
        'Generator Availability Rate',
        'Line Availability Rate',
        'Generator Repair Rate',
        'Line Repair Rate',
    ]
    row_cmaps  = ['Greens', 'Greens', 'Reds', 'Blues']
    row_keys   = ['avl_gen', 'avl_line', 'rep_gen', 'rep_line']
    row_ncomps = [n_gen, n_lines, n_gen, n_lines]
    row_labels = ['Gen', 'Line', 'Gen', 'Line']

    # The repair rows share one vmax across all methods (the largest repair
    # rate, at least 0.01), so the color scale is comparable across columns and
    # a method without repairs (e.g. NoRepair) shows as near-white.
    global_rep_gen_max  = max((float(d['rep_gen'].max())  for _, d in methods_data), default=0.0)
    global_rep_line_max = max((float(d['rep_line'].max()) for _, d in methods_data), default=0.0)
    row_vmaxes = [1.0, 1.0,
                  max(global_rep_gen_max,  0.01),
                  max(global_rep_line_max, 0.01)]

    for col, (name, data) in enumerate(methods_data):
        for row in range(4):
            ax   = axes[row, col]
            mat  = data[row_keys[row]]          # (T, n_comp)
            nc   = row_ncomps[row]
            vmax = row_vmaxes[row]

            im = ax.imshow(mat.T, aspect='auto', cmap=row_cmaps[row],
                           vmin=0, vmax=vmax, interpolation='nearest',
                           origin='lower', extent=[0, T, -0.5, nc - 0.5])

            ax.set_xticks(range(T))
            ax.set_xticklabels(stages, fontsize=7)
            ax.set_xlim(-0.5, T - 0.5)
            ax.set_yticks(range(nc))
            ax.set_yticklabels([f'{row_labels[row]}{k+1}' for k in range(nc)], fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

            if row == 0:
                ax.set_title(name, fontsize=13, fontweight='bold', pad=8)
            if col == 0:
                ax.set_ylabel(row_titles[row], fontsize=10)
            if row == 3:
                ax.set_xlabel('Stage', fontsize=10)

    n_sc_label = n_scenarios if n_scenarios is not None else '?'
    fig.suptitle(
        f'Repair & Availability Heatmaps — All Methods  '
        f'(averaged over {n_sc_label} test scenarios)',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    _save(fig, save_path, 'repair_heatmap_comparison.png')


# ---------------------------------------------------------------------------
# 6. SDDiP convergence (LB / UB over iterations)
# ---------------------------------------------------------------------------

def plot_sddip_convergence(sddip_solution, save_path=None):
    """Plot the SDDiP lower bound and the upper-bound estimate (the cost of each
    iteration's forward pass) against the iteration number. The UB curve is
    drawn only when it has one value per LB value."""
    conv = sddip_solution.get('convergence', {})
    lbs  = conv.get('lower_bounds') or conv.get('lower_bound_history')
    ubs  = conv.get('upper_bounds') or conv.get('upper_bound_history')

    if not lbs:
        return   # no convergence history stored

    iters = np.arange(1, len(lbs) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, lbs, color='#2ecc71', linewidth=2, label='Lower bound (LB)')
    if ubs and len(ubs) == len(lbs):
        ax.plot(iters, ubs, color='#e74c3c', linewidth=2, linestyle='--',
                label='Upper bound (UB)')
        ax.fill_between(iters, lbs, ubs, alpha=0.15, color='grey',
                        label='Optimality gap')

    ax.set_xlabel('SDDP iteration', fontsize=12)
    ax.set_ylabel('Bound ($)', fontsize=12)
    ax.set_title('SDDiP Convergence: Lower & Upper Bounds',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'sddip_convergence.png')


# ---------------------------------------------------------------------------
# 7. Repair rate by stage (line plot)
# ---------------------------------------------------------------------------

def plot_repair_rate_by_stage(methods_data, network, params,
                               n_scenarios=None, save_path=None):
    """
    For each method, plot the conditional repair rate at each stage.

    Conditional repair rate at stage t =
        (sum_k rep_gen[t, k] + sum_k rep_line[t, k]) /
        max(sum_k (1 - avl_gen[t, k]) + sum_k (1 - avl_line[t, k]), ε)

    with ε = 1e-6. The inputs are scenario averages, so this is the expected
    number of repairs ordered at t divided by the expected number of
    unavailable components at t (a ratio of means): the fraction of failed
    components that the policy repairs.

    Parameters
    ----------
    methods_data : list of (name, data_dict), same format as in
                   plot_repair_heatmap_comparison
    """
    if not methods_data:
        return

    T      = params.T
    stages = list(range(1, T + 1))
    eps    = 1e-6

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, data in methods_data:
        total_rep    = data['rep_gen'].sum(axis=1)  + data['rep_line'].sum(axis=1)
        total_failed = ((1 - data['avl_gen']).sum(axis=1) +
                        (1 - data['avl_line']).sum(axis=1))
        rate = total_rep / np.maximum(total_failed, eps)
        ax.plot(stages, rate, marker='o', linewidth=2, markersize=5,
                color=_color(name), label=name)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(stages)
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Fraction of failed components repaired', fontsize=12)
    n_sc_label = n_scenarios if n_scenarios is not None else '?'
    ax.set_title(
        f'Conditional Repair Rate by Stage  '
        f'(averaged over {n_sc_label} test scenarios)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(0.0, color='grey', linestyle=':', linewidth=1, alpha=0.6)
    plt.tight_layout()
    _save(fig, save_path, 'repair_rate_by_stage.png')


# ---------------------------------------------------------------------------
# 8. Stage-by-stage cost profile (line plot)
# ---------------------------------------------------------------------------

def plot_stage_cost_profile(methods_data, params, n_scenarios=None, save_path=None):
    """
    Mean total cost (OPF + repair) per stage for each method.

    Shows at which stages the methods differ, e.g. whether early repair
    spending is followed by lower dispatch cost than in the baselines.

    Parameters
    ----------
    methods_data : list of (name, data_dict); entries without 'stage_costs'
                   are skipped
    """
    if not methods_data:
        return

    T      = params.T
    stages = list(range(1, T + 1))

    # Filter to methods that have stage_costs
    valid = [(n, d) for n, d in methods_data if 'stage_costs' in d]
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for name, data in valid:
        sc = data['stage_costs']
        ax.plot(stages, sc, marker='s', linewidth=2, markersize=5,
                color=_color(name), label=name)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_xticks(stages)
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Mean stage cost ($)', fontsize=12)
    n_sc_label = n_scenarios if n_scenarios is not None else '?'
    ax.set_title(
        f'Stage-by-Stage Cost Profile  '
        f'(averaged over {n_sc_label} test scenarios)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'stage_cost_profile.png')


# ---------------------------------------------------------------------------
# 9. Cost share: 100 % stacked bars (generation / shedding / repair)
# ---------------------------------------------------------------------------

def plot_cost_share_pct(results_dict, save_path=None):
    """
    100% stacked horizontal bar chart: generation dispatch, load shedding and
    repair as percentages of the mean total cost, for every method that has
    the full breakdown.

    Requires results_dict entries to have:
      mean_opf_cost, mean_repair_cost, mean_load_shed_mwh

    The load-shedding part is mean_load_shed_mwh * config.LOAD_SHEDDING_COST
    ($/MWh); generation is the rest of mean_opf_cost, clipped at 0. The split
    assumes the runs used the config value of the shedding penalty
    (Pseudo_DFL_main.py --load-shedding-cost can change it).
    """
    from config import LOAD_SHEDDING_COST

    ordered = [k for k in METHOD_ORDER if k in results_dict
               and isinstance(results_dict[k], dict)
               and 'mean_opf_cost' in results_dict[k]
               and 'mean_load_shed_mwh' in results_dict[k]]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict)
               and 'mean_opf_cost' in results_dict[k]
               and 'mean_load_shed_mwh' in results_dict[k]]
    labels  = ordered + extras

    if not labels:
        return

    shed_costs  = []
    gen_costs   = []
    repair_costs = []

    for k in labels:
        r           = results_dict[k]
        shed_dollar = r['mean_load_shed_mwh'] * LOAD_SHEDDING_COST
        gen_dollar  = max(r['mean_opf_cost'] - shed_dollar, 0.0)
        rep_dollar  = r['mean_repair_cost']
        total       = shed_dollar + gen_dollar + rep_dollar
        if total <= 0:
            gen_costs.append(0.0); shed_costs.append(0.0); repair_costs.append(0.0)
        else:
            gen_costs.append(100.0 * gen_dollar  / total)
            shed_costs.append(100.0 * shed_dollar / total)
            repair_costs.append(100.0 * rep_dollar  / total)

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * len(labels) + 1.5)))

    gen_pct    = np.array(gen_costs)
    shed_pct   = np.array(shed_costs)
    repair_pct = np.array(repair_costs)

    b1 = ax.barh(y_pos, gen_pct,   color='#3498db', edgecolor='black',
                 linewidth=0.7, alpha=0.85, label='Generation dispatch')
    b2 = ax.barh(y_pos, shed_pct,  left=gen_pct, color='#e74c3c',
                 edgecolor='black', linewidth=0.7, alpha=0.85, label='Load shedding')
    b3 = ax.barh(y_pos, repair_pct, left=gen_pct + shed_pct,
                 color='#f39c12', edgecolor='black',
                 linewidth=0.7, alpha=0.85, label='Repair')

    # Annotate each segment if wide enough
    for i, (g, s, r) in enumerate(zip(gen_pct, shed_pct, repair_pct)):
        if g > 5:
            ax.text(g / 2, i, f'{g:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if s > 5:
            ax.text(g + s / 2, i, f'{s:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if r > 5:
            ax.text(g + s + r / 2, i, f'{r:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of mean total cost (%)', fontsize=12)
    ax.set_title('Cost Composition — Generation / Load Shedding / Repair  (% of total, test set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'cost_share_pct.png')


def plot_cost_share_pct_normalized(results_dict, save_path=None):
    """
    Same layout as plot_cost_share_pct but with per-component normalization.

    Each cost component (generation dispatch, load shedding, repair) is divided
    by its maximum over the plotted methods before the shares are computed, so
    the method with the largest value of a component gets 1 for it. This keeps
    a component with a small raw share (for example repair at a low repair-cost
    scale s) visible and comparable across methods.

    Each bar still sums to 100 %: a segment is the method's normalized value
    of that component divided by the sum of its three normalized values.
    """
    from config import LOAD_SHEDDING_COST

    ordered = [k for k in METHOD_ORDER if k in results_dict
               and isinstance(results_dict[k], dict)
               and 'mean_opf_cost' in results_dict[k]
               and 'mean_load_shed_mwh' in results_dict[k]]
    extras  = [k for k in results_dict if k not in ordered
               and isinstance(results_dict[k], dict)
               and 'mean_opf_cost' in results_dict[k]
               and 'mean_load_shed_mwh' in results_dict[k]]
    labels  = ordered + extras

    if not labels:
        return

    # Raw dollar amounts per method
    gen_raw    = []
    shed_raw   = []
    repair_raw = []
    for k in labels:
        r           = results_dict[k]
        shed_dollar = r['mean_load_shed_mwh'] * LOAD_SHEDDING_COST
        gen_dollar  = max(r['mean_opf_cost'] - shed_dollar, 0.0)
        rep_dollar  = r['mean_repair_cost']
        gen_raw.append(gen_dollar)
        shed_raw.append(shed_dollar)
        repair_raw.append(rep_dollar)

    # Normalize each component by its maximum over the methods. A maximum of 0
    # (the component is zero for every method) is replaced by 1 to avoid
    # dividing by zero.
    gen_max    = max(gen_raw)    or 1.0
    shed_max   = max(shed_raw)   or 1.0
    repair_max = max(repair_raw) or 1.0

    gen_norm    = [v / gen_max    for v in gen_raw]
    shed_norm   = [v / shed_max   for v in shed_raw]
    repair_norm = [v / repair_max for v in repair_raw]

    # Shares of the normalized total, in percent
    gen_pct    = np.empty(len(labels))
    shed_pct   = np.empty(len(labels))
    repair_pct = np.empty(len(labels))
    for i in range(len(labels)):
        total = gen_norm[i] + shed_norm[i] + repair_norm[i]
        if total <= 0:
            gen_pct[i] = shed_pct[i] = repair_pct[i] = 0.0
        else:
            gen_pct[i]    = 100.0 * gen_norm[i]    / total
            shed_pct[i]   = 100.0 * shed_norm[i]   / total
            repair_pct[i] = 100.0 * repair_norm[i] / total

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * len(labels) + 1.5)))

    ax.barh(y_pos, gen_pct,   color='#3498db', edgecolor='black',
            linewidth=0.7, alpha=0.85, label='Generation dispatch')
    ax.barh(y_pos, shed_pct,  left=gen_pct, color='#e74c3c',
            edgecolor='black', linewidth=0.7, alpha=0.85, label='Load shedding')
    ax.barh(y_pos, repair_pct, left=gen_pct + shed_pct,
            color='#f39c12', edgecolor='black',
            linewidth=0.7, alpha=0.85, label='Repair')

    # Annotate each segment if wide enough
    for i, (g, s, r) in enumerate(zip(gen_pct, shed_pct, repair_pct)):
        if g > 5:
            ax.text(g / 2, i, f'{g:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if s > 5:
            ax.text(g + s / 2, i, f'{s:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
        if r > 5:
            ax.text(g + s + r / 2, i, f'{r:.1f}%', ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Normalised share (%)', fontsize=12)
    ax.set_title(
        'Cost Composition — Normalised by per-component maximum across methods\n'
        '(each component scaled so its worst-case method = 100 % of that component)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'cost_share_pct_normalized.png')


# ---------------------------------------------------------------------------
# 10. Computation time comparison
# ---------------------------------------------------------------------------

def plot_computation_time(timing_dict, save_path=None):
    """
    Horizontal log-scale bar chart comparing computation times across methods.

    Parameters
    ----------
    timing_dict : dict of the form
        {
          'DFL training (s)':          float,
          'DFL inference / scenario (s)': float,
          'SAA solve (s)':             float,
          'SDDiP training (s)':        float,
          ...
        }
    Keys are free-form labels; values are seconds. Entries that are None, NaN
    or not positive are skipped.
    """
    if not timing_dict:
        return

    labels = [k for k, v in timing_dict.items()
              if v is not None and not (isinstance(v, float) and np.isnan(v)) and v > 0]
    values = [timing_dict[k] for k in labels]

    if not labels:
        return

    # Color each bar by the method named in its label ('dfl', 'saa' or 'sddip',
    # case-insensitive); other labels get _DEFAULT_COLOR
    bar_colors = []
    for lbl in labels:
        lbl_lower = lbl.lower()
        if 'dfl' in lbl_lower:
            bar_colors.append(COLORS['Pseudo_DFL'])
        elif 'saa' in lbl_lower:
            bar_colors.append(COLORS['SAA'])
        elif 'sddip' in lbl_lower:
            bar_colors.append(COLORS['SDDiP'])
        else:
            bar_colors.append(_DEFAULT_COLOR)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.6 * len(labels) + 1.5)))
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=bar_colors, edgecolor='black',
                   linewidth=0.7, alpha=0.85)

    # Annotate each bar with a human-readable time
    for bar, val in zip(bars, values):
        if val >= 3600:
            label = f'{val/3600:.1f} h'
        elif val >= 60:
            label = f'{val/60:.1f} min'
        elif val >= 1:
            label = f'{val:.2f} s'
        else:
            label = f'{val*1000:.1f} ms'
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                label, va='center', fontsize=10)

    ax.set_xscale('log')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Computation time (seconds, log scale)', fontsize=11)
    ax.set_title('Computation Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path, 'computation_time.png')


# ---------------------------------------------------------------------------
# 11. Single-method repair heatmap (saved to the method's own folder)
# ---------------------------------------------------------------------------

def plot_single_repair_heatmap(name, heatmap_data, network, params,
                                n_scenarios=None, save_path=None):
    """
    4-row averaged availability/repair heatmap for a single method.

    Layout (same as the columns in plot_repair_heatmap_comparison):
      Row 0: Generator availability rate  (Greens)
      Row 1: Line availability rate       (Greens)
      Row 2: Generator repair rate        (Reds)
      Row 3: Line repair rate             (Blues)
    Unlike the comparison figure, the repair rows are scaled to this method's
    own maximum repair rate.

    Parameters
    ----------
    name          : str   method label (used in the title and default file name)
    heatmap_data  : dict  {avl_gen, avl_line, rep_gen, rep_line}, all (T, n_comp)
    n_scenarios   : int or None, shown in the title
    save_path     : file path, or a directory (then repair_heatmap_<name>.png
                    with the name lowercased and spaces replaced by '_')
    """
    T       = params.T
    n_gen   = network.n_generators
    n_lines = network.n_lines
    stages  = list(range(1, T + 1))

    h_gen   = max(1.5, n_gen   * 0.8)
    h_lines = max(2.0, n_lines * 0.8)
    height_ratios = [h_gen, h_lines, h_gen, h_lines]
    total_h = sum(height_ratios) + 2.0

    fig, axes = plt.subplots(4, 1, figsize=(14, total_h),
                             gridspec_kw={'height_ratios': height_ratios})

    row_titles = ['Generator Availability Rate', 'Line Availability Rate',
                  'Generator Repair Rate',       'Line Repair Rate']
    row_cmaps  = ['Greens', 'Greens', 'Reds', 'Blues']
    row_keys   = ['avl_gen', 'avl_line', 'rep_gen', 'rep_line']
    row_ncomps = [n_gen, n_lines, n_gen, n_lines]
    row_labels = ['Gen', 'Line', 'Gen', 'Line']
    # Repair rows: color scale up to this method's own maximum (at least 0.01);
    # a method without repairs (NoRepair) shows as near-white.
    rep_gen_vmax  = max(float(heatmap_data['rep_gen'].max()),  0.01)
    rep_line_vmax = max(float(heatmap_data['rep_line'].max()), 0.01)
    row_vmaxes = [1.0, 1.0, rep_gen_vmax, rep_line_vmax]

    for row in range(4):
        ax  = axes[row]
        mat = heatmap_data[row_keys[row]]     # (T, n_comp)
        nc  = row_ncomps[row]

        im = ax.imshow(mat.T, aspect='auto', cmap=row_cmaps[row],
                       vmin=0, vmax=row_vmaxes[row], interpolation='nearest',
                       origin='lower', extent=[0, T, -0.5, nc - 0.5])

        ax.set_xticks(range(T))
        ax.set_xticklabels(stages, fontsize=8)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_yticks(range(nc))
        ax.set_yticklabels([f'{row_labels[row]}{k+1}' for k in range(nc)], fontsize=9)
        ax.set_ylabel(row_titles[row], fontsize=11)
        ax.set_title(row_titles[row], fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        if row == 3:
            ax.set_xlabel('Stage', fontsize=11)

    n_sc_label = n_scenarios if n_scenarios is not None else '?'
    fig.suptitle(
        f'{name}: Avg Availability & Repair Heatmap  '
        f'(averaged over {n_sc_label} test scenarios)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    default_name = f'repair_heatmap_{name.lower().replace(" ", "_")}.png'
    _save(fig, save_path, default_name)


# ---------------------------------------------------------------------------
# Master: create comparison folder
# ---------------------------------------------------------------------------

def create_comparison_folder(results_dict, rollouts_dict, scenarios,
                              network, params, out_dir,
                              sddip_solution=None, saa_solution=None,
                              method_save_dirs=None, timing_dict=None,
                              selection_labels=None):
    """
    Write the cross-method comparison plots and a summary JSON to
    out_dir/comparison/:

      cost_bars_with_bounds.png      mean cost ± std per method, SDDiP LB/UB lines
      computation_time.png           log-scale time comparison (only with timing_dict)
      cost_share_pct.png             generation / shedding / repair shares of mean cost
      cost_share_pct_normalized.png  the same shares after per-component normalization
      repair_heatmap_comparison.png  4-row heatmap, one column per method
      repair_rate_by_stage.png       conditional repair rate per stage (line)
      stage_cost_profile.png         mean cost per stage (line)
      comparison_summary.json        cost statistics per method and the SDDiP bounds

    The heatmap, repair-rate and stage-cost plots use the first
    min(50, len(scenarios)) scenarios. Methods in rollouts_dict are simulated
    with their rollout_scenario. SAA and SDDiP, when their solution has a
    'repairs' entry, are simulated with the rounded schedule of
    _extract_policy_from_solution, and that simulation also supplies their bars
    in the two cost-share plots. A simulated method with an entry in
    method_save_dirs also gets its single-method heatmap in that directory.

    Parameters
    ----------
    results_dict     : {label: eval_result_dict}  (_cost_stats format of
                       Pseudo_DFL_solver.py)
    rollouts_dict    : {label: rollout_object}; None values and objects
                       without rollout_scenario are skipped
    scenarios        : list of Scenario  (test set)
    network          : NetworkData
    params           : ModelParameters
    out_dir          : str or Path  (parent experiment directory)
    sddip_solution   : dict from sddip_solution.json, or None
    saa_solution     : dict from solution.json (SAA), or None
    method_save_dirs : dict {label: Path}, individual heatmap destinations
    timing_dict      : dict {label: seconds}, for computation_time.png
    selection_labels : dict {label: str}, second tick-label line in
                       cost_bars_with_bounds.png
    """
    cmp_dir = Path(out_dir) / 'comparison'
    cmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Creating comparison folder → {cmp_dir}/")

    # ── 1. Cost bars (all methods + SDDiP LB/UB) ─────────────────────────────
    plot_cost_bars_with_bounds(results_dict, sddip_solution,
                               save_path=cmp_dir / 'cost_bars_with_bounds.png',
                               selection_labels=selection_labels)

    # results_dict_aug: per-entry copy of results_dict for the two cost-share
    # plots. The SAA and SDDiP breakdowns are filled in from the schedule
    # simulation below.
    results_dict_aug = {k: dict(v) if isinstance(v, dict) else v
                        for k, v in results_dict.items()}

    # ── 2. Computation time ───────────────────────────────────────────────────
    if timing_dict:
        plot_computation_time(timing_dict,
                              save_path=cmp_dir / 'computation_time.png')

    # ── 3–5. Rollout-derived plots (heatmap, repair rate, stage cost) ─────────
    n_sc_heatmap = min(50, len(scenarios))
    methods_data = []
    method_save_dirs = method_save_dirs or {}

    def _save_individual(label, hm):
        mdir = method_save_dirs.get(label)
        if mdir is not None:
            plot_single_repair_heatmap(
                label, hm, network, params,
                n_scenarios=n_sc_heatmap,
                save_path=Path(mdir) / f'repair_heatmap_{label.lower().replace(" ", "_")}.png'
            )

    # Methods with a rollout object (e.g. Pseudo_DFL, AlwaysRepair, NoRepair,
    # NearSightedMPC)
    for label, rollout_obj in rollouts_dict.items():
        if rollout_obj is None or not hasattr(rollout_obj, 'rollout_scenario'):
            continue
        hm = _avg_heatmap_data(rollout_obj, scenarios, n_eval=n_sc_heatmap)
        methods_data.append((label, hm))
        _save_individual(label, hm)

    # SAA: rounded mean-repair schedule, simulated on the same test scenarios
    if saa_solution is not None and 'repairs' in saa_solution:
        try:
            pg, pl = _extract_policy_from_solution(saa_solution)
            hm = _rollout_policy_on_scenarios(pg, pl, scenarios[:n_sc_heatmap],
                                              network, params)
            methods_data.append(('SAA', hm))
            _save_individual('SAA', hm)
            # Inject cost breakdown so SAA appears in both cost-share plots
            saa_aug = results_dict_aug.get('SAA', {})
            saa_aug.update({k: hm[k] for k in
                            ('mean_opf_cost', 'mean_repair_cost', 'mean_load_shed_mwh')})
            results_dict_aug['SAA'] = saa_aug
        except Exception as exc:
            print(f"  Warning: could not simulate SAA heatmap ({exc})")

    # SDDiP: rounded mean-repair schedule, simulated on the same test scenarios
    if sddip_solution is not None and 'repairs' in sddip_solution:
        try:
            pg, pl = _extract_policy_from_solution(sddip_solution)
            hm = _rollout_policy_on_scenarios(pg, pl, scenarios[:n_sc_heatmap],
                                              network, params)
            methods_data.append(('SDDiP', hm))
            _save_individual('SDDiP', hm)
            # Inject cost breakdown so SDDiP appears in both cost-share plots
            sddip_aug = results_dict_aug.get('SDDiP', {})
            sddip_aug.update({k: hm[k] for k in
                              ('mean_opf_cost', 'mean_repair_cost', 'mean_load_shed_mwh')})
            results_dict_aug['SDDiP'] = sddip_aug
        except Exception as exc:
            print(f"  Warning: could not simulate SDDiP heatmap ({exc})")

    # ── 1b. Cost shares, raw and normalized (need the SAA/SDDiP breakdowns) ───
    plot_cost_share_pct(results_dict_aug,
                        save_path=cmp_dir / 'cost_share_pct.png')
    plot_cost_share_pct_normalized(results_dict_aug,
                                   save_path=cmp_dir / 'cost_share_pct_normalized.png')

    if methods_data:
        # Sort columns by METHOD_ORDER; other labels go last
        order = {n: i for i, n in enumerate(METHOD_ORDER)}
        methods_data.sort(key=lambda x: order.get(x[0], 99))

        plot_repair_heatmap_comparison(
            methods_data, network, params,
            n_scenarios=n_sc_heatmap,
            save_path=cmp_dir / 'repair_heatmap_comparison.png'
        )
        plot_repair_rate_by_stage(
            methods_data, network, params,
            n_scenarios=n_sc_heatmap,
            save_path=cmp_dir / 'repair_rate_by_stage.png'
        )
        plot_stage_cost_profile(
            methods_data, params,
            n_scenarios=n_sc_heatmap,
            save_path=cmp_dir / 'stage_cost_profile.png'
        )

    # ── 6. Summary JSON ───────────────────────────────────────────────────────
    summary = {}
    for label, res in results_dict.items():
        if not isinstance(res, dict) or 'mean_cost' not in res:
            continue
        entry = {
            'mean_cost':  res['mean_cost'],
            'std_cost':   res.get('std_cost', 0.0),
            'median':     res.get('median'),
            'min_cost':   res.get('min_cost'),
            'max_cost':   res.get('max_cost'),
        }
        if 'mean_opf_cost' in res:
            entry['mean_opf_cost']    = res['mean_opf_cost']
            entry['mean_repair_cost'] = res['mean_repair_cost']
        if 'mean_load_shed_mwh' in res:
            entry['mean_load_shed_mwh'] = res['mean_load_shed_mwh']
        if 'eval_time_s' in res:
            entry['eval_time_s'] = res['eval_time_s']
        summary[label] = entry

    if sddip_solution is not None:
        summary['_sddip_bounds'] = {
            'lower_bound': sddip_solution.get('lower_bound'),
            'upper_bound': sddip_solution.get('upper_bound'),
            'gap_pct':     sddip_solution.get('gap', 0) * 100,
            'iterations':  sddip_solution.get('iterations'),
        }

    with open(cmp_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {cmp_dir}/comparison_summary.json")
    print(f"  Comparison folder complete: {cmp_dir}/")
