"""
Plots for a DFL run (Pseudo_DFL_main.py): training history, cost comparison
across the methods in the evaluation results, and availability and repair
heatmaps of the DFL policy on the test scenarios.

Figures generated
-----------------
1. training_curves.png          Fenchel-Young (FY) loss on the training and validation
                                splits and the validation repair ratio, per epoch
2. cost_distribution.png        Box and violin plots of the per-scenario cost of each method
3. cost_comparison_bar.png      Mean cost ± 1 std dev of each method
4. repair_heatmap.png           One scenario: availability (Gantt) and repair decisions
5. availability_trajectory.png  One scenario: availability under DFL and under NoRepair
6. repair_decisions.png         Repair rate per stage (Reds/Blues heatmaps), averaged over
                                up to 50 scenarios
7. repair_heatmap_avg_50.png    4-panel layout of repair_heatmap.png, averaged over up to
                                50 scenarios

Style conventions
-----------------
Figures are saved with dpi=300 and bbox_inches='tight'. Axis labels use fontsize 12,
subplot titles fontsize 14 bold, suptitles fontsize 16 bold, legends fontsize 10 and
grids alpha 0.3. The availability panels of repair_heatmap.png use the Gantt bars of
visualization.plot_availability_gantt (green = available, red = failed), and repair
heatmaps use Reds for generators and Blues for lines, as in
visualization.plot_repair_decisions.
"""

import numpy as np
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Method keys match the labels of results_dict (other labels are drawn in gray);
# 'train' and 'val' color the loss curves. NearSightedMPC has the same color as in
# compare_visualization.py; the other method colors reuse colors of
# visualization.py and SDDiP_visualization.py:
#   Pseudo_DFL    steelblue   generator 1 in plot_generation_per_generator
#   SDDiP         #2ecc71     lower-bound bar in plot_sddip_cost_breakdown
#   AlwaysRepair  darkorange  generator 2 in plot_generation_per_generator
#   NoRepair      #e74c3c     upper-bound bar in plot_sddip_cost_breakdown
COLORS = {
    'Pseudo_DFL':          'steelblue',
    'SDDiP':        '#2ecc71',
    'AlwaysRepair': 'darkorange',
    'NoRepair':     '#e74c3c',
    'NearSightedMPC': '#16a085',
    'train':        '#1565C0',   # deep blue  (steelblue family)
    'val':          '#e67e22',   # orange     (solve-time bar in plot_algorithm_statistics)
}

# Per-component colors; the first four are the plot_generation_per_generator
# palette of visualization.py.
_COMP_COLORS = ['steelblue', 'darkorange', 'seagreen', 'firebrick',
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(trainer, save_path=None):
    """Plot the FY train/val loss and the validation repair ratio over epochs.

    The repair ratio (stored in trainer.val_accs) is the number of failed components
    with predicted theta > 0 on the validation split divided by the number of repairs
    in its labels. The dashed line marks 1.0.
    """
    train_l = trainer.train_losses
    val_l   = trainer.val_losses
    val_a   = trainer.val_accs

    if not train_l:
        print("  No training history — skipping training curve plot.")
        return

    epochs = np.arange(1, len(train_l) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax = axes[0]
    ax.plot(epochs, train_l, color=COLORS['train'], label='Train FY loss',
            linewidth=2)
    ax.plot(epochs, val_l, color=COLORS['val'], label='Val FY loss',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Fenchel-Young loss', fontsize=12)
    ax.set_title('Pseudo_DFL Training: FY Loss  (training split)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Validation repair ratio
    ax2 = axes[1]
    if val_a:
        ax2.plot(epochs, val_a, color=COLORS['Pseudo_DFL'], linewidth=2)
        ax2.axhline(1.0, color='grey', linestyle='--', linewidth=1, alpha=0.6)
        ax2.set_ylim(0, None)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Repair ratio  (DFL / optimal)', fontsize=12)
    ax2.set_title('Pseudo_DFL Validation: Repair Ratio  (validation split)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Pseudo_DFL Training History  (training & validation splits)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path, 'training_curves.png')


# ---------------------------------------------------------------------------
# 2. Cost distribution (box + violin)
# ---------------------------------------------------------------------------

def plot_cost_distribution(results_dict, save_path=None):
    """
    Box + violin comparison of cost distributions.

    Parameters
    ----------
    results_dict : dict  {label: evaluation_result_dict}
                   e.g. {'Pseudo_DFL': dfl_res, 'SDDiP': sddip_res, ...}.
                   Entries without a 'costs' list, such as the 'timing' entry
                   of evaluation.json, are skipped.
    """
    labels = [k for k, v in results_dict.items()
              if isinstance(v, dict) and 'costs' in v]
    data   = [np.array(results_dict[k]['costs']) for k in labels]

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(labels)), 6))

    positions = np.arange(1, len(labels) + 1)
    colors    = [COLORS.get(k, '#9E9E9E') for k in labels]

    # Violin
    vp = ax.violinplot(data, positions=positions, showmedians=True,
                       showextrema=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.4)
    for part in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
        vp[part].set_color('black')
        vp[part].set_linewidth(1.5)

    # Box
    bp = ax.boxplot(data, positions=positions, widths=0.25,
                    patch_artist=True, notch=False,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Total scenario cost ($)', fontsize=12)
    ax.set_title('Cost Distribution by Method  (test set)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # Mean markers + annotations
    for pos, d in zip(positions, data):
        ax.scatter(pos, np.mean(d), marker='D', color='black', s=50, zorder=5)
        ax.text(pos + 0.17, np.mean(d), f'${np.mean(d):,.0f}',
                va='center', fontsize=9, color='black')

    plt.tight_layout()
    _save_or_show(fig, save_path, 'cost_distribution.png')


# ---------------------------------------------------------------------------
# 3. Mean ± std bar chart
# ---------------------------------------------------------------------------

def plot_cost_comparison_bar(results_dict, save_path=None):
    """Bar chart of mean cost ± 1 std dev per method, in the style of the SDDiP cost breakdown.

    Entries of results_dict without a 'mean_cost' key are skipped.
    """
    labels = [k for k, v in results_dict.items()
              if isinstance(v, dict) and 'mean_cost' in v]
    means  = [results_dict[k]['mean_cost'] for k in labels]
    stds   = [results_dict[k]['std_cost']  for k in labels]
    colors = [COLORS.get(k, '#9E9E9E') for k in labels]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(labels)), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Mean total cost ($)', fontsize=12)
    ax.set_title('Mean Cost Comparison ± 1 Std Dev  (test set)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.annotate(f'${mean:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    _save_or_show(fig, save_path, 'cost_comparison_bar.png')


# ---------------------------------------------------------------------------
# 4. Repair heatmap (one scenario)
# ---------------------------------------------------------------------------

def plot_repair_heatmap(rollout_result, network, params, save_path=None):
    """
    4-panel figure for one scenario's DFL rollout:
      1. Generator availability:     Gantt barh (green = available, red = failed)
      2. Line availability:          Gantt barh
      3. Generator repair decisions: Reds heatmap
      4. Line repair decisions:      Blues heatmap

    Same style as visualization.plot_availability_gantt and
    visualization.plot_repair_decisions.

    Parameters
    ----------
    rollout_result : dict from PseudoDFLRollout.rollout_scenario
    """
    n_gen   = network.n_generators
    n_lines = network.n_lines
    T       = params.T

    rep_gen  = rollout_result['repairs_gen']        # (T, n_gen)
    rep_line = rollout_result['repairs_line']        # (T, n_lines)
    avl_gen  = rollout_result['availability_gen']   # (T, n_gen)
    avl_line = rollout_result['availability_line']  # (T, n_lines)

    stages = list(range(T))

    # Height ratios: Gantt rows scale with component count
    h_gen   = max(1.5, n_gen   * 0.8)
    h_lines = max(2.5, n_lines * 0.8)
    height_ratios = [h_gen, h_lines, h_gen, h_lines]
    total_h = sum(height_ratios) + 1.5   # extra for suptitle

    fig, axes = plt.subplots(4, 1, figsize=(14, total_h),
                             gridspec_kw={'height_ratios': height_ratios})

    # Panel 1: Generator availability (Gantt)
    ax = axes[0]
    for g in range(n_gen):
        for t in stages:
            color = 'green' if avl_gen[t, g] == 1 else 'red'
            ax.barh(g, 1, left=t, height=0.8, color=color,
                    alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(n_gen))
    ax.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)], fontsize=10)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylabel('Generator', fontsize=12)
    ax.set_title('Generator Availability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Available')
    red_patch   = mpatches.Patch(color='red',   alpha=0.7, label='Failed')
    ax.legend(handles=[green_patch, red_patch], loc='upper right', fontsize=10)

    # Panel 2: Line availability (Gantt)
    ax = axes[1]
    for e in range(n_lines):
        for t in stages:
            color = 'green' if avl_line[t, e] == 1 else 'red'
            ax.barh(e, 1, left=t, height=0.8, color=color,
                    alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels([f'Line {e+1}' for e in range(n_lines)], fontsize=10)
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_ylabel('Line', fontsize=12)
    ax.set_title('Line Availability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 3: Generator repair decisions (Reds heatmap)
    ax = axes[2]
    gen_max = max(float(rep_gen.max()), 0.01)
    # extent=[0, T, ...]: cell t spans [t, t+1], same as barh bar t spanning [t, t+1]
    # origin='lower': row 0 (Gen 1) at the bottom, like barh y=0
    im1 = ax.imshow(rep_gen.T, aspect='auto', cmap='Reds',
                    vmin=0, vmax=gen_max, interpolation='nearest',
                    origin='lower', extent=[0, T, -0.5, n_gen - 0.5])
    ax.set_yticks(range(n_gen))
    ax.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)], fontsize=10)
    ax.set_xlim(-0.5, T - 0.5)   # identical to Gantt xlim
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Generator', fontsize=12)
    ax.set_title('Generator Repair Decisions', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax, fraction=0.03, pad=0.02, label='Repair decision')

    # Panel 4: Line repair decisions (Blues heatmap)
    ax = axes[3]
    line_max = max(float(rep_line.max()), 0.01)
    im2 = ax.imshow(rep_line.T, aspect='auto', cmap='Blues',
                    vmin=0, vmax=line_max, interpolation='nearest',
                    origin='lower', extent=[0, T, -0.5, n_lines - 0.5])
    ax.set_yticks(range(n_lines))
    ax.set_yticklabels([f'Line {e+1}' for e in range(n_lines)], fontsize=10)
    ax.set_xlim(-0.5, T - 0.5)   # identical to Gantt xlim
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Line', fontsize=12)
    ax.set_title('Line Repair Decisions', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax, fraction=0.03, pad=0.02, label='Repair decision')

    fig.suptitle('Pseudo_DFL Policy: Availability & Repair Decisions  (test set - 1 scenario)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path, 'repair_heatmap.png')


# ---------------------------------------------------------------------------
# 5. Availability trajectory comparison
# ---------------------------------------------------------------------------

def plot_availability_trajectory(dfl_result, nr_result, network, params,
                                  save_path=None):
    """Step plot of each component's availability under DFL and under NoRepair for one scenario."""
    n_gen   = network.n_generators
    n_lines = network.n_lines
    T       = params.T
    stages  = np.arange(1, T + 1)

    n_comp = n_gen + n_lines
    fig, axes = plt.subplots(n_comp, 1,
                              figsize=(14, 2.5 * n_comp), sharex=True)
    if n_comp == 1:
        axes = [axes]

    comp_labels = (
        [f'Gen {g+1}' for g in range(n_gen)] +
        [f'Line {e+1}' for e in range(n_lines)]
    )

    for idx, ax in enumerate(axes):
        if idx < n_gen:
            dfl_av = dfl_result['availability_gen'][:, idx]
            nr_av  = nr_result ['availability_gen'][:, idx]
        else:
            e = idx - n_gen
            dfl_av = dfl_result['availability_line'][:, e]
            nr_av  = nr_result ['availability_line'][:, e]

        ax.step(stages, dfl_av, where='mid', color=COLORS['Pseudo_DFL'],
                label='Pseudo_DFL', linewidth=2)
        ax.step(stages, nr_av, where='mid', color=COLORS['NoRepair'],
                label='No Repair', linewidth=2, linestyle='--', alpha=0.8)
        ax.set_ylim(-0.15, 1.3)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Down', 'Up'], fontsize=10)
        ax.set_ylabel(comp_labels[idx], fontsize=12, rotation=0,
                      labelpad=60, va='center')
        ax.grid(True, axis='x', alpha=0.3)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    axes[-1].set_xlabel('Stage', fontsize=12)
    fig.suptitle('Availability: Pseudo_DFL vs No-Repair  (test set - 1 scenario)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path, 'availability_trajectory.png')


# ---------------------------------------------------------------------------
# 6. Expected repair rate heatmap (same layout as the SAA repair_decisions.png)
# ---------------------------------------------------------------------------

def plot_repair_decisions(scenarios, dfl_rollout, network, params,
                          save_path=None, n_eval=50):
    """
    Expected repair rate per component and stage, averaged over the first
    min(n_eval, len(scenarios)) scenarios. Two heatmaps stacked vertically,
    Reds for generators and Blues for lines, in the style of
    visualization.plot_repair_decisions.
    """
    n_gen   = network.n_generators
    n_lines = network.n_lines
    T       = params.T
    n_sc    = min(n_eval, len(scenarios))

    rep_gen_sum  = np.zeros((T, n_gen))
    rep_line_sum = np.zeros((T, n_lines))

    for sc in scenarios[:n_sc]:
        res = dfl_rollout.rollout_scenario(sc)
        rep_gen_sum  += res['repairs_gen']
        rep_line_sum += res['repairs_line']

    rep_gen_mean  = rep_gen_sum  / n_sc   # (T, n_gen)
    rep_line_mean = rep_line_sum / n_sc   # (T, n_lines)

    stages = list(range(1, T + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Generator repair rate (Reds)
    im1 = ax1.imshow(rep_gen_mean.T, aspect='auto', cmap='Reds',
                     vmin=0, vmax=max(rep_gen_mean.max(), 0.01),
                     interpolation='nearest')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Generator', fontsize=12)
    ax1.set_title('Generator Repair Decisions  (test set - expected rate)',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(T))
    ax1.set_xticklabels(stages)
    ax1.set_yticks(range(n_gen))
    ax1.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)])
    plt.colorbar(im1, ax=ax1, label='Repair Rate')

    # Line repair rate (Blues)
    im2 = ax2.imshow(rep_line_mean.T, aspect='auto', cmap='Blues',
                     vmin=0, vmax=max(rep_line_mean.max(), 0.01),
                     interpolation='nearest')
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Line', fontsize=12)
    ax2.set_title('Line Repair Decisions  (test set - expected rate)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(range(T))
    ax2.set_xticklabels(stages)
    ax2.set_yticks(range(n_lines))
    ax2.set_yticklabels([f'Line {e+1}' for e in range(n_lines)])
    plt.colorbar(im2, ax=ax2, label='Repair Rate')

    fig.suptitle(f'Pseudo_DFL Expected Repair Rate  (test set - {n_sc} scenarios)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path, 'repair_decisions.png')


# ---------------------------------------------------------------------------
# 7. Averaged 4-panel heatmap (availability and repairs, up to 50 scenarios)
# ---------------------------------------------------------------------------

def plot_repair_heatmap_avg(scenarios, dfl_rollout, network, params,
                             save_path=None, n_eval=50):
    """
    Same 4-panel layout as plot_repair_heatmap, averaged over the first
    min(n_eval, len(scenarios)) scenarios.

    Averaged availability is fractional, so all four panels are heatmaps
    (no Gantt bars):
      1. Generator availability rate: Greens heatmap (0 = failed in every scenario,
                                      1 = available in every scenario)
      2. Line availability rate:      Greens heatmap
      3. Generator repair rate:       Reds heatmap
      4. Line repair rate:            Blues heatmap

    Parameters
    ----------
    scenarios   : list of Scenario
    dfl_rollout : PseudoDFLRollout
    n_eval      : int  maximum number of scenarios to average over
    """
    n_gen   = network.n_generators
    n_lines = network.n_lines
    T       = params.T
    n_sc    = min(n_eval, len(scenarios))

    avl_gen_sum  = np.zeros((T, n_gen))
    avl_line_sum = np.zeros((T, n_lines))
    rep_gen_sum  = np.zeros((T, n_gen))
    rep_line_sum = np.zeros((T, n_lines))

    for sc in scenarios[:n_sc]:
        res = dfl_rollout.rollout_scenario(sc)
        avl_gen_sum  += res['availability_gen']
        avl_line_sum += res['availability_line']
        rep_gen_sum  += res['repairs_gen']
        rep_line_sum += res['repairs_line']

    avl_gen_mean  = avl_gen_sum  / n_sc   # (T, n_gen),   values in [0, 1]
    avl_line_mean = avl_line_sum / n_sc   # (T, n_lines),  values in [0, 1]
    rep_gen_mean  = rep_gen_sum  / n_sc   # (T, n_gen),   values in [0, 1]
    rep_line_mean = rep_line_sum / n_sc   # (T, n_lines),  values in [0, 1]

    stages = list(range(1, T + 1))

    h_gen   = max(1.5, n_gen   * 0.8)
    h_lines = max(2.5, n_lines * 0.8)
    height_ratios = [h_gen, h_lines, h_gen, h_lines]
    total_h = sum(height_ratios) + 1.5

    fig, axes = plt.subplots(4, 1, figsize=(14, total_h),
                             gridspec_kw={'height_ratios': height_ratios})

    def _heatmap(ax, data, cmap, vmax, ylabel_base, n_comp, title, cbar_label):
        """Draw a single heatmap panel with shared tick style."""
        im = ax.imshow(data.T, aspect='auto', cmap=cmap,
                       vmin=0, vmax=max(vmax, 0.01), interpolation='nearest',
                       origin='lower', extent=[0, T, -0.5, n_comp - 0.5])
        ax.set_yticks(range(n_comp))
        ax.set_yticklabels([f'{ylabel_base} {k+1}' for k in range(n_comp)],
                           fontsize=10)
        ax.set_xticks(range(T))
        ax.set_xticklabels(stages, fontsize=8)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylabel(ylabel_base, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)

    _heatmap(axes[0], avl_gen_mean,  'Greens', 1.0,
             'Gen',  n_gen,   'Generator Availability Rate',    'Avg availability')
    _heatmap(axes[1], avl_line_mean, 'Greens', 1.0,
             'Line', n_lines, 'Line Availability Rate',         'Avg availability')
    _heatmap(axes[2], rep_gen_mean,  'Reds',   rep_gen_mean.max(),
             'Gen',  n_gen,   'Generator Repair Rate',          'Avg repair rate')
    _heatmap(axes[3], rep_line_mean, 'Blues',  rep_line_mean.max(),
             'Line', n_lines, 'Line Repair Rate',               'Avg repair rate')

    fig.suptitle(
        f'Pseudo_DFL Policy: Avg Availability & Repair Decisions  (test set - {n_sc} scenarios)',
        fontsize=16, fontweight='bold')
    plt.tight_layout()
    _save_or_show(fig, save_path, 'repair_heatmap_avg_50.png')


# ---------------------------------------------------------------------------
# Master report
# ---------------------------------------------------------------------------

def create_pseudo_dfl_report(trainer, results_dict, dfl_rollout, one_scenario,
                       nr_rollout, scenarios, network, params,
                       output_dir='results_Pseudo_DFL'):
    """
    Generate all DFL plots and save them to output_dir.

    Parameters
    ----------
    trainer       : PseudoDFLTrainer (for training curves)
    results_dict  : {label: eval_result_dict}   all methods
    dfl_rollout   : PseudoDFLRollout   (for the single-scenario and averaged plots)
    one_scenario  : Scenario     (for the repair heatmap and availability trajectory)
    nr_rollout    : NoRepairRollout  (the availability trajectory is skipped if it
                    has no rollout_scenario method)
    scenarios     : list of Scenario  (for the averaged plots; the first 50 are used)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving Pseudo_DFL report plots to {out}/")

    dfl_sc_result = dfl_rollout.rollout_scenario(one_scenario)
    nr_sc_result  = nr_rollout.rollout_scenario(one_scenario) \
        if hasattr(nr_rollout, 'rollout_scenario') else None

    plot_training_curves(trainer,
                         save_path=out / 'training_curves.png')

    plot_cost_distribution(results_dict,
                           save_path=out / 'cost_distribution.png')

    plot_cost_comparison_bar(results_dict,
                             save_path=out / 'cost_comparison_bar.png')

    plot_repair_heatmap(dfl_sc_result, network, params,
                        save_path=out / 'repair_heatmap.png')

    if nr_sc_result is not None:
        plot_availability_trajectory(dfl_sc_result, nr_sc_result,
                                     network, params,
                                     save_path=out / 'availability_trajectory.png')

    plot_repair_decisions(scenarios, dfl_rollout, network, params,
                          save_path=out / 'repair_decisions.png',
                          n_eval=min(50, len(scenarios)))

    plot_repair_heatmap_avg(scenarios, dfl_rollout, network, params,
                            save_path=out / 'repair_heatmap_avg_50.png',
                            n_eval=min(50, len(scenarios)))

    print(f"  All Pseudo_DFL plots saved to {out}/")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path, default_name):
    if save_path is None:
        plt.show()
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / default_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def load_sddip_costs(sddip_results_dir):
    """
    Try to load simulation costs from an SDDiP results directory.

    Checks simulation_results.json, results.json and sddip_results.json, in
    that order, for a 'costs', 'simulation_costs' or 'total_costs' key and
    returns the first match as an array. Returns None if none is found.
    """
    results_dir = Path(sddip_results_dir)
    for fname in ('simulation_results.json', 'results.json',
                  'sddip_results.json'):
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)
            for key in ('costs', 'simulation_costs', 'total_costs'):
                if key in data:
                    return np.array(data[key])
    return None
