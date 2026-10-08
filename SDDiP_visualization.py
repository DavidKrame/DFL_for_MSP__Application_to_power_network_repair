"""
Plots of an SDDiP solution from SDDP.jl (SDDiP_main.py): convergence of the
bounds and gap, repair heatmaps, a bound and gap summary, and algorithm
statistics. Layout and style follow visualization.py.

The solution dict comes from SDDPJuliaBridge.extract_results in
SDDiP_julia_bridge.py. Its 'convergence' entry holds per-iteration lower
bounds, upper bounds (the forward-pass simulation value in the SDDP.jl
training log) and gaps. After the policy simulation, 'repairs' holds for each
stage and component the share of simulated trajectories that repair it.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the failure probability plotting from SAA visualization
from visualization import plot_failure_probabilities


def plot_sddip_convergence(solution: Dict, save_path: Optional[str] = None) -> None:
    """Lower and upper bounds and the optimality gap per training iteration.

    The dashed 1% line is a visual reference only; training stops on the
    iteration or time limit.
    """
    conv = solution.get('convergence', {})
    if not conv or 'upper_bounds' not in conv:
        print("No convergence data available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = range(1, len(conv['upper_bounds']) + 1)

    axes[0].plot(iterations, conv['upper_bounds'], 'b-o', linewidth=2,
                 markersize=6, label='Upper Bound')
    axes[0].plot(iterations, conv['lower_bounds'], 'r-s', linewidth=2,
                 markersize=6, label='Lower Bound')

    axes[0].fill_between(iterations, conv['lower_bounds'], conv['upper_bounds'],
                         alpha=0.2, color='gray')

    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Cost ($)', fontsize=12)
    axes[0].set_title('SDDiP Convergence: Bounds  (algorithm internal)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    if 'gaps' in conv:
        gaps_pct = [g * 100 for g in conv['gaps']]
        axes[1].plot(iterations, gaps_pct, 'g-^', linewidth=2, markersize=6)
        axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
                        label='1% Target')

        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Optimality Gap (%)', fontsize=12)
        axes[1].set_title('SDDiP Convergence: Optimality Gap  (algorithm internal)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(bottom=0)
        axes[1].legend(loc='best', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_sddip_repair_decisions(solution: Dict, network_data,
                                save_path: Optional[str] = None) -> None:
    """Heatmaps of the repair probability per component and stage.

    Each value is the share of simulated trajectories that repair the component
    at that stage.
    """
    repairs = solution.get('repairs', {})
    if not repairs or 'gen' not in repairs:
        print("No repair data available")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    stages = sorted([int(k) for k in repairs['gen'].keys()])
    n_gen = network_data.n_generators
    n_lines = network_data.n_lines

    gen_repairs = np.zeros((len(stages), n_gen))
    line_repairs = np.zeros((len(stages), n_lines))

    for i, t in enumerate(stages):
        # Stage and component keys are strings when the solution comes from
        # JSON (as the bridge output does) and ints in hand-built dicts.
        t_key = str(t) if str(t) in repairs['gen'] else t
        for g in range(n_gen):
            g_key = str(g) if str(g) in repairs['gen'][t_key] else g
            gen_repairs[i, g] = repairs['gen'][t_key][g_key]
        for e in range(n_lines):
            e_key = str(e) if str(e) in repairs['line'][t_key] else e
            line_repairs[i, e] = repairs['line'][t_key][e_key]

    im1 = ax1.imshow(gen_repairs.T, aspect='auto', cmap='Reds',
                     vmin=0, vmax=1, interpolation='nearest')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Generator', fontsize=12)
    ax1.set_title('Generator Repair Decisions  (training scenarios)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages)
    ax1.set_yticks(range(n_gen))
    ax1.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)])
    plt.colorbar(im1, ax=ax1, label='Repair Probability')

    im2 = ax2.imshow(line_repairs.T, aspect='auto', cmap='Blues',
                     vmin=0, vmax=1, interpolation='nearest')
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Line', fontsize=12)
    ax2.set_title('Line Repair Decisions  (training scenarios)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages)
    ax2.set_yticks(range(n_lines))
    ax2.set_yticklabels([f'Line {e+1}' for e in range(n_lines)])
    plt.colorbar(im2, ax=ax2, label='Repair Probability')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _compute_smoothed_bounds(solution, tail_pct=3):
    """Average the last tail_pct% of convergence iterations (at least one) to reduce noise.

    The smoothed gap is (upper - lower) / upper, or 0 when upper <= 0.
    Returns (smoothed_lower, smoothed_upper, smoothed_gap, n_tail), or None
    without convergence data.
    """
    conv = solution.get('convergence', {})
    upper_bounds = conv.get('upper_bounds', [])
    lower_bounds = conv.get('lower_bounds', [])
    if not upper_bounds or not lower_bounds:
        return None
    n_tail = max(1, int(np.ceil(len(upper_bounds) * tail_pct / 100)))
    smoothed_upper = np.mean(upper_bounds[-n_tail:])
    smoothed_lower = np.mean(lower_bounds[-n_tail:])
    smoothed_gap = (smoothed_upper - smoothed_lower) / smoothed_upper if smoothed_upper > 0 else 0
    return smoothed_lower, smoothed_upper, smoothed_gap, n_tail


def plot_sddip_cost_breakdown(solution: Dict, save_path: Optional[str] = None) -> None:
    """Bound bars with their gap: last iteration (left) and 3% tail average (right)."""
    lower_bound = solution.get('lower_bound', 0)
    upper_bound = solution.get('upper_bound', 0)
    gap = solution.get('gap', 0)
    smoothed = _compute_smoothed_bounds(solution)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    categories = ['Lower Bound\n(Last Iter)', 'Upper Bound\n(Last Iter)']
    values = [lower_bound, upper_bound]
    colors = ['#2ecc71', '#e74c3c']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'${val:,.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.annotate(f'Gap: {gap*100:.2f}%',
                xy=(0.5, (lower_bound + upper_bound) / 2),
                fontsize=14, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Last Iteration  (algorithm bounds)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    if smoothed:
        sm_lower, sm_upper, sm_gap, n_tail = smoothed
        categories = [f'Lower Bound\n(Avg last {n_tail})', f'Upper Bound\n(Avg last {n_tail})']
        values = [sm_lower, sm_upper]

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'${val:,.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.annotate(f'Smoothed Gap: {sm_gap*100:.2f}%',
                    xy=(0.5, (sm_lower + sm_upper) / 2),
                    fontsize=14, ha='center',
                    bbox=dict(boxstyle='round', facecolor='#87CEEB', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No convergence\ndata available',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Smoothed 3% Tail Avg  (algorithm bounds)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('SDDiP Solution Cost Summary  (algorithm bounds)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_sddip_summary(solution: Dict, model_params, network_data,
                       save_path: Optional[str] = None) -> None:
    """6-panel summary figure of the SDDiP solution.

    Panels (3 x 2 grid, row by row): bounds, gap, generator and line failure
    probabilities, generator repair heatmap, and bound bars with the last and
    smoothed gaps.
    """
    fig = plt.figure(figsize=(16, 12))

    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # convergence bounds (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    conv = solution.get('convergence', {})
    if conv and 'upper_bounds' in conv:
        iterations = range(1, len(conv['upper_bounds']) + 1)
        ax1.plot(iterations, conv['upper_bounds'], 'b-o', label='Upper Bound', markersize=4)
        ax1.plot(iterations, conv['lower_bounds'], 'r-s', label='Lower Bound', markersize=4)
        ax1.fill_between(iterations, conv['lower_bounds'], conv['upper_bounds'],
                         alpha=0.2, color='gray')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Convergence: Bounds  (algorithm internal)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # gap convergence (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    if conv and 'gaps' in conv:
        gaps_pct = [g * 100 for g in conv['gaps']]
        iterations = range(1, len(gaps_pct) + 1)
        ax2.plot(iterations, gaps_pct, 'g-^', markersize=4)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Target')
        ax2.set_ylim(bottom=0)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gap (%)')
    ax2.set_title('Convergence: Optimality Gap  (algorithm internal)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # generator failure probabilities (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    stages = np.arange(1, model_params.T + 1)
    for g in range(model_params.n_generators):
        ax3.plot(stages, model_params.lambda_g[g, :],
                 marker='o', markersize=3, label=f'Gen {g+1}')
    ax3.set_xlabel('Stage')
    ax3.set_ylabel('Failure Probability')
    ax3.set_title('Generator Failure Probabilities  (model parameters)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # line failure probabilities (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    for e in range(model_params.n_lines):
        ax4.plot(stages, model_params.lambda_e[e, :],
                 marker='s', markersize=3, label=f'Line {e+1}')
    ax4.set_xlabel('Stage')
    ax4.set_ylabel('Failure Probability')
    ax4.set_title('Line Failure Probabilities  (model parameters)')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    # generator repair heatmap (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    repairs = solution.get('repairs', {})
    if repairs and 'gen' in repairs:
        stages_list = sorted([int(k) for k in repairs['gen'].keys()])
        n_gen = network_data.n_generators
        gen_repairs = np.zeros((len(stages_list), n_gen))
        for i, t in enumerate(stages_list):
            t_key = str(t) if str(t) in repairs['gen'] else t
            for g in range(n_gen):
                g_key = str(g) if str(g) in repairs['gen'][t_key] else g
                gen_repairs[i, g] = repairs['gen'][t_key][g_key]
        im = ax5.imshow(gen_repairs.T, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        ax5.set_xlabel('Stage')
        ax5.set_ylabel('Generator')
        ax5.set_yticks(range(n_gen))
        ax5.set_yticklabels([f'G{g+1}' for g in range(n_gen)])
        plt.colorbar(im, ax=ax5, label='Prob')
    ax5.set_title('Generator Repair Decisions  (training scenarios)')

    # bound bars with last-iteration and smoothed gap (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])
    lower = solution.get('lower_bound', 0)
    upper = solution.get('upper_bound', 0)
    gap = solution.get('gap', 0)
    smoothed = _compute_smoothed_bounds(solution)

    bars = ax6.bar(['Lower\nBound', 'Upper\nBound'], [lower, upper],
                   color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, [lower, upper]):
        ax6.annotate(f'${val:,.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    ax6.set_ylabel('Cost ($)')
    title = f'Cost Summary  (algorithm bounds · gap: {gap*100:.2f}%'
    if smoothed:
        title += f' | smoothed: {smoothed[2]*100:.2f}%'
    title += ')'
    ax6.set_title(title)
    ax6.grid(True, alpha=0.3, axis='y')

    fig.suptitle('SDDiP Solution Summary  (algorithm internal)', fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_algorithm_statistics(solution: Dict, save_path: Optional[str] = None) -> None:
    """Iteration and cut counts, total and per-iteration solve time, and a gauge of the gap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iterations = solution.get('iterations', 0)
    solve_time = solution.get('solve_time', 0)
    n_cuts = solution.get('n_cuts', 0)
    gap = solution.get('gap', 0)

    ax1 = axes[0]
    categories = ['Iterations', 'Cuts Generated']
    values = [iterations, n_cuts]
    colors = ['#3498db', '#9b59b6']
    bars = ax1.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, values):
        ax1.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.set_title('Algorithm Statistics  (algorithm metadata)')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2 = axes[1]
    ax2.bar(['Total Time'], [solve_time], color='#e67e22', alpha=0.8, edgecolor='black')
    ax2.annotate(f'{solve_time:.1f}s', xy=(0, solve_time),
                 xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
    if iterations > 0:
        time_per_iter = solve_time / iterations
        ax2.bar(['Per Iteration'], [time_per_iter], color='#f39c12', alpha=0.8, edgecolor='black')
        ax2.annotate(f'{time_per_iter:.2f}s', xy=(1, time_per_iter),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Solve Time  (algorithm metadata)')
    ax2.grid(True, alpha=0.3, axis='y')

    # semicircle gauge for the optimality gap
    ax3 = axes[2]
    gap_pct = gap * 100

    theta = np.linspace(0, np.pi, 100)
    r = 1
    ax3.plot(r * np.cos(theta), r * np.sin(theta), 'k-', linewidth=2)

    gap_angle = np.pi * (1 - min(gap_pct / 20, 1))  # gap 0% -> pi (left), >= 20% -> 0 (right)
    ax3.plot([0, 0.8 * np.cos(gap_angle)], [0, 0.8 * np.sin(gap_angle)],
             'r-', linewidth=4)

    ax3.text(0, -0.3, f'{gap_pct:.2f}%', ha='center', fontsize=16, fontweight='bold')
    ax3.text(-1, 0, '20%', ha='center', fontsize=10)
    ax3.text(1, 0, '0%', ha='center', fontsize=10)
    ax3.text(0, 1.1, 'Optimality Gap', ha='center', fontsize=12, fontweight='bold')

    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-0.5, 1.3)
    ax3.set_aspect('equal')
    ax3.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_sddip_results_report(solution: Dict, network_data, model_params,
                                output_dir: str = 'results_SDDiP') -> None:
    """Generate all SDDiP plots and save to output_dir."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CREATING SDDiP VISUAL RESULTS REPORT")
    print("=" * 70)

    print("Plotting failure probabilities...")
    plot_failure_probabilities(model_params,
                               save_path=output_path / 'sddip_failure_probabilities.png')

    print("Plotting convergence...")
    plot_sddip_convergence(solution,
                           save_path=output_path / 'sddip_convergence.png')

    print("Plotting repair decisions...")
    plot_sddip_repair_decisions(solution, network_data,
                                save_path=output_path / 'sddip_repair_decisions.png')

    print("Plotting cost breakdown...")
    plot_sddip_cost_breakdown(solution,
                              save_path=output_path / 'sddip_cost_breakdown.png')

    print("Plotting algorithm statistics...")
    plot_algorithm_statistics(solution,
                              save_path=output_path / 'sddip_algorithm_stats.png')

    print("Creating summary plot...")
    plot_sddip_summary(solution, model_params, network_data,
                       save_path=output_path / 'sddip_summary.png')

    print()
    print(f"All figures saved to {output_path}/")
    print("=" * 70)


if __name__ == "__main__":
    print("Testing SDDiP visualization functions...")

    from network_data import create_network
    from model_parameters import create_model_parameters

    network = create_network()
    params = create_model_parameters(network, seed=42)

    mock_solution = {
        'lower_bound': 5000.0,
        'upper_bound': 5500.0,
        'gap': 0.1,
        'iterations': 10,
        'solve_time': 45.3,
        'n_cuts': 50,
        'repairs': {
            'gen': {1: {0: 0.0, 1: 0.0}, 2: {0: 0.1, 1: 0.0}},
            'line': {1: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                     2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}}
        },
        'convergence': {
            'upper_bounds': [8000, 7000, 6500, 6000, 5800, 5600, 5550, 5520, 5510, 5500],
            'lower_bounds': [4000, 4500, 4800, 4900, 4950, 4980, 4990, 4995, 4998, 5000],
            'gaps': [0.5, 0.36, 0.26, 0.18, 0.15, 0.11, 0.10, 0.095, 0.093, 0.1]
        }
    }

    plot_failure_probabilities(params)
    plot_sddip_convergence(mock_solution)

    print("SDDiP visualization test complete!")
