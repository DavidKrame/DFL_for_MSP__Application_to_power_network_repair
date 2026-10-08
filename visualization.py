"""
Plots of an SAA solution (main.py): failure probabilities, dispatch, load
shedding, availability, repair rates and summary statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def plot_failure_probabilities(model_params, save_path=None):
    """Plot the per-stage failure probabilities lambda_g (generators) and lambda_e (lines)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    stages = np.arange(1, model_params.T + 1)

    for g in range(model_params.n_generators):
        ax1.plot(stages, model_params.lambda_g[g, :], 
                marker='o', label=f'Gen {g+1}', linewidth=2)
    
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Failure Probability', fontsize=12)
    ax1.set_title('Generator Failure Probabilities  (model parameters)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for e in range(model_params.n_lines):
        ax2.plot(stages, model_params.lambda_e[e, :],
                marker='s', label=f'Line {e+1}', linewidth=2)
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Failure Probability', fontsize=12)
    ax2.set_title('Line Failure Probabilities  (model parameters)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_generation_dispatch(solution, network_data, scenario_id=0, save_path=None):
    """Stacked area chart of generation dispatch for a single scenario."""
    if scenario_id not in solution['generation']:
        print(f"Scenario {scenario_id} not found in solution")
        return
    
    stages = sorted(solution['generation'][scenario_id].keys())
    n_gen = network_data.n_generators

    gen_data = np.zeros((len(stages), n_gen))
    for i, t in enumerate(stages):
        for g in range(n_gen):
            gen_data[i, g] = solution['generation'][scenario_id][t][g]

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, n_gen))

    ax.stackplot(stages, gen_data.T, labels=[f'Gen {g+1}' for g in range(n_gen)],
                colors=colors, alpha=0.8)

    total_demand = sum(network_data.load_nominal)
    ax.axhline(total_demand, color='red', linestyle='--', linewidth=2,
              label=f'Total Demand ({total_demand:.1f} MW)')
    
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Generation (MW)', fontsize=12)
    ax.set_title(f'Generation Dispatch  (training scenario {scenario_id})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_load_shedding(solution, network_data, scenario_id=0, save_path=None):
    """Stacked bar chart of unserved demand (nominal minus served) per load for one scenario."""
    if scenario_id not in solution['load_served']:
        print(f"Scenario {scenario_id} not found in solution")
        return
    
    stages = sorted(solution['load_served'][scenario_id].keys())
    n_loads = network_data.n_loads

    shedding_data = np.zeros((len(stages), n_loads))
    for i, t in enumerate(stages):
        for l in range(n_loads):
            demand = network_data.load_nominal[l]
            served = solution['load_served'][scenario_id][t][l]
            shedding_data[i, l] = demand - served

    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_loads))
    
    bottom = np.zeros(len(stages))
    for l in range(n_loads):
        ax.bar(stages, shedding_data[:, l], bottom=bottom,
              label=f'Load {l+1}', color=colors[l], alpha=0.8)
        bottom += shedding_data[:, l]
    
    ax.set_xlabel('Stage', fontsize=12)
    ax.set_ylabel('Load Shedding (MW)', fontsize=12)
    ax.set_title(f'Load Shedding  (training scenario {scenario_id})',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_availability_gantt(availability, model_params, save_path=None):
    """Gantt chart of availability per component and stage (green = available, red = failed).

    availability holds 0/1 arrays 'gen' of shape (T, n_gen) and 'line' of shape (T, n_lines).
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    T = model_params.T
    n_gen = availability['gen'].shape[1]
    n_lines = availability['line'].shape[1]

    for g in range(n_gen):
        for t in range(T):
            if availability['gen'][t, g] == 1:
                ax1.barh(g, 1, left=t, height=0.8, 
                        color='green', alpha=0.7, edgecolor='black')
            else:
                ax1.barh(g, 1, left=t, height=0.8, 
                        color='red', alpha=0.7, edgecolor='black')
    
    ax1.set_yticks(range(n_gen))
    ax1.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)])
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Generator', fontsize=12)
    ax1.set_title('Generator Availability  (training scenario)', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, T - 0.5)
    ax1.grid(True, alpha=0.3, axis='x')

    for e in range(n_lines):
        for t in range(T):
            if availability['line'][t, e] == 1:
                ax2.barh(e, 1, left=t, height=0.8, 
                        color='green', alpha=0.7, edgecolor='black')
            else:
                ax2.barh(e, 1, left=t, height=0.8, 
                        color='red', alpha=0.7, edgecolor='black')
    
    ax2.set_yticks(range(n_lines))
    ax2.set_yticklabels([f'Line {e+1}' for e in range(n_lines)])
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Line', fontsize=12)
    ax2.set_title('Line Availability  (training scenario)', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.5, T - 0.5)
    ax2.grid(True, alpha=0.3, axis='x')

    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Available')
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Failed')
    ax1.legend(handles=[green_patch, red_patch], loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_solution_statistics(stats, model_params, save_path=None, objective_value=None,
                             solver_metadata=None):
    """Mean total generation and load shedding per stage across scenarios, with ±1 std dev bars.

    The objective value, if given, goes in the title and solver_metadata in a footer.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    stages = np.arange(1, model_params.T + 1)

    ax1.errorbar(stages, stats['mean_generation'], yerr=stats['std_generation'],
                marker='o', linewidth=2, capsize=5, capthick=2, label='Generation')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Generation (MW)', fontsize=12)
    ax1.set_title('Mean Generation ± 1 Std Dev  (training scenarios)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.errorbar(stages, stats['mean_shedding'], yerr=stats['std_shedding'],
                marker='s', linewidth=2, capsize=5, capthick=2,
                color='red', label='Load Shedding')
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Load Shedding (MW)', fontsize=12)
    ax2.set_title('Mean Load Shedding ± 1 Std Dev  (training scenarios)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if objective_value is not None:
        fig.suptitle(f'SAA Solution Statistics  (training scenarios)  —  Objective: ${objective_value:,.2f}',
                     fontsize=14, fontweight='bold', y=1.02)

    if solver_metadata is not None:
        meta = solver_metadata
        gap_target = f"{meta.get('mip_gap_target', 0)*100:.1f}%"
        gap_val = meta.get('mip_gap_achieved')
        gap_achieved = f"{gap_val*100:.2f}%" if gap_val is not None else "N/A"
        tc = meta.get('termination_condition', '')
        status_str = f"MIP gap: {gap_achieved} (target: {gap_target})"
        if gap_val is None and tc:
            status_str = f"Status: {tc} (target gap: {gap_target})"
        # In tree mode the model size is the tree's B^T sample paths. The stored
        # n_scenarios is the --n-scenarios argument, which tree mode ignores.
        if meta.get('mode') == 'tree':
            n_paths = meta.get('n_sample_paths')
            size_str = (f"Sample paths: {n_paths:,} (tree B={meta.get('branching', '?')})"
                        if n_paths else "Sample paths: ?")
        else:
            size_str = f"Scenarios: {meta.get('n_scenarios', '?')}"
        meta_text = (
            f"Solver: {meta.get('solver_name', '?')}  |  "
            f"{size_str}  |  "
            f"{status_str}  |  "
            f"Time: {meta.get('solve_time', 0):.0f}s / {meta.get('time_limit', 0):.0f}s"
        )
        fig.text(0.5, -0.02, meta_text, ha='center', va='top',
                 fontsize=9, fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_generation_per_generator(solution, network_data, model_params, save_path=None):
    """One subplot per generator: mean dispatch across scenarios, ±1 std dev band, P_max line."""
    n_gen = network_data.n_generators
    scenarios = sorted(solution['generation'].keys())
    stages = sorted(solution['generation'][scenarios[0]].keys())
    n_scenarios = len(scenarios)
    T = len(stages)

    gen_data = np.zeros((n_scenarios, T, n_gen))
    for i, s in enumerate(scenarios):
        for j, t in enumerate(stages):
            for g in range(n_gen):
                gen_data[i, j, g] = solution['generation'][s][t][g]

    mean_gen = gen_data.mean(axis=0)   # (T, n_gen)
    std_gen  = gen_data.std(axis=0)    # (T, n_gen)

    colors = ['steelblue', 'darkorange', 'seagreen', 'firebrick']

    fig, axes = plt.subplots(1, n_gen, figsize=(7 * n_gen, 5), sharey=False)
    if n_gen == 1:
        axes = [axes]

    for g, ax in enumerate(axes):
        c = colors[g % len(colors)]
        ax.fill_between(stages,
                        np.maximum(0, mean_gen[:, g] - std_gen[:, g]),
                        mean_gen[:, g] + std_gen[:, g],
                        alpha=0.25, color=c, label='±1 std dev')
        ax.plot(stages, mean_gen[:, g], marker='o', linewidth=2,
                color=c, label='Mean dispatch')
        ax.axhline(network_data.gen_pmax[g], color='gray', linestyle='--',
                   linewidth=1.5,
                   label=f'P$_{{\\mathrm{{max}}}}$ = {network_data.gen_pmax[g]:.0f} MW')
        bus_label = network_data.gen_buses[g] + 1
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Dispatch (MW)', fontsize=12)
        ax.set_title(f'Generator {g + 1}  (Bus {bus_label})',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(stages[0] - 0.5, stages[-1] + 0.5)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Mean Generation Dispatch per Generator ± 1 Std Dev  (training scenarios)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_repair_decisions(solution, network_data, save_path=None):
    """Heatmaps of expected repair rates per component and stage.

    solution['repairs'] holds the scenario mean (ws mode) or the path-probability-weighted
    mean (tree mode) of the repair decisions at each stage.
    """
    repairs = solution.get('repairs', {})
    if not repairs or 'gen' not in repairs:
        print("No repair data available")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    stages = sorted(repairs['gen'].keys())
    n_gen = network_data.n_generators
    n_lines = network_data.n_lines

    gen_repairs = np.zeros((len(stages), n_gen))
    line_repairs = np.zeros((len(stages), n_lines))

    for i, t in enumerate(stages):
        for g in range(n_gen):
            gen_repairs[i, g] = repairs['gen'][t].get(g, 0)
        for e in range(n_lines):
            line_repairs[i, e] = repairs['line'][t].get(e, 0)

    im1 = ax1.imshow(gen_repairs.T, aspect='auto', cmap='Reds',
                     vmin=0, vmax=max(0.01, gen_repairs.max()), interpolation='nearest')
    ax1.set_xlabel('Stage', fontsize=12)
    ax1.set_ylabel('Generator', fontsize=12)
    ax1.set_title('Generator Repair Decisions  (training scenarios)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(stages)))
    ax1.set_xticklabels(stages)
    ax1.set_yticks(range(n_gen))
    ax1.set_yticklabels([f'Gen {g+1}' for g in range(n_gen)])
    plt.colorbar(im1, ax=ax1, label='Repair Rate')

    im2 = ax2.imshow(line_repairs.T, aspect='auto', cmap='Blues',
                     vmin=0, vmax=max(0.01, line_repairs.max()), interpolation='nearest')
    ax2.set_xlabel('Stage', fontsize=12)
    ax2.set_ylabel('Line', fontsize=12)
    ax2.set_title('Line Repair Decisions  (training scenarios)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages)
    ax2.set_yticks(range(n_lines))
    ax2.set_yticklabels([f'Line {e+1}' for e in range(n_lines)])
    plt.colorbar(im2, ax=ax2, label='Repair Rate')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def create_results_report(solution, network_data, model_params, scenarios=None,
                         output_dir='results'):
    """Generate all SAA result plots and save them to output_dir.

    scenarios is optional. The availability plot uses solution['availability'][0] when
    present. Otherwise it shows the availability of scenarios[0] without any repairs
    (compute_availability_trajectories with no repair decisions, so failures only).
    It is skipped if neither exists.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CREATING VISUAL RESULTS REPORT")
    print("=" * 70)

    print("Plotting failure probabilities...")
    plot_failure_probabilities(model_params,
                               save_path=output_path / 'failure_probabilities.png')

    print("Plotting generation dispatch...")
    plot_generation_dispatch(solution, network_data, scenario_id=0,
                            save_path=output_path / 'generation_dispatch_s0.png')

    print("Plotting load shedding...")
    plot_load_shedding(solution, network_data, scenario_id=0,
                      save_path=output_path / 'load_shedding_s0.png')

    print("Plotting availability...")
    if 'availability' in solution and 0 in solution['availability']:
        availability = solution['availability'][0]
    elif scenarios is not None:
        from state_dynamics import create_state_dynamics
        state_dyn = create_state_dynamics(network_data, model_params)
        availability = state_dyn.compute_availability_trajectories(scenarios[0])
    else:
        availability = None
    if availability is not None:
        plot_availability_gantt(availability, model_params,
                               save_path=output_path / 'availability_s0.png')
    else:
        print("  Skipped (no availability data)")

    print("Plotting generation per generator...")
    plot_generation_per_generator(solution, network_data, model_params,
                                  save_path=output_path / 'generation_per_generator.png')

    print("Plotting repair decisions...")
    plot_repair_decisions(solution, network_data,
                         save_path=output_path / 'repair_decisions.png')

    print("Plotting solution statistics...")
    from utils import compute_solution_statistics
    stats = compute_solution_statistics(solution, network_data, model_params)
    obj_val = solution.get('objective_value', None)
    solver_metadata = solution.get('solver_metadata', None)
    plot_solution_statistics(stats, model_params,
                            save_path=output_path / 'solution_statistics.png',
                            objective_value=obj_val,
                            solver_metadata=solver_metadata)

    print()
    print(f"All figures saved to {output_path}/")
    print("=" * 70)


if __name__ == "__main__":
    print("Testing visualization functions...")

    from network_data import create_network
    from model_parameters import create_model_parameters

    network = create_network()
    params = create_model_parameters(network, seed=42)

    plot_failure_probabilities(params)

    print("Visualization test complete!")
