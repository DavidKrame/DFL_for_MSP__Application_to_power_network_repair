"""
SDDiP_julia_bridge.py: Python interface to SDDP.jl for the multistage stochastic ED-OPF.

Transfers network/parameter data to Julia via juliacall, builds the SDDP.jl model
(binary state variables for component availability and repair pipeline), runs
training, and extracts results back into Python.

Requirements:
    pip install juliacall
    (Julia packages are managed via juliapkg.json; Gurobi.jl is installed
    separately and only for solver='gurobi', see setup_julia)
"""

import json
import os
import time
import numpy as np


class SDDPJuliaBridge:
    """Wraps SDDP.jl: data transfer, model construction, training, and result extraction."""

    def __init__(self, network, params, seed=None, solver='highs'):
        self.network = network
        self.params = params
        self.seed = seed
        # Layer B solver selector: 'highs' (default) or 'gurobi'. Any gurobi
        # variant name ('gurobi', 'gurobi_direct', ...) maps to Gurobi.Optimizer,
        # any other name to HiGHS.
        self.solver = (solver or 'highs').lower()
        self._use_gurobi = 'gurobi' in self.solver
        self.jl = None
        self._model_built = False
        self._trained = False
        self._simulated = False
        self._train_time = 0.0

    def setup_julia(self):
        """Import juliacall, load SDDP.jl, JuMP, JSON and the chosen optimizer, push data."""
        print("  Setting up Julia environment...")
        print("  (First run may take several minutes to install Julia packages)")

        from juliacall import Main as jl
        self.jl = jl

        # Only the optimizer in use is loaded. Gurobi.jl is not in juliapkg.json,
        # so HiGHS-only environments never need it installed or licensed. Using
        # Gurobi for SDDiP requires adding Gurobi.jl to the Julia environment once
        # (with GUROBI_HOME set):
        #   python -c "import juliapkg; juliapkg.add('Gurobi', \
        #       '2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()"
        # Without that step, `using Gurobi` fails with an unhelpful Julia error,
        # which is re-raised below with these instructions.
        if self._use_gurobi:
            try:
                jl.seval("using SDDP, JuMP, Gurobi, JSON")
            except Exception as e:
                raise RuntimeError(
                    "SDDiP with Gurobi needs Gurobi.jl in the Julia environment, "
                    "which is not installed by default. It is added once "
                    "(GUROBI_HOME must point at the Gurobi install):\n"
                    "    python -c \"import juliapkg; juliapkg.add('Gurobi', "
                    "'2e9cd046-0924-5485-92f1-d5272153d98b'); juliapkg.resolve()\"\n"
                    "HiGHS needs no extra setup (--solver highs / "
                    "ED_OPF_SDDIP_SOLVER=highs).\n"
                    f"  Underlying Julia error: {e}"
                ) from e
        else:
            jl.seval("using SDDP, JuMP, HiGHS, JSON")
        # Julia's global RNG draws the sampled failure realizations and SDDP.jl's
        # forward-pass and simulation samples.
        if self.seed is not None:
            jl.seval(f"import Random; Random.seed!({self.seed})")
        print(f"  Julia packages loaded successfully (optimizer: "
              f"{'Gurobi' if self._use_gurobi else 'HiGHS'}).")

        self._transfer_data()

    def _optimizer_julia_code(self):
        """Julia snippet defining `const OPTIMIZER` for the selected solver.

        Both branches produce an `optimizer_with_attributes(...)` factory that
        SDDP.LinearPolicyGraph consumes. SOLVER_THREADS is defined just before
        this runs. The Gurobi branch shares a single `Gurobi.Env()` across all
        subproblem nodes, the usual Gurobi.jl pattern, so the license check runs
        once instead of once per node.
        """
        if self._use_gurobi:
            return """
            const GRB_ENV = Gurobi.Env()
            const OPTIMIZER = optimizer_with_attributes(
                () -> Gurobi.Optimizer(GRB_ENV),
                "OutputFlag"     => 0,
                "Threads"        => SOLVER_THREADS,
                "FeasibilityTol" => 1e-6,
                "OptimalityTol"  => 1e-6,
                # DualReductions=0 forces a definite infeasible/unbounded status:
                # the default (1) can report the ambiguous INFEASIBLE_OR_UNBOUNDED on a
                # well-posed backward-pass LP and abort the node solve. InfUnbdInfo=1
                # retains a certificate if a genuine bad status ever arises.
                "DualReductions" => 0,
                "InfUnbdInfo"    => 1,
            )
            """
        return """
            const OPTIMIZER = optimizer_with_attributes(
                HiGHS.Optimizer,
                "output_flag"                  => false,
                "threads"                      => SOLVER_THREADS,
                "primal_feasibility_tolerance" => 1e-6,
                "dual_feasibility_tolerance"   => 1e-6,
                "ipm_optimality_tolerance"     => 1e-6,
            )
            """

    def _transfer_data(self):
        """Transfer all Python data into the Julia namespace.

        All power quantities are normalized to per-unit on a capacity base
        S_cap = max(sum(gen_pmax), sum(load_nominal)), floored at 1, before
        being sent to Julia. This is not the MATPOWER sbase_mva (100 MVA); it
        is the total installed generation capacity, or the total load if larger.

        Reason for the capacity base instead of sbase_mva (case14 numbers):
          - With sbase_mva = 100: B_pu = 1/x_pu, between 1.8 and 24 (up to 247
            on stiff lines of case118).
          - With S_cap (399 MW for case14): B_pu = 100/(x_pu·399), between 0.45
            and 6.
          - All power quantities (p_g, d_l, F) become O(0.01–1).
          - The susceptances and the big-M coefficients M_e = 2π·B_pu (at most
            about 37 on case14) stay well within the numerical range of HiGHS,
            Gurobi and SDDP.jl.

        The scaling is self-consistent:
          - Cost coefficients are multiplied by S_cap so that
            cost_pu × p_pu = cost_MW × p_MW (objective stays in dollar units).
          - Repair costs are fixed $ amounts on binary variables, so they are
            passed unscaled.
          - Objective is in dollars throughout; no unscaling of bounds needed.
          - Generation and load simulation results are multiplied by S_cap in
            extract_results() to restore MW units.
        """
        jl = self.jl
        net = self.network
        par = self.params

        # Capacity base: total installed generation capacity, or total load if larger.
        # S_cap is used instead of sbase_mva=100 for numerical conditioning:
        #   B stored as sbase_mva/x_pu [MW/rad].  After dividing by S_cap:
        #     B_pu = sbase_mva / (x_pu · S_cap)
        #   For case14 (S_cap=399 MW, x_pu=0.059): B_pu = 100/(0.059·399) ≈ 4.2
        #   For case118 (S_cap=6515 MW, same line):  B_pu = 100/(0.059·6515) ≈ 0.26
        #   Compare to standard per-unit (sbase_mva):  B_pu = 1/x_pu ≈ 16.9
        # The capacity base keeps B_pu close to the power quantities (B_pu about
        # 0.04–6 and power below 2 on case14 and case118), whereas standard
        # per-unit gives B_pu up to a few hundred on large networks (247 on
        # case118).
        # A floor of 1.0 prevents division by zero on degenerate inputs.
        sbase = float(max(net.gen_pmax.sum(), net.load_nominal.sum()))
        if sbase < 1.0:
            sbase = 1.0

        # Scalars
        jl.seval(f"const T = {par.T}")
        jl.seval(f"const tau = {par.tau}")
        jl.seval(f"const N_G = {net.n_generators}")
        jl.seval(f"const N_E = {net.n_lines}")
        jl.seval(f"const N_B = {net.n_buses}")
        jl.seval(f"const N_L = {net.n_loads}")
        jl.seval(f"const SLACK = {net.slack_bus + 1}")  # 1-indexed

        # Cost parameters: $/MWh times sbase, so that cost_pu * p_pu = cost_MW * p_MW in $
        jl.seval(f"const load_shedding_cost = {par.load_shedding_cost * sbase}")
        jl.seval(f"const gen_cost = {_to_julia_vec(net.gen_cost * sbase)}")

        # Power limits: MW divided by sbase gives per-unit
        jl.seval(f"const gen_pmin = {_to_julia_vec(net.gen_pmin / sbase)}")
        jl.seval(f"const gen_pmax = {_to_julia_vec(net.gen_pmax / sbase)}")
        jl.seval(f"const load_nominal = {_to_julia_vec(net.load_nominal / sbase)}")

        # Repair costs are in $ per repair (not $/MWh), independent of the power scale
        jl.seval(f"const gen_repair_cost = {_to_julia_vec(par.gen_repair_cost)}")
        jl.seval(f"const line_repair_cost = {_to_julia_vec(par.line_repair_cost)}")

        # Per-stage repair budget in $ (phi times the total fleet repair cost, set
        # by ModelParameters). None gives HAS_BUDGET=false and no budget constraint.
        _budget = getattr(par, 'repair_budget_per_stage', None)
        jl.seval(f"const HAS_BUDGET = {'true' if _budget is not None else 'false'}")
        jl.seval(f"const repair_budget = {float(_budget) if _budget is not None else 0.0}")

        # Bus mappings (convert to 1-indexed)
        gen_buses_jl = [b + 1 for b in net.gen_buses]
        load_buses_jl = [b + 1 for b in net.load_buses]
        jl.seval(f"const gen_buses = {gen_buses_jl}")
        jl.seval(f"const load_buses = {load_buses_jl}")

        # Line topology (convert to 1-indexed)
        line_from = [c[0] + 1 for c in net.line_connections]
        line_to = [c[1] + 1 for c in net.line_connections]
        jl.seval(f"const line_from = {line_from}")
        jl.seval(f"const line_to = {line_to}")

        # Susceptance: b_MW/rad / S_cap = sbase_mva/(x_pu·S_cap)
        # (about 0.04–6 on the PGLib cases case14 and case118)
        jl.seval(f"const line_susceptances = {_to_julia_vec(net.line_susceptances / sbase)}")

        # Flow limits: MW divided by sbase gives per-unit
        jl.seval(f"const line_flow_min = {_to_julia_vec(net.line_flow_min / sbase)}")
        jl.seval(f"const line_flow_max = {_to_julia_vec(net.line_flow_max / sbase)}")

        # Failure probability matrices: lambda_g (N_G x T), lambda_e (N_E x T)
        jl.seval(f"const lambda_g = {_to_julia_matrix(par.lambda_g)}")
        jl.seval(f"const lambda_e = {_to_julia_matrix(par.lambda_e)}")

        max_b = float((net.line_susceptances / sbase).max())
        print(f"  Data transferred to Julia  (power_scale={sbase:.1f} MW, "
              f"max_susceptance_pu={max_b:.2f}).")

    def build_model(self, enumerate_all=True, n_samples=30):
        """Build the SDDP.jl LinearPolicyGraph model.

        enumerate_all=True enumerates all 2^(N_G+N_E) failure outcomes per stage;
        set False and provide n_samples to sample instead (faster for large networks).
        The samples are drawn once per stage when the model is built, with equal
        probabilities 1/n_samples, and stay fixed for training and simulation.
        """
        jl = self.jl
        print("  Building SDDP.jl model...")

        if enumerate_all:
            jl.seval(_ENUMERATE_REALISATIONS_CODE)
        else:
            jl.seval(_SAMPLE_REALISATIONS_CODE.format(n_samples=n_samples))

        # Pass the allocated CPU count to the optimizer inside Julia so it does
        # not size itself to the full physical node. SLURM_CPUS_PER_TASK is set
        # by SLURM; os.cpu_count() is the fallback elsewhere.
        _slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        _n_threads   = int(_slurm_cpus) if _slurm_cpus else (os.cpu_count() or 1)
        jl.seval(f"const SOLVER_THREADS = {_n_threads}")

        # Define the OPTIMIZER factory that _MODEL_DEFINITION_CODE references
        # (setup_julia has loaded the matching Julia package).
        jl.seval(self._optimizer_julia_code())

        jl.seval(_MODEL_DEFINITION_CODE)

        self._model_built = True
        print("  SDDP.jl model built successfully.")

    def train(self, iteration_limit=50, time_limit=172800.0,
              duality_handler="lagrangian", log_frequency=1):
        """Run SDDP forward-backward iterations until iteration_limit or time_limit is hit.

        These two limits are the only stopping rules; there is no gap-based stop.
        time_limit is in seconds; SDDP.jl tests it after each iteration, so
        training can run past it by up to one iteration.
        duality_handler: 'lagrangian' | 'conic' | 'strengthened' | 'bandit'
        (an unknown name falls back to 'lagrangian').
        """
        if not self._model_built:
            raise RuntimeError("Model not built yet. Call build_model() first.")

        handler_map = {
            "lagrangian": "SDDP.LagrangianDuality()",
            "conic": "SDDP.ContinuousConicDuality()",
            "strengthened": "SDDP.StrengthenedConicDuality()",
            "bandit": "SDDP.BanditDuality()",
        }
        handler_str = handler_map.get(duality_handler, "SDDP.LagrangianDuality()")

        print(f"  Training with duality_handler={duality_handler}, "
              f"iteration_limit={iteration_limit}, time_limit={time_limit}s...")

        start_time = time.time()

        self.jl.seval(f"""
SDDP.train(
    model;
    iteration_limit = {iteration_limit},
    time_limit = {time_limit},
    duality_handler = {handler_str},
    log_frequency = {log_frequency},
    print_level = 1,
    run_numerical_stability_report = false,
)
""")

        self._train_time = time.time() - start_time
        self._trained = True
        print(f"  Training completed in {self._train_time:.1f}s.")

    def save_cuts(self, path: str) -> None:
        """Write all Benders/Lagrangian cuts to a JSON file via SDDP.write_cuts_to_file."""
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        self.jl.seval(f'SDDP.write_cuts_to_file(model, "{path}")')
        print(f"  Cuts saved to: {path}")

    def lower_bound(self) -> float:
        """Return the trained policy's deterministic lower bound.

        Computed with SDDP.calculate_bound: fast, requires no simulation, and
        valid any time after train().
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        return float(self.jl.seval("SDDP.calculate_bound(model)"))

    def simulate(self, n_replications=500):
        """Monte Carlo rollout of the trained policy (n_replications trajectories).

        Records p_g, d, r_g, r_e, a_bar_g and a_bar_e per stage. extract_results()
        averages r_g, r_e, p_g and d into the expected repairs, generation and
        load served; a_bar_g and a_bar_e are not used. The upper bound comes
        from the training log, not from these trajectories.
        """
        if not self._trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        print(f"  Running {n_replications} simulation replications...")

        self.jl.seval(f"""
simulations = SDDP.simulate(
    model,
    {n_replications},
    [:p_g, :d, :r_g, :r_e, :a_bar_g, :a_bar_e],
)
""")

        self._simulated = True
        print("  Simulations completed.")

    def extract_results(self):
        """Pull training log and simulation results from Julia into a solution dict.

        lower_bound is the last SDDP.jl bound, and objective_value equals it.
        upper_bound is the cost of the last training forward pass, one sampled
        trajectory, so it is a noisy estimate of the policy's expected cost;
        gap = |UB - LB| / max(|UB|, |LB|, 1). solve_time is the training time.
        Repairs, expected generation and expected load served stay empty until
        simulate() has run; they are keyed by the stage as a string ('1' to 'T'),
        then by the 0-based component or load index as a string.
        Returned keys match the sddip_solution.json schema read by
        SDDiP_visualization.py and compare_methods.py.
        """
        jl = self.jl

        log_json_str = str(jl.seval("""
log_entries = model.most_recent_training_results.log
log_data = Dict(
    "lower_bounds" => Float64[entry.bound for entry in log_entries],
    "simulation_values" => Float64[
        entry.simulation_value for entry in log_entries
    ],
    "times" => Float64[entry.time for entry in log_entries],
)
JSON.json(log_data)
"""))
        log_data = json.loads(log_json_str)

        lower_bounds = log_data["lower_bounds"]
        sim_values = log_data["simulation_values"]
        times = log_data["times"]
        n_iterations = len(lower_bounds)

        upper_bounds = sim_values  # cost of each iteration's sampled forward pass

        gaps = []
        for lb, ub in zip(lower_bounds, upper_bounds):
            denom = max(abs(ub), abs(lb), 1.0)
            gaps.append(abs(ub - lb) / denom)

        final_lower = lower_bounds[-1] if lower_bounds else 0.0
        final_upper = upper_bounds[-1] if upper_bounds else 0.0
        final_gap = gaps[-1] if gaps else 1.0

        repairs = {"gen": {}, "line": {}}
        if self._simulated:
            repairs_json_str = str(jl.seval("""
n_sims = length(simulations)
repairs_gen = Dict{String, Dict{String, Float64}}()
repairs_line = Dict{String, Dict{String, Float64}}()

for t in 1:T
    t_key = string(t)
    repairs_gen[t_key] = Dict{String, Float64}()
    repairs_line[t_key] = Dict{String, Float64}()

    for i in 1:N_G
        g_key = string(i - 1)
        total = 0.0
        for sim in simulations
            total += sim[t][:r_g][i]
        end
        repairs_gen[t_key][g_key] = total / n_sims
    end

    for j in 1:N_E
        e_key = string(j - 1)
        total = 0.0
        for sim in simulations
            total += sim[t][:r_e][j]
        end
        repairs_line[t_key][e_key] = total / n_sims
    end
end
JSON.json(Dict("gen" => repairs_gen, "line" => repairs_line))
"""))
            repairs = json.loads(repairs_json_str)

        expected_generation = {}
        expected_load_served = {}
        if self._simulated:
            gen_load_json_str = str(jl.seval("""
n_sims = length(simulations)
exp_gen = Dict{String, Dict{String, Float64}}()
exp_load = Dict{String, Dict{String, Float64}}()

for t in 1:T
    t_key = string(t)
    exp_gen[t_key] = Dict{String, Float64}()
    exp_load[t_key] = Dict{String, Float64}()

    for i in 1:N_G
        g_key = string(i - 1)
        total = 0.0
        for sim in simulations
            total += sim[t][:p_g][i]
        end
        exp_gen[t_key][g_key] = total / n_sims
    end

    for l in 1:N_L
        l_key = string(l - 1)
        total = 0.0
        for sim in simulations
            total += sim[t][:d][l]
        end
        exp_load[t_key][l_key] = total / n_sims
    end
end
JSON.json(Dict("gen" => exp_gen, "load" => exp_load))
"""))
            gen_load = json.loads(gen_load_json_str)
            expected_generation = gen_load.get("gen", {})
            expected_load_served = gen_load.get("load", {})

            # Restore MW units: Julia variables were scaled by the same factor used
            # in _transfer_data (S_cap = max(total generation capacity, total load)).
            net = self.network
            sbase = float(max(net.gen_pmax.sum(), net.load_nominal.sum()))
            if sbase < 1.0:
                sbase = 1.0
            if sbase != 1.0:
                for t_key in expected_generation:
                    for g_key in expected_generation[t_key]:
                        expected_generation[t_key][g_key] *= sbase
                for t_key in expected_load_served:
                    for l_key in expected_load_served[t_key]:
                        expected_load_served[t_key][l_key] *= sbase

        # Total cuts across all nodes. If the count fails, assume one cut per
        # iteration at each of the first T-1 stages.
        try:
            n_cuts = int(jl.seval("""
let total_cuts = 0
    for (key, node) in model.nodes
        if hasproperty(node, :bellman_function) && !isnothing(node.bellman_function)
            total_cuts += length(node.bellman_function.global_theta.cuts)
        end
    end
    total_cuts
end
"""))
        except Exception:
            n_cuts = n_iterations * (self.params.T - 1)

        solution = {
            "objective_value": final_lower,
            "lower_bound": final_lower,
            "upper_bound": final_upper,
            "gap": final_gap,
            "iterations": n_iterations,
            "solve_time": self._train_time,
            "n_cuts": n_cuts,
            "repairs": repairs,
            "expected_generation": expected_generation,
            "expected_load_served": expected_load_served,
            "convergence": {
                "upper_bounds": upper_bounds,
                "lower_bounds": lower_bounds,
                "gaps": gaps,
                "times": times,
            },
        }
        return solution


# ---------------------------------------------------------------------------
# Julia code templates
# ---------------------------------------------------------------------------

# Both realization templates define generate_failure_realisations(t), which
# returns the stage-t outcomes and their probabilities. An outcome is a 0/1
# vector xi of length N_G + N_E (generators first, then lines); xi = 1 means the
# component fails at stage t, independently with probability lambda_g[g, t] or
# lambda_e[e, t].
_ENUMERATE_REALISATIONS_CODE = """
function generate_failure_realisations(t::Int)
    n_comp = N_G + N_E
    n_real = 2^n_comp
    omegas = Vector{Vector{Float64}}(undef, n_real)
    probs = Vector{Float64}(undef, n_real)

    for idx in 0:(n_real - 1)
        xi = Float64[(idx >> k) & 1 for k in 0:(n_comp - 1)]
        omegas[idx + 1] = xi

        p = 1.0
        for g in 1:N_G
            lam = lambda_g[g, t]
            p *= xi[g] == 1.0 ? lam : (1.0 - lam)
        end
        for e in 1:N_E
            lam = lambda_e[e, t]
            p *= xi[N_G + e] == 1.0 ? lam : (1.0 - lam)
        end
        probs[idx + 1] = p
    end
    return omegas, probs
end
"""

_SAMPLE_REALISATIONS_CODE = """
function generate_failure_realisations(t::Int)
    n_samp = {n_samples}
    omegas = Vector{{Vector{{Float64}}}}(undef, n_samp)
    probs = fill(1.0 / n_samp, n_samp)

    for s in 1:n_samp
        xi = Float64[]
        for g in 1:N_G
            push!(xi, rand() < lambda_g[g, t] ? 1.0 : 0.0)
        end
        for e in 1:N_E
            push!(xi, rand() < lambda_e[e, t] ? 1.0 : 0.0)
        end
        omegas[s] = xi
    end
    return omegas, probs
end
"""

# Stage-t subproblem. The stage-t failures xi are known when it is solved
# (SDDP.parameterize); a repair ordered at stage t enters position tau of the
# pipeline and returns the component to service at stage t + tau.
_MODEL_DEFINITION_CODE = """
model = SDDP.LinearPolicyGraph(;
    stages = T,
    sense = :Min,
    lower_bound = 0.0,
    optimizer = OPTIMIZER,
) do sp, t

    # ================================================================
    # STATE VARIABLES (all binary)
    # ================================================================
    # Component availability
    @variable(sp, a_g[i=1:N_G], SDDP.State, Bin, initial_value = 1)
    @variable(sp, a_e[j=1:N_E], SDDP.State, Bin, initial_value = 1)

    # Repair pipeline (tau positions per component)
    @variable(sp, q_g[i=1:N_G, k=1:tau], SDDP.State, Bin, initial_value = 0)
    @variable(sp, q_e[j=1:N_E, k=1:tau], SDDP.State, Bin, initial_value = 0)

    # ================================================================
    # CONTROL VARIABLES
    # ================================================================
    @variable(sp, 0 <= p_g[i=1:N_G] <= gen_pmax[i])
    @variable(sp, -pi <= theta[n=1:N_B] <= pi)
    @variable(sp, f[e=1:N_E])            # explicit line flows (limits set by the thermal constraints below)
    @variable(sp, 0 <= d[l=1:N_L] <= load_nominal[l])
    @variable(sp, r_g[i=1:N_G], Bin)
    @variable(sp, r_e[j=1:N_E], Bin)

    # ================================================================
    # AUXILIARY VARIABLES
    # ================================================================
    @variable(sp, a_bar_g[i=1:N_G], Bin)       # effective availability
    @variable(sp, a_bar_e[j=1:N_E], Bin)
    @variable(sp, 0 <= w_g[i=1:N_G] <= 1)      # repair activation (McCormick)
    @variable(sp, 0 <= w_e[j=1:N_E] <= 1)
    @variable(sp, R_g[i=1:N_G], Bin)            # repair completion signal
    @variable(sp, R_e[j=1:N_E], Bin)

    # ================================================================
    # REPAIR PIPELINE TRANSITIONS
    # ================================================================
    # Pipeline entry: new repair enters at position tau
    for i in 1:N_G
        @constraint(sp, q_g[i, tau].out == r_g[i])
    end
    for j in 1:N_E
        @constraint(sp, q_e[j, tau].out == r_e[j])
    end

    # Pipeline shift: q^(k).out = q^(k+1).in for k=1..tau-1
    for i in 1:N_G
        for k in 1:(tau - 1)
            @constraint(sp, q_g[i, k].out == q_g[i, k + 1].in)
        end
    end
    for j in 1:N_E
        for k in 1:(tau - 1)
            @constraint(sp, q_e[j, k].out == q_e[j, k + 1].in)
        end
    end

    # Repair completion signal: R = q^(1).in
    for i in 1:N_G
        @constraint(sp, R_g[i] == q_g[i, 1].in)
    end
    for j in 1:N_E
        @constraint(sp, R_e[j] == q_e[j, 1].in)
    end

    # McCormick linearization: w = (1 - a.in) * R
    for i in 1:N_G
        @constraint(sp, w_g[i] <= 1 - a_g[i].in)
        @constraint(sp, w_g[i] <= R_g[i])
        @constraint(sp, w_g[i] >= (1 - a_g[i].in) + R_g[i] - 1)
    end
    for j in 1:N_E
        @constraint(sp, w_e[j] <= 1 - a_e[j].in)
        @constraint(sp, w_e[j] <= R_e[j])
        @constraint(sp, w_e[j] >= (1 - a_e[j].in) + R_e[j] - 1)
    end

    # Availability transition: a_bar = (1 - xi) * a.in + w
    # The (1-xi) coefficient is set dynamically in parameterize.
    # Default coefficient on a.in is -1 (i.e., no failure: xi=0).
    @constraint(sp, avail_trans_g[i=1:N_G],
        a_bar_g[i] - a_g[i].in - w_g[i] == 0
    )
    @constraint(sp, avail_trans_e[j=1:N_E],
        a_bar_e[j] - a_e[j].in - w_e[j] == 0
    )

    # Link a.out = a_bar (effective availability becomes next state)
    for i in 1:N_G
        @constraint(sp, a_g[i].out == a_bar_g[i])
    end
    for j in 1:N_E
        @constraint(sp, a_e[j].out == a_bar_e[j])
    end

    # ================================================================
    # OPF CONSTRAINTS
    # ================================================================
    # Slack bus
    @constraint(sp, theta[SLACK] == 0)

    # Power balance at each bus
    for n in 1:N_B
        gen_injection = AffExpr(0.0)
        for i in 1:N_G
            if gen_buses[i] == n
                add_to_expression!(gen_injection, 1.0, p_g[i])
            end
        end

        load_withdrawal = AffExpr(0.0)
        for l in 1:N_L
            if load_buses[l] == n
                add_to_expression!(load_withdrawal, 1.0, d[l])
            end
        end

        # Net flow out of bus n, in flow form: +f on lines leaving, -f entering.
        # A failed line (f = 0) drops out of the balance automatically.
        flow_out = AffExpr(0.0)
        for e in 1:N_E
            if line_from[e] == n
                add_to_expression!(flow_out, 1.0, f[e])
            elseif line_to[e] == n
                add_to_expression!(flow_out, -1.0, f[e])
            end
        end

        @constraint(sp, gen_injection - load_withdrawal == flow_out)
    end

    # Conditional DC power-flow law (big-M). When line e is available the flow
    # equals b_e*(θ_from - θ_to); when failed the law relaxes so its endpoint
    # angles decouple. M_e = 2π*b_e bounds |b_e*Δθ| over θ ∈ [-π, π].
    for e in 1:N_E
        b_e = line_susceptances[e]
        M_e = 2 * pi * b_e
        @constraint(sp,
            f[e] - b_e * (theta[line_from[e]] - theta[line_to[e]]) <=
            M_e * (1 - a_bar_e[e]))
        @constraint(sp,
            f[e] - b_e * (theta[line_from[e]] - theta[line_to[e]]) >=
            -M_e * (1 - a_bar_e[e]))
    end

    # Generation limits: 0 <= p_g[i] <= a_bar_g[i] * pmax
    for i in 1:N_G
        @constraint(sp, p_g[i] <= a_bar_g[i] * gen_pmax[i])
        @constraint(sp, p_g[i] >= a_bar_g[i] * gen_pmin[i])
    end

    # Line thermal limits on the explicit flow variable, scaled by availability.
    # A failed line (a_bar_e = 0) is forced to zero flow.
    for e in 1:N_E
        @constraint(sp, f[e] <= a_bar_e[e] * line_flow_max[e])
        @constraint(sp, f[e] >= a_bar_e[e] * line_flow_min[e])
    end

    # Repair feasibility: can only repair failed components
    for i in 1:N_G
        @constraint(sp, r_g[i] <= 1 - a_bar_g[i])
    end
    for j in 1:N_E
        @constraint(sp, r_e[j] <= 1 - a_bar_e[j])
    end

    # Per-stage repair budget in dollars; the constraint exists only when a
    # budget is active (HAS_BUDGET).
    if HAS_BUDGET
        @constraint(sp,
            sum(gen_repair_cost[i] * r_g[i] for i in 1:N_G) +
            sum(line_repair_cost[j] * r_e[j] for j in 1:N_E) <= repair_budget)
    end

    # ================================================================
    # STOCHASTIC PARAMETERIZATION
    # ================================================================
    omegas_t, probs_t = generate_failure_realisations(t)

    SDDP.parameterize(sp, omegas_t, probs_t) do omega
        xi_g = omega[1:N_G]
        xi_e = omega[(N_G+1):(N_G+N_E)]

        # Update coefficient of a_g[i].in in availability transition.
        # Constraint in standard form: a_bar_g[i] + coeff * a_g[i].in - w_g[i] = 0
        # so coeff = -(1 - xi_g[i]).
        for i in 1:N_G
            JuMP.set_normalized_coefficient(
                avail_trans_g[i], a_g[i].in, -(1.0 - xi_g[i])
            )
        end
        for j in 1:N_E
            JuMP.set_normalized_coefficient(
                avail_trans_e[j], a_e[j].in, -(1.0 - xi_e[j])
            )
        end
    end

    # ================================================================
    # STAGE OBJECTIVE
    # ================================================================
    @stageobjective(sp,
        sum(gen_cost[i] * p_g[i] for i in 1:N_G) +
        load_shedding_cost * sum(load_nominal[l] - d[l] for l in 1:N_L) +
        sum(gen_repair_cost[i] * r_g[i] for i in 1:N_G) +
        sum(line_repair_cost[j] * r_e[j] for j in 1:N_E)
    )
end
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _to_julia_vec(arr):
    """numpy array → Julia vector literal, e.g. [1.0, 2.0, 3.0]"""
    vals = [f"{float(v)}" for v in arr]
    return "[" + ", ".join(vals) + "]"


def _to_julia_matrix(mat):
    """2D numpy array → Julia matrix literal [r0c0 r0c1; r1c0 r1c1; ...]"""
    rows = []
    for r in range(mat.shape[0]):
        row_vals = " ".join(f"{float(mat[r, c])}" for c in range(mat.shape[1]))
        rows.append(row_vals)
    return "[" + "; ".join(rows) + "]"
