#!/usr/bin/env python3
"""Recompute the numbers reported in the paper from the stored results.

Run from the repository root:

    python3 paper/reproduce_paper_numbers.py

The results are read from ``experiments/``; nothing is trained or solved. Each
check prints the value stated in the paper next to the value recomputed from the
result files, and the script exits with status 1 if any of them disagree beyond
rounding. A few quantities, such as the SAA solver statistics and the
sensitivity table, are only printed.

The SDDiP policy means are recomputed from the saved mean actions; this needs
each run's network and repair costs (create_network, create_model_parameters).
network_data.py downloads case14 from PGLib-OPF on first use; apart from that
download cache, nothing is written.

Naming used in the result files: the 5-bus network is ``toy`` and the DFL policy
is stored under ``Pseudo_DFL``. The repair-cost scale s is 100 for ``toy``, 45
for ``case14`` and 55 for ``case14_scale55`` (SCALE below).
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "exact_dp"))

from exact_dp_toy import reconstruct_saved_sddip_mean  # noqa: E402
from model_parameters import create_model_parameters  # noqa: E402
from network_data import create_network  # noqa: E402

UNB, BUD = "paper_unbudgeted", "paper_budget0.3"
NETS = ["toy", "case14", "case14_scale55"]
NETWORK = {"toy": "toy", "case14": "case14", "case14_scale55": "case14"}
SCALE = {"toy": 100, "case14": 45, "case14_scale55": 55}

# Training size reported for each configuration (the best size of its sweep).
BEST = {(UNB, "toy"): 15000, (UNB, "case14"): 30000, (UNB, "case14_scale55"): 100000,
        (BUD, "toy"): 60000, (BUD, "case14"): 15000, (BUD, "case14_scale55"): 80000}

N_FAILED = 0


def load(path):
    return json.loads(Path(path).read_text())


def evaluation(variant, net, n=None):
    n = BEST[(variant, net)] if n is None else n
    return load(EXP / variant / "dfl_scaling_multiseed" / net / f"dfl_train{n}" / "evaluation.json")


def check(label, recomputed, paper, tol=0.5, fmt="{:,.0f}"):
    """Print one comparison; an absolute difference above tol counts as a mismatch."""
    global N_FAILED
    ok = abs(recomputed - paper) <= tol
    N_FAILED += not ok
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label:56s} paper={fmt.format(paper):>13s}"
          f"  recomputed={fmt.format(recomputed):>13s}")


def sweep(variant, net):
    rows = []
    for p in glob.glob(str(EXP / variant / "dfl_scaling_multiseed" / net / "dfl_train*" / "evaluation.json")):
        rows.append((int(p.split("dfl_train")[1].split("/")[0]), load(p)))
    return sorted(rows)


def section(title):
    print("=" * 100)
    print(title)


def main():
    section("Failure model constants")
    k_star = 0.1
    # For t* = 1 the failure model needs delta = (1 + K) ln(1 + K) - 1 > 0, i.e. K > K*.
    for _ in range(60):  # Newton's method on (1 + K) ln(1 + K) = 1
        k_star -= ((1 + k_star) * math.log(1 + k_star) - 1) / (math.log(1 + k_star) + 1)
    check("K* solving (1+K) ln(1+K) = 1", k_star, 0.763, tol=0.0005, fmt="{:.4f}")
    check("delta at K = 1.5", 2.5 * math.log(2.5) - 1, 1.29, tol=0.005, fmt="{:.4f}")

    section("Mean out-of-sample cost by method (unbudgeted and budgeted tables)")
    table = {
        (UNB, "toy"): (2511488, 1509596, 1445096, 2028014, 1406419, 6.8),
        (UNB, "case14"): (3614867, 2112099, 1949190, 3193445, 2271956, -7.6),
        (UNB, "case14_scale55"): (3614867, 2385736, 2107215, 3255115, 2318269, 2.8),
        (BUD, "toy"): (2511488, 1940415, 2403624, 2069534, 1520472, 21.6),
        (BUD, "case14"): (3614867, 2856136, 2600141, 3240817, 2235973, 21.7),
        (BUD, "case14_scale55"): (3614867, 3069309, 3390760, 3214677, 2340615, 23.7),
    }
    seed_std = {(BUD, "toy"): 2011, (BUD, "case14"): 68890, (BUD, "case14_scale55"): 49014}
    for (v, net), (nr, ar, mpc, saa, dfl, savings) in table.items():
        d = evaluation(v, net)
        s = load(EXP / v / net / "SAA" / "solution.json")["out_of_sample_evaluation"]
        print(f"  -- {v} / {net} (n_train = {BEST[(v, net)]:,})")
        check("NoRepair", d["NoRepair"]["mean_cost"], nr)
        check("AlwaysRepair", d["AlwaysRepair"]["mean_cost"], ar)
        check("Near-sighted MPC", d["NearSightedMPC"]["mean_cost"], mpc)
        check("SAA (nearest-child rollout)", s["mean_cost"], saa)
        check("DFL (five-seed mean)", d["Pseudo_DFL"]["mean_cost"], dfl)
        check("Savings vs AlwaysRepair, % of AR cost",
              (d["AlwaysRepair"]["mean_cost"] - d["Pseudo_DFL"]["mean_cost"]) / d["AlwaysRepair"]["mean_cost"] * 100,
              savings, tol=0.05, fmt="{:+.1f}")
        if (v, net) in seed_std:
            check("DFL across-seed std", d["Pseudo_DFL"]["std_cost"], seed_std[(v, net)])
        best_n = min(sweep(v, net), key=lambda r: r[1]["Pseudo_DFL"]["mean_cost"])[0]
        check("reported n_train is the best size of the sweep", best_n, BEST[(v, net)])

    section("Cost decompositions quoted in the text")
    decomposition = {
        (UNB, "case14"): (540798, 1714, 1231367, 857),
        (UNB, "toy"): (381209, 988, 740842, 726),
        (BUD, "case14"): (379670, 1840, 959277, 1878),
        (BUD, "toy"): (342405, 1144, 556445, 1356),
        (BUD, "case14_scale55"): (434735, 1889, 1172450, 1878),
    }
    for (v, net), (dfl_rep, dfl_shed, ar_rep, ar_shed) in decomposition.items():
        d = evaluation(v, net)
        print(f"  -- {v} / {net}")
        check("DFL repair spend ($)", d["Pseudo_DFL"]["mean_repair_cost"], dfl_rep)
        check("DFL load shed (MWh)", d["Pseudo_DFL"]["mean_load_shed_mwh"], dfl_shed)
        check("AlwaysRepair repair spend ($)", d["AlwaysRepair"]["mean_repair_cost"], ar_rep)
        check("AlwaysRepair load shed (MWh)", d["AlwaysRepair"]["mean_load_shed_mwh"], ar_shed)

    section("Across-scenario standard deviation (DFL: average over the five seeds)")
    budget_std = {"toy": (153482, 216843), "case14": (247306, 298915), "case14_scale55": (215595, 307964)}
    for v in (UNB, BUD):
        for net in NETS:
            n = BEST[(v, net)]
            d = evaluation(v, net)
            per_seed = [load(p)["Pseudo_DFL"]["std_cost"] for p in
                        glob.glob(str(EXP / v / "dfl_scaling_multiseed" / net / f"dfl_train{n}" / "seed_*" / "evaluation.json"))]
            dfl_std = float(np.mean(per_seed))
            print(f"  {v:16s} {net:15s} DFL {dfl_std:>9,.0f}   AlwaysRepair {d['AlwaysRepair']['std_cost']:>9,.0f}"
                  f"   MPC {d['NearSightedMPC']['std_cost']:>9,.0f}")
            if v == BUD:
                check(f"budgeted {net}: DFL std", dfl_std, budget_std[net][0])
                check(f"budgeted {net}: AlwaysRepair std", d["AlwaysRepair"]["std_cost"], budget_std[net][1])

    section("Deployment time per 14-stage test scenario (unbudgeted runs)")
    for net, (dfl_ms, ratio) in {"toy": (30.0, 196), "case14": (35.2, 2707)}.items():
        d = evaluation(UNB, net)
        s = load(EXP / UNB / net / "SAA" / "solution.json")["out_of_sample_evaluation"]
        t = {k: d[k]["mean_time_per_scenario_s"] for k in ("NoRepair", "AlwaysRepair", "Pseudo_DFL", "NearSightedMPC")}
        print(f"  {net:7s} NoRepair {t['NoRepair'] * 1e3:.1f} ms | AlwaysRepair {t['AlwaysRepair'] * 1e3:.1f} ms"
              f" | DFL {t['Pseudo_DFL'] * 1e3:.1f} ms | SAA {s['mean_time_per_scenario_s'] * 1e3:.1f} ms"
              f" | MPC {t['NearSightedMPC']:.1f} s")
        check(f"{net}: DFL time (ms)", t["Pseudo_DFL"] * 1e3, dfl_ms, tol=0.05, fmt="{:.1f}")
        check(f"{net}: MPC / DFL time ratio", t["NearSightedMPC"] / t["Pseudo_DFL"], ratio, tol=0.5)

    section("SAA solver statistics")
    for v in (UNB, BUD):
        for net in NETS:
            m = load(EXP / v / net / "SAA" / "solution.json")
            md = m["solver_metadata"]
            print(f"  {v:16s} {net:15s} solver {md['solver_name']}, gap {md['mip_gap_achieved'] * 100:5.1f}%,"
                  f" incumbent {m['objective_value']:>11,.0f}, {md['solve_time'] / 3600:6.1f} h"
                  f" ({md['termination_condition']})")

    section("SDDiP: lower bound, iterations and 500-simulation policy mean (SDDiP table)")
    exact = {v: load(EXP / "exact_dp_toy" / name / "exact_dp_solution.json")["solution"]["exact_optimal_value"]
             for v, name in ((UNB, "unbudgeted"), (BUD, "budgeted"))}
    sddip = {(UNB, "toy"): (1024482, 782, 1584476), (BUD, "toy"): (1296240, 755, 1651543),
             (UNB, "case14"): (671289, 158, 2110204), (BUD, "case14"): (831741, 150, 2208569),
             (UNB, "case14_scale55"): (740410, 150, 2283357), (BUD, "case14_scale55"): (927540, 147, 2304480)}
    # Price each run's saved mean actions with its own network and repair costs
    # (generator and line bases 550 s and 350 s).
    networks = {}
    for (v, net), (lb, iters, mean) in sddip.items():
        network = networks.setdefault(NETWORK[net], create_network(NETWORK[net]))
        params = create_model_parameters(network, seed=42, t=14, tau=1,
                                         gen_repair_base=550.0 * SCALE[net],
                                         line_repair_base=350.0 * SCALE[net],
                                         repair_budget_fraction=None if v == UNB else 0.3)
        sol = load(EXP / v / net / "SDDiP" / "sddip_solution.json")
        recon = reconstruct_saved_sddip_mean(sol, network, params)
        print(f"  -- {v} / {net} ({sol['solve_time'] / 3600:.1f} h, {sol['n_cuts']} cuts)")
        check("lower bound", sol["lower_bound"], lb)
        check("iterations", sol["iterations"], iters)
        check("policy mean", recon, mean)
        check("estimated gap (mean - LB) / mean, %", (recon - sol["lower_bound"]) / recon * 100,
              (mean - lb) / mean * 100, tol=0.05, fmt="{:.1f}")
        if net == "toy":
            check("policy mean above the exact optimum, %", (recon / exact[v] - 1) * 100,
                  14.6 if v == UNB else 10.6, tol=0.05, fmt="{:.1f}")
            check("lower bound below the exact optimum, %", (1 - sol["lower_bound"] / exact[v]) * 100,
                  25.9 if v == UNB else 13.2, tol=0.05, fmt="{:.1f}")

    section("Exact dynamic programming on the 5-bus network")
    check("unbudgeted exact optimum ($)", exact[UNB], 1382625)
    check("budgeted exact optimum ($)", exact[BUD], 1493319)
    for v, gap in ((UNB, 1.7), (BUD, 1.8)):
        check(f"{v}: DFL mean above the exact optimum, %",
              (evaluation(v, "toy")["Pseudo_DFL"]["mean_cost"] / exact[v] - 1) * 100, gap, tol=0.05, fmt="{:.1f}")

    section("Sample-complexity statements")
    rows = sweep(UNB, "toy")
    first = next(n for n, d in rows if d["Pseudo_DFL"]["mean_cost"] < d["AlwaysRepair"]["mean_cost"])
    check("5-bus: first size where DFL beats AlwaysRepair", first, 2000)
    print(f"  5-bus: largest across-seed std for n >= 15,000: "
          f"{max(d['Pseudo_DFL']['std_cost'] for n, d in rows if n >= 15000):,.0f} (paper: below $30k)")
    best = min(sweep(UNB, "case14_scale55"), key=lambda r: r[1]["Pseudo_DFL"]["mean_cost"])
    check("case14 s=55: lowest DFL mean ($)", best[1]["Pseudo_DFL"]["mean_cost"], 2318269)
    check("case14 s=55: size of the lowest mean", best[0], 100000)

    section("Repair-cost sensitivity (AR - DFL) / DFL, %")
    lines = (EXP / "sensitivity" / "sensitivity_curve.txt").read_text().splitlines()
    for line in lines:
        print("   " + line)
    rows = {float(l.split()[0]): l.split()[1:] for l in lines[2:] if l.split()}
    y50, y70 = float(rows[50.0][1].rstrip("%")), float(rows[70.0][1].rstrip("%"))
    print(f"  case14 zero crossing by linear interpolation between s = 50 and s = 70: s = {50 + 20 * -y50 / (y70 - y50):.1f}")

    print("=" * 100)
    print("All checks agree with the paper." if N_FAILED == 0 else f"{N_FAILED} check(s) disagree with the paper.")
    sys.exit(1 if N_FAILED else 0)


if __name__ == "__main__":
    main()
