# Reported numbers and figures

The two scripts in this folder rebuild the numbers and figures reported in the paper from the result files stored in [`experiments/`](../experiments/README.md). They read only stored files and do not train, solve or simulate anything. Neither needs a MIP solver, a solver license, Julia or a GPU.

| Script | What it does | Writes | Run time |
|---|---|---|---|
| [`reproduce_paper_numbers.py`](reproduce_paper_numbers.py) | Recomputes the reported numbers and compares each one with the value stated in the paper (112 checks) | nothing, apart from caching the case14 file on first use ([below](#the-case14-network-file)) | about 3 s |
| [`make_paper_figures.py`](make_paper_figures.py) | Redraws the 10 figures in [`figures/`](../figures) | `figures/`, or the folder given with `--out` | about 3 s |

## Quick start

Run with the Python environment of the [top-level README](../README.md#python-environment) active:

```bash
python paper/reproduce_paper_numbers.py                       # ends with "All checks agree with the paper."
python paper/make_paper_figures.py --out /tmp/paper_figures   # redraws the figures, leaves figures/ unchanged
python paper/make_paper_figures.py                            # overwrites the PNGs in figures/
```

[`scripts/quick_check.sh`](../scripts/quick_check.sh) runs both scripts as part of its check of a checkout and compares the redrawn figures with `figures/` ([`scripts/README.md`](../scripts/README.md)).

## `reproduce_paper_numbers.py`

The script checks the failure-model constants, the mean costs of every method in the six configurations, the cost decompositions and standard deviations quoted in the text, the deployment times, the SDDiP bounds and policy means, the exact optima on the 5-bus network and the sample-complexity statements. It prints the SAA solver statistics and the repair-cost sensitivity table for reference. [From files to reported numbers](../experiments/README.md#from-files-to-reported-numbers) gives the file and field behind each quantity.

Each checked number prints one line:

```text
  [ok] DFL (five-seed mean)                                     paper=    1,406,419  recomputed=    1,406,419
```

A check passes when the two values differ by no more than the rounding of the stated value. A failed check shows `[MISMATCH]`. The last line is `All checks agree with the paper.` with exit status 0, or `<N> check(s) disagree with the paper.` with exit status 1. A mismatch means that a result file differs from the stored one, for example after rerunning experiments into `experiments/`. Budgeted DFL training, the SAA and SDDiP runs that stopped at their time limit, and all timings change when rerun, so new runs are compared with the stored files rather than expected to pass each line.

pandapower prints a harmless note that "3 branches ... are considered as trafos" each time the script builds the case14 network.

## `make_paper_figures.py`

### The figures

| File in `figures/` | What it shows | Data |
|---|---|---|
| `failure_probability_profiles.png` | Failure probability of an operational component over the 14 stages, for the Standard and Elevated vulnerability classes | `config.py` |
| `paper/<lane>/convergence_multiseed.png` | DFL mean cost against the training size (five-seed mean, ± one across-seed std), with AlwaysRepair, NoRepair and Near-sighted MPC as horizontal lines; `<lane>` is `toy` (the 5-bus network), `case14` or `case14_scale55` (case14 at $`s = 55`$) | aggregated `evaluation.json` of every `experiments/paper_unbudgeted/dfl_scaling_multiseed/<lane>/dfl_train*/` |
| `paper/<lane>/convergence_multiseed_budget.png` | The same with the per-stage budget $`\varphi = 0.3`$ | `experiments/paper_budget0.3/dfl_scaling_multiseed/<lane>/` |
| `sensitivity/cost_vs_scale_{toy,case14}.png` | $`(\bar c_{\mathrm{AR}} - \bar c_{\mathrm{DFL}})/\bar c_{\mathrm{DFL}}`$ in % against the repair-cost scale $`s`$ (one DFL model, seed 42, 120,000 training scenarios) | `experiments/sensitivity/sensitivity_curve.txt` |
| `paper/toy/sddip_convergence.png` | SDDiP lower bound and forward-pass cost per iteration on the 5-bus network without budget, and their relative difference | `convergence` of `experiments/paper_unbudgeted/toy/SDDiP/sddip_solution.json` |

The case14 ($`s = 45`$) curves appear in the paper's tables only. The budgeted case14 ($`s = 55`$) curve ends at 80,000 training scenarios because the two larger sizes did not finish within the wall-clock limit. [From files to figures](../experiments/README.md#from-files-to-figures) lists the plotted fields.

`--out DIR` writes the same files and subfolders into `DIR`. With `requirements-lock.txt`, the redrawn figures can differ from the stored ones by a few pixels, from text rasterization. The plotted data are the same.

## The case14 network file

`reproduce_paper_numbers.py` builds the IEEE 14-bus network to price the saved SDDiP decisions on case14. The first time this happens, `network_data.py` downloads `pglib_opf_case14_ieee.m` from the master branch of PGLib-OPF and caches it in `~/.cache/pglib_opf/`. Later runs need no internet connection.

The experiments used the file of PGLib-OPF v23.07, with SHA-256 `bd5c568621de65e4b0922317010868bc7fa94173807faa10ea8fdbbe77c28106`, which is identical to the master-branch file at the time of writing. To pin this version, or to prepare a machine without internet access:

```bash
mkdir -p ~/.cache/pglib_opf
curl -L -o ~/.cache/pglib_opf/pglib_opf_case14_ieee.m \
    https://raw.githubusercontent.com/power-grid-lib/pglib-opf/v23.07/pglib_opf_case14_ieee.m
shasum -a 256 ~/.cache/pglib_opf/pglib_opf_case14_ieee.m    # on Linux: sha256sum
```

The download is not checked against a checksum. If a download was interrupted, delete the cached file and run the script again.
