"""Arms side by side, each the mean of its runs, scored the way the main tables are.

    python scripts/arm_table.py --group double --device cuda base='results/runs/nEp0fm_f3_s*' P='results/runs/nEp0fmP_f3_s*' D='results/runs/nEp0fmD_f3_s*' PD='results/runs/nEp0fmPD_f3_s*'
    python scripts/arm_table.py --group all --device cuda base='results/runs/cDfm_s[012]' P='results/runs/cDfmP_s*' D='results/runs/cDfmD_s*' PD='results/runs/cDfmPD_s*'

The main-table rule, per run:
- L2 over the 1,000 scanpy-HVG genes, from N_p soft cells (scDFM eq. 15, --n-cells 0).
- MSE, MAE, Pearson delta and DS from celleval_soft_bounded_n128/ce0542.
- DE-Spearman from celleval_sample_bounded_n128/ce0542.

Produce both folders first (scripts/metric_arms.sh does it after each run), with
--profile full: cell-eval's default 'minimal' has no de_spearman_sig.

The first arm is the reference: every other arm's row is followed by its difference from
it, and the last column counts the columns it improves. --per-run adds each run's row
under its arm; the seeds pair across arms. The scorer correction for the
published rows (MSE x1.042, DS +0.0126) is left out: it cancels in a difference.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_table import celleval_means, compute_l2  # noqa: E402
from run_celleval import celleval_dir  # noqa: E402

COLUMNS = [("L2", None, False), ("MSE", "mse", False), ("MAE", "mae", False),
           ("DE", "de_spearman_sig", True), ("PD", "pearson_delta", True),
           ("DS", "discrimination_score_l1", True)]


def score_run(run_dir: str, group: str, device: str, tag: str) -> dict[str, float]:
    soft = os.path.join(celleval_dir("soft", None, True, 128), tag)
    sample = os.path.join(celleval_dir("sample", None, True, 128), tag)
    soft_values = celleval_means(run_dir, "soft", group, None, True, soft) or {}
    sample_values = celleval_means(run_dir, "sample", group, None, True, sample) or {}
    row = {"L2": compute_l2(run_dir, device, 0, "soft", 1000, group, None, "beta", True)}
    for name, key, _ in COLUMNS[1:]:
        source = sample_values if name == "DE" else soft_values
        row[name] = float(source[key]) if key in source else float("nan")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arms", nargs="+", help="NAME='glob of run folders'")
    parser.add_argument("--group", default="double", choices=["double", "single", "all"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-tag", default="ce0542")
    parser.add_argument("--per-run", action="store_true",
                        help="also print each run's row under its arm (seeds pair across arms)")
    args = parser.parse_args()

    arms = []
    for spec in args.arms:
        name, _, pattern = spec.partition("=")
        runs = sorted(r for r in glob.glob(pattern.strip("'\""))
                      if os.path.exists(os.path.join(r, "checkpoint.pt")))
        if not runs:
            raise SystemExit(f"{name}: no finished run matches {pattern}")
        arms.append((name, runs))

    means = {}
    for name, runs in arms:
        rows = []
        for run in runs:
            row = score_run(run, args.group, args.device, args.results_tag)
            missing = [c for c, v in row.items() if not np.isfinite(v)]
            if missing:
                hint = ("  (scored with --profile minimal? rescore it with --score-only "
                        "--profile full)" if missing == ["DE"] else "")
                print(f"  [warn] {os.path.basename(run)} has no {', '.join(missing)}{hint}")
            rows.append(row)
        mean = {}
        for c, _, _ in COLUMNS:
            finite = [r[c] for r in rows if np.isfinite(r[c])]
            mean[c] = float(np.mean(finite)) if finite else float("nan")
        means[name] = (mean, len(runs), list(zip(runs, rows)))

    def cells(values: dict, signed: bool = False) -> str:
        sign = "+" if signed else ""
        return " ".join(format(values[c], f"{sign}9.5f" if c in ("MSE", "MAE") else f"{sign}9.4f")
                        for c, _, _ in COLUMNS)

    header = f"{'arm':8s} {'runs':>4s} " + " ".join(f"{c:>9s}" for c, _, _ in COLUMNS)
    print(header + "   better")
    print("-" * (len(header) + 9))
    reference = means[arms[0][0]][0]
    for name, _ in arms:
        values, count, per_run = means[name]
        print(f"{name:8s} {count:4d} " + cells(values))
        if name != arms[0][0]:
            better = sum(1 for c, _, high in COLUMNS
                         if np.isfinite(values[c]) and np.isfinite(reference[c])
                         and values[c] != reference[c] and (values[c] > reference[c]) == high)
            difference = {c: values[c] - reference[c] for c, _, _ in COLUMNS}
            print(f"{'  diff':8s} {'':4s} " + cells(difference, signed=True) + f"   {better}/6")
        if args.per_run:
            for run, row in per_run:
                # The part after the arm's prefix: s0, or f3_s0.
                label = "_".join(os.path.basename(run).split("_")[1:]) or os.path.basename(run)
                print(f"    {label:9s} " + cells(row))
    print("\nlower is better: L2, MSE, MAE.  higher is better: DE, PD (Pearson delta), DS.")


if __name__ == "__main__":
    main()
