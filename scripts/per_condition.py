"""One validation family, condition by condition: a val_queue.py arm against scDFM.

    python scripts/per_condition.py t3v0 --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/per_condition.py t2v --block double --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/per_condition.py t3v0 --ours v0_base --scdfm ... --iteration 140000

Where a family mean is lost, this shows which conditions lose it. Per condition, from the
results.csv files the tables are averaged from:
  MSE, PD (pearson_delta), DS (discrimination_score_l1) - ours from the soft-gate scoring,
      mean over seeds; scDFM from the checkpoint given by --iteration (default: the last);
  real/pred - significant DE genes per condition, real cells then predicted cells, once
      as our scoring counts them (sample gate, the one the DE column is read from; mean
      over seeds) and once as scDFM's does. Each model's genes and controls are its own,
      so the two real counts differ. DE-Spearman is the rank correlation of a model's real
      and pred columns across conditions.
Rows are sorted by how much more MSE ours has than scDFM, worst first. The last row is the
plain mean over conditions, which is the table's value - a check that nothing was dropped.

Labels are matched after removing 'control'/'ctrl' and sorting the parts: scDFM writes a
single 'A+control' or 'control+A', our cell-eval export writes 'A'.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_celleval import celleval_dir  # noqa: E402

BLOCKS = {"t1v": ["double"], "t2v": ["single", "double"],
          "t3v0": ["all"], "t3v1": ["all"], "t3v2": ["all"]}
SOFT = [("MSE", "mse"), ("PD", "pearson_delta"), ("DS", "discrimination_score_l1")]
NSIG_REAL, NSIG_PRED = "de_nsig_counts_real", "de_nsig_counts_pred"


def key(label: str) -> str:
    return "+".join(sorted(p for p in label.split("+") if p not in ("control", "ctrl")))


def read(path: str) -> dict[str, dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        out = {}
        for row in csv.DictReader(handle):
            if row["perturbation"] == "control":
                continue
            values = {}
            for name in [k for _, k in SOFT] + [NSIG_REAL, NSIG_PRED]:
                try:
                    values[name] = float(row[name])
                except (KeyError, TypeError, ValueError):
                    values[name] = float("nan")
            out[key(row["perturbation"])] = values
    return out


def ours(prefix: str, family: str, block: str, tag: str) -> tuple[list[dict], dict]:
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{prefix}_{family}_s*")))
    if not runs:
        raise SystemExit(f"no runs results/runs/{prefix}_{family}_s*")
    soft, sample = [], []
    for run in runs:
        for gate, bucket in (("soft", soft), ("sample", sample)):
            path = os.path.join(run, celleval_dir(gate, None, True, 128, None, block), tag,
                                "results.csv")
            if not os.path.exists(path):
                raise SystemExit(f"not scored: {os.path.relpath(path, ROOT)}")
            bucket.append(read(path))
    return soft, sample[0] if len(sample) == 1 else _mean(sample)


def _mean(tables: list[dict]) -> dict:
    out = {}
    for condition in tables[0]:
        out[condition] = {k: float(np.nanmean([t[condition][k] for t in tables]))
                          for k in tables[0][condition]}
    return out


def scdfm(result: str, family: str, block: str, iteration: str) -> tuple[dict, int]:
    found = glob.glob(os.path.join(result, family, "*", "iteration_*", f"scpkr2_{block}",
                                   "results.csv"))
    by_iteration = {int(re.search(r"iteration_(\d+)", p).group(1)): p for p in found}
    if not by_iteration:
        raise SystemExit(f"no scored checkpoints under {os.path.join(result, family)}")
    chosen = max(by_iteration) if iteration == "last" else int(iteration)
    if chosen not in by_iteration:
        raise SystemExit(f"iteration {chosen} not scored; have {sorted(by_iteration)}")
    return read(by_iteration[chosen]), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=list(BLOCKS))
    parser.add_argument("--block", default=None, help="default: the family's first block")
    parser.add_argument("--ours", required=True, help="a val_queue.py arm, <name>_<arm>")
    parser.add_argument("--scdfm", required=True, help="result dir of a scdfm_queue.py run")
    parser.add_argument("--iteration", default="last", help="scDFM checkpoint: 'last' or a number")
    parser.add_argument("--results-tag", default="ce0542")
    args = parser.parse_args()
    block = args.block or BLOCKS[args.family][0]
    if block not in BLOCKS[args.family]:
        parser.error(f"{args.family} has blocks {BLOCKS[args.family]}")

    soft_runs, sample = ours(args.ours, args.family, block, args.results_tag)
    soft = _mean(soft_runs)
    theirs, chosen = scdfm(args.scdfm, args.family, block, args.iteration)
    missing = sorted(set(soft) ^ set(theirs))
    if missing:
        raise SystemExit(f"conditions not in both: {missing}")

    conditions = sorted(soft, key=lambda c: theirs[c]["mse"] - soft[c]["mse"])
    width = max(len(c) for c in conditions) + 2
    print(f"\n=== {args.family} {block}: {args.ours} ({len(soft_runs)} seeds, mean) vs scDFM "
          f"iteration {chosen} ===")
    print(f"{'condition':<{width}}" + "".join(f"{f'{n} ours':>10}{f'{n} scDFM':>11}"
                                              for n, _ in SOFT)
          + f"{'real/ours':>11}{'pred':>6}{'real/scDFM':>12}{'pred':>6}")
    rows = conditions + ["mean"]
    for condition in rows:
        if condition == "mean":
            o = {k: np.mean([soft[c][k] for c in conditions]) for _, k in SOFT}
            t = {k: np.mean([theirs[c][k] for c in conditions]) for _, k in SOFT}
            n = ("", "", "", "")
            print("-" * (width + 21 * len(SOFT) + 35))
        else:
            o, t = soft[condition], theirs[condition]
            n = tuple(f"{v:.0f}" for v in (sample[condition][NSIG_REAL],
                                           sample[condition][NSIG_PRED],
                                           theirs[condition][NSIG_REAL],
                                           theirs[condition][NSIG_PRED]))
        mse = f"{o['mse']:>10.5f}{t['mse']:>11.5f}"
        rest = "".join(f"{o[k]:>10.4f}{t[k]:>11.4f}" for _, k in SOFT[1:])
        print(f"{condition:<{width}}{mse}{rest}{n[0]:>11}{n[1]:>6}{n[2]:>12}{n[3]:>6}")


if __name__ == "__main__":
    main()
