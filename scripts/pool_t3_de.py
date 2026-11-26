"""Table 3's DE-Spearman over the three ComboSciPlex validation folds pooled: 15 conditions.

    python scripts/pool_t3_de.py --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/pool_t3_de.py --ours v0_base
    python scripts/pool_t3_de.py --scdfm /mnt/data/dev/scDFM/result/v1 --ours v0_base v1_x

WHY POOL. cell-eval's de_spearman_sig is one number per scored file, not per condition:
the Spearman correlation, ACROSS the file's perturbations, between the number of
significant DE genes in the real cells (de_nsig_counts_real) and in the predicted ones
(de_nsig_counts_pred). Every row of results.csv carries the same value. A validation fold
scores 5 conditions, and a rank correlation over 5 points takes a handful of values: an
untrained scDFM scored 0.9 on t3v0, trained checkpoints range from -0.87 to 0.97.

Each condition's two counts are computed inside its own fold's file - its own control
cells, its own genes - so they can be gathered without rescoring anything. The pooled
value is the same statistic over the 15 conditions of t3v0, t3v1 and t3v2, each predicted
by the model that held it out. Both models are pooled the same way.

CHECK. Before pooling, each fold's own 5 counts are run through the same correlation and
compared with the de_spearman_sig cell-eval wrote for that fold. A disagreement means the
statistic is not what this script computes, and it stops.

WHERE THE ROWS COME FROM
  --scdfm <result dir of a scdfm_queue.py run>: t3v<k>/*/iteration_<n>/scpkr2_all/results.csv,
      the 128-cell scoring of every checkpoint. One row per iteration all three folds have;
      then BEST, LAST and the mean over iterations >= --late.
  --ours <name>_<arm> (a val_queue.py arm): results/runs/<name>_<arm>_t3v<k>_s<seed>/
      celleval_sample_bounded_n128/ce0542/results.csv - the sample-gate scoring the DE
      column is read from. Pooled per seed; then the mean and spread over seeds.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_celleval import celleval_dir  # noqa: E402

FOLDS = ["t3v0", "t3v1", "t3v2"]
REAL, PRED, REPORTED = "de_nsig_counts_real", "de_nsig_counts_pred", "de_spearman_sig"
NOTED: set[tuple[str, ...]] = set()


def spearman(real: list[float], pred: list[float]) -> float:
    if len(set(real)) < 2 or len(set(pred)) < 2:
        return float("nan")  # a constant side has no ranking
    return float(spearmanr(real, pred)[0])


def read_fold(path: str) -> list[tuple[str, float, float]]:
    """(condition, real count, predicted count) per perturbation, checked against the
    fold's own de_spearman_sig."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["perturbation"] != "control"]
    if not rows:
        raise SystemExit(f"{path}: no perturbation rows")
    out = [(r["perturbation"], float(r[REAL]), float(r[PRED])) for r in rows]
    reported = float(rows[0][REPORTED])
    mine = spearman([c[1] for c in out], [c[2] for c in out])
    same = (np.isnan(mine) and np.isnan(reported)) or abs(mine - reported) < 1e-9
    if not same:
        raise SystemExit(f"{path}: Spearman of the counts is {mine:.6f}, cell-eval wrote "
                         f"{reported:.6f} - de_spearman_sig is not what this script assumes")
    return out


def pool(paths: list[str]) -> tuple[float, list[float], int]:
    """(pooled value, each fold's own value, conditions pooled)."""
    conditions, own = [], []
    for path in paths:
        rows = read_fold(path)
        own.append(spearman([c[1] for c in rows], [c[2] for c in rows]))
        conditions += rows
    names = [c[0] for c in conditions]
    twice = tuple(sorted({n for n in names if names.count(n) > 1}))
    if twice and twice not in NOTED:
        # Each fold's model held the condition out, so both predictions are kept.
        NOTED.add(twice)
        print(f"  note: held out by two folds, both predictions kept: {', '.join(twice)}")
    return (spearman([c[1] for c in conditions], [c[2] for c in conditions]), own,
            len(conditions))


def scdfm_rows(result: str, late: int) -> None:
    by_fold: dict[str, dict[int, str]] = {}
    for fold in FOLDS:
        found = glob.glob(os.path.join(result, fold, "*", "iteration_*", "scpkr2_all",
                                       "results.csv"))
        by_fold[fold] = {int(re.search(r"iteration_(\d+)", p).group(1)): p for p in found}
        if not found:
            raise SystemExit(f"no scored checkpoints under {os.path.join(result, fold)}")
    iterations = sorted(set.intersection(*(set(v) for v in by_fold.values())))
    print(f"\n=== scDFM  ({result}) ===")
    print(f"{'iteration':>10} {'n':>3} {'pooled':>8}   " + " ".join(f"{f:>7}" for f in FOLDS))
    values = {}
    for iteration in iterations:
        value, own, n = pool([by_fold[f][iteration] for f in FOLDS])
        values[iteration] = value
        print(f"{iteration:>10} {n:>3} {value:>8.4f}   " + " ".join(f"{v:>7.3f}" for v in own))
    # BEST over trained checkpoints: iteration 0 is the untrained model, which no one
    # would pick, and on 5-15 conditions it can out-rank every trained one by chance.
    finite = {i: v for i, v in values.items() if np.isfinite(v) and i > 0}
    late_values = [v for i, v in finite.items() if i >= late]
    print(f"{'BEST':>10} {'':>3} {max(finite.values()):>8.4f}   (iteration "
          f"{max(finite, key=finite.get)}; iteration 0 not counted)")
    print(f"{'LAST':>10} {'':>3} {values[iterations[-1]]:>8.4f}")
    if late_values:
        print(f"{'late mean':>10} {'':>3} {np.mean(late_values):>8.4f}   "
              f"({len(late_values)} iterations >= {late})")


def our_rows(prefix: str, folder: str) -> None:
    seeds = sorted({int(re.search(r"_s(\d+)$", p).group(1))
                    for p in glob.glob(os.path.join(ROOT, "results", "runs",
                                                    f"{prefix}_{FOLDS[0]}_s*"))})
    print(f"\n=== {prefix}  (results/runs/{prefix}_t3v<k>_s<seed>/{folder}) ===")
    if not seeds:
        print(f"  no run folders results/runs/{prefix}_{FOLDS[0]}_s* yet")
        return
    print(f"{'seed':>10} {'n':>3} {'pooled':>8}   " + " ".join(f"{f:>7}" for f in FOLDS))
    values = []
    for seed in seeds:
        paths = [os.path.join(ROOT, "results", "runs", f"{prefix}_{f}_s{seed}", folder,
                              "results.csv") for f in FOLDS]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"{seed:>10}   not scored yet: {', '.join(os.path.relpath(p, ROOT) for p in missing)}")
            continue
        value, own, n = pool(paths)
        values.append(value)
        print(f"{seed:>10} {n:>3} {value:>8.4f}   " + " ".join(f"{v:>7.3f}" for v in own))
    finite = [v for v in values if np.isfinite(v)]
    if finite:
        spread = f" +- {np.std(finite, ddof=1):.4f}" if len(finite) > 1 else ""
        print(f"{'mean':>10} {'':>3} {np.mean(finite):>8.4f}{spread}   ({len(finite)} seeds)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scdfm", nargs="*", default=[],
                        help="result dirs of scdfm_queue.py runs (holding t3v0, t3v1, t3v2)")
    parser.add_argument("--ours", nargs="*", default=[],
                        help="val_queue.py arms as <name>_<arm>, e.g. v0_base")
    parser.add_argument("--late", type=int, default=100000,
                        help="scDFM's late mean starts at this iteration")
    parser.add_argument("--results-tag", default="ce0542")
    parser.add_argument("--effect-scale", type=float, default=None,
                        help="read our runs' rescored export (run_celleval.py --effect-scale)")
    args = parser.parse_args()
    if not args.scdfm and not args.ours:
        parser.error("give --scdfm and/or --ours")
    folder = os.path.join(celleval_dir("sample", None, True, 128, None, None,
                                       args.effect_scale), args.results_tag)
    for result in args.scdfm:
        scdfm_rows(result, args.late)
    for prefix in args.ours:
        our_rows(prefix, folder)


if __name__ == "__main__":
    main()
