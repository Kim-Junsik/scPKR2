"""Why DE-Spearman is what it is: the per-perturbation counts behind the one number.

    python scripts/de_rank.py results/runs/nEfm_comb_f*_s* --group single --group-export --gate soft --bound-mean --pred-cells 128 --results-tag ce0542
    python scripts/de_rank.py results/runs/nEfm_f*_s* --group double --gate soft --bound-mean --pred-cells 128 --results-tag ce0542

WHAT THE COLUMN IS. cell-eval's de_spearman_sig (FINDINGS 16): for every perturbation in
the scored file, count the genes pdex calls significant against the control (Wilcoxon,
FDR < 0.05, no fold-change threshold) - once on the real cells, once on the predicted -
and take the Spearman correlation of the two counts ACROSS perturbations. A perturbation
with no significant gene on either side drops out of the join.

THREE WAYS IT CAN BE LOW, and the correlations printed per fold tell them apart:

  magnitude   the model orders the perturbations' effect sizes wrongly.
              rho(pred |d|, real |d|) is low.
  conversion  the order of effect sizes is right, but the predicted cells turn it into
              counts differently from the real cells - the soft gate never emits an exact
              zero (generator.py), and 128 predicted cells meet several hundred real ones.
              rho(pred |d|, real |d|) is high while rho(pred n, real n) is low, and
              rho(pred |d|, real n) beats rho(pred n, real n).
  power       the real counts follow how many cells a perturbation has, which no model
              is told. rho(real n, real cells) is high.

|d| is the norm over the scored genes of (condition mean - control mean), from the
export's own h5ad files, so it is the effect as cell-eval saw it.

The table under each fold lists every perturbation with how much training data its genes
had: the number of TRAINING DOUBLES containing the gene, and '*' where the gene's single
was not in training. A gene with 0 and '*' was predicted from the knowledge encoder alone.

It reads only what is on disk - real_de.csv and pred_de.csv, which cell-eval writes beside
results.csv, and the export's pred.h5ad and real.h5ad - so it needs neither cell-eval nor
a GPU. Seeds of one fold share their perturbations and are averaged in the table.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_celleval import celleval_dir  # noqa: E402  the writer names the folder

CONTROL_LABEL = "non-targeting"   # src/eval/celleval.py
FDR = 0.05                        # cell-eval's default for de_spearman_sig


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def sig_counts(path: str) -> pd.DataFrame:
    """Per target: genes with FDR < 0.05, and how many of them go up."""
    de = pd.read_csv(path)
    missing = {"target", "feature", "fdr"} - set(de.columns)
    if missing:
        raise SystemExit(f"{path} has no {sorted(missing)} column; it has {list(de.columns)}")
    de = de[de["target"].astype(str) != CONTROL_LABEL]
    sig = de[de["fdr"] < FDR]
    out = sig.groupby("target").size().rename("n").to_frame()
    if "fold_change" in sig.columns:
        out["up"] = sig[sig["fold_change"] > 1].groupby("target").size()
    out["up"] = out.get("up", pd.Series(dtype=float)).fillna(0)
    return out


def effects(path: str) -> pd.DataFrame:
    """Per label: cell count, |condition mean - control mean|, fraction of exact zeros."""
    import anndata as ad
    adata = ad.read_h5ad(path)
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    labels = adata.obs["target"].astype(str).to_numpy()
    control = x[labels == CONTROL_LABEL].mean(0)
    rows = {}
    for label in np.unique(labels):
        if label == CONTROL_LABEL:
            continue
        cells = x[labels == label]
        rows[label] = {"cells": cells.shape[0],
                       "d": float(np.linalg.norm(cells.mean(0) - control)),
                       "zeros": float((cells == 0).mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def training_support(run_dir: str) -> tuple[tuple, dict]:
    """(method, fold), and per gene (training doubles containing it, single in training)."""
    import torch
    from src.data import splits
    from src.data.conventions import ConditionNaming

    config = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location="cpu",
                        weights_only=False)["config"]
    method, index = config["split"]["method"], int(config["split"]["fold"])
    fold = splits.folds(config, method)[index]
    naming = ConditionNaming.from_config(config)
    doubles = fold.get("train_doubles", fold["train"])
    held = set(fold.get("held_out_genes", ()))
    count: dict[str, int] = defaultdict(int)
    for condition in doubles:
        for gene in naming.genes(condition):
            count[gene] += 1
    support = {}
    for condition in fold["test"]:
        for gene in naming.genes(condition):
            support[gene] = (count[gene], gene not in held)
    return (method, index), support


def describe(label: str, support: dict) -> str:
    parts = []
    for gene in label.split("+"):
        n, single = support.get(gene, (0, False))
        parts.append(f"{n}{'' if single else '*'}")
    return "/".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--group", default="double", choices=["double", "single", "all"])
    parser.add_argument("--group-export", action="store_true",
                        help="the export was made per block (run_celleval.py --group)")
    parser.add_argument("--gate", default="soft")
    parser.add_argument("--bound-mean", action="store_true")
    parser.add_argument("--pred-cells", type=int, default=None)
    parser.add_argument("--results-tag", default=None)
    parser.add_argument("--no-table", action="store_true",
                        help="print the per-fold correlations only")
    args = parser.parse_args()

    runs = sorted({r for pattern in args.runs for r in (glob.glob(pattern) or [pattern])})
    export_name = celleval_dir(args.gate, None, args.bound_mean, args.pred_cells, None,
                               args.group if args.group_export else None)
    keep = {"double": lambda t: "+" in t, "single": lambda t: "+" not in t,
            "all": lambda t: True}[args.group]

    by_fold: dict[tuple, list] = defaultdict(list)
    supports: dict[tuple, dict] = {}
    for run in runs:
        export_dir = os.path.join(run, export_name)
        score_dir = os.path.join(export_dir, args.results_tag or "")
        need = [os.path.join(score_dir, "real_de.csv"), os.path.join(score_dir, "pred_de.csv"),
                os.path.join(export_dir, "pred.h5ad"), os.path.join(export_dir, "real.h5ad")]
        absent = [p for p in need if not os.path.exists(p)]
        if absent:
            print(f"  skip {os.path.basename(run)}: missing {', '.join(absent)}")
            continue
        key, support = training_support(run)
        supports[key] = support
        real_n, pred_n = sig_counts(need[0]), sig_counts(need[1])
        real_e, pred_e = effects(need[3]), effects(need[2])

        # cell-eval's join: every target significant on either side, the other filled 0.
        joined = real_n[["n"]].join(pred_n[["n"]], how="outer", lsuffix="_real",
                                    rsuffix="_pred").fillna(0)
        rho_celleval = spearman(joined["n_real"], joined["n_pred"])

        table = pd.DataFrame(index=real_e.index)
        table["cells"] = real_e["cells"]
        table["real_n"] = real_n["n"].reindex(table.index).fillna(0)
        table["real_up"] = real_n["up"].reindex(table.index).fillna(0)
        table["pred_n"] = pred_n["n"].reindex(table.index).fillna(0)
        table["pred_up"] = pred_n["up"].reindex(table.index).fillna(0)
        table["real_d"] = real_e["d"]
        table["pred_d"] = pred_e["d"].reindex(table.index)
        table["pred_zeros"] = pred_e["zeros"].reindex(table.index)
        table["real_zeros"] = real_e["zeros"]
        table = table[[keep(t) for t in table.index]]

        reported = float("nan")
        results = os.path.join(score_dir, "results.csv")
        if os.path.exists(results):
            column = pd.read_csv(results).get("de_spearman_sig")
            if column is not None and len(column):
                reported = float(column.iloc[0])
        by_fold[key].append({"run": os.path.basename(run.rstrip("/\\")), "table": table,
                             "rho": rho_celleval, "reported": reported})

    if not by_fold:
        raise SystemExit("nothing to read")

    names = ["n: pred~real", "|d|: pred~real", "pred |d|~real n", "real n~cells",
             "real n~real |d|", "pred n~pred |d|"]
    summary = []
    for key in sorted(by_fold):
        entries = by_fold[key]
        method, index = key
        print("")
        print(f"=== {method} fold {index}: {len(entries)} run(s), {args.group} block ===")
        rows = []
        for e in entries:
            t = e["table"]
            values = [spearman(t["pred_n"], t["real_n"]), spearman(t["pred_d"], t["real_d"]),
                      spearman(t["pred_d"], t["real_n"]), spearman(t["real_n"], t["cells"]),
                      spearman(t["real_n"], t["real_d"]), spearman(t["pred_n"], t["pred_d"])]
            rows.append(values)
            print(f"  {e['run']:24s} cell-eval {e['reported']:7.4f}  reproduced "
                  f"{e['rho']:7.4f}  " + "  ".join(f"{n} {v:+.3f}"
                                                   for n, v in zip(names, values)))
        summary.append(np.nanmean(np.asarray(rows, float), axis=0))

        if args.no_table:
            continue
        mean = sum(e["table"] for e in entries) / len(entries)
        spread_n = pd.concat([e["table"]["pred_n"] for e in entries], axis=1)
        mean = mean.sort_values("real_n", ascending=False)
        print("")
        print(f"  {'perturbation':18s} {'train':>7s} {'cells':>6s} {'real n':>7s} {'up':>5s} "
              f"{'pred n':>7s} {'range':>11s} {'up':>5s} {'real |d|':>9s} {'pred |d|':>9s} "
              f"{'0s real':>8s} {'0s pred':>8s}")
        for label, r in mean.iterrows():
            lo, hi = spread_n.loc[label].min(), spread_n.loc[label].max()
            print(f"  {label[:18]:18s} {describe(label, supports[key]):>7s} {r['cells']:6.0f} "
                  f"{r['real_n']:7.0f} {r['real_up']:5.0f} {r['pred_n']:7.1f} "
                  f"{f'{lo:.0f}-{hi:.0f}':>11s} {r['pred_up']:5.1f} {r['real_d']:9.3f} "
                  f"{r['pred_d']:9.3f} {r['real_zeros']:8.3f} {r['pred_zeros']:8.3f}")

    print("")
    print("=== mean over folds (each fold the mean of its seeds) ===")
    total = np.nanmean(np.asarray(summary, float), axis=0)
    for name, value in zip(names, total):
        print(f"  {name:18s} {value:+.3f}")
    print("")
    print("  train: training doubles containing each gene; '*' = its single was not in")
    print("  training. n = genes with FDR < 0.05; |d| = |condition mean - control mean|.")


if __name__ == "__main__":
    main()
