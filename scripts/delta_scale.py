"""Size and direction of each predicted effect against the real one, and what resizing would do.

    python scripts/delta_scale.py t3v0 --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/delta_scale.py t2v --block double --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/delta_scale.py t3v0 t3v1 t3v2 --ours v0_base --sweep

Read from the exported cells the tables were scored on (the soft-gate export for ours,
the 128-cell prediction for scDFM). Per condition, with D = mean(real) - mean(real control)
and P = mean(pred) - mean(pred control) on the scored genes:

  |D|        size of the real effect
  size       |P| / |D|: 1 is the right size, below 1 too small, above too large
  cos        direction agreement of P with D (Pearson delta is close to this)
  s*         the factor on P that would minimise this condition's MSE, <P,D>/<P,P>
  DS         cell-eval 0.5.42's discrimination_score_l1, recomputed: |P| against every
             condition's |D| by L1 distance, leaving out a gene named exactly like the
             condition label (cell-eval's exclude_target_gene), score 1 - rank/N with rank
             0-based. The script checks its mean against the scored results.csv.
  nearest    the condition whose |D| the prediction is closer to, when it is not its own

The exclusion is by exact name, so it is not applied evenly: our export labels a single
'BAK1' and drops gene BAK1 - for a CRISPRa single the most changed gene - while scDFM's
'BAK1+control' matches no gene name (its files carry none) and keeps it. --keep-target
recomputes ours as scDFM's is scored. Doubles ('A+B') match no gene in either.

--sweep: every prediction's effect resized by s = a (|P| / m0)^b before MSE, DS and L2 are
recomputed from the means, where m0 is the median |P| over the families given (printed,
since a rule chosen here would carry it). b = 0 is one shared factor; b > 0 enlarges large
predicted effects more than small ones. The resized mean is c + s P clipped at 0, which is
what resizing every predicted cell's change from its control cell gives on average, up to
that clipping. PD (pearson delta) does not change under s. DE-Spearman needs the cells
themselves and is not estimated.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_celleval import celleval_dir  # noqa: E402

BLOCKS = {"t1v": ["double"], "t2v": ["single", "double"],
          "t3v0": ["all"], "t3v1": ["all"], "t3v2": ["all"]}
CONTROLS = ("control", "non-targeting", "ctrl")
GRID_A = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
GRID_B = [0.0, 0.25, 0.5, 0.75, 1.0]
KEEP_TARGET = False  # --keep-target: DS over every gene, as when no gene matches a label


def key(label: str) -> str:
    return "+".join(sorted(p for p in label.split("+") if p not in CONTROLS))


def bulk(path: str):
    """(control mean, {key: (raw label, mean)}, gene names)."""
    import anndata as ad

    adata = ad.read_h5ad(path)
    column = "target" if "target" in adata.obs else "perturbation"
    labels = adata.obs[column].astype(str).to_numpy()
    X = adata.X
    out, control = {}, None
    for label in np.unique(labels):
        mean = np.asarray(X[labels == label].mean(axis=0)).ravel().astype(np.float64)
        if label in CONTROLS:
            control = mean
        else:
            out[key(label)] = (label, mean)
    if control is None:
        raise SystemExit(f"{path}: no control cells")
    return control, out, np.asarray(adata.var_names).astype(str)


def load(pred_path: str, real_path: str) -> dict:
    c_real, real, genes = bulk(real_path)
    c_pred, pred, _ = bulk(pred_path)
    if set(pred) != set(real):
        raise SystemExit(f"{pred_path}: conditions differ from {real_path}")
    names = sorted(real)
    return {"names": names, "labels": [real[n][0] for n in names], "genes": genes,
            "c_pred": c_pred, "c_real": c_real,
            "Mp": np.stack([pred[n][1] for n in names]),
            "Mr": np.stack([real[n][1] for n in names])}


def evaluate(d: dict, scale: np.ndarray | None = None) -> dict:
    P = d["Mp"] - d["c_pred"]
    if scale is not None:
        P = np.maximum(d["c_pred"] + scale[:, None] * P, 0.0) - d["c_pred"]
    D = d["Mr"] - d["c_real"]
    Mp = d["c_pred"] + P
    eff_real, eff_pred = np.abs(D), np.abs(P)
    n = len(d["names"])
    rows = {}
    for i, name in enumerate(d["names"]):
        keep = (np.ones(len(d["genes"]), bool) if KEEP_TARGET
                else d["genes"] != d["labels"][i])
        distances = np.abs(eff_real[:, keep] - eff_pred[i, keep]).sum(axis=1)
        order = np.argsort(distances, kind="stable")
        rank = int(np.where(order == i)[0][0])
        nP, nD = np.linalg.norm(P[i]), np.linalg.norm(D[i])
        rows[name] = {
            "|D|": float(nD), "|P|": float(nP),
            "size": float(nP / max(nD, 1e-12)),
            "cos": float(P[i] @ D[i] / max(nP * nD, 1e-12)),
            "s*": float(P[i] @ D[i] / max(P[i] @ P[i], 1e-12)),
            "mse": float(np.mean((Mp[i] - d["Mr"][i]) ** 2)),
            "l2": float(np.linalg.norm(Mp[i] - d["Mr"][i])),
            "DS": 1.0 - rank / n,
            "nearest": d["names"][order[0]] if rank > 0 else "",
        }
    return {"rows": rows,
            **{k: float(np.mean([r[k] for r in rows.values()])) for k in ("mse", "l2", "DS")}}


def our_runs(prefix: str, family: str, block: str, tag: str) -> list[tuple[dict, float]]:
    """(loaded export, the soft-gate DS mean cell-eval wrote) per seed."""
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{prefix}_{family}_s*")))
    if not runs:
        raise SystemExit(f"no runs results/runs/{prefix}_{family}_s*")
    folder = celleval_dir("soft", None, True, 128, None, block)
    out = []
    for run in runs:
        d = load(os.path.join(run, folder, "pred.h5ad"), os.path.join(run, folder, "real.h5ad"))
        out.append((d, scored_ds(os.path.join(run, folder, tag, "results.csv"))))
    return out


def scored_ds(path: str) -> float:
    if not os.path.exists(path):
        return float("nan")
    with open(path, newline="", encoding="utf-8") as handle:
        values = [float(r["discrimination_score_l1"]) for r in csv.DictReader(handle)
                  if r["perturbation"] not in CONTROLS]
    return float(np.mean(values))


def scdfm_run(result: str, family: str, block: str, iteration: str) -> tuple[dict, float, int]:
    found = glob.glob(os.path.join(result, family, "*", "iteration_*", "pred.h5ad"))
    by_iteration = {int(re.search(r"iteration_(\d+)", p).group(1)): os.path.dirname(p)
                    for p in found}
    if not by_iteration:
        raise SystemExit(f"no predictions under {os.path.join(result, family)}")
    chosen = max(by_iteration) if iteration == "last" else int(iteration)
    folder = by_iteration[chosen]
    scored = os.path.join(folder, f"scpkr2_{block}")
    if block != "all":
        folder = scored  # the block's subset files, written by score_scdfm.py
    d = load(os.path.join(folder, "pred.h5ad"), os.path.join(folder, "real.h5ad"))
    return d, scored_ds(os.path.join(scored, "results.csv")), chosen


def check(label: str, mine: float, scored: float) -> None:
    if np.isfinite(scored) and abs(mine - scored) > 1e-6 and not KEEP_TARGET:
        print(f"  WARNING {label}: DS recomputed {mine:.4f}, cell-eval wrote {scored:.4f} - "
              f"the sweep's DS is not cell-eval's")


def table(family: str, block: str, runs, theirs, chosen, prefix) -> None:
    ours = [evaluate(d) for d, _ in runs]
    for (d, scored), result in zip(runs, ours):
        check(f"{prefix} {family}", result["DS"], scored)
    t = evaluate(theirs[0])
    check(f"scDFM {family}", t["DS"], theirs[1])
    names = sorted(ours[0]["rows"], key=lambda n: -ours[0]["rows"][n]["|D|"])

    def avg(name: str, field: str) -> float:
        return float(np.mean([r["rows"][name][field] for r in ours]))

    width = max(len(n) for n in names) + 2
    print(f"\n=== {family} {block}: {prefix} ({len(ours)} seeds, mean) | scDFM iteration "
          f"{chosen} ===  (largest real effect first)")
    print(f"{'condition':<{width}}{'|D|':>7}{'size':>7}{'cos':>7}{'s*':>7}{'DS':>7}"
          f"  {'nearest (ours)':<22}|{'|D|':>7}{'size':>7}{'cos':>7}{'s*':>7}{'DS':>7}")
    for name in names:
        nearest = Counter(r["rows"][name]["nearest"] for r in ours if r["rows"][name]["nearest"])
        near = nearest.most_common(1)[0][0] if nearest else ""
        r = t["rows"][name]
        print(f"{name:<{width}}{avg(name, '|D|'):>7.3f}{avg(name, 'size'):>7.2f}"
              f"{avg(name, 'cos'):>7.2f}{avg(name, 's*'):>7.2f}{avg(name, 'DS'):>7.3f}"
              f"  {near[:22]:<22}|{r['|D|']:>7.3f}{r['size']:>7.2f}{r['cos']:>7.2f}"
              f"{r['s*']:>7.2f}{r['DS']:>7.3f}")
    print(f"\n  ours : MSE {np.mean([o['mse'] for o in ours]):.5f}  DS "
          f"{np.mean([o['DS'] for o in ours]):.4f}")
    print(f"  scDFM: MSE {t['mse']:.5f}  DS {t['DS']:.4f}")


def sweep(groups: list[list[tuple[dict, float]]], theirs: list[dict], prefix: str) -> None:
    """groups: one list of seeds per family. Metrics are averaged over seeds, then families."""
    norms = [np.linalg.norm(d["Mp"] - d["c_pred"], axis=1) for runs in groups for d, _ in runs]
    m0 = float(np.median(np.concatenate(norms)))
    print(f"\n=== sweep, {prefix}: s = a (|P| / m0)^b with m0 = {m0:.4f} ===")
    results = {}
    for b in GRID_B:
        for a in GRID_A:
            per_family = []
            for runs in groups:
                seeds = []
                for d, _ in runs:
                    norm = np.linalg.norm(d["Mp"] - d["c_pred"], axis=1)
                    s = np.clip(a * (norm / m0) ** b, 0.25, 4.0)
                    seeds.append(evaluate(d, s))
                per_family.append({k: np.mean([e[k] for e in seeds]) for k in ("mse", "l2", "DS")})
            results[(a, b)] = {k: float(np.mean([f[k] for f in per_family]))
                               for k in ("mse", "l2", "DS")}
    for metric, fmt in (("mse", "{:>9.5f}"), ("DS", "{:>9.4f}"), ("l2", "{:>9.4f}")):
        print(f"\n  {metric} (rows b, columns a; a=1 b=0 is the model as scored)")
        print("  " + f"{'b':>6}" + "".join(f"{a:>9.2f}" for a in GRID_A))
        for b in GRID_B:
            print("  " + f"{b:>6.2f}" + "".join(fmt.format(results[(a, b)][metric])
                                                for a in GRID_A))
    if theirs:
        t = [evaluate(d) for d in theirs]
        print(f"\n  scDFM (as scored, same families): mse {np.mean([e['mse'] for e in t]):.5f}"
              f"  DS {np.mean([e['DS'] for e in t]):.4f}  l2(128 cells) "
              f"{np.mean([e['l2'] for e in t]):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("families", nargs="+", choices=list(BLOCKS))
    parser.add_argument("--block", default=None)
    parser.add_argument("--ours", required=True, help="a val_queue.py arm, <name>_<arm>")
    parser.add_argument("--scdfm", default=None, help="result dir of a scdfm_queue.py run")
    parser.add_argument("--iteration", default="last")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--results-tag", default="ce0542")
    parser.add_argument("--keep-target", action="store_true",
                        help="DS without cell-eval's target-gene exclusion. It drops a gene "
                             "named like the label: our single 'BAK1' loses gene BAK1, "
                             "scDFM's 'BAK1+control' matches no gene and keeps it")
    args = parser.parse_args()
    global KEEP_TARGET
    KEEP_TARGET = args.keep_target

    groups, theirs = [], []
    for family in args.families:
        block = args.block or BLOCKS[family][0]
        if block not in BLOCKS[family]:
            parser.error(f"{family} has blocks {BLOCKS[family]}")
        runs = our_runs(args.ours, family, block, args.results_tag)
        groups.append(runs)
        if args.scdfm:
            d, scored, chosen = scdfm_run(args.scdfm, family, block, args.iteration)
            theirs.append(d)
            if not args.sweep:
                table(family, block, runs, (d, scored), chosen, args.ours)
        elif not args.sweep:
            parser.error("the per-condition table needs --scdfm")
    if args.sweep:
        for runs, family in zip(groups, args.families):
            for d, scored in runs:
                check(f"{args.ours} {family}", evaluate(d)["DS"], scored)
        sweep(groups, theirs, args.ours)


if __name__ == "__main__":
    main()
