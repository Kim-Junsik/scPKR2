"""Size and direction of each predicted effect against the real one, for one validation family.

    python scripts/delta_scale.py t3v0 --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1
    python scripts/delta_scale.py t2v --block double --ours v0_base --scdfm /mnt/data/dev/scDFM/result/v1

Read from the exported cells the tables were scored on (the soft-gate export for ours,
the 128-cell prediction for scDFM). With c the mean of the control cells, and per condition
D = mean(real) - c and P = mean(pred) - c on the scored genes:

  |D|        size of the real effect
  size       |P| / |D|: 1 is the right size, below 1 too small, above too large
  cos        direction agreement of P with D (Pearson delta is close to this)
  s*         the factor on P that would minimise this condition's MSE, <P,D>/<P,P>
  rank       where the condition's own real mean comes among all real means, by L1
             distance to its prediction - 1 is what the discrimination score rewards
  nearest    the real condition the prediction is closer to, when rank > 1

The summary line gives each model's MSE as scored, and its MSE had every P in the family
been multiplied by one shared factor (fitted here, on these conditions - a measure of how
much of the error is size alone, not a setting to use). Each model is compared with its
own real cells: genes and preprocessing differ between the two.
"""

from __future__ import annotations

import argparse
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


def key(label: str) -> str:
    return "+".join(sorted(p for p in label.split("+") if p not in CONTROLS))


def means(path: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import anndata as ad

    adata = ad.read_h5ad(path)
    column = "target" if "target" in adata.obs else "perturbation"
    labels = adata.obs[column].astype(str).to_numpy()
    X = adata.X
    out, control = {}, None
    for label in np.unique(labels):
        rows = X[labels == label]
        mean = np.asarray(rows.mean(axis=0)).ravel().astype(np.float64)
        if label in CONTROLS:
            control = mean
        else:
            out[key(label)] = mean
    if control is None:
        raise SystemExit(f"{path}: no control cells")
    return control, out


def measure(pred_path: str, real_path: str) -> dict:
    control, real = means(real_path)
    _, pred = means(pred_path)
    if set(pred) != set(real):
        raise SystemExit(f"{pred_path}: conditions differ from {real_path}")
    names = sorted(real)
    out = {"rows": {}, "cross": 0.0, "self": 0.0, "real": 0.0, "genes": len(control)}
    stacked = np.stack([real[n] for n in names])
    for name in names:
        D, P = real[name] - control, pred[name] - control
        distances = np.abs(stacked - pred[name]).sum(axis=1)
        order = np.argsort(distances)
        rank = int(np.where(order == names.index(name))[0][0]) + 1
        out["rows"][name] = {
            "|D|": float(np.linalg.norm(D)),
            "size": float(np.linalg.norm(P) / max(np.linalg.norm(D), 1e-12)),
            "cos": float(P @ D / max(np.linalg.norm(P) * np.linalg.norm(D), 1e-12)),
            "s*": float(P @ D / max(P @ P, 1e-12)),
            "mse": float(np.mean((P - D) ** 2)),
            "rank": rank,
            "nearest": names[order[0]] if rank > 1 else "",
        }
        out["cross"] += float(P @ D)
        out["self"] += float(P @ P)
        out["real"] += float(D @ D)
    # MSE under one shared factor s on every P: mean over conditions of |sP - D|^2 / genes.
    s = out["cross"] / max(out["self"], 1e-12)
    n, g = len(names), out["genes"]
    out["mse"] = float(np.mean([r["mse"] for r in out["rows"].values()]))
    out["s"] = s
    out["mse_s"] = float((s * s * out["self"] - 2 * s * out["cross"] + out["real"]) / (n * g))
    return out


def ours(prefix: str, family: str, block: str) -> list[dict]:
    runs = sorted(glob.glob(os.path.join(ROOT, "results", "runs", f"{prefix}_{family}_s*")))
    if not runs:
        raise SystemExit(f"no runs results/runs/{prefix}_{family}_s*")
    folder = celleval_dir("soft", None, True, 128, None, block)
    return [measure(os.path.join(r, folder, "pred.h5ad"), os.path.join(r, folder, "real.h5ad"))
            for r in runs]


def scdfm(result: str, family: str, block: str, iteration: str) -> tuple[dict, int]:
    found = glob.glob(os.path.join(result, family, "*", "iteration_*", "pred.h5ad"))
    by_iteration = {int(re.search(r"iteration_(\d+)", p).group(1)): os.path.dirname(p)
                    for p in found}
    if not by_iteration:
        raise SystemExit(f"no predictions under {os.path.join(result, family)}")
    chosen = max(by_iteration) if iteration == "last" else int(iteration)
    folder = by_iteration[chosen]
    if block != "all":
        folder = os.path.join(folder, f"scpkr2_{block}")  # written by score_scdfm.py
    return measure(os.path.join(folder, "pred.h5ad"), os.path.join(folder, "real.h5ad")), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", choices=list(BLOCKS))
    parser.add_argument("--block", default=None)
    parser.add_argument("--ours", required=True, help="a val_queue.py arm, <name>_<arm>")
    parser.add_argument("--scdfm", required=True, help="result dir of a scdfm_queue.py run")
    parser.add_argument("--iteration", default="last")
    args = parser.parse_args()
    block = args.block or BLOCKS[args.family][0]

    runs = ours(args.ours, args.family, block)
    theirs, chosen = scdfm(args.scdfm, args.family, block, args.iteration)
    names = sorted(runs[0]["rows"], key=lambda n: -runs[0]["rows"][n]["|D|"])
    if set(names) != set(theirs["rows"]):
        raise SystemExit("the two models scored different conditions")

    def avg(name: str, field: str) -> float:
        return float(np.mean([r["rows"][name][field] for r in runs]))

    width = max(len(n) for n in names) + 2
    print(f"\n=== {args.family} {block}: {args.ours} ({len(runs)} seeds, mean) | scDFM "
          f"iteration {chosen} ===  (largest real effect first)")
    print(f"{'condition':<{width}}{'|D|':>7}{'size':>7}{'cos':>7}{'s*':>7}{'rank':>6}"
          f"  {'nearest (ours)':<22}|{'|D|':>7}{'size':>7}{'cos':>7}{'s*':>7}{'rank':>6}")
    for name in names:
        nearest = Counter(r["rows"][name]["nearest"] for r in runs if r["rows"][name]["nearest"])
        near = nearest.most_common(1)[0][0] if nearest else ""
        t = theirs["rows"][name]
        print(f"{name:<{width}}{avg(name, '|D|'):>7.3f}{avg(name, 'size'):>7.2f}"
              f"{avg(name, 'cos'):>7.2f}{avg(name, 's*'):>7.2f}{avg(name, 'rank'):>6.1f}"
              f"  {near[:22]:<22}|{t['|D|']:>7.3f}{t['size']:>7.2f}{t['cos']:>7.2f}"
              f"{t['s*']:>7.2f}{t['rank']:>6d}")
    mse = np.mean([r["mse"] for r in runs])
    mse_s = np.mean([r["mse_s"] for r in runs])
    s = np.mean([r["s"] for r in runs])
    print(f"\n  ours : MSE {mse:.5f}; with one shared factor {s:.2f} on every effect: {mse_s:.5f}")
    print(f"  scDFM: MSE {theirs['mse']:.5f}; with one shared factor {theirs['s']:.2f} on every "
          f"effect: {theirs['mse_s']:.5f}")


if __name__ == "__main__":
    main()
