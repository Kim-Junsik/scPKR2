"""Score scDFM's saved predictions the way scPKR2's are scored: one row per checkpoint.

    python scripts/scdfm/score_scdfm.py /mnt/data/dev/scDFM/result/val/t1v --blocks double --threads 16
    python scripts/scdfm/score_scdfm.py /mnt/data/dev/scDFM/result/val/t2v --blocks single double --threads 16
    python scripts/scdfm/score_scdfm.py /mnt/data/dev/scDFM/result/val/t3v0 --blocks all --threads 16

The patched scDFM run with SCDFM_NO_EVAL=1 writes pred.h5ad and real.h5ad into every
iteration_<n>/ folder it saves (scripts/scdfm/scdfm_val.patch). This scores each of them
with the scorer scPKR2's tables use - run_celleval.run_scorer: cell-eval 0.5.42 and pdex
0.1.21 in their own interpreter, --profile full, pdex's shared matrix kept as a file so a
small /dev/shm cannot kill it - and averages the per-condition rows the way
paper_table.celleval_means does. Finished scorings are skipped, so a rerun picks up the
checkpoints saved since.

BLOCKS. Table 2 scores Single and Double apart because DE-Spearman and DS are computed over
every perturbation in the scored file; a block here is the same thing - the block's
perturbations plus the control cells, written to their own pair of files and scored alone.
A single is a condition one of whose two parts is 'control'.

L2 is computed here from the same files: the mean over conditions of ||mean(pred) -
mean(real)|| on the scored genes. It is read at however many cells were predicted - 128
during training, N_p from a SCDFM_PRED_CELLS=0 pass (scDFM eq. 15) - and the n column says
which.

The last two rows are BEST (each column's best TRAINED checkpoint, separately - the strictest
target, since no single checkpoint need reach all of them) and LAST.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_celleval import run_scorer  # noqa: E402

COLUMNS = [("MSE", "mse", False), ("MAE", "mae", False), ("DE", "de_spearman_sig", True),
           ("PD", "pearson_delta", True), ("DS", "discrimination_score_l1", True)]


def block_of(label: str) -> str | None:
    if label == "control":
        return None
    parts = label.split("+")
    return "single" if "control" in parts else "double"


def block_files(folder: str, block: str) -> tuple[str, str, str]:
    """(pred, real, outdir) for one block, subset files written once."""
    import anndata as ad

    outdir = os.path.join(folder, f"scpkr2_{block}")
    os.makedirs(outdir, exist_ok=True)
    if block == "all":
        return os.path.join(folder, "pred.h5ad"), os.path.join(folder, "real.h5ad"), outdir
    paths = []
    for name in ("pred", "real"):
        path = os.path.join(outdir, f"{name}.h5ad")
        if not os.path.exists(path):
            adata = ad.read_h5ad(os.path.join(folder, f"{name}.h5ad"))
            labels = adata.obs["perturbation"].astype(str)
            keep = (labels == "control") | (labels.map(block_of) == block)
            adata[keep.to_numpy()].copy().write_h5ad(path)
        paths.append(path)
    return paths[0], paths[1], outdir


def l2_and_cells(pred_path: str, real_path: str) -> tuple[float, int]:
    import anndata as ad

    pred, real = ad.read_h5ad(pred_path), ad.read_h5ad(real_path)
    distances, cells = [], []
    pred_labels = pred.obs["perturbation"].astype(str).to_numpy()
    real_labels = real.obs["perturbation"].astype(str).to_numpy()
    def dense(matrix):
        return (matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
                ).astype(np.float64)

    for label in np.unique(real_labels):
        if label == "control":
            continue
        p = dense(pred.X[pred_labels == label])
        r = dense(real.X[real_labels == label])
        distances.append(float(np.linalg.norm(p.mean(axis=0) - r.mean(axis=0))))
        cells.append(len(p))
    return float(np.mean(distances)), int(np.median(cells))


def condition_means(outdir: str) -> dict[str, float]:
    path = os.path.join(outdir, "results.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for name, key, _ in COLUMNS:
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        out[name] = float(np.mean(values)) if values else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", help="the --result_path of one scDFM run")
    parser.add_argument("--blocks", nargs="+", default=["all"],
                        choices=["all", "single", "double"])
    parser.add_argument("--celleval-python", default=".env-ce-scdfm/bin/python")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    interpreter = args.celleval_python
    if not os.path.isabs(interpreter):
        interpreter = os.path.join(ROOT, interpreter)
    found = glob.glob(os.path.join(args.result_dir, "**", "iteration_*", "pred.h5ad"),
                      recursive=True)
    folders = sorted((os.path.dirname(p) for p in found),
                     key=lambda f: int(re.search(r"iteration_(\d+)$", f).group(1)))
    if not folders:
        raise SystemExit(f"no iteration_*/pred.h5ad under {args.result_dir}")

    for block in args.blocks:
        rows = []
        for folder in folders:
            iteration = int(re.search(r"iteration_(\d+)$", folder).group(1))
            pred, real, outdir = block_files(folder, block)
            if not os.path.exists(os.path.join(outdir, "results.csv")):
                print(f"  scoring {block} iteration {iteration} ...", flush=True)
                done = run_scorer(interpreter, outdir, [pred, real, outdir, "full", "control",
                                                        "perturbation", str(args.threads)])
                if done.returncode != 0 or not os.path.exists(os.path.join(outdir, "results.csv")):
                    print(done.stdout[-2000:], done.stderr[-4000:], sep="\n")
                    raise SystemExit(f"scoring failed: {outdir}")
            l2, cells = l2_and_cells(pred, real)
            rows.append({"iteration": iteration, "n": cells, "L2": l2, **condition_means(outdir)})

        names = ["L2"] + [name for name, _, _ in COLUMNS]
        higher = {"L2": False, **{name: up for name, _, up in COLUMNS}}
        print(f"\n=== {block}  ({args.result_dir}) ===")
        print(f"{'iteration':>10} {'n':>5} " + " ".join(f"{n:>8}" for n in names))
        for row in rows:
            print(f"{row['iteration']:>10} {row['n']:>5} "
                  + " ".join(f"{row[n]:>8.4f}" for n in names))
        # Trained checkpoints only: iteration 0 is the untrained model, and on DE-Spearman
        # over a few conditions it can out-rank every trained one by chance (t3v0: 0.9).
        trained = [r for r in rows if r["iteration"] > 0] or rows
        best = {n: (max if higher[n] else min)(r[n] for r in trained) for n in names}
        print(f"{'BEST':>10} {'':>5} " + " ".join(f"{best[n]:>8.4f}" for n in names)
              + "   (iteration 0 not counted)")
        print(f"{'LAST':>10} {rows[-1]['n']:>5} " + " ".join(f"{rows[-1][n]:>8.4f}" for n in names))

        summary = os.path.join(args.result_dir, f"scpkr2_scores_{block}.csv")
        with open(summary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["iteration", "n"] + names)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {summary}")


if __name__ == "__main__":
    main()
