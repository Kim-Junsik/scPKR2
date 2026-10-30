"""Score a run on the Virtual Cell Challenge 2025 test set the way its leaderboard did.

    python scripts/vcc_celleval.py export results/runs/vEp0fm_s0 --gate sample --bound-mean --device cuda --root nvcc/data/vcc/2025
    python scripts/vcc_celleval.py baseline --root nvcc/data/vcc/2025
    python scripts/vcc_celleval.py score results/runs/vEp0fm_s0/vcc_sample_bounded --celleval-python .env-ce-scdfm/bin/python --threads 16 --root nvcc/data/vcc/2025
    python scripts/vcc_celleval.py score results/vcc_cellmean --celleval-python .env-ce-scdfm/bin/python --threads 16 --root nvcc/data/vcc/2025
    python scripts/vcc_celleval.py table results/runs/vEp0fm_s0/vcc_sample_bounded --base results/vcc_cellmean

--root is where the release sits: data/vcc/2025 by default, nvcc/data/vcc/2025 on the
server.

WHAT THE LEADERBOARD SCORED. cell-eval's 'vcc' profile: three metrics over all 18,080
genes, each averaged over the 100 test perturbations.

    DES  overlap_at_N             DE genes recovered (Wilcoxon against the controls, BH)
    PDS  discrimination_score_l1  1 - rank/T of the true effect among all T true effects
    MAE  mae                      mean absolute error of the mean profile

`cell-eval score` puts each against the cell-mean baseline: (u - b) / (1 - b) for DES
and PDS, 1 - u / b for MAE, NaN to 0, clipped below at 0. Overall is 100 x their mean.
With DES_b 0.097, PDS_b 0.509 and the MAE term at 0 for every team, that formula
reproduces the published top ten to 0.1 point (fitted 2026-10-30 from the Final
Results tab).

THE FILES.
- pred.h5ad holds, for each test perturbation, as many cells as the release's
  pert_counts_Test.csv. Each is a training control cell, transported by the model on
  the model's genes and left as it was on the rest. The control cells themselves
  follow.
- The three released files carry the same 38,176 control cells, identical in every
  count and in the same order. They are therefore read from the test file by default
  (--controls), which scoring needs anyway. Only test/ of the release has to be on the
  machine.
- Values are log1p(count / library * S), with S the training median library, which is
  the cache's recipe. cell-eval leaves log-normalised input alone.
- The real side is the released test file as it is: raw counts, which cell-eval
  normalises itself.

THE BASELINE. This is cell-eval's own cell-mean baseline (_baseline.py): the unweighted
mean of the training perturbations' mean profiles and the control's. Every predicted
cell gets that same vector, and the control cells are appended.
- It is scored through this same pipeline, so the comparison does not mix two scorers.
- Its DES and PDS, set against 0.097 and 0.509, also say whether this pipeline
  reproduces the leaderboard's scoring at all.

Both exports are dense float32, gzip-compressed on disk. cell-eval holds each in memory
whole, about 12.4 GB, beside the real file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# run_celleval keeps its module level light on purpose, so this import costs nothing.
from run_celleval import SCORING_SCRIPT, check_runnable, find_celleval_python

PERT_COL = "target_gene"
CONTROL = "non-targeting"
TRAIN_FILE = "train/adata_Training.h5ad"
TEST_FILE = "test/adata_Test.h5ad"
COUNTS_FILE = "test/pert_counts_Test.csv"
# Fitted from the leaderboard's Final Results tab; see the docstring.
LEADERBOARD_BASE = {"DES": 0.097, "PDS": 0.509}
METRICS = {"DES": ("overlap_at_N", "one"),
           "PDS": ("discrimination_score_l1", "one"),
           "MAE": ("mae", "zero")}


def require(root: str, relative: str) -> str:
    """The path of one released file, or a message that says where to point --root."""
    path = os.path.join(root, relative)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}. --root names the folder holding the "
                         f"release's test/ (and train/, for --controls train); on the "
                         f"server it is nvcc/data/vcc/2025 (scripts/vcc_prepare.py's "
                         f"docstring)")
    return path


def control_cells(root: str, median: float, source: str, chunk: int = 20000):
    """The release's control cells over every gene, normalised as the cache is.

    `source` names the file they are read from. The test file is the default because
    scoring needs that file anyway, so the 15.5 GB training file need not be on the
    machine at all. The three files carry the same 38,176 control cells, identical in
    every count (compared in full, 2026-10-30), so either source gives the same values.

    The whole file is streamed rather than the control rows read one by one. They are
    spread through it, and one read per row of a chunked HDF5 dataset is far slower than
    a pass over all of it.
    """
    import scipy.sparse as sp
    from vcc_prepare import CountFile

    relative = TEST_FILE if source == "test" else TRAIN_FILE
    handle = CountFile(require(root, relative))
    genes = handle.genes
    pieces = []
    t0 = time.time()
    for start, stop, matrix in handle.chunks(chunk):
        keep = np.flatnonzero(handle.target[start:stop] == CONTROL)
        if keep.size:
            block = matrix[keep]
            library = np.asarray(block.sum(axis=1)).ravel()
            block = (sp.diags(median / library) @ block).tocsr()
            block.data = np.log1p(block.data)
            pieces.append(block.toarray().astype(np.float32))
        print(f"  reading controls  {stop:,}/{handle.shape[0]:,}  {time.time() - t0:.0f}s",
              flush=True)
    handle.close()
    return genes, np.concatenate(pieces)


def write_h5ad(path: str, labels: list[str], genes, blocks) -> None:
    """Write X dense, block by block, so the matrix never sits in memory whole."""
    import anndata as ad
    import h5py

    n = len(labels)
    obs = pd.DataFrame({PERT_COL: pd.Categorical(labels)},
                       index=[f"c{i}" for i in range(n)])
    ad.AnnData(obs=obs, var=pd.DataFrame(index=[str(g) for g in genes])).write_h5ad(path)
    row = 0
    with h5py.File(path, "r+") as out:
        if "X" in out:
            del out["X"]
        x = out.create_dataset("X", shape=(n, len(genes)), dtype=np.float32,
                               chunks=(min(256, n), len(genes)),
                               compression="gzip", compression_opts=1)
        x.attrs["encoding-type"] = "array"
        x.attrs["encoding-version"] = "0.2.0"
        for block in blocks:
            x[row:row + block.shape[0]] = block
            row += block.shape[0]
    if row != n:
        raise SystemExit(f"wrote {row:,} rows for {n:,} labels")


def read_counts(root: str) -> pd.DataFrame:
    counts = pd.read_csv(require(root, COUNTS_FILE))
    if CONTROL in set(counts["target_gene"]):
        counts = counts[counts["target_gene"] != CONTROL]
    return counts.reset_index(drop=True)


def export_model(args) -> None:
    from src.eval.diagnostics import load_run
    from src.eval.predict import predict_cells

    overrides = {}
    if args.gate == "sample":
        overrides["realisation"] = "beta"
    if args.bound_mean:
        overrides["bound_mean"] = True
    config, data, stats, fold, model = load_run(
        args.run_dir, args.device, gate=args.gate, eval_overrides=overrides or None)
    if config["split"].get("source") != "vcc":
        raise SystemExit(f"{args.run_dir} was not trained on the VCC split")
    if config["split"].get("validation"):
        raise SystemExit(f"{args.run_dir} is a validation run; its held-out genes are "
                         f"not the test set")

    store = np.load(args.means, allow_pickle=False)
    genes, controls = control_cells(args.root, float(store["library_median"]),
                                    args.controls)
    column = {str(g): i for i, g in enumerate(genes)}
    model_columns = np.array([column[str(g)] for g in data.gene_names])
    condition_of = {str(data.naming.genes(c)[0]): c for c in fold["test"]}
    counts = read_counts(args.root)
    missing = [g for g in counts["target_gene"] if g not in condition_of]
    if missing:
        raise SystemExit(f"test genes the run does not hold out: {missing[:5]}")

    rng = np.random.default_rng(config["eval"]["seed"])
    highest = [0.0]

    def blocks():
        for i, (gene, n) in enumerate(zip(counts["target_gene"], counts["n_cells"])):
            pick = rng.choice(controls.shape[0], size=int(n), replace=False)
            block = controls[pick]
            predicted = predict_cells(model, np.ascontiguousarray(block[:, model_columns]),
                                      condition_of[gene], data.pert_index, args.device,
                                      data.naming)
            if not np.isfinite(predicted).all() or (predicted < 0).any():
                raise SystemExit(f"{gene}: the prediction holds non-finite or negative "
                                 f"values")
            highest[0] = max(highest[0], float(predicted.max()))
            block[:, model_columns] = predicted
            print(f"  {i + 1:3d}/{len(counts)}  {gene:12s} {int(n):5d} cells", flush=True)
            yield block
        yield controls

    out_dir = os.path.join(args.run_dir, f"vcc_{args.gate}"
                           + ("_bounded" if args.bound_mean else ""))
    os.makedirs(out_dir, exist_ok=True)
    labels = [g for g, n in zip(counts["target_gene"], counts["n_cells"])
              for _ in range(int(n))] + [CONTROL] * controls.shape[0]
    path = os.path.join(out_dir, "pred.h5ad")
    write_h5ad(path, labels, genes, blocks())
    with open(os.path.join(out_dir, "export.json"), "w") as handle:
        json.dump({"run": os.path.abspath(args.run_dir), "gate": args.gate,
                   "bound_mean": args.bound_mean, "cells": len(labels),
                   "model_genes": int(len(model_columns)), "genes": int(len(genes)),
                   "max_predicted": highest[0]}, handle, indent=2)
    print(f"-> {path}: {len(labels):,} cells x {len(genes):,} genes, "
          f"{len(model_columns):,} of them predicted, highest value {highest[0]:.2f}")


def export_baseline(args) -> None:
    store = np.load(args.means, allow_pickle=False)
    sets = store["vcc_set"].astype(str)
    # The 150 training perturbations and the control, each weighted once, as
    # cell-eval's _build_pert_baseline does with as_delta=False.
    rows = np.isin(sets, ["train", "control"])
    vector = store["means"][rows].astype(np.float64).mean(axis=0).astype(np.float32)
    genes, controls = control_cells(args.root, float(store["library_median"]),
                                    args.controls)
    if not np.array_equal(np.asarray(genes, dtype=str),
                          np.asarray(store["gene_names"], dtype=str)):
        raise SystemExit("the means file and the training file disagree on the genes")
    counts = read_counts(args.root)

    def blocks():
        for n in counts["n_cells"]:
            yield np.repeat(vector[None, :], int(n), axis=0)
        yield controls

    os.makedirs(args.out, exist_ok=True)
    labels = [g for g, n in zip(counts["target_gene"], counts["n_cells"])
              for _ in range(int(n))] + [CONTROL] * controls.shape[0]
    path = os.path.join(args.out, "pred.h5ad")
    write_h5ad(path, labels, genes, blocks())
    with open(os.path.join(args.out, "export.json"), "w") as handle:
        json.dump({"baseline": "cell mean", "conditions_averaged": int(rows.sum()),
                   "cells": len(labels)}, handle, indent=2)
    print(f"-> {path}: {len(labels):,} cells, the mean of {int(rows.sum())} "
          f"training profiles")


def score(args) -> None:
    interpreter = find_celleval_python(args.celleval_python)
    check_runnable(interpreter)
    pred = os.path.abspath(os.path.join(args.export_dir, "pred.h5ad"))
    if not os.path.exists(pred):
        raise SystemExit(f"missing {pred} - run the export step first")
    real = os.path.abspath(args.real or require(args.root, TEST_FILE))
    if not os.path.exists(real):
        raise SystemExit(f"missing {real}")
    out_dir = os.path.abspath(os.path.join(args.export_dir, args.results_tag))
    os.makedirs(out_dir, exist_ok=True)
    script = os.path.join(out_dir, "_score.py")
    with open(script, "w", encoding="utf-8") as handle:
        handle.write(SCORING_SCRIPT)
    print(f"scoring {pred} against {real} (profile vcc) ...", flush=True)
    completed = subprocess.run(
        [interpreter, script, pred, real, out_dir, "vcc", CONTROL, PERT_COL,
         str(args.threads)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(completed.stdout)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"cell-eval failed (exit {completed.returncode})")
    print(f"-> {out_dir}")


def read_scores(export_dir: str, tag: str) -> dict[str, float]:
    path = os.path.join(export_dir, tag, "agg_results.csv")
    if not os.path.exists(path):
        raise SystemExit(f"no scores at {path} - run the score step first")
    row = pd.read_csv(path).set_index("statistic").loc["mean"]
    return {key: float(row[name]) for key, (name, _) in METRICS.items()}


def scaled(user: float, base: float, best: str) -> float:
    """cell-eval score's normalisation, before the clip."""
    value = (user - base) / (1.0 - base) if best == "one" else 1.0 - user / base
    return 0.0 if np.isnan(value) else float(value)


def table(args) -> None:
    model = read_scores(args.export_dir, args.results_tag)
    base = read_scores(args.base, args.results_tag)
    leaderboard = {**LEADERBOARD_BASE, "MAE": base["MAE"]}
    print("")
    print(f"  {'metric':6s} {'model':>9s} {'baseline':>9s} {'board b':>9s} "
          f"{'vs base':>9s} {'vs board':>9s}")
    print("  " + "-" * 56)
    local, board = [], []
    for key, (_, best) in METRICS.items():
        a = scaled(model[key], base[key], best)
        b = scaled(model[key], leaderboard[key], best)
        local.append(max(a, 0.0))
        board.append(max(b, 0.0))
        print(f"  {key:6s} {model[key]:9.4f} {base[key]:9.4f} {leaderboard[key]:9.4f} "
              f"{a:+9.4f} {b:+9.4f}")
    print("  " + "-" * 56)
    print(f"  Overall against this pipeline's baseline     {100 * np.mean(local):6.1f}")
    print(f"  Overall against the leaderboard's DES/PDS b  {100 * np.mean(board):6.1f}")
    print("")
    print("  The scaled columns are before the clip at 0; Overall clips each at 0, as")
    print("  cell-eval score does. 'board b' for MAE is this pipeline's baseline: the")
    print("  leaderboard's own MAE baseline cannot be recovered from its table.")
    print(f"  This pipeline's baseline against the leaderboard's: DES {base['DES']:.4f} vs "
          f"{LEADERBOARD_BASE['DES']}, PDS {base['PDS']:.4f} vs {LEADERBOARD_BASE['PDS']}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # On every step, after its name, so `export <run> --root ...` reads the way it is typed.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="data/vcc/2025",
                        help="the released files, as scripts/vcc_prepare.py reads them; "
                             "on the server they sit at nvcc/data/vcc/2025")
    common.add_argument("--means", default="data/vcc/vcc2025_means.npz")
    common.add_argument("--controls", default="test", choices=["test", "train"],
                        help="the released file the control cells are read from. They "
                             "are the same cells in both; test by default, so the "
                             "training file is not needed")
    steps = parser.add_subparsers(dest="step", required=True)

    export = steps.add_parser("export", parents=[common],
                              help="the model's prediction, as pred.h5ad")
    export.add_argument("run_dir")
    export.add_argument("--gate", default="sample", choices=["soft", "sample"],
                        help="sample by default: DES is a DE test on cells, and the soft "
                             "gate emits no zeros")
    export.add_argument("--bound-mean", action="store_true")
    export.add_argument("--device", default="cuda")

    baseline = steps.add_parser("baseline", parents=[common],
                                help="cell-eval's cell-mean baseline")
    baseline.add_argument("--out", default="results/vcc_cellmean")

    scoring = steps.add_parser("score", parents=[common],
                               help="run cell-eval's vcc profile on an export")
    scoring.add_argument("export_dir")
    scoring.add_argument("--real", default=None,
                         help="the released test file; defaults to <root>/" + TEST_FILE)
    scoring.add_argument("--celleval-python", default=None)
    scoring.add_argument("--threads", type=int, default=1)
    scoring.add_argument("--results-tag", default="ce0542")

    tabulate = steps.add_parser("table", help="DES, PDS, MAE and Overall, as the "
                                              "leaderboard computes them")
    tabulate.add_argument("export_dir")
    tabulate.add_argument("--base", default="results/vcc_cellmean")
    tabulate.add_argument("--results-tag", default="ce0542")

    args = parser.parse_args()
    {"export": export_model, "baseline": export_baseline, "score": score,
     "table": table}[args.step](args)


if __name__ == "__main__":
    main()
