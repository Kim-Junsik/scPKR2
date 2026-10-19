"""How much of each learned generator lies outside its backbone graph - DESIGN_PRIOR.md H2.

    python scripts/offgraph_share.py results/runs/hK_f*_s* results/runs/hP_f*_s*
    python scripts/offgraph_share.py results/runs/hK_f*_s* --top 20

For every run, over the perturbations its training conditions contain:

    s = sum_a ||(1 - G) * A_a||^2 / sum_a ||A_a||^2        K x K block, G its own graph

Runs are then paired by (fold, seed): the KEGG backbone (generator_mask=kegg) against
its shuffle (generator_mask=permuted). H2 passes when the KEGG run's s is lower in at
least 10 of the 12 pairs on folds 0, 1, 2, 4 - a one-sided sign test, p = 0.019. Fold 3
runs are printed but do not count, because development looked at fold 3 repeatedly.

--top lists the off-graph couplings the KEGG runs share most strongly: the candidates
for pathway relations the data supports and KEGG's maps do not draw. Candidates only.
"""

from __future__ import annotations

import argparse
import os
import sys
from math import comb

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval import baselines
from src.eval.baselines import training_conditions
from src.eval.diagnostics import load_run

DECIDING_FOLDS = (0, 1, 2, 4)
NEEDED = 10


def trained_perturbations(config, data, stats, fold) -> list[int]:
    """The perturbation indices the run's training conditions contain, as build_model
    chose them - subsampling included, so a run that saw fewer conditions is measured on
    the generators it actually trained."""
    method = config["split"]["method"]
    conditions = baselines.subsample_conditions(
        training_conditions(stats, fold, method), data.naming,
        float(config["split"].get("train_condition_fraction", 1.0)),
        int(config["split"].get("train_condition_seed", 0)))
    return sorted({data.pert_index[gene] for condition in conditions
                   if not data.naming.is_control(condition)
                   for gene in data.naming.genes(condition)
                   if gene in data.pert_index})


def sign_test(wins: int, n: int) -> float:
    """P(at least `wins` of `n` fair coin flips), one-sided."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top", type=int, default=0,
                        help="list the N strongest off-graph couplings of the kegg runs")
    args = parser.parse_args()

    rows, energy, names = [], None, None
    for run_dir in args.runs:
        config, data, stats, fold, model = load_run(run_dir, device=args.device)
        response = model.modulation
        if getattr(response, "residual", None) is None:
            raise SystemExit(f"{run_dir} has no residual branch - not a soft-prior run")
        perts = trained_perturbations(config, data, stats, fold)
        with torch.no_grad():
            share = response.offgraph_share(perts)
            k = response.observable_dim
            backbone = residual = 0.0
            outside = (1.0 - response.graph_mask)[:k, :k]
            run_energy = torch.zeros(k, k)
            for pert in perts:
                a = response.generator(pert)[:k, :k].cpu()
                w = (response.residual.matrix(pert) * response.row_mask)[:k, :k].cpu()
                residual += float(w.square().sum())
                backbone += float(a.square().sum())
                run_energy += a.square() * outside.cpu()
        cfg = config["model"]
        row = {"run": os.path.basename(os.path.normpath(run_dir)),
               "kind": cfg.get("generator_mask"), "source": cfg.get("generator_mask_source"),
               "fold": int(config["split"]["fold"]), "seed": int(config["train"]["seed"]),
               "lambda": cfg.get("generator_residual_penalty"), "share": share,
               "residual": residual / backbone if backbone else float("nan")}
        rows.append(row)
        if row["kind"] == "kegg" and args.top:
            normalised = run_energy / max(backbone, 1e-12)
            energy = normalised if energy is None else energy + normalised
            names = list(model.observables.names)
        print(f"  {row['run']:18s} {row['kind']:12s} fold {row['fold']} seed {row['seed']}  "
              f"lambda {row['lambda']}  off-graph {share:.4f}  "
              f"residual/total {row['residual']:.4f}")

    by_key = {(r["kind"], r["fold"], r["seed"]): r for r in rows}
    pairs = sorted({(r["fold"], r["seed"]) for r in rows
                    if ("kegg", r["fold"], r["seed"]) in by_key
                    and ("permuted", r["fold"], r["seed"]) in by_key})
    if pairs:
        print("")
        print(f"  {'fold':>4s} {'seed':>4s} {'KEGG':>8s} {'shuffle':>8s}  KEGG lower  counts")
        print("  " + "-" * 50)
        wins = n = 0
        for fold_index, seed in pairs:
            kegg = by_key[("kegg", fold_index, seed)]["share"]
            shuffle = by_key[("permuted", fold_index, seed)]["share"]
            counts = fold_index in DECIDING_FOLDS
            lower = kegg < shuffle
            if counts:
                n += 1
                wins += int(lower)
            print(f"  {fold_index:4d} {seed:4d} {kegg:8.4f} {shuffle:8.4f}  "
                  f"{'yes' if lower else 'no':>10s}  {'yes' if counts else 'no (fold 3)'}")
        print("")
        print(f"  KEGG lower in {wins} of {n} deciding pairs; one-sided sign test "
              f"p = {sign_test(wins, n):.4f}" if n else "  no deciding pairs")
        if n == 12:
            print(f"  H2 {'PASSES' if wins >= NEEDED else 'FAILS'} "
                  f"(needs {NEEDED} of 12, DESIGN_PRIOR.md 4)")
        elif n:
            print(f"  H2 is decided on 12 pairs (folds 0, 1, 2, 4 x seeds 0, 1, 2); "
                  f"{n} here - not a verdict")

    if energy is not None and args.top:
        kegg_runs = sum(1 for r in rows if r["kind"] == "kegg")
        mean = (energy / kegg_runs).numpy()
        order = np.argsort(-mean, axis=None)[:args.top]
        print("")
        print(f"  strongest off-graph couplings over {kegg_runs} kegg run(s) "
              f"(share of each run's generator energy, averaged) - CANDIDATES only")
        for flat in order:
            i, j = np.unravel_index(flat, mean.shape)
            print(f"  {mean[i, j]:.5f}  {names[i][:45]:45s} <- {names[j][:45]}")


if __name__ == "__main__":
    main()
