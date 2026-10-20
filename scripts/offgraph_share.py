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

    # Keyed by coordinate NAME, not position: folds keep slightly different pathway sets
    # (210-212 on norman), so position i is not the same pathway in two folds.
    rows, energy, seen = [], {}, {}
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
            normalised = (run_energy / max(backbone, 1e-12)).numpy()
            names = list(model.observables.names)
            rows_i, cols_j = np.nonzero(normalised)
            for i, j in zip(rows_i.tolist(), cols_j.tolist()):
                key = (names[i], names[j])
                energy[key] = energy.get(key, 0.0) + float(normalised[i, j])
                seen[key] = seen.get(key, 0) + 1
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

    if energy and args.top:
        kegg_runs = sum(1 for r in rows if r["kind"] == "kegg")
        # Averaged over EVERY kegg run, a run without the pair counting as zero: a
        # coupling only some folds can even express is weaker evidence, and this keeps it so.
        ranked = sorted(energy.items(), key=lambda item: -item[1])[:args.top]
        print("")
        print(f"  strongest off-graph couplings over {kegg_runs} kegg run(s) "
              f"(share of each run's generator energy, averaged) - CANDIDATES only")
        for (name_i, name_j), total in ranked:
            print(f"  {total / kegg_runs:.5f}  in {seen[(name_i, name_j)]:2d}/{kegg_runs} runs  "
                  f"{name_i[:42]:42s} <- {name_j[:42]}")


if __name__ == "__main__":
    main()
