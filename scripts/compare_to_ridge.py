"""The bar this model has to clear: closed-form ridge, on the same split, same protocol.

    python scripts/compare_to_ridge.py results/runs/<run> --device cuda

docs/DESIGN.md C1 is settled here and nowhere else. ridge fits one vector per
perturbation in closed form and adds them; it scores 1.6690 / 1.4160 / 2.2510 / 1.8580
on the four reported blocks, which is at or ahead of scDFM on two of them and ahead of
CellFlow on three. A learned model that does not clear it has not earned the word
"learned", however much the loss fell during training.

BOTH NUMBERS COME FROM THIS ONE SCRIPT, on the same conditions, the same control sample,
the same 1,000 genes and the same cell count. Quoting a model's L2 from one pipeline
against a baseline's from another is how a comparison stops meaning anything, and the
two differ by more than the gap being measured: v2's own L2 moved 0.52 between gates and
0.13 between cell counts.

The run's own gate is used as saved. Under `soft` this reports the mean the model
states; under `sample` it reports the mean of cells drawn from it. The ridge row is a
mean vector either way, so a `sample` comparison is not like for like - prefer `soft`
for C1 and read docs/inherited/FINDINGS.md on why that gap existed in v2.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import baselines
from src.eval.conditions import condition_groups, scdfm_eval_genes
from src.eval.baselines import training_conditions
from src.eval.diagnostics import load_run, measure_transport


def ridge_l2(data, stats, train_conditions, conditions, genes, alpha):
    """The additive baseline's L2 per condition, as a mean profile against the truth.

    No cells are transported: w_A + w_B IS a mean, so the prediction is the control mean
    plus the sum, and drawing cells around it would only add variance to a quantity that
    is already exact.
    """
    fit = baselines.fit_ridge_additive(stats, train_conditions, data.perturbations,
                                       alpha=alpha)
    index, weights = fit["index"], fit["w"]
    control = stats.mean[data.control_condition][genes]
    scores = []
    for condition in conditions:
        targets = [g for g in data.naming.genes(condition)]
        if any(g not in index for g in targets):
            continue
        shift = sum(weights[index[g]][genes] for g in targets)
        scores.append(float(np.linalg.norm(control + shift
                                           - stats.mean[condition][genes])))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--gate", default="soft", choices=["soft", "hard", "sample"],
                        help="soft is the default here because the ridge row is a mean "
                             "vector, so only soft compares like with like")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=4096)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--group", default="all",
                        choices=["double", "single", "all"])
    args = parser.parse_args()

    config, data, stats, fold, model = load_run(args.run_dir, args.device, args.gate)
    rng = np.random.default_rng(config["eval"]["seed"])
    genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    conditions = {"double": groups["test doubles"],
                  "single": groups["test singles"],
                  "all": groups["test doubles"] + groups["test singles"]}[args.group]
    if not conditions:
        raise SystemExit(f"no {args.group} held-out condition in this fold")

    # The SAME set the model trained on, from the same helper, so the ridge row is not
    # quietly fitted on conditions the model never saw.
    train_conditions = training_conditions(stats, fold, config["split"]["method"])
    rows = measure_transport(model, data, stats, conditions, config, rng,
                             args.device, args.n_cells, genes=genes)
    model_scores = [r["l2"] for r in rows]
    ridge_scores = ridge_l2(data, stats, train_conditions, conditions, genes,
                            args.ridge_alpha)

    gate = args.gate
    print(f"\n{args.group} held-out conditions, {len(conditions)} of them, "
          f"gate={gate}, {args.n_cells} cells, {len(genes)} genes\n")
    print(f"  {'ridge (closed form)':24s} {np.mean(ridge_scores):.4f}"
          f"   over {len(ridge_scores)}")
    print(f"  {'this model':24s} {np.mean(model_scores):.4f}"
          f"   over {len(model_scores)}")
    gap = np.mean(model_scores) - np.mean(ridge_scores)
    print(f"  {'difference':24s} {gap:+.4f}   "
          f"{'CLEARS the bar' if gap < 0 else 'DOES NOT clear it - see C1'}")
    if gate != "soft":
        print(f"\n  gate is {gate!r}: this compares drawn cells against a mean vector. "
              f"Rerun with --gate soft for C1.")


if __name__ == "__main__":
    main()
