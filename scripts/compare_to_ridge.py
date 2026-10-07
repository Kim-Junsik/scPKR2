"""The bar this model has to clear: closed-form ridge, on the same split, same protocol.

    python scripts/compare_to_ridge.py results/runs/<run>... --device cuda

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

PASS EVERY FOLD. The reported protocol holds each of the five out in turn and averages
them, and the folds are not equally hard - ridge scores 1.5473 on norman additive fold 0
against 1.6690 over all five. A single fold is a direction while a design is being
changed; it is not a C1 verdict.

The gate defaults to soft. The ridge row is a mean vector, so only soft compares like
with like; under sample this would put drawn cells against a mean, and the difference
would be the realisation's rather than the model's.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import baselines
from src.eval.baselines import training_conditions
from src.eval.conditions import condition_groups, scdfm_eval_genes
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
        targets = list(data.naming.genes(condition))
        if any(g not in index for g in targets):
            continue
        shift = sum(weights[index[g]][genes] for g in targets)
        scores.append(float(np.linalg.norm(control + shift
                                           - stats.mean[condition][genes])))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+",
                        help="one run, or every fold of a split; the reported protocol "
                             "holds each fold out in turn and averages them")
    parser.add_argument("--gate", default="soft", choices=["soft", "hard", "sample"],
                        help="soft by default because the ridge row is a mean vector")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=4096)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--group", default="all",
                        choices=["double", "single", "all"])
    args = parser.parse_args()

    print("")
    print(f"{args.group} held-out conditions, gate={args.gate}, "
          f"{args.n_cells} cells, {args.infer_top_gene} genes")
    print("")
    print(f"  {'run':34s} {'ridge':>8s} {'model':>8s} {'diff':>8s}  n")
    print("  " + "-" * 64)

    # PER FOLD, then averaged over folds. Pooling the conditions instead would weight a
    # fold by how many held-out conditions it happens to have, while the protocol is a
    # mean over folds.
    ridge_folds, model_folds = [], []
    for run_dir in args.run_dirs:
        config, data, stats, fold, model = load_run(run_dir, args.device, args.gate)
        rng = np.random.default_rng(config["eval"]["seed"])
        genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
        groups = condition_groups(data, stats, fold, config["split"]["method"])
        conditions = {"double": groups["test doubles"],
                      "single": groups["test singles"],
                      "all": groups["test doubles"] + groups["test singles"]}[args.group]
        name = os.path.basename(os.path.normpath(run_dir))[:34]
        if not conditions:
            print(f"  {name:34s} {'no ' + args.group + ' held out':>26s}")
            continue
        # The SAME set the model trained on, from the same helper, so the ridge row is
        # not quietly fitted on conditions the model never saw.
        train_conditions = training_conditions(stats, fold, config["split"]["method"])
        rows = measure_transport(model, data, stats, conditions, config, rng,
                                 args.device, args.n_cells, genes=genes)
        this_model = float(np.mean([r["l2"] for r in rows]))
        this_ridge = float(np.mean(ridge_l2(data, stats, train_conditions, conditions,
                                            genes, args.ridge_alpha)))
        ridge_folds.append(this_ridge)
        model_folds.append(this_model)
        print(f"  {name:34s} {this_ridge:8.4f} {this_model:8.4f} "
              f"{this_model - this_ridge:+8.4f}  {len(conditions)}")

    if not model_folds:
        raise SystemExit("nothing scored")
    ridge_mean = float(np.mean(ridge_folds))
    model_mean = float(np.mean(model_folds))
    gap = model_mean - ridge_mean
    print("  " + "-" * 64)
    label = f"mean over {len(model_folds)} fold(s)"
    print(f"  {label:34s} {ridge_mean:8.4f} {model_mean:8.4f} {gap:+8.4f}")
    print("")
    print("  " + ("CLEARS the bar" if gap < 0 else "DOES NOT clear it - see C1"))
    if len(model_folds) == 1:
        print("  One fold only, and the folds are not equally hard: ridge scores 1.5473")
        print("  on norman additive fold 0 against 1.6690 over all five. A direction,")
        print("  not a verdict.")
    if args.gate != "soft":
        print("")
        print(f"  gate is {args.gate!r}: this puts drawn cells against a mean vector.")
        print("  Rerun with --gate soft for C1.")


if __name__ == "__main__":
    main()
