"""Where a model's error on a held-out double comes from. No training, minutes to run.

    python scripts/decompose_gap.py results/runs/<run>... --device cuda

On norman additive, five folds and three seeds, this model scores 1.6388 against
ridge's 1.6695 and v2's 1.5527. It beats ridge on three folds and loses on two, and v2
beats it by about 0.10 on four of five. That says the model is behind; it does not say
WHERE, and the two candidate places need opposite repairs:

  ITS SINGLES ARE WRONG. Every single perturbation is in training under this split. If
  the model's prediction for A alone is off, then nothing built on it can be right, and
  the deficit is in how a perturbation's own effect is learned - not in how two combine.

  ITS COMBINATION IS WRONG. If the singles are accurate and the double is still off, the
  deficit is in composition: how the decoder turns e_A + e_B into a joint effect.

Four numbers per fold separate them, all L2 against the true mean of the double:

  true additive    m_A + m_B - m_control, from the TRUE single means. The naive
                   baseline, and the ceiling for anything that only adds singles.
  ridge            the same idea with every training condition pooled into the fit.
  model, added     pred_A + pred_B - control: the MODEL's singles, added by hand. Close
                   to "true additive" means its singles are accurate.
  model, direct    what the model actually predicts for the pair. Below "model, added"
                   means its composition learned something adding does not.

and one more, the fit on the singles themselves: ||pred_A - m_A||. These are TRAINING
conditions, so this is not generalisation - it is whether the model has learned what it
was shown.
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
from src.eval.diagnostics import load_run
from src.eval.predict import predict_cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=2048)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    columns = ("true add", "ridge", "model add", "model dir", "single fit")
    print("")
    print(f"  {'run':24s} " + " ".join(f"{c:>10s}" for c in columns) + "    n")
    print("  " + "-" * 86)

    totals = []
    for run_dir in args.run_dirs:
        config, data, stats, fold, model = load_run(run_dir, args.device, "soft")
        rng = np.random.default_rng(config["eval"]["seed"])
        genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
        method = config["split"]["method"]
        doubles = condition_groups(data, stats, fold, method)["test doubles"]
        train = training_conditions(stats, fold, method)
        fit = baselines.fit_ridge_additive(stats, train, data.perturbations,
                                           alpha=args.ridge_alpha)

        control = data.cells(data.control_condition)
        sample = control[rng.choice(control.shape[0],
                                    size=min(args.n_cells, control.shape[0]),
                                    replace=False)]
        base = sample.mean(axis=0)[genes]
        m_control = stats.mean[data.control_condition][genes]

        def predicted(condition: str) -> np.ndarray:
            cells = predict_cells(model, sample, condition, data.pert_index,
                                  args.device, data.naming)
            return cells.mean(axis=0)[genes]

        single_cache: dict[str, np.ndarray] = {}
        rows = []
        for double in doubles:
            a, b = data.naming.genes(double)
            singles = [stats.single_of(a), stats.single_of(b)]
            # Without both singles observed there is no "true additive" to compare with,
            # and a row that silently dropped that column would make the others look
            # like they were measured on the same conditions when they were not.
            if not all(stats.has(s) for s in singles) or any(
                    g not in fit["index"] for g in (a, b)):
                continue
            for s in singles:
                if s not in single_cache:
                    single_cache[s] = predicted(s)
            truth = stats.mean[double][genes]
            m_a, m_b = (stats.mean[s][genes] for s in singles)
            p_a, p_b = (single_cache[s] for s in singles)
            ridge = m_control + sum(fit["w"][fit["index"][g]][genes] for g in (a, b))
            rows.append((
                np.linalg.norm(m_a + m_b - m_control - truth),
                np.linalg.norm(ridge - truth),
                np.linalg.norm(p_a + p_b - base - truth),
                np.linalg.norm(predicted(double) - truth),
                0.5 * (np.linalg.norm(p_a - m_a) + np.linalg.norm(p_b - m_b)),
            ))

        name = os.path.basename(os.path.normpath(run_dir))[:24]
        if not rows:
            print(f"  {name:24s} no double with both singles observed")
            continue
        mean = np.mean(np.asarray(rows), axis=0)
        totals.append(mean)
        print(f"  {name:24s} " + " ".join(f"{v:10.4f}" for v in mean)
              + f"  {len(rows):3d}")

    if totals:
        mean = np.mean(np.asarray(totals), axis=0)
        print("  " + "-" * 86)
        print(f"  {'mean over runs':24s} " + " ".join(f"{v:10.4f}" for v in mean))
        print("")
        print("  model add  vs true add   how accurate the model's SINGLES are")
        print("  model dir  vs model add  what its COMBINATION adds over summing them")
        print("  single fit               its error on singles it was TRAINED on")


if __name__ == "__main__":
    main()
