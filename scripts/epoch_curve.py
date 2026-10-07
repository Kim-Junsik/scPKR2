"""How long to train, chosen on validation conditions rather than assumed.

    python scripts/epoch_curve.py results/runs/<validation run> --device cuda

WHY THIS EXISTS. Training length is a hyperparameter and the two datasets disagree about
it, in opposite directions:

                  400 epochs   2,500 epochs
  ComboSciPlex      1.9101        1.4900
  norman additive   1.4955        1.7041

One is undertrained at 400 and the other is overfitted by 2,500. Picking a number per
dataset by looking at those test figures would be selecting on the test set, so the
choice has to be made against HELD-OUT TRAINING conditions - which is what
split.validation=true carves out - and then applied to the real runs.

IT DOES NOT ANSWER "HOW MANY EPOCHS SHOULD A RUN BE", AND IT CANNOT. The learning rate
is a cosine from 1e-3 to 1e-6 across the run's TOTAL length, so epoch 250 of a 2,500
epoch run sits at about 9e-4 while epoch 250 of a 250 epoch run has annealed to 1e-6.
They are different models, and the second scores better: a standalone 250-epoch run
reaches 1.6759 on combosciplex validation where the best point anywhere inside a
2,500-epoch run is 1.9227.

So this reads one schedule's trajectory, which is worth seeing - it is how overfitting
shows itself - but choosing a length means training separate runs that each anneal in
full and comparing those with scripts/compare_to_ridge.py.

THE RUN PASSED HERE MUST BE A VALIDATION RUN. Its "test" conditions are then validation
conditions held out of training, and nothing in the reported test set has been touched.
Pointing this at a real run would select the epoch count by reading the test set, which
is the one thing this file exists to prevent.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.conditions import condition_groups, scdfm_eval_genes
from src.eval.diagnostics import load_run, measure_transport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--gate", default="soft", choices=["soft", "hard", "sample"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=2048)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--group", default="all",
                        choices=["double", "single", "all"])
    args = parser.parse_args()

    if not os.path.basename(os.path.normpath(args.run_dir)).lower().count("val"):
        print("  NOTE: the run's name does not say 'val'. If its held-out conditions are")
        print("  the reported test set, this selects an epoch count by reading the test")
        print("  set, and the number it produces cannot be used.")

    names = sorted(glob.glob(os.path.join(args.run_dir, "checkpoint_e*.pt")))
    if os.path.exists(os.path.join(args.run_dir, "checkpoint.pt")):
        names.append(os.path.join(args.run_dir, "checkpoint.pt"))
    if not names:
        raise SystemExit(
            f"no per-epoch checkpoint in {args.run_dir}. Train with train.save_every set "
            f"- without it the run keeps only its last state and the curve cannot be "
            f"recovered without training again.")

    print("")
    print(f"{args.group} held-out conditions of this run, gate={args.gate}, "
          f"{args.n_cells} cells")
    print("")
    print(f"  {'epoch':>8s} {'L2':>10s}")
    print("  " + "-" * 20)

    best = (float("inf"), None)
    for path in names:
        match = re.search(r"checkpoint_e(\d+)\.pt$", path)
        epoch = int(match.group(1)) if match else -1
        config, data, stats, fold, model = load_run(
            args.run_dir, args.device, args.gate,
            checkpoint_name=os.path.basename(path))
        rng = np.random.default_rng(config["eval"]["seed"])
        genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
        groups = condition_groups(data, stats, fold, config["split"]["method"])
        conditions = {"double": groups["test doubles"],
                      "single": groups["test singles"],
                      "all": groups["test doubles"] + groups["test singles"]}[args.group]
        rows = measure_transport(model, data, stats, conditions, config, rng,
                                 args.device, args.n_cells, genes=genes)
        score = float(np.mean([r["l2"] for r in rows]))
        label = str(epoch) if epoch > 0 else "final"
        print(f"  {label:>8s} {score:10.4f}")
        if score < best[0]:
            best = (score, label)

    print("  " + "-" * 20)
    print(f"  best at epoch {best[1]} with {best[0]:.4f}")
    print("")
    print("  That is the best point ON THIS SCHEDULE, not the length a run should be.")
    print("  The learning rate anneals over the run's total length, so a shorter run is")
    print("  a different model and usually a better one. To choose a length, train")
    print("  separate runs and compare them with scripts/compare_to_ridge.py.")


if __name__ == "__main__":
    main()
