"""train.shrink_weight for a given share of the training loss, read off trained runs.

    python scripts/shrink_weight.py results/runs/cDfmP_s0 results/runs/cDfmP_s1 results/runs/cDfmP_s2 --share 0.05

The shrink term (src/config.py, train.shrink_weight) is in operator units, so its weight
cannot be written down on the scale of the other terms. It is set here, once, before
any run that uses it, from runs trained WITHOUT it:

  loss      the run's last-epoch mean loss per step (results.json, train_loss.loss)
  penalty   what its trained generators would pay per step at weight 1: the mean over
            its training conditions of shrink_penalty, the quantity the loop adds

weight = share x loss / penalty, averaged over the runs. The weight is printed last, on a
line of its own, so a script can read it with tail -n 1.

WHY A SMALL SHARE IS ENOUGH. The term is a ridge prior on how far each generator sits
from the average one. Along a direction the data pin down, the data gradient outweighs
it and the generator barely moves. Along a direction the data leave free (how a
combination splits between two drugs that are seen together only), the data gradient is
zero, and over a long run any positive weight decides it. So the share only has to be
large enough to act, not large enough to compete with the fit.

Reads training conditions only: the generators and the training loss, nothing scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import splits  # noqa: E402
from src.eval.diagnostics import _dataset, build_model  # noqa: E402
from src.train.loop import appearance_counts, train_conditions_used  # noqa: E402


@torch.no_grad()
def measure(run_dir: str, device: str) -> tuple[float, float, dict]:
    """(loss per step, penalty per step at weight 1, details) for one finished run."""
    checkpoint = torch.load(os.path.join(run_dir, "checkpoint.pt"), map_location=device,
                            weights_only=False)
    config = checkpoint["config"]
    config["train"]["device"] = config["eval"]["device"] = device
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as handle:
        loss = float(json.load(handle)["train_loss"]["loss"])

    data, stats = _dataset(config)
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]
    model, train_conditions, _ = build_model(config, data, stats, fold, method, device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if not model.has_generators:
        raise SystemExit(f"{run_dir}: not a generator model, nothing to shrink")

    conditions = train_conditions_used(data, stats, train_conditions)
    counts = appearance_counts(data, conditions)
    members = sorted(counts)
    centre = model.modulation.mean_generator(members)
    penalties = []
    for condition in conditions:
        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
        penalties.append(float(model.modulation.shrink_penalty(perturbations, centre,
                                                               counts)))
    # Each perturbation's own distance from the average, to see who the prior acts on.
    distance = {data.perturbations[p]: float((model.modulation._masked(p) - centre)
                                             .square().sum()) for p in members}
    average = float(centre.square().sum())
    details = {"conditions": len(conditions), "perturbations": len(members),
               "once": sorted(data.perturbations[p] for p, n in counts.items() if n == 1),
               "distance": distance, "centre": average}
    return loss, float(np.mean(penalties)), details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="finished runs trained without the term")
    parser.add_argument("--share", type=float, default=0.05,
                        help="the penalty's share of the per-step loss at those runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights = []
    for run in args.runs:
        loss, penalty, details = measure(run, args.device)
        weight = args.share * loss / penalty
        weights.append(weight)
        print(f"{os.path.basename(run)}: loss/step {loss:.4f}  penalty/step at weight 1 "
              f"{penalty:.4f}  ->  weight {weight:.4g}   ({details['conditions']} "
              f"conditions, {details['perturbations']} perturbations, "
              f"||A_bar||^2 {details['centre']:.4f})", file=sys.stderr)
        ranked = sorted(details["distance"].items(), key=lambda kv: -kv[1])
        print("    ||A_a - A_bar||^2: " + ", ".join(f"{name} {value:.3f}"
                                                   for name, value in ranked),
              file=sys.stderr)
        if details["once"]:
            print(f"    in one training condition only: {', '.join(details['once'])}",
                  file=sys.stderr)
    weight = float(np.mean(weights))
    print(f"shrink_weight = {weight:.4g}  (share {args.share:g}, mean of {len(weights)} "
          f"runs)", file=sys.stderr)
    print(f"{weight:.4g}")


if __name__ == "__main__":
    main()
