"""Does the model predict cells outside the range their genes occupy, and what would
bounding the multiplicative term cost?

    python scripts/range_check.py results/runs/<run> --device cuda

All fifteen norman exports were refused at the door: min 0.0000, max between 16.51 and
29.77, where cell-eval rejects log1p data above 15. The minimum says the structural
non-negativity holds. The maximum says some predicted cells sit far above anything
their gene reaches in the data.

THE SUSPECTED MECHANISM. Norman is CRISPR activation: a target gene is near zero in
most control cells and high after perturbation. x * exp(u) cannot move a cell that is
at zero, so to raise the gene's MEAN it multiplies the few control cells that already
express it by up to exp(log_factor_max) = 20. The mean comes out right and those cells
come out at 25. Turning a gene on is softplus(v)'s job, and the multiplicative term has
taken it over.

This measures three things on finished checkpoints, without training anything:

  how many predicted entries exceed their gene's training maximum, and by how much
  whether the worst genes are the perturbation's own targets
  the L2 with x * exp(u) stopped at that maximum, against the L2 as trained

The last is the one that decides the repair. If stopping the term costs nothing, it can
be part of the model's definition; if it costs a lot, the mean was leaning on values no
cell can take, and the turn-on has to be learned by the right term instead.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.conditions import condition_groups, scdfm_eval_genes
from src.eval.diagnostics import load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-cells", type=int, default=2048)
    parser.add_argument("--infer-top-gene", type=int, default=1000)
    parser.add_argument("--group", default="all", choices=["double", "single", "all"])
    args = parser.parse_args()

    print("")
    print(f"  {'run':18s} {'over %':>8s} {'max mu':>8s} {'L2':>8s} {'L2 bounded':>11s} "
          f"{'change':>8s} {'max bounded':>12s} {'targets in worst 5':>19s}")
    print("  " + "-" * 100)

    for run_dir in args.run_dirs:
        config, data, stats, fold, model = load_run(run_dir, args.device, "soft")
        rng = np.random.default_rng(config["eval"]["seed"])
        genes = scdfm_eval_genes(data, fold, args.infer_top_gene)
        groups = condition_groups(data, stats, fold, config["split"]["method"])
        conditions = {"double": groups["test doubles"],
                      "single": groups["test singles"],
                      "all": groups["test doubles"] + groups["test singles"]}[args.group]
        control = data.cells(data.control_condition)
        sample = control[rng.choice(control.shape[0],
                                    size=min(args.n_cells, control.shape[0]),
                                    replace=False)]
        x = torch.as_tensor(sample, device=args.device)
        ceiling = model.modulation.ceiling
        names = np.asarray(data.gene_names)

        over, peak, peak_bounded, plain, bounded, hits, total = [], 0.0, 0.0, [], [], 0, 0
        gene_peak = torch.zeros(len(names), device=args.device)
        with torch.no_grad():
            for condition in conditions:
                targets = list(data.naming.genes(condition))
                index = [data.pert_index[g] for g in targets]
                params = model(x, index)
                mean = params["mean"]
                modulated = x * torch.exp(params["u"])
                turn_on = mean - modulated
                # The multiplicative term alone, stopped where its gene's data stops.
                # softplus(v) is left as it is: it is the same for every cell of a gene
                # up to the decoder's view of the cell, so it carries the mean.
                capped = torch.minimum(modulated, ceiling.expand_as(modulated)) + turn_on

                over.append(float((mean > ceiling).float().mean()))
                peak = max(peak, float(mean.max()))
                peak_bounded = max(peak_bounded, float(capped.max()))
                truth = stats.mean[condition][genes]
                plain.append(float(np.linalg.norm(
                    mean.mean(dim=0).cpu().numpy()[genes] - truth)))
                bounded.append(float(np.linalg.norm(
                    capped.mean(dim=0).cpu().numpy()[genes] - truth)))
                gene_peak = torch.maximum(gene_peak, mean.max(dim=0).values)
                worst = torch.topk(mean.max(dim=0).values, 5).indices.cpu().numpy()
                hits += sum(1 for j in worst if names[j] in targets)
                total += len(targets)

        name = os.path.basename(os.path.normpath(run_dir))[:18]
        print(f"  {name:18s} {100 * np.mean(over):8.4f} {peak:8.2f} {np.mean(plain):8.4f} "
              f"{np.mean(bounded):11.4f} {np.mean(bounded) - np.mean(plain):+8.4f} "
              f"{peak_bounded:12.2f} {hits:>12d} / {total}")
        # WHICH genes, and what they look like in the control. A gene the control rarely
        # expresses is where a multiplicative term has the least to multiply.
        detection = (x > 0).float().mean(dim=0)
        for j in torch.topk(gene_peak, 5).indices.cpu().numpy():
            print(f"      {str(names[j])[:14]:14s} peak mu {float(gene_peak[j]):6.2f}   "
                  f"training max {float(ceiling[j]):5.2f}   control mean "
                  f"{float(x[:, j].mean()):6.3f}   detected in {100 * float(detection[j]):5.1f} %")

    print("")
    print("  over %        predicted entries above their gene's training maximum")
    print("  L2 bounded    with x*exp(u) stopped at that maximum; 'change' is its cost")
    print("  targets       how many of each condition's targets are among its 5 worst genes")


if __name__ == "__main__":
    main()
