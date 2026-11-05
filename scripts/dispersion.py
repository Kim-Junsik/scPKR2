"""How widely the predicted cells spread around their own mean, against real cells.

    python scripts/dispersion.py results/runs/nEp0fm_f3_s0 results/runs/nEp0fm_f3_s1 --device cuda
    python scripts/dispersion.py 'results/runs/nEp0fm_f*_s*' --device cuda --quiet

Read-only. It changes no model, export or score, and it is not used by any other script.

WHY. A finite set of predicted cells scores worse than the mean they come from:
E||mean_n(pred) - m||^2 = ||mu_hat - m||^2 + tr(Sigma_pred) / n. So every column that
averages 128 cells (MSE, MAE, PD, DS), and L2 at N_p cells, pays tr(Sigma_pred) / n on
top of the error of the mean. On Table 1 the soft L2 is 1.5653 at 4,096 cells, 1.6670 at
N_p and 1.7124 at 128. That puts tr(Sigma_pred) near 62 on the 1,000 scored genes, against
about 42 for the real control cells (scDFM's Control row, 4.0132 -> 4.0541 at 128).

WHAT IT MEASURES, per run, on the soft gate with the mean bounded (the main tables):

  tr_pred   sum over genes of the predicted cells' variance, from --n-cells control cells
  tr_real   the same for the condition's real cells (TRAINING conditions only)
  tr_ctrl   the same for the real control cells, the population the model transports

  ratio     sum tr_pred / sum tr_real over the training conditions
  lambda    sqrt(1 / ratio): the factor that would pull every predicted cell towards its
            condition's predicted mean (x -> mu_hat + lambda (x - mu_hat)) so that the
            training conditions' spread matches the real one. The mean is unchanged.

Held-out conditions are predicted but their real cells are NEVER read: their row reports
tr_pred alone, to show whether the model spreads held-out cells as it spreads training
ones. The gene space is scDFM's 1,000 scored genes (--infer-top-gene 1000, the protocol's
own selection) or every modelled gene (--infer-top-gene 0).

The last stdout line is the mean lambda over the runs, so a later script can read it with
tail -n 1. Nothing here applies it.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import condition_groups, load_run, scdfm_eval_genes  # noqa: E402
from src.eval.predict import predict_cells  # noqa: E402

TRAIN = ("train singles", "train doubles")
TEST = ("test singles", "test doubles")


def spread(cells: np.ndarray) -> float:
    """tr(Sigma): the summed per-gene variance."""
    return float(cells.var(axis=0, ddof=1).sum()) if cells.shape[0] > 1 else float("nan")


@torch.no_grad()
def measure(run_dir: str, device: str, n_cells: int, n_top: int, min_real: int) -> dict:
    config, data, stats, fold, model = load_run(run_dir, device, "soft",
                                                eval_overrides={"bound_mean": True})
    rng = np.random.default_rng(config["eval"]["seed"])
    groups = condition_groups(data, stats, fold, config["split"]["method"])
    genes = scdfm_eval_genes(data, fold, n_top) if n_top else np.arange(data.n_genes)

    control = data.cells(data.control_condition)
    out = {"tr_ctrl": spread(control[:, genes]), "genes": len(genes), "groups": {}}
    for name in TRAIN + TEST:
        rows = []
        for condition in groups[name]:
            n_real = int(stats.n[condition])
            if name in TRAIN and n_real < min_real:
                continue
            pick = rng.choice(control.shape[0], size=min(n_cells, control.shape[0]),
                              replace=False)
            predicted = predict_cells(model, control[pick], condition, data.pert_index,
                                      device, data.naming)[:, genes]
            row = {"condition": condition, "tr_pred": spread(predicted)}
            if name in TRAIN:
                row["tr_real"] = spread(data.cells(condition)[:, genes])
            rows.append(row)
        out["groups"][name] = rows

    train_rows = [r for name in TRAIN for r in out["groups"][name]]
    pred = sum(r["tr_pred"] for r in train_rows)
    real = sum(r["tr_real"] for r in train_rows)
    out["ratio"] = pred / real if real else float("nan")
    out["lambda"] = float(np.sqrt(1.0 / out["ratio"])) if out["ratio"] > 0 else float("nan")
    out["train_pred"] = pred / max(len(train_rows), 1)
    return out


def report(run_dir: str, result: dict, quiet: bool) -> None:
    print(f"\n{os.path.basename(run_dir)}   ({result['genes']} genes, real control "
          f"tr {result['tr_ctrl']:.2f})", file=sys.stderr)
    print(f"  {'group':<14}{'n':>5}{'tr_pred':>10}{'tr_real':>10}{'pred/real':>11}"
          f"{'pred/ctrl':>11}", file=sys.stderr)
    for name, rows in result["groups"].items():
        if not rows:
            continue
        pred = float(np.mean([r["tr_pred"] for r in rows]))
        if "tr_real" in rows[0]:
            real = float(np.mean([r["tr_real"] for r in rows]))
            real_text = f"{real:>10.2f}{pred / real:>11.3f}"
        else:
            real_text = f"{'(unread)':>10}{'':>11}"
        print(f"  {name:<14}{len(rows):>5}{pred:>10.2f}{real_text}"
              f"{pred / result['tr_ctrl']:>11.3f}", file=sys.stderr)
    print(f"  training conditions pooled: pred/real {result['ratio']:.3f}  ->  lambda "
          f"{result['lambda']:.3f}", file=sys.stderr)
    if quiet:
        return
    rows = [r for name in TRAIN for r in result["groups"][name]]
    rows.sort(key=lambda r: -r["tr_pred"] / r["tr_real"])
    shown = rows[:5] + rows[-5:] if len(rows) > 10 else rows
    print("  training conditions, most and least over-spread (pred/real):", file=sys.stderr)
    for r in shown:
        print(f"    {r['condition']:<28}{r['tr_pred']:>9.2f}{r['tr_real']:>9.2f}"
              f"{r['tr_pred'] / r['tr_real']:>8.3f}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="finished runs; quoted globs are expanded")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-cells", type=int, default=512,
                        help="control cells transported per condition")
    parser.add_argument("--infer-top-gene", type=int, default=1000,
                        help="scDFM's scored genes (1000); 0 = every modelled gene")
    parser.add_argument("--min-real", type=int, default=20,
                        help="skip training conditions with fewer real cells than this")
    parser.add_argument("--quiet", action="store_true", help="no per-condition lines")
    args = parser.parse_args()

    runs = []
    for pattern in args.runs:
        found = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        runs += [r for r in found if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    if not runs:
        raise SystemExit("no finished run matched")

    results = []
    for run in runs:
        result = measure(run, args.device, args.n_cells, args.infer_top_gene, args.min_real)
        report(run, result, args.quiet)
        results.append(result)

    ratios = [r["ratio"] for r in results]
    lambdas = [r["lambda"] for r in results]
    print(f"\n{len(results)} runs: pred/real {np.mean(ratios):.3f} (sd {np.std(ratios):.3f}), "
          f"lambda {np.mean(lambdas):.3f} (sd {np.std(lambdas):.3f})", file=sys.stderr)
    for name in TEST:
        held = [np.mean([r["tr_pred"] for r in res["groups"][name]]) / res["train_pred"]
                for res in results if res["groups"][name]]
        if held:
            print(f"  {name}: predicted spread {np.mean(held):.3f} x the training "
                  f"conditions' (real cells not read)", file=sys.stderr)
    print(f"{np.mean(lambdas):.4f}")


if __name__ == "__main__":
    main()
