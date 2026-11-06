"""Does averaging the seeds of one fold beat the seeds alone? Single runs against ensembles.

    python scripts/ensemble_check.py 'results/runs/nEp0fm_f3_s*' --device cuda
    python scripts/ensemble_check.py 'results/runs/nEp0fm_comb_f*_s*' --device cuda
    python scripts/ensemble_check.py 'results/runs/cDfmP_s?' --device cuda

Read-only. It changes no model, export or score, and it is not used by any other script.

Runs are grouped by name without the _s<seed> suffix (nEp0fm_f3_s0-2 -> nEp0fm_f3), so
each group is one fold trained with several seeds. Within a group every model transports
THE SAME control cells, picked exactly as scripts/paper_table.py compute_l2 picks them, so
the "single runs" row reproduces arm_table.py's L2 (soft gate, mean bounded, N_p cells,
1,000 scored genes). Two ensembles are built from those cells:

  avg   each control cell's prediction averaged over the seeds. The mean is the average
        of the seeds' means, and the cells are also pulled together.
  mix   cell i taken from seed i mod K. The same averaged mean, but each cell keeps one
        model's spread: the ensemble as a mixture of the seeds.

If avg wins and mix does not, the gain comes from tighter cells rather than from a better
mean. Our cells already spread slightly LESS than real ones (scripts/dispersion.py), so
only a gain that mix also shows counts as a better prediction.

COLUMNS. L2 at N_p cells is the table's own L2. MSE, MAE and PD (Pearson delta) are
computed HERE from 128 cells against the real condition mean - in-script approximations of
the cell-eval columns, good for single vs ensemble, not for quoting against scDFM. spread
is tr(Sigma) of the N_p predicted cells (scripts/dispersion.py). DE and DS need cell-eval
and are not here.

Blocks follow condition_groups: "single" and "double" (Table 2 reads each), and "all"
when a fold holds out both (Table 3 reads all). Table 1 holds out doubles only.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import condition_groups, load_run, scdfm_eval_genes  # noqa: E402
from src.eval.predict import predict_cells  # noqa: E402

COLUMNS = ("L2", "MSE", "MAE", "PD", "spread")
FORMATS = {"L2": "{:>9.4f}", "MSE": "{:>10.5f}", "MAE": "{:>9.5f}", "PD": "{:>8.4f}",
           "spread": "{:>8.2f}"}


def group_runs(patterns: list[str]) -> dict[str, list[str]]:
    runs = []
    for pattern in patterns:
        found = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[") else [pattern]
        runs += [r for r in found if os.path.exists(os.path.join(r, "checkpoint.pt"))]
    groups: dict[str, list[str]] = {}
    for run in sorted(set(runs)):
        key = re.sub(r"_s\d+$", "", os.path.basename(os.path.normpath(run)))
        groups.setdefault(key, []).append(run)
    return groups


def blocks(groups: dict) -> dict[str, list[str]]:
    """The held-out conditions per block, in compute_l2's order."""
    out = {}
    if groups["test singles"]:
        out["single"] = groups["test singles"]
    if groups["test doubles"]:
        out["double"] = groups["test doubles"]
    if groups["test singles"] and groups["test doubles"]:
        out["all"] = groups["test doubles"] + groups["test singles"]
    return out


def picks(conditions: list[str], stats, n_control: int, seed: int, size: int) -> list:
    """compute_l2's draw (size 0 = N_p): one generator per block, conditions in order."""
    rng = np.random.default_rng(seed)
    out = []
    for condition in conditions:
        n = int(stats.n[condition]) if size <= 0 else size
        out.append(rng.choice(n_control, size=min(n, n_control), replace=False))
    return out


def metrics(np_cells: list, cells128: list, conditions: list[str], stats,
            genes: np.ndarray) -> dict:
    control = stats.control[genes]
    rows = {name: [] for name in COLUMNS}
    for condition, full, small in zip(conditions, np_cells, cells128):
        true = stats.mean[condition][genes]
        rows["L2"].append(float(np.linalg.norm(full.mean(axis=0) - true)))
        error = small.mean(axis=0) - true
        rows["MSE"].append(float(np.mean(error ** 2)))
        rows["MAE"].append(float(np.mean(np.abs(error))))
        rows["PD"].append(float(np.corrcoef(small.mean(axis=0) - control,
                                            true - control)[0, 1]))
        rows["spread"].append(float(full.var(axis=0, ddof=1).sum())
                              if full.shape[0] > 1 else float("nan"))
    return {name: float(np.nanmean(values)) for name, values in rows.items()}


def mixture(per_seed: list[np.ndarray]) -> np.ndarray:
    k = len(per_seed)
    out = np.empty_like(per_seed[0])
    for i in range(out.shape[0]):
        out[i] = per_seed[i % k][i]
    return out


@torch.no_grad()
def check_group(name: str, runs: list[str], device: str, n_top: int, n_small: int) -> dict:
    """{block: {"single": [per-seed dicts], "avg": dict, "mix": dict, "n": conditions}}."""
    reference = None
    cells: dict = {}
    for run in runs:
        config, data, stats, fold, model = load_run(run, device, "soft",
                                                    eval_overrides={"bound_mean": True})
        identity = (config["split"]["method"], config["split"]["fold"],
                    config["data"]["cache_h5ad"])
        if reference is None:
            reference = identity
            seed = int(config["eval"]["seed"])
            control = data.cells(data.control_condition)
            genes = scdfm_eval_genes(data, fold, n_top) if n_top else np.arange(data.n_genes)
            plan = {}
            for block, conditions in blocks(condition_groups(
                    data, stats, fold, config["split"]["method"])).items():
                plan[block] = (conditions,
                               picks(conditions, stats, control.shape[0], seed, 0),
                               picks(conditions, stats, control.shape[0], seed + 1, n_small))
        elif identity != reference:
            raise SystemExit(f"{run} is not the same split as {runs[0]}: {identity} vs "
                             f"{reference} - group by fold")
        for block, (conditions, full_picks, small_picks) in plan.items():
            for condition, full, small in zip(conditions, full_picks, small_picks):
                for kind, pick in (("np", full), ("128", small)):
                    predicted = predict_cells(model, control[pick], condition,
                                              data.pert_index, device, data.naming)[:, genes]
                    cells.setdefault((block, condition, kind), []).append(predicted)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    result = {}
    for block, (conditions, _, _) in plan.items():
        def gather(kind, combine):
            return [combine(cells[(block, c, kind)]) for c in conditions]

        single = [metrics([cells[(block, c, "np")][s] for c in conditions],
                          [cells[(block, c, "128")][s] for c in conditions],
                          conditions, stats, genes) for s in range(len(runs))]
        average = lambda per_seed: np.mean(per_seed, axis=0)  # noqa: E731
        result[block] = {
            "n": len(conditions), "single": single,
            "avg": metrics(gather("np", average), gather("128", average),
                           conditions, stats, genes),
            "mix": metrics(gather("np", mixture), gather("128", mixture),
                           conditions, stats, genes),
        }
    return result


def line(label: str, values: dict, extra: str = "") -> str:
    return f"  {label:<14}" + "".join(FORMATS[c].format(values[c]) for c in COLUMNS) + extra


def header() -> str:
    return (f"  {'':<14}{'L2(N_p)':>9}{'MSE128':>10}{'MAE128':>9}{'PD128':>8}{'spread':>8}")


def mean_of(rows: list[dict]) -> dict:
    return {c: float(np.mean([r[c] for r in rows])) for c in COLUMNS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="finished runs; quoted globs are expanded")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--infer-top-gene", type=int, default=1000,
                        help="scDFM's scored genes (1000); 0 = every modelled gene")
    parser.add_argument("--pred-cells", type=int, default=128,
                        help="cells behind the MSE, MAE and PD columns")
    args = parser.parse_args()

    groups = group_runs(args.runs)
    if not groups:
        raise SystemExit("no finished run matched")
    lonely = [k for k, v in groups.items() if len(v) < 2]
    if lonely:
        raise SystemExit(f"groups with one seed cannot be averaged: {', '.join(lonely)}")

    totals: dict = {}
    for name, runs in groups.items():
        seeds = " ".join(os.path.basename(os.path.normpath(r)).rsplit("_s", 1)[-1]
                         for r in runs)
        result = check_group(name, runs, args.device, args.infer_top_gene, args.pred_cells)
        for block, res in result.items():
            print(f"\n{name}   {block}, {res['n']} conditions, seeds {seeds}")
            print(header())
            per_seed = " ".join(f"{r['L2']:.4f}" for r in res["single"])
            print(line("single runs", mean_of(res["single"]), f"   L2 per seed {per_seed}"))
            print(line("ensemble avg", res["avg"]))
            print(line("ensemble mix", res["mix"]))
            bucket = totals.setdefault(block, {"single": [], "avg": [], "mix": []})
            bucket["single"] += res["single"]
            bucket["avg"].append(res["avg"])
            bucket["mix"].append(res["mix"])

    print(f"\n=== over {len(groups)} group(s): the mean of every single run against the "
          f"mean of the ensembles ===")
    for block, bucket in totals.items():
        single = mean_of(bucket["single"])
        print(f"\n{block}   ({len(bucket['single'])} single runs, {len(bucket['avg'])} "
              f"ensembles)")
        print(header())
        print(line("single runs", single))
        for kind in ("avg", "mix"):
            ensemble = mean_of(bucket[kind])
            diff = {c: ensemble[c] - single[c] for c in COLUMNS}
            print(line(f"ensemble {kind}", ensemble))
            print(f"  {'  diff':<14}" + "".join(
                ("{:>+9.4f}" if c == "L2" else "{:>+10.5f}" if c == "MSE" else
                 "{:>+9.5f}" if c == "MAE" else "{:>+8.4f}" if c == "PD" else
                 "{:>+8.2f}").format(diff[c]) for c in COLUMNS))
    print("\nlower is better: L2, MSE, MAE.  higher is better: PD.  spread: tr(Sigma) of "
          "the N_p cells (real cells spread more, scripts/dispersion.py).")


if __name__ == "__main__":
    main()
