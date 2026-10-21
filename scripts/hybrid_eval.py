"""Decide docs/DESIGN_HYBRID.md: H0 on validation runs, F1 / F2 / O1 on the fold runs.

    python scripts/hybrid_eval.py results/runs/yLS_val_sig* --device cuda --bound-mean
    python scripts/hybrid_eval.py results/runs/y*_f*_s* results/runs/s2fin_f[0124]_s* \
        --device cuda --bound-mean

Two numbers per run:
  L2        compare_to_ridge.py's, recomputed here the same way: soft gate, the scDFM
            1,000 genes, 4,096 control cells, mean over the held-out doubles
  edist     the energy distance train.py's evaluation wrote to results.json (edist_rel),
            the same procedure for every arm - FM's primary metric (DESIGN_HYBRID.md 5)

The arm is read from each run's saved config, not its name:

    latent  fm   sigma    arm
    0       off  -        D        KEGG, no FM
    0       on   0        B        KEGG, deterministic FM
    0       on   >0       S        KEGG, stochastic FM
    >0      off  -        L        KEGG + z, no FM
    >0      on   0        LF       KEGG + z, deterministic FM
    >0      on   >0       LS       KEGG + z, stochastic FM

Runs are paired by (fold, seed). Folds 0, 1, 2 and 4 decide; fold 3 is printed only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.conditions import condition_groups, scdfm_eval_genes
from src.eval.diagnostics import load_run, measure_transport

DECIDING = (0, 1, 2, 4)
NEEDED, PAIRS = 10, 12
MARGIN = 0.02


def arm_of(config: dict) -> str:
    latent = int(config["model"].get("latent_dim", 0)) > 0
    fm = float(config["train"].get("fm_weight", 0)) > 0
    sigma = float(config["train"].get("fm_sigma", 0.0)) > 0
    if not fm:
        return "L" if latent else "D"
    return ("LS" if sigma else "LF") if latent else ("S" if sigma else "B")


def sign_p(wins: int, n: int) -> float:
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def score(run_dir: str, args) -> dict:
    config, data, stats, fold, model = load_run(
        run_dir, args.device, "soft",
        eval_overrides={"bound_mean": True} if args.bound_mean else None)
    rng = np.random.default_rng(config["eval"]["seed"])
    genes = scdfm_eval_genes(data, fold, 1000)
    doubles = condition_groups(data, stats, fold, config["split"]["method"])["test doubles"]
    rows = measure_transport(model, data, stats, doubles, config, rng, args.device,
                             args.n_cells, genes=genes)
    with open(os.path.join(run_dir, "results.json"), encoding="utf-8") as handle:
        edist = float(json.load(handle)["results"]["edist_rel"])
    return {"run": os.path.basename(os.path.normpath(run_dir)), "arm": arm_of(config),
            "fold": int(config["split"]["fold"]), "seed": int(config["train"]["seed"]),
            "validation": bool(config["split"].get("validation")),
            "sigma": float(config["train"].get("fm_sigma", 0.0)),
            "l2": float(np.mean([r["l2"] for r in rows])), "edist": edist}


def h0(rows: list[dict]) -> None:
    candidates = [r for r in rows if r["arm"] == "LS"]
    if not candidates:
        raise SystemExit("H0 needs LS validation runs")
    best = min(r["l2"] for r in candidates)
    print(f"\nH0: LS on the validation split - sigma by edist among L2 within {MARGIN} "
          f"of the best ({best:.4f})")
    print(f"  {'sigma':>7s} {'L2':>8s} {'edist':>8s}  eligible")
    for r in sorted(candidates, key=lambda r: r["sigma"]):
        print(f"  {r['sigma']:7.3g} {r['l2']:8.4f} {r['edist']:8.4f}  "
              f"{'yes' if r['l2'] <= best + MARGIN else 'no'}")
    eligible = [r for r in candidates if r["l2"] <= best + MARGIN]
    chosen = min(eligible, key=lambda r: r["edist"])
    grid = sorted(r["sigma"] for r in candidates)
    print(f"\n  H0 chooses sigma {chosen['sigma']:g}: SIGMA={chosen['sigma']:g}")
    if chosen["sigma"] in (grid[0], grid[-1]):
        side = "below" if chosen["sigma"] == grid[0] else "above"
        extra = chosen["sigma"] / 3 if side == "below" else chosen["sigma"] * 3
        print(f"  it sits at the grid's edge: extend one step {side} (sigma {extra:.3g}) on "
              f"the validation split before the fold runs (DESIGN_HYBRID.md 6)")


def judge(rows: list[dict]) -> None:
    by = {(r["arm"], r["fold"], r["seed"]): r for r in rows}
    arms = sorted({r["arm"] for r in rows}, key="D B S L LF LS".split().index)
    folds = sorted({r["fold"] for r in rows})
    print(f"\n  per arm, mean over seeds (L2 / edist)")
    print("  " + f"{'arm':4s}" + "".join(f"   fold {f:<10d}" for f in folds) + "   deciding mean")
    for arm in arms:
        cells, deciding_l2, deciding_e = [], [], []
        for f in folds:
            here = [r for r in rows if r["arm"] == arm and r["fold"] == f]
            if here:
                l2 = np.mean([r["l2"] for r in here]); ed = np.mean([r["edist"] for r in here])
                cells.append(f"{l2:7.4f}/{ed:6.3f}")
                if f in DECIDING:
                    deciding_l2 += [r["l2"] for r in here]; deciding_e += [r["edist"] for r in here]
            else:
                cells.append(f"{'-':>14s}")
        tail = (f"{np.mean(deciding_l2):7.4f}/{np.mean(deciding_e):6.3f}"
                if deciding_l2 else "-")
        print("  " + f"{arm:4s}" + "".join(f"   {c:14s}" for c in cells) + f"   {tail}")

    def compare(name, a, b, metric):
        keys = sorted({(f, s) for (arm, f, s) in by if arm == a and f in DECIDING}
                      & {(f, s) for (arm, f, s) in by if arm == b and f in DECIDING})
        if not keys:
            print(f"\n  {name}: no {a}/{b} pairs on folds {DECIDING}")
            return None
        wins_e = sum(by[(a, f, s)]["edist"] < by[(b, f, s)]["edist"] for f, s in keys)
        wins_l = sum(by[(a, f, s)]["l2"] < by[(b, f, s)]["l2"] for f, s in keys)
        dl2 = np.mean([by[(a, f, s)]["l2"] - by[(b, f, s)]["l2"] for f, s in keys])
        print(f"\n  {name} ({a} vs {b}, {len(keys)} pairs): edist lower in {wins_e} "
              f"(p {sign_p(wins_e, len(keys)):.3f}), L2 lower in {wins_l} "
              f"(p {sign_p(wins_l, len(keys)):.3f}), mean L2 difference {dl2:+.4f}")
        if len(keys) != PAIRS:
            print(f"    decided on {PAIRS} pairs; {len(keys)} here - not a verdict")
            return None
        if metric == "edist":
            return wins_e >= NEEDED and dl2 <= MARGIN
        return dl2 < -MARGIN or wins_l >= NEEDED

    verdicts = {"F1": compare("F1 FM contributes", "LS", "L", "edist"),
                "F2 ": compare("F2 stochastic is needed", "LS", "LF", "edist"),
                "O1": compare("O1 the latent contributes", "LS", "S", "l2")}
    print("")
    for name, verdict in verdicts.items():
        if verdict is not None:
            print(f"  {name.strip()}: {'PASSES' if verdict else 'FAILS'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=4096)
    parser.add_argument("--bound-mean", action="store_true")
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        r = score(run_dir, args)
        rows.append(r)
        print(f"  {r['run']:20s} {r['arm']:3s} fold {r['fold']} seed {r['seed']}  "
              f"sigma {r['sigma']:g}  L2 {r['l2']:.4f}  edist {r['edist']:.4f}")
    if all(r["validation"] for r in rows):
        h0(rows)
    elif any(r["validation"] for r in rows):
        raise SystemExit("validation and fold runs cannot be judged together")
    else:
        judge(rows)


if __name__ == "__main__":
    main()
