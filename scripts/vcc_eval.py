"""Score runs on the Virtual Cell Challenge 2025 split - docs/DESIGN_ENCODER.md 5 (V1, V2).

    python scripts/vcc_eval.py results/runs/vE_s* results/runs/vS_s* results/runs/vN_s* \
        --means data/vcc/vcc2025_means.npz --device cuda --bound-mean

For every scored gene (the test genes, or the validation genes for a split.validation run)
and every run: the predicted mean expression over ALL measured genes, against the true
mean, as an L2 distance. Genes outside the model's set are predicted as the control mean
- unchanged - which is what every arm predicts for them, so they move no comparison. The
same control cells (a fixed sample) are transported in every run, so runs are paired.

Two closed-form rows need no model:
  control   the control mean for every gene
  mean      the average of the TRAINING genes' means - the baseline a model of unseen
            perturbations has to beat (the challenge's own cell-mean baseline)

Runs are grouped into arms by their saved config (knowledge source, shuffled or not) and
averaged over seeds per gene before any comparison. Then, paired over the scored genes,
one-sided Wilcoxon signed-rank tests of the real-knowledge arm against each other row.

  V1 passes  E beats S with p < 0.01 and a positive median improvement
  V2 passes  E beats the mean baseline the same way

Only V1 and V2 are decided here. The challenge's official cell-eval metrics (V3) are a
separate export.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.diagnostics import load_run
from src.eval.predict import predict_cells

ALPHA = 0.01


def arm_of(config: dict) -> str:
    model = config["model"]
    kind = str(model.get("perturbation_knowledge", "none"))
    if kind == "none":
        return "table"
    return kind + ("-shuffled" if model.get("knowledge_shuffle") else "")


def paired(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> tuple[float, float, int]:
    """One-sided: is a's L2 lower than b's? (p, median improvement, genes a wins)."""
    gain = b - a
    if np.allclose(gain, 0.0):
        return 1.0, 0.0, 0
    p = float(wilcoxon(a, b, alternative="less", zero_method="wilcox").pvalue)
    return p, float(np.median(gain)), int((gain > 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--means", default="data/vcc/vcc2025_means.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-cells", type=int, default=4096)
    parser.add_argument("--bound-mean", action="store_true")
    parser.add_argument("--per-gene", action="store_true")
    args = parser.parse_args()

    store = np.load(args.means, allow_pickle=False)
    conditions = [str(c) for c in store["conditions"]]
    row_of = {c: i for i, c in enumerate(conditions)}
    means, sets = store["means"].astype(np.float64), [str(s) for s in store["vcc_set"]]
    all_genes = [str(g) for g in store["gene_names"]]
    gene_column = {g: i for i, g in enumerate(all_genes)}
    control_mean = means[row_of["ctrl"]]
    train_rows = [i for i, s in enumerate(sets) if s == "train"]
    mean_baseline = means[train_rows].mean(axis=0)

    per_arm: dict[str, list[np.ndarray]] = {}
    scored, covered, validation = None, None, None
    by_setting: dict[tuple, list[float]] = {}
    for run_dir in args.runs:
        config, data, stats, fold, model = load_run(
            run_dir, device=args.device, gate="soft",
            eval_overrides={"bound_mean": True} if args.bound_mean else None)
        test = list(fold["test"])
        if scored is None:
            scored = test
        elif test != scored:
            raise SystemExit(f"{run_dir} scores a different gene set from the first run - "
                             f"validation and test runs cannot be mixed")
        column = [gene_column[str(g)] for g in data.gene_names]
        control = data.cells(data.control_condition)
        pick = np.random.default_rng(0).choice(control.shape[0],
                                               size=min(args.n_cells, control.shape[0]),
                                               replace=False)
        sample = control[pick]
        distances = []
        for condition in test:
            predicted = control_mean.copy()
            predicted[column] = predict_cells(model, sample, condition, data.pert_index,
                                              args.device, data.naming).mean(axis=0)
            distances.append(float(np.linalg.norm(predicted - means[row_of[condition]])))
        per_arm.setdefault(arm_of(config), []).append(np.asarray(distances))
        validation = bool(config["split"].get("validation"))
        setting = (float(config["model"].get("perturbation_dropout", 0.0)),
                   int(config["train"]["epochs"]))
        by_setting.setdefault(setting, []).append(float(np.mean(distances)))
        operators = model.modulation.operators
        if covered is None and getattr(operators, "knowledge", None) is not None:
            k = operators.knowledge.detach().cpu().numpy()
            covered = np.array([np.abs(k[data.pert_index[data.naming.genes(c)[0]]]).sum() > 0
                                for c in test])
        print(f"  {os.path.basename(os.path.normpath(run_dir)):18s} {arm_of(config):18s} "
              f"seed {config['train']['seed']}  mean L2 {np.mean(distances):.4f}")

    if scored is None:
        raise SystemExit("no runs")
    if validation:
        # H0 (DESIGN_ENCODER.md 5): the setting whose runs - real AND shuffled knowledge
        # together - score lowest on the validation genes. Choosing on the real arm
        # alone would hand it the setting that suits it.
        print(f"\nH0 on {len(scored)} validation genes: mean L2 over every run of a setting")
        print(f"  {'dropout':>8s} {'epochs':>7s} {'mean L2':>9s}  runs")
        ranked = sorted(by_setting.items(), key=lambda item: np.mean(item[1]))
        for (dropout, epochs), values in ranked:
            print(f"  {dropout:8.2f} {epochs:7d} {np.mean(values):9.4f}  {len(values)}")
        (dropout, epochs), _ = ranked[0]
        print(f"\n  H0 chooses dropout {dropout} and {epochs} epochs: "
              f"P={dropout} EPOCHS={epochs} for the test runs")
        return
    truth = means[[row_of[c] for c in scored]]
    rows = {arm: np.mean(np.stack(seeds), axis=0) for arm, seeds in per_arm.items()}
    rows["control"] = np.linalg.norm(truth - control_mean, axis=1)
    rows["mean"] = np.linalg.norm(truth - mean_baseline, axis=1)
    seeds = {arm: len(v) for arm, v in per_arm.items()}

    print(f"\n{len(scored)} scored genes, L2 over all {len(all_genes):,} measured genes"
          + (", mean bounded" if args.bound_mean else ""))
    print(f"  {'arm':20s} {'mean L2':>9s} {'median':>9s}  seeds")
    for arm, values in sorted(rows.items(), key=lambda item: item[1].mean()):
        print(f"  {arm:20s} {values.mean():9.4f} {np.median(values):9.4f}  "
              f"{seeds.get(arm, '-')}")

    real = next((a for a in rows if a in ("string", "moa")), None)
    if real is None:
        print("\nno real-knowledge arm among the runs - nothing to decide")
        return
    print(f"\n  {real} against each other row, paired over {len(scored)} genes "
          f"(one-sided Wilcoxon, {real} lower):")
    verdict = {}
    for other in sorted(rows):
        if other == real:
            continue
        p, gain, wins = paired(real, rows[real], other, rows[other])
        verdict[other] = p < ALPHA and gain > 0
        print(f"    vs {other:20s} p {p:.2e}  median gain {gain:+.4f}  "
              f"{real} better on {wins}/{len(scored)}")
        if covered is not None and other.endswith("-shuffled"):
            for label, mask in (("covered", covered), ("not covered", ~covered)):
                if mask.any():
                    sub = rows[other][mask] - rows[real][mask]
                    print(f"       {label:12s} n={int(mask.sum()):3d}  "
                          f"median gain {np.median(sub):+.4f}")
    shuffled = f"{real}-shuffled"
    if shuffled in verdict:
        print(f"\n  V1 ({real} beats {shuffled}): {'PASSES' if verdict[shuffled] else 'FAILS'}")
    print(f"  V2 ({real} beats the mean baseline): "
          f"{'PASSES' if verdict['mean'] else 'FAILS'}")
    print(f"  (decided at p < {ALPHA}; seeds per arm {seeds}; "
          f"DESIGN_ENCODER.md 5 needs 3 seeds for a verdict)")

    if args.per_gene:
        print("")
        for i, condition in enumerate(scored):
            print(f"  {condition:20s} " + "  ".join(f"{arm} {rows[arm][i]:.4f}"
                                                   for arm in sorted(rows)))


if __name__ == "__main__":
    main()
