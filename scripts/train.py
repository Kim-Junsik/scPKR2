"""Train one configuration end to end.

    python scripts/train.py --tag ftA --set model.composition=anticommutator
    python scripts/train.py --tag smoke --set train.epochs=5 train.device=cpu

Every value lives in src/config.py and any key can be overridden with dot notation. The
resolved config is written next to the results, so a run is reproducible from its own
directory.

THE PREMISE IS CHECKED BEFORE TRAINING, ON THIS RUN'S OWN DATA. An untrained model must
predict exactly the additive baseline: W starts at zero and the head is mean-preserving,
so the residual is identically zero and the prediction is m_ctrl + sum_a w_a. That is
worth a unit test (tests/test_structure.py has it) and also worth checking here, every
run, because the test uses a synthetic observable space and this uses the real one. The
check reads TRAINING conditions only.

It is not a formality. The first version of the head clamped the mean at zero per cell,
which broke the premise on real data - 2.7318 where ridge_additive scores 1.8577 - while
every structural test still passed, because those are about the displacement and that bug
was about the mean of a nonlinearity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module
from src.data import splits
from src.eval import baselines
from src.eval.diagnostics import _dataset, build_model, training_rows
from src.eval.predict import evaluate_model
from src.train.loop import train


def build_logger(path: str):
    handle = open(path, "a", encoding="utf-8")

    def log(message: str = "") -> None:
        print(message)
        handle.write(message + "\n")
        handle.flush()

    return log, handle


@torch.no_grad()
def verify_premise(model, data, stats, conditions: list[str], log,
                   tolerance: float = 1e-3) -> None:
    """What this model guarantees by construction, checked as numbers before training.

    v2's premise was that an untrained model IS the additive baseline, 1.6690. That
    protection is gone on purpose - it came from a closed-form term that also carried
    88 % of the signal and left the learned part with nothing to do on single
    perturbations. What is left is weaker and still worth asserting:

      mu >= 0 ALWAYS. The whole reason for x exp(u) + softplus(v). 33.8 % of v2's
      per-cell predictions on ComboSciPlex were negative, which no non-negative
      realisation can reproduce - flooring them cost 0.80 of L2 there. A single
      negative here means the parameterisation is not doing what it is for.

      THE UNTRAINED MODEL IS THE CONTROL PLUS EXACTLY softplus(turn_on_init). The output
      maps start at zero, so every output is its bias: u = 0, and the turn-on term is the
      same constant on every gene of every cell. That constant is a CHOSEN quantity, not
      a tolerance - asserting the deviation equals it is a stronger check than asserting
      the deviation is small, and it is what lets turn_on_init be swept at all.

      The starting point is Control, 3.9937, and the run has to climb from there to beat
      ridge's 1.6690. Asserting it means the run begins somewhere nameable rather than at
      a random gene-space field.

      ORDER DOES NOT MATTER. e_S is a sum, so A+B and B+A are the same prediction. A
      simultaneous perturbation has no order and no training signal would impose this.
    """
    device = model.modulation.to_u.bias.device
    control = torch.as_tensor(data.cells(data.control_condition), device=device)[:256]
    offset = float(torch.nn.functional.softplus(
        torch.tensor(float(model.modulation.to_v.bias[0]))))
    worst_negative, worst_control, worst_order = 0.0, 0.0, 0.0
    for condition in conditions[:8]:
        perturbations = [data.pert_index[g] for g in data.naming.genes(condition)]
        mu = model(control, perturbations)["mean"]
        worst_negative = min(worst_negative, float(mu.min()))
        worst_control = max(worst_control, float((mu - control - offset).abs().max()))
        if len(perturbations) > 1:
            flipped = model(control, list(reversed(perturbations)))["mean"]
            worst_order = max(worst_order, float((mu - flipped).abs().max()))

    # With model.init_from_ridge the untrained model is the control scaled by fitted
    # fold changes, so it is NOT the control and must not be asserted to be. The other
    # two claims - non-negativity and order invariance - hold either way and are the
    # ones that matter.
    from_ridge = bool(model.modulation.direct
                      and float(model.modulation.u_direct.abs().max()) > 0)
    log(f"  premise: min mu {worst_negative:.2e}, "
        + (f"started from a ridge fit of the log fold changes, so it is the control "
           f"scaled rather than the control ({worst_control:.3f} away), "
           if from_ridge else
           f"untrained prediction equals the control plus softplus(v)={offset:.2e} "
           f"to {worst_control:.2e}, ")
        + f"order-invariant to {worst_order:.2e}")
    if from_ridge:
        worst_control = 0.0
    if worst_negative < 0.0:
        raise SystemExit(
            f"mu went negative ({worst_negative:.3e}). x exp(u) + softplus(v) cannot do "
            f"that, so the parameterisation is not what is running. This is the defect "
            f"the whole rebuild exists to remove - see docs/DESIGN.md C2.")
    if worst_control > tolerance or worst_order > tolerance:
        raise SystemExit(
            f"an untrained model must be the control plus softplus(turn_on_init) "
            f"({worst_control:.3e} off it) and must ignore the order of a combination "
            f"({worst_order:.3e}); tolerance {tolerance:g}. Starting anywhere else means "
            f"the run begins at an unnamed point and nothing downstream can be compared "
            f"to a baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--tag", default=None, help="run directory name")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing checkpoint.pt. Off by default: "
                             "scPKFM opened run directories with exist_ok and "
                             "overwrote checkpoints without a word, so a reused tag "
                             "destroyed the run it was being compared against.")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    device = config["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable, falling back to cpu")
        device = config["train"]["device"] = config["eval"]["device"] = "cpu"

    tag = args.tag or f"scpkr_{time.strftime('%m%d-%H%M')}"
    run_dir = os.path.join(config["train"]["out_dir"], tag)
    checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(checkpoint_path) and not args.force:
        raise SystemExit(f"{checkpoint_path} exists. Use a new --tag, or --force to "
                         f"overwrite it deliberately.")
    os.makedirs(run_dir, exist_ok=True)
    log, handle = build_logger(os.path.join(run_dir, "train.log"))

    # The weights are drawn under init_seed and everything after under train.seed, so
    # the two sources can be moved independently. build_model runs below, after this.
    init_seed = config["train"]["init_seed"]
    torch.manual_seed(config["train"]["seed"] if init_seed is None else int(init_seed))
    rng = np.random.default_rng(config["train"]["seed"])

    log(f"run {tag}")
    log(f"config {json.dumps(config['model'])}")
    log("")

    data, stats = _dataset(config)
    method = config["split"]["method"]
    fold = splits.folds(config, method)[config["split"]["fold"]]
    log(f"data: {data.x.shape[0]:,} cells x {data.n_genes:,} genes, "
        f"{data.n_perturbations} perturbations, {len(data.conditions)} conditions")

    log("\n=== the additive component (closed form, not learned) ===")
    model, train_conditions, rows = build_model(config, data, stats, fold, method, device)
    # Back to train.seed for everything downstream of construction: the batch order and
    # the coupling must not inherit init_seed, or the two would move together again.
    torch.manual_seed(config["train"]["seed"])
    # Leak accounting, in the log so it is in the artifact rather than in someone's head.
    held_out = data.x.shape[0] - len(rows)
    log(f"  cells visible to training: {len(rows):,} / {data.x.shape[0]:,}  "
        f"({held_out:,} held-out cells excluded from the ridge fit, from Phi's "
        f"standardisation and from its anchor ranking)")
    log(f"  training conditions: {len(train_conditions)}")
    seen = {data.pert_index[g] for c in train_conditions if stats.has(c)
            for g in data.naming.genes(c) if g in data.pert_index}
    log(f"  perturbations with training data: {len(seen)} / {data.n_perturbations}"
        + ("" if len(seen) == data.n_perturbations else
           "  <- the rest keep their initial embedding and predict the control"))

    log("\n=== the observables (fixed) ===")
    for key, value in model.observables.summary().items():
        log(f"  {key:20s} {value}")

    log("\n=== the model ===")
    log(f"  {model.extra_repr()}")
    for key, value in model.learned_parameters().items():
        log(f"  {key:32s} {value:,}")

    log("\n=== premise check (training conditions only) ===")
    verify_premise(model, data, stats,
                   [c for c in train_conditions if stats.has(c)], log)

    log("\n=== training ===")
    parts = train(model, data, stats, train_conditions, config, device, rng, log,
                  run_dir=run_dir)

    log("\n=== evaluation (same protocol as the baselines) ===")
    results = evaluate_model(model, data, stats, [fold], method, config, rng)
    for key, value in results.items():
        log(f"  {key:24s} {value}")
    log("  resid_R2 is measured against the ADDITIVE arithmetic, so 0.0 means tying it. "
        "An untrained run of this model sits there by construction, and the additive "
        "component is a stronger predictor than that: see edist_rel_control.")

    torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as out:
        json.dump({"config": config, "results": results, "train_loss": parts}, out,
                  indent=2)
    log(f"-> {run_dir}")
    handle.close()


if __name__ == "__main__":
    main()
