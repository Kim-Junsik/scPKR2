"""The validation campaign: the final models and their arms on the validation splits, scored
on all six table columns, one shell queue per GPU slot.

    python scripts/val_queue.py --name v0 --arms base= --gpus 0 1 2 3 --slots 2

then, from the repository root, one tmux pane per printed queue:

    sh results/val/v0/gpu0a.sh

and, when the queues are done, the comparison commands in results/val/v0/compare.sh.

WHAT IS SCORED. Validation sets only - no reported test condition is ever trained on,
used to choose genes, or scored here:

  t1v    Norman additive fold 0, NORMAN_VALIDATION[0]: 15 training doubles held out.
         Stands in for Table 1. Scored block: double.
  t2v    Norman combinations fold 0, the same 15 doubles AND the singles of their
         genes held out (splits.norman_validation_combinations). Stands in for Table 2.
         Scored blocks: single and double, exported and scored apart as Table 2 is.
  t3v0-2 ComboSciPlex validation folds 0-2, four combinations each plus one held-out
         single (split.validation_singles). Stand in for Table 3. Scored block: all.

The real test conditions of each split go into the fold's `excluded` list: neither
trained on nor scored, and data.exclude_test_from_hvg keeps them out of gene selection.
Every family gets its own cache for that reason (assets/val_<family>.h5ad).

WHAT EVERY ARM STARTS FROM. The final models, setting for setting:
  Norman      scripts/final_p0.sh (nEp0fm): STRING knowledge, perturbation dropout 0,
              250 epochs.
  ComboSciPlex scripts/best_fm_table3.sh ARM=D + train.pearson_weight=10 (cDfmP): no
              knowledge, dropout 0.25, 2,500 epochs.
An arm is a list of config overrides applied after those, so it can change anything.
--arms-for restricts an arm to some tables (t1 t2 t3), e.g. a Pearson weight that only
means something on ComboSciPlex.

SCORING follows the main tables exactly (scripts/arm_table.py reads it): cell-eval 0.5.42
under both gates, --bound-mean, 1,000 test-subset HVGs, 128 predicted cells, --profile full
(the default profile has no de_spearman_sig). L2 is computed by arm_table.py itself from
N_p cells. A marker file per (block, gate) makes a restarted queue skip finished scoring,
and an existing checkpoint makes it skip finished training.

RESCORING UNDER AN EFFECT SCALE. --effect-scale t1=1.3 t3=1.1 --out v0e writes queues that
rescore the existing runs of --name with run_celleval.py --effect-scale (eval.effect_scale:
every predicted mean becomes x + S (mu - x)), into their own folders and markers, and a
compare.sh that shows each arm as trained and rescaled side by side. Chosen per table from
scripts/delta_scale.py --sweep on these same validation runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NAME = re.compile(r"^[A-Za-z0-9]+$")

COMMON_TRAIN = [
    "train.device=cuda",
    "train.gene_endpoint_weight=10",
    "train.fm_weight=1.0",
    "train.endpoint_weight=0",
    "train.fm_sigma=0.0",
    "train.coupling=uot",
    "model.observable_whiten=zca",
    "model.response=generator",
]

NORMAN_DATA = [
    "data.raw_h5ad=data/norman/norman.h5ad",
    "data.n_hvg=5000",
    "data.hvg_criterion=scanpy",
    "data.force_include_targets=true",
    "data.exclude_test_from_hvg=true",
    "split.source=reference_pkl",
    "split.fold=0",
    "split.validation=true",
]
NORMAN_MODEL = COMMON_TRAIN + [
    "train.epochs=250",
    "model.perturbation_dropout=0",
    "model.perturbation_knowledge=string",
]

COMBO_DATA = [
    "data.raw_h5ad=data/combosciplex/combosciplex.h5ad",
    "data.control_label=control",
    "data.normalise_from_counts=counts",
    "data.n_hvg=5000",
    "data.hvg_criterion=scanpy",
    "data.force_include_targets=true",
    "data.exclude_test_from_hvg=true",
    "split.source=list",
    "split.method=additive",
    "split.fold=0",
    "split.validation=true",
    "split.validation_singles=true",
]
COMBO_MODEL = COMMON_TRAIN + [
    "train.epochs=2500",
    "model.perturbation_dropout=0.25",
    "model.perturbation_knowledge=none",
    "train.pearson_weight=10",
]

# family -> (table, data overrides, model overrides, scored blocks)
FAMILIES = {
    "t1v": ("t1", NORMAN_DATA + ["split.method=additive"], NORMAN_MODEL, ["double"]),
    "t2v": ("t2", NORMAN_DATA + ["split.method=combinations"], NORMAN_MODEL,
            ["single", "double"]),
    **{f"t3v{k}": ("t3", COMBO_DATA + [f"split.validation_fold={k}"], COMBO_MODEL, ["all"])
       for k in range(3)},
}
TABLES = ("t1", "t2", "t3")


def cache_path(family: str) -> str:
    return f"assets/val_{family}.h5ad"


def plan(name: str, arms: dict[str, list[str]], arms_for: dict[str, set[str]],
         tables: list[str], seeds: list[int], queues: list[tuple[int, str]]) -> list[dict]:
    for label in [name, *arms]:
        if not NAME.match(label):
            raise ValueError(f"names must be letters and digits only, got {label!r}")
    jobs = []
    for family, (table, data, model, blocks) in FAMILIES.items():
        if table not in tables:
            continue
        cache = cache_path(family)
        for arm, overrides in arms.items():
            if arm in arms_for and table not in arms_for[arm]:
                continue
            for seed in seeds:
                jobs.append({
                    "tag": f"{name}_{arm}_{family}_s{seed}",
                    "family": family, "table": table, "arm": arm, "seed": seed,
                    "cache": cache, "blocks": blocks, "data": data,
                    # Arm overrides last, so an arm can replace any final-model setting.
                    "overrides": data + [f"data.cache_h5ad={cache}"] + model
                                 + [f"train.seed={seed}"] + overrides,
                })
    if not jobs:
        raise ValueError("no jobs: check --tables and --arms-for")
    # Longest first (ComboSciPlex trains ten times the epochs), then round-robin, so the
    # queues finish at about the same time.
    jobs.sort(key=lambda j: (j["table"] != "t3", j["family"], j["arm"], j["seed"]))
    for index, job in enumerate(jobs):
        gpu, slot = queues[index % len(queues)]
        job["gpu"], job["slot"] = gpu, slot
    return jobs


def queue_lines(jobs: list[dict], gpu: int, slot: str, first: bool,
                caches: dict[str, list[str]]) -> list[str]:
    lines = ["#!/bin/sh",
             "# Generated by scripts/val_queue.py. One queue, sequential. Run from the repo root.",
             f"# {len(jobs)} runs on GPU {gpu}, slot {slot}.",
             "set -e",
             'cd "$(dirname "$0")/../../.."',
             'CELLEVAL_PY="${CELLEVAL_PY:-.env-ce-scdfm/bin/python}"',
             'THREADS="${THREADS:-8}"',
             'if [ ! -x "$CELLEVAL_PY" ]; then echo "no scoring interpreter at $CELLEVAL_PY" >&2; exit 2; fi',
             ""]
    if first:
        lines += ["# Caches first, on this queue only; every other queue waits for them.",
                  "# Two queues building the same cache at once would read a half-written file.",
                  ""]
        for cache, data in sorted(caches.items()):
            lines += [f'if [ ! -f "{cache}" ]; then',
                      f'  python scripts/build_data.py --set {" ".join(data)} data.cache_h5ad={cache}',
                      "fi", ""]
    else:
        lines += ["# Wait for the caches the first queue builds.", ""]
        for cache in sorted(caches):
            lines += [f'while [ ! -f "{cache}" ]; do sleep 30; done']
        lines += [""]
    for job in jobs:
        run = f'results/runs/{job["tag"]}'
        lines += [f'echo "=== {job["tag"]} ==="',
                  f'if [ ! -f "{run}/checkpoint.pt" ]; then',
                  f'  CUDA_VISIBLE_DEVICES={gpu} python scripts/train.py --tag {job["tag"]} '
                  f'--set {" ".join(job["overrides"])}',
                  "fi"]
        effect = job.get("effect")
        scaled = f" --effect-scale {effect:g}" if effect else ""
        suffix = f"_e{effect:g}".replace(".", "p") if effect else ""
        for block in job["blocks"]:
            for gate in ("soft", "sample"):
                marker = f"{run}/.val_scored_{block}_{gate}{suffix}"
                lines += [f'if [ ! -f "{marker}" ]; then',
                          f'  CUDA_VISIBLE_DEVICES={gpu} python scripts/run_celleval.py {run} '
                          f'--gate {gate} --bound-mean --infer-top-gene 1000 --pred-cells 128 '
                          f'--results-tag ce0542 --profile full --celleval-python "$CELLEVAL_PY" '
                          f'--threads "$THREADS" --group {block}{scaled}',
                          f'  touch "{marker}"',
                          "fi"]
        lines += [""]
    lines += ['echo "=== queue done ==="', ""]
    return lines


def compare_lines(name: str, jobs: list[dict]) -> list[str]:
    """arm_table.py commands, one per (family, block): every arm of the family side by side,
    the first arm as the reference, each run's row under its arm."""
    lines = ["#!/bin/sh", "# Generated by scripts/val_queue.py. Run from the repo root.",
             'cd "$(dirname "$0")/../../.."', ""]
    for family, (_, _, _, blocks) in FAMILIES.items():
        arms, effect = [], None
        for job in jobs:
            if job["family"] == family and job["arm"] not in arms:
                arms.append(job["arm"])
            if job["family"] == family and job.get("effect"):
                effect = job["effect"]
        if not arms:
            continue
        # The final model first: arm_table.py takes the first arm as the reference that
        # every other arm's diff row is measured from (the jobs are sorted by arm name).
        arms.sort(key=lambda arm: arm != "base")
        # Under --effect-scale every arm is shown as trained and rescaled, side by side.
        labels = [(arm, "") for arm in arms] + (
            [(arm, f":{effect:g}") for arm in arms] if effect else [])
        globs = " ".join(f"{arm}{scale}='results/runs/{name}_{arm}_{family}_s*'"
                         for arm, scale in labels)
        for block in blocks:
            # Every queue scores with --group <block>, so a single or double block is always
            # in its own export folder (t1v included), and the table must read that one.
            export = " --group-export" if block in ("single", "double") else ""
            lines += [f'echo "=== {family} {block} ==="',
                      f"python scripts/arm_table.py --group {block}{export} --device cuda "
                      f"--per-run {globs}", ""]
    return lines


def write(name: str, jobs: list[dict], queues: list[tuple[int, str]], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    caches: dict[str, list[str]] = {}
    for job in jobs:
        caches.setdefault(job["cache"], job["data"])
    paths = []
    for index, (gpu, slot) in enumerate(queues):
        mine = [job for job in jobs if job["gpu"] == gpu and job["slot"] == slot]
        path = os.path.join(out_dir, f"gpu{gpu}{slot}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(queue_lines(mine, gpu, slot, index == 0, caches)))
        os.chmod(path, 0o755)
        paths.append((path, len(mine)))
    compare = os.path.join(out_dir, "compare.sh")
    with open(compare, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(compare_lines(name, jobs)))
    os.chmod(compare, 0o755)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"name": name, "jobs": jobs}, handle, indent=2)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="letters and digits only")
    parser.add_argument("--arms", nargs="+", required=True,
                        help='label="key=value key=value"; base= is the final model as is')
    parser.add_argument("--arms-for", nargs="*", default=[],
                        help="label=t1,t3 restricts an arm to those tables")
    parser.add_argument("--tables", nargs="+", default=list(TABLES), choices=TABLES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--effect-scale", nargs="*", default=[], metavar="TABLE=S",
                        help="t1=1.3 t3=1.1: rescore the runs of --name under "
                             "eval.effect_scale S (training is skipped where a checkpoint "
                             "exists). Needs --out, so the training queues are not overwritten")
    parser.add_argument("--out", default=None,
                        help="the folder under results/val/ for the queues and compare.sh "
                             "(default: --name)")
    parser.add_argument("--slots", type=int, default=2,
                        help="queues per GPU. One run uses a few GB, so several fit in 46 GB; "
                             "cell-eval scoring is CPU-bound, so THREADS x queues should not "
                             "exceed the cores")
    args = parser.parse_args()
    os.chdir(ROOT)  # the queues and compare.sh land in results/val/ whatever the shell's folder

    arms: dict[str, list[str]] = {}
    for item in args.arms:
        if "=" not in item:
            raise SystemExit(f'--arms takes label="overrides", got {item!r}')
        label, value = item.split("=", 1)
        arms[label] = value.split()
    arms_for: dict[str, set[str]] = {}
    for item in args.arms_for:
        label, value = item.split("=", 1)
        if label not in arms:
            raise SystemExit(f"--arms-for names an unknown arm {label!r}")
        arms_for[label] = set(value.split(","))
    if not 1 <= args.slots <= 26:
        raise SystemExit("--slots must be 1-26")
    queues = [(gpu, chr(ord("a") + s)) for s in range(args.slots) for gpu in args.gpus]

    effects: dict[str, float] = {}
    for item in args.effect_scale:
        table, _, value = item.partition("=")
        if table not in TABLES or not value:
            raise SystemExit(f"--effect-scale takes t1=S t2=S t3=S, got {item!r}")
        effects[table] = float(value)
    out = args.out or args.name
    if effects and out == args.name:
        raise SystemExit("--effect-scale needs --out: its queues would replace the training ones")
    if not NAME.match(out):
        raise SystemExit(f"--out must be letters and digits only, got {out!r}")

    jobs = plan(args.name, arms, arms_for, args.tables, args.seeds, queues)
    for job in jobs:
        job["effect"] = effects.get(job["table"])
    out_dir = os.path.join("results", "val", out)
    paths = write(args.name, jobs, queues, out_dir)
    print(f"{len(jobs)} runs over {len(queues)} queues -> {out_dir}")
    for path, count in paths:
        print(f"  sh {path}   ({count} runs)")
    print(f"  then: sh {os.path.join(out_dir, 'compare.sh')}")


if __name__ == "__main__":
    main()
