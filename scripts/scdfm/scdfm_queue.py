"""scDFM on our five validation splits: one shell queue per GPU (or per GPU pair).

    python scripts/scdfm/scdfm_queue.py --name v0

then, one tmux pane each, the first queue first (it prepares the data the others wait for):

    sh results/scdfm/v0/gpu0.sh
    sh results/scdfm/v0/gpu1.sh   ...

WHAT EACH FAMILY RUNS. scDFM's released recipe (its run.sh), with the patch in
scripts/scdfm/scdfm_val.patch applied to /mnt/data/dev/scDFM:
  - the split: results/scdfm/splits/<family>.json from export_splits.py - the real test
    conditions dropped before scDFM computes anything, our validation set scored;
  - batch 48, 200,000 steps (the loop now stops there; the released one ran on to the end
    of its pass over the data), lr 5e-5 cosine to 1e-6, MMD weight 0.5, signed top-30
    gene mask, RK4 sampling, bf16;
  - Norman: CRISPR perturbations embedded through the gene encoder (crisper);
    ComboSciPlex: the dedicated drug-embedding table (any value but crisper selects it).
Its model takes hidden 512, 4 layers, 8 heads from its own defaults: run.sh's --d_model
only names the result folder.

BATCH 48 ON A 44 GB GPU. One GPU does not hold 48. --mode single (default) runs 2 x 24 on
one GPU with gradient accumulation (SCDFM_ACCUM=2): no NCCL, so no /dev/shm, which kang
does not have enough of for NCCL. --mode pairs runs 2 GPUs x 24 under DDP instead - the
same update, half the wall time per run, half the runs at once - and needs a /dev/shm
large enough for NCCL.

Checkpoints are written every --print-every steps and at the last step. Each one predicts
128 cells per condition (SCDFM_NO_EVAL=1: prediction only) and score_scdfm.py scores them
with scPKR2's scorer once training ends, so every checkpoint gets a row.

PREPARATION happens once, in the first queue, single process, before any training: the
split files, then each family's gene mask, vocabulary and (ComboSciPlex, the first time)
processed.h5ad - without scDFM's PubChem annotation (SCDFM_SKIP_ANNOTATE), whose columns
nothing in scDFM reads. The other queues wait for its marker; no file is written twice.
"""

from __future__ import annotations

import argparse
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FAMILIES = {
    # family: (data_name, perturbation_function, blocks to score, conditions scored)
    "t1v": ("norman", "crisper", ["double"], 15),
    "t2v": ("norman", "crisper", ["single", "double"], 36),
    "t3v0": ("combosciplex", "drug", ["all"], 5),
    "t3v1": ("combosciplex", "drug", ["all"], 5),
    "t3v2": ("combosciplex", "drug", ["all"], 5),
}
ORDER = ["t2v", "t1v", "t3v0", "t3v1", "t3v2"]
# Measured on kang (Norman, 2 GPUs x 24): 5,000 steps in 65 min, and one prediction of 24
# cells per condition in 19 s. Used only to balance the queues.
TRAIN_HOURS = {"pairs": 200000 / 5000 * 65 / 60, "single": 2 * 200000 / 5000 * 65 / 60}


def hours(family: str, mode: str, steps: int, print_every: int) -> float:
    predictions = steps // print_every + 2
    conditions = FAMILIES[family][3]
    return TRAIN_HOURS[mode] * steps / 200000 + predictions * conditions * 6 * 19 / 3600


def run_args(family: str, steps: int, print_every: int, name: str) -> str:
    data_name, function, _, _ = FAMILIES[family]
    return " ".join([
        "src/script/run.py", "--batch_size=24", "--model_type=origin", "--lr=5e-5",
        f"--steps={steps}", f"--print_every={print_every}", f"--data_name={data_name}",
        "--d_model=128", "--eta_min=1e-6", "--fusion_method=differential_perceiver",
        "--infer_top_gene=1000", "--n_top_genes=5000",
        f"--result_path=./result/{name}/{family}", f"--perturbation_function={function}",
        "--noise_type=Gaussian", "--mode=predict_y", "--gamma=0.5",
        "--split_method=additive", "--use_mmd_loss", "--fold=0", "--topk=30",
        "--use_negative_edge"])


def env(family: str, split_dir: str) -> str:
    # SCDFM_SKIP_ANNOTATE: ComboSciPlex's first preparation would query PubChem through
    # pertpy for drug annotations that nothing in scDFM reads (drugs are a learned table).
    return (f"SCDFM_SPLIT_JSON={split_dir}/{family}.json SCDFM_TAG=_{family} "
            "SCDFM_SKIP_ANNOTATE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scdfm-dir", default="/mnt/data/dev/scDFM")
    parser.add_argument("--families", nargs="+", default=ORDER, choices=ORDER)
    parser.add_argument("--mode", choices=["single", "pairs"], default="single")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"],
                        help="single: one queue per GPU. pairs: one queue per two GPUs")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--print-every", type=int, default=20000,
                        help="checkpoint and predict every this many steps. scDFM's 5000 "
                             "made prediction cost more than training on 36 conditions")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    os.chdir(ROOT)

    if args.mode == "single":
        slots = [[gpu] for gpu in args.gpus]
    else:
        if len(args.gpus) % 2:
            raise SystemExit("--mode pairs needs an even number of --gpus")
        slots = [args.gpus[i:i + 2] for i in range(0, len(args.gpus), 2)]

    # Longest first, each to the queue that would finish soonest.
    families = [f for f in ORDER if f in args.families]
    load = [0.0] * len(slots)
    mine: list[list[str]] = [[] for _ in slots]
    for family in sorted(families, key=lambda f: -hours(f, args.mode, args.steps,
                                                          args.print_every)):
        index = min(range(len(slots)), key=lambda i: load[i])
        mine[index].append(family)
        load[index] += hours(family, args.mode, args.steps, args.print_every)

    out_dir = os.path.join("results", "scdfm", args.name)
    os.makedirs(out_dir, exist_ok=True)
    split_dir = os.path.join(ROOT, "results", "scdfm", "splits")
    marker = os.path.join(ROOT, out_dir, "prep.done")
    scdfm = args.scdfm_dir

    for index, gpus in enumerate(slots):
        label = "gpu" + "".join(gpus)
        visible = ",".join(gpus)
        lines = ["#!/bin/sh",
                 f"# Generated by scripts/scdfm/scdfm_queue.py ({args.mode}). GPUs {visible}.",
                 "set -e", 'cd "$(dirname "$0")/../../.."', ""]
        if index == 0:
            lines += ["# Preparation, single process, before any training.",
                      "python scripts/scdfm/export_splits.py"]
            for family in families:
                lines += [f'echo "=== prepare {family} ==="',
                          f"SCDFM_PREP_ONLY=1 CUDA_VISIBLE_DEVICES={gpus[0]} {env(family, split_dir)} "
                          f"env -C {scdfm} .venv/bin/python "
                          f"{run_args(family, args.steps, args.print_every, args.name)}"]
            lines += [f'touch "{marker}"', ""]
        else:
            lines += ["# Wait for the first queue's preparation.",
                      f'while [ ! -f "{marker}" ]; do sleep 30; done', ""]
        for family in mine[index]:
            blocks = FAMILIES[family][2]
            result = f"{scdfm}/result/{args.name}/{family}"
            common = (f"SCDFM_NO_EVAL=1 SCDFM_WORKERS=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                      f"CUDA_VISIBLE_DEVICES={visible} {env(family, split_dir)} env -C {scdfm}")
            if args.mode == "single":
                launch = (f"SCDFM_ACCUM=2 ACCELERATE_MIXED_PRECISION=bf16 {common} .venv/bin/python")
            else:
                launch = (f"{common} .venv/bin/accelerate launch --multi_gpu --num_processes 2 "
                          f"--num_machines 1 --mixed_precision bf16 --dynamo_backend no "
                          f"--main_process_port {29500 + index}")
            lines += [f'echo "=== train {family} on GPU {visible} ==="',
                      f"{launch} {run_args(family, args.steps, args.print_every, args.name)}",
                      f'echo "=== score {family} ==="',
                      f"python scripts/scdfm/score_scdfm.py {result} --blocks {' '.join(blocks)} "
                      f"--threads {args.threads}", ""]
        lines += ['echo "=== queue done ==="', ""]
        path = os.path.join(out_dir, f"{label}.sh")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
        os.chmod(path, 0o755)
        print(f"  sh {path}   (GPU {visible}: {', '.join(mine[index]) or '-'}; "
              f"about {load[index] / 24:.1f} days)")


if __name__ == "__main__":
    main()
