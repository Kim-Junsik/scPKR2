#!/usr/bin/env bash
# The shrink term on Table 3's model: one check on ComboSciPlex.
#
#   bash scripts/shrink_arms.sh                         arm PS, seeds 0-2, the weight set from cDfmP_s0-2
#   SEEDS="3 4 5 6 7 8 9" SHRINK=<w> bash scripts/shrink_arms.sh   more seeds at the weight already printed
#
# THE ARM. cDfmP (scripts/metric_arms.sh, DATA=combo ARMS=P: the final Table 3 model, pearson
# weight 10) with one change, train.shrink_weight > 0:
#
#   PS  + sum_a ||A_a - A_bar||_F^2, each drug charged once per epoch (src/config.py)
#
# WHY. Table 3's Pearson-delta gap is two conditions, Alvespimycin and SRT2104+Alvespimycin
# (0.28 and 0.14 under P, the other five 0.94-0.99). Alvespimycin is never trained alone.
# It appears in Panobinostat+Alvespimycin, which the HDAC response dominates, and in
# Alvespimycin+Pirarubicin, and Pirarubicin appears in no other condition. Nothing stops
# Pirarubicin from taking up that whole combination. The shrink term makes each drug pay
# for its distance from the average drug, so a combination's response is split between its
# two drugs unless the data say otherwise.
#
# THE WEIGHT is fixed BEFORE any PS run: scripts/shrink_weight.py sets it so that the term
# is SHARE (default 5 %) of the per-step loss at the trained cDfmP_s0-2, which never had
# it. It is printed with each drug's distance from the average. Pass SHRINK=<w> to reuse
# it for more seeds.
#
# Each run is exported and scored right after it trains, under both gates (DE is read from
# sample, the rest from soft). Then compare, on the same seeds:
#
#   python scripts/arm_table.py --group all --device cuda --per-run P='results/runs/cDfmP_s[012]' PS='results/runs/cDfmPS_s*'

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-2500}"
SHARE="${SHARE:-0.05}"
REF="${REF:-results/runs/cDfmP_s0 results/runs/cDfmP_s1 results/runs/cDfmP_s2}"
CELLEVAL_PY="${CELLEVAL_PY:-.env-ce-scdfm/bin/python}"
THREADS="${THREADS:-16}"
CACHE="assets/combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad"

if [ ! -x "$CELLEVAL_PY" ]; then
  echo "no scoring interpreter at $CELLEVAL_PY (python -m venv .env-ce-scdfm && .env-ce-scdfm/bin/pip install -r requirements-celleval-scdfm.txt)" >&2
  exit 2
fi
if [ ! -f "$CACHE" ]; then
  echo "missing $CACHE - the reference runs were trained on it (scripts/best_fm_table3.sh builds it)" >&2
  exit 2
fi

if [ -z "${SHRINK:-}" ]; then
  echo ""
  echo "=== shrink weight from ${REF} (share ${SHARE}) ==="
  # shellcheck disable=SC2086
  SHRINK="$(python scripts/shrink_weight.py $REF --share "$SHARE" --device "$DEVICE" | tail -n 1)"
fi
echo "train.shrink_weight=${SHRINK}"

for seed in $SEEDS; do
  tag="cDfmPS_s${seed}"
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [combo, pearson 10, shrink ${SHRINK}] ==="
  python scripts/train.py --tag "$tag" --force --set \
    "train.device=${DEVICE}" \
    train.gene_endpoint_weight=10 \
    train.fm_weight=1.0 \
    train.endpoint_weight=0 \
    train.fm_sigma=0.0 \
    train.coupling=uot \
    model.observable_whiten=zca \
    model.response=generator \
    data.raw_h5ad=data/combosciplex/combosciplex.h5ad \
    data.control_label=control \
    data.normalise_from_counts=counts \
    data.n_hvg=5000 \
    data.hvg_criterion=scanpy \
    split.source=list \
    split.method=additive \
    split.fold=0 \
    "data.cache_h5ad=${CACHE}" \
    model.perturbation_dropout=0.25 \
    model.perturbation_knowledge=none \
    "train.seed=${seed}" \
    "train.epochs=${EPOCHS}" \
    train.pearson_weight=10 \
    train.discrim_weight=0 \
    "train.shrink_weight=${SHRINK}"
  for gate in soft sample; do
    # --profile full: the default 'minimal' has no de_spearman_sig, the DE column.
    python scripts/run_celleval.py "results/runs/${tag}" --gate "$gate" --bound-mean \
      --infer-top-gene 1000 --pred-cells 128 --results-tag ce0542 --profile full \
      --celleval-python "$CELLEVAL_PY" --threads "$THREADS"
  done
done
