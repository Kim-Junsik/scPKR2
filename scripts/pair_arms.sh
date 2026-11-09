#!/usr/bin/env bash
# The hurdle likelihood on the OT pairs (train.head_on_pairs=true): one check on the dev sets.
#
#   bash scripts/pair_arms.sh                                Table 1 dev: Norman additive fold 3
#   SPLIT=combinations bash scripts/pair_arms.sh             Table 2 dev: Norman combinations fold 3
#   DATA=combo bash scripts/pair_arms.sh                     Table 3: ComboSciPlex, seeds 0-2
#   SEEDS="0" bash scripts/pair_arms.sh                      part of it, one pane each
#
# WHAT IS BEING FIXED. The training step draws OT pairs (rows[i], columns[i]). Flow matching
# uses them. The hurdle likelihood did not: it compared the prediction for control cell i, in
# batch order, with target cell columns[i] - the partner of rows[i] - so each prediction met
# a random cell of the condition. Every run up to best-2026-11-06 was trained that way, and
# the tables are what those models score. This arm pairs the likelihood with the draw. Every
# other setting is the final model's (scripts/final_p0.sh; scripts/best_fm_table3.sh ARM=D
# + train.pearson_weight=10 for Table 3), so the comparison is one switch.
#
# THE REFERENCE RUNS EXIST: nEp0fm_f3_s0-2, nEp0fm_comb_f3_s0-2, cDfmP_s0-2.
#
# THE RULE, fixed before any of these runs is seen. Four blocks: T1, T2 Single, T2 Double, T3.
#   Adopt - and retrain all three tables - only if L2 is no worse in every block AND more
#   than half of the 24 columns (13 or more) improve. Otherwise the code stays as it is:
#   head_on_pairs stays false and the tables stand.
#
# Each run is exported and scored right after it trains, under both gates (the tables read
# DE from sample and the rest from soft). Then compare:
#
#   python scripts/arm_table.py --group double --device cuda --per-run base='results/runs/nEp0fm_f3_s[012]' H='results/runs/nEp0fmH_f3_s*'
#   python scripts/arm_table.py --group single --group-export --device cuda --per-run base='results/runs/nEp0fm_comb_f3_s[012]' H='results/runs/nEp0fmH_comb_f3_s*'
#   python scripts/arm_table.py --group double --group-export --device cuda --per-run base='results/runs/nEp0fm_comb_f3_s[012]' H='results/runs/nEp0fmH_comb_f3_s*'
#   python scripts/arm_table.py --group all --device cuda --per-run base='results/runs/cDfmP_s[012]' H='results/runs/cDfmPH_s*'

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DATA="${DATA:-norman}"
SPLIT="${SPLIT:-additive}"
SEEDS="${SEEDS:-0 1 2}"
FOLD="${FOLD:-3}"
CELLEVAL_PY="${CELLEVAL_PY:-.env-ce-scdfm/bin/python}"
THREADS="${THREADS:-16}"

if [ ! -x "$CELLEVAL_PY" ]; then
  echo "no scoring interpreter at $CELLEVAL_PY (python -m venv .env-ce-scdfm && .env-ce-scdfm/bin/pip install -r requirements-celleval-scdfm.txt)" >&2
  exit 2
fi

blocks="all"
case "$DATA" in
  norman)
    case "$SPLIT" in
      additive) prefix="nEp0fmH" ;;
      combinations) prefix="nEp0fmH_comb"; blocks="single double" ;;
      *) echo "unknown SPLIT '$SPLIT' (additive combinations)" >&2; exit 2 ;;
    esac
    EPOCHS="${EPOCHS:-250}"
    cache="assets/norman_scanpy5000_${SPLIT}_fold${FOLD}.h5ad"
    SETTINGS=(
      data.raw_h5ad=data/norman/norman.h5ad
      split.source=reference
      "split.method=${SPLIT}"
      "split.fold=${FOLD}"
      "data.cache_h5ad=${cache}"
      model.perturbation_dropout=0
      model.perturbation_knowledge=string
    )
    ;;
  combo)
    if [ "$SPLIT" != additive ]; then
      echo "SPLIT is for Norman only; ComboSciPlex has the one Table 3 split" >&2; exit 2
    fi
    prefix="cDfmPH"
    EPOCHS="${EPOCHS:-2500}"
    cache="assets/combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad"
    SETTINGS=(
      data.raw_h5ad=data/combosciplex/combosciplex.h5ad
      data.control_label=control
      data.normalise_from_counts=counts
      data.n_hvg=5000
      data.hvg_criterion=scanpy
      split.source=list
      split.method=additive
      split.fold=0
      "data.cache_h5ad=${cache}"
      model.perturbation_dropout=0.25
      model.perturbation_knowledge=none
      train.pearson_weight=10
    )
    ;;
  *) echo "unknown DATA '$DATA' (norman or combo)" >&2; exit 2 ;;
esac
if [ ! -f "$cache" ]; then
  echo "missing $cache - the reference runs were trained on it; build it with the script that made them" >&2
  exit 2
fi

for seed in $SEEDS; do
  if [ "$DATA" = norman ]; then tag="${prefix}_f${FOLD}_s${seed}"; else tag="${prefix}_s${seed}"; fi
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [${DATA} ${SPLIT}, hurdle on the OT pairs] ==="
  python scripts/train.py --tag "$tag" --force --set \
    "train.device=${DEVICE}" \
    train.gene_endpoint_weight=10 \
    train.fm_weight=1.0 \
    train.endpoint_weight=0 \
    train.fm_sigma=0.0 \
    train.coupling=uot \
    model.observable_whiten=zca \
    model.response=generator \
    "${SETTINGS[@]}" \
    "train.seed=${seed}" \
    "train.epochs=${EPOCHS}" \
    train.head_on_pairs=true
  for block in $blocks; do
    for gate in soft sample; do
      # --profile full: the default 'minimal' has no de_spearman_sig, the DE column.
      python scripts/run_celleval.py "results/runs/${tag}" --gate "$gate" --bound-mean \
        --infer-top-gene 1000 --pred-cells 128 --results-tag ce0542 --profile full \
        --celleval-python "$CELLEVAL_PY" --threads "$THREADS" --group "$block"
    done
  done
done
