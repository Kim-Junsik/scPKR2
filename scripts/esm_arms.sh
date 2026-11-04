#!/usr/bin/env bash
# ESM-2 protein embeddings beside STRING in the knowledge encoder: one check on the two
# Norman dev folds.
#
#   bash scripts/esm_arms.sh                       arm SE: additive fold 3, then combinations fold 3, seeds 0-2
#   SPLIT=combinations bash scripts/esm_arms.sh    one split per pane
#   SPLIT=additive bash scripts/esm_arms.sh
#   ARMS="SE E" bash scripts/esm_arms.sh           E as well
#
# THE ARMS. The final model (nEp0fm, scripts/final_p0.sh) with one change, the knowledge:
#
#   SE  model.perturbation_knowledge=string_esm   STRING's 64 dims + ESM-2's 64, side by side
#   E   model.perturbation_knowledge=esm          ESM-2 alone, to see which half carries it
#
# The ESM-2 table is assets/knowledge/norman_esm2_650m.npz, committed
# (scripts/build_esm_knowledge.py). It holds one mean-pooled facebook/esm2_t33_650M_UR50D
# vector per perturbed gene's protein.
#
# WHY. Under the combinations split a test double's genes are seen in training only
# beside other partners, and 2-4 genes per fold are in no training condition at all, so
# STRING is all the model has for them. Table 2 is where the model loses most (DE and DS).
# ESM-2 describes a gene by its protein, with or without network neighbours.
#
# THE REFERENCE RUNS already exist: nEp0fm_f3_s0-2 (additive) and nEp0fm_comb_f3_s0-2
# (combinations). Fold 3 is the hardest fold of both.
#
# Each run is exported and scored right after it trains, under both gates (DE is read from
# sample, the rest from soft). Combinations runs are exported per block, single and
# double apart, as Table 2 is. Then compare:
#
#   python scripts/arm_table.py --group double --device cuda --per-run base='results/runs/nEp0fm_f3_s*' SE='results/runs/nEp0fmSE_f3_s*'
#   python scripts/arm_table.py --group single --group-export --device cuda --per-run base='results/runs/nEp0fm_comb_f3_s*' SE='results/runs/nEp0fmSE_comb_f3_s*'
#   python scripts/arm_table.py --group double --group-export --device cuda --per-run base='results/runs/nEp0fm_comb_f3_s*' SE='results/runs/nEp0fmSE_comb_f3_s*'

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SPLIT="${SPLIT:-additive combinations}"
ARMS="${ARMS:-SE}"
SEEDS="${SEEDS:-0 1 2}"
FOLD="${FOLD:-3}"
EPOCHS="${EPOCHS:-250}"
CELLEVAL_PY="${CELLEVAL_PY:-.env-ce-scdfm/bin/python}"
THREADS="${THREADS:-16}"
ESM_TABLE="assets/knowledge/norman_esm2_650m.npz"

if [ ! -x "$CELLEVAL_PY" ]; then
  echo "no scoring interpreter at $CELLEVAL_PY (python -m venv .env-ce-scdfm && .env-ce-scdfm/bin/pip install -r requirements-celleval-scdfm.txt)" >&2
  exit 2
fi
if [ ! -f "$ESM_TABLE" ]; then
  echo "missing $ESM_TABLE - git pull (it is committed)" >&2
  exit 2
fi

for method in $SPLIT; do
  case "$method" in
    additive) prefix=nEp0fm; tail="" ;;
    combinations) prefix=nEp0fm; tail="_comb" ;;
    *) echo "unknown SPLIT '$method' (additive combinations)" >&2; exit 2 ;;
  esac
  cache="assets/norman_scanpy5000_${method}_fold${FOLD}.h5ad"
  if [ ! -f "$cache" ]; then
    echo "missing $cache - the reference runs were trained on it (scripts/final_p0.sh builds it)" >&2
    exit 2
  fi
  for arm in $ARMS; do
    case "$arm" in
      SE) knowledge=string_esm ;;
      E)  knowledge=esm ;;
      *) echo "unknown arm '$arm' (SE E)" >&2; exit 2 ;;
    esac
    for seed in $SEEDS; do
      tag="${prefix}${arm}${tail}_f${FOLD}_s${seed}"
      echo ""
      echo "=== ${tag}  ->  results/runs/${tag}   [${method}, knowledge=${knowledge}] ==="
      python scripts/train.py --tag "$tag" --force --set \
        "train.device=${DEVICE}" \
        train.gene_endpoint_weight=10 \
        train.fm_weight=1.0 \
        train.endpoint_weight=0 \
        train.fm_sigma=0.0 \
        train.coupling=uot \
        model.observable_whiten=zca \
        model.response=generator \
        data.raw_h5ad=data/norman/norman.h5ad \
        split.source=reference \
        "split.method=${method}" \
        "split.fold=${FOLD}" \
        "data.cache_h5ad=${cache}" \
        "train.seed=${seed}" \
        "train.epochs=${EPOCHS}" \
        model.perturbation_dropout=0 \
        "model.perturbation_knowledge=${knowledge}"
      if [ "$method" = additive ]; then blocks="all"; else blocks="single double"; fi
      for block in $blocks; do
        for gate in soft sample; do
          # --profile full: the default 'minimal' has no de_spearman_sig, the DE column.
          python scripts/run_celleval.py "results/runs/${tag}" --gate "$gate" --bound-mean \
            --infer-top-gene 1000 --pred-cells 128 --results-tag ce0542 --profile full \
            --celleval-python "$CELLEVAL_PY" --threads "$THREADS" --group "$block"
        done
      done
    done
  done
done
