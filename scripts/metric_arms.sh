#!/usr/bin/env bash
# Metric-aligned loss terms on the final model: one check on the two dev sets.
#
#   bash scripts/metric_arms.sh                      Norman additive fold 3, arms P D PD, seeds 0-2
#   DATA=combo bash scripts/metric_arms.sh           ComboSciPlex (Table 3 split), the same arms
#   ARMS="P" SEEDS="0" bash scripts/metric_arms.sh   part of it, one pane each
#   DATA=combo ARMS="PD" PW=30 bash scripts/metric_arms.sh   another weight: tag cDfmPDp30_s*
#   SPLIT=combinations ARMS="D PD" bash scripts/metric_arms.sh   Table 2's split: tag nEp0fmD_comb_f3_s*
#
# PW and DW are the pearson and discrim weights (default 10). A weight other than 10 is
# written into the tag (p30, d30), so the weight-10 runs keep their names.
#
# THE IDEA comes from the VCC 2025 winners, who trained on their metrics directly. The
# final model (nEp0fm / cDfm) trains its mean on L2 alone, and the columns it loses are
# the ones L2 does not ask for: DS on Tables 1-2, Pearson delta on Table 3 (0.715
# against scDFM's 0.893). Two terms on the condition's predicted mean, both on the scale
# of the L2 term (src/config.py):
#
#   P   train.pearson_weight=10   1 - Pearson(mu_hat - m_ctrl, m_S - m_ctrl)
#   D   train.discrim_weight=10   soft rank of S among the training conditions (DS)
#   PD  both
#
# Every other setting is the final model's: scripts/final_p0.sh (TABLE=1, or TABLE=2 for
# SPLIT=combinations) for Norman, scripts/best_fm_table3.sh (ARM=D) for ComboSciPlex. The
# reference runs already exist: nEp0fm_f3_s0-2, nEp0fm_comb_f3_s0-2 and cDfm_s0-2.
#
# TABLE 2 (SPLIT=combinations). Its Double block loses to scDFM by 0.011 on L2 and 0.0002 on
# MAE (five-fold means). On additive fold 3's doubles, D lowered L2 by 0.019 and PD by 0.031.
# A combinations run is scored per block (Single, Double), like the reference runs.
#
# Each run is exported and scored right after it trains, under both gates (the main
# tables read DE from sample and the rest from soft). Then compare:
#
#   python scripts/arm_table.py --group double --device cuda base='results/runs/nEp0fm_f3_s*' P='results/runs/nEp0fmP_f3_s*' D='results/runs/nEp0fmD_f3_s*' PD='results/runs/nEp0fmPD_f3_s*'
#   python scripts/arm_table.py --group all --device cuda base='results/runs/cDfm_s[012]' P='results/runs/cDfmP_s*' D='results/runs/cDfmD_s*' PD='results/runs/cDfmPD_s*'
#   python scripts/arm_table.py --group double --group-export --device cuda --per-run base='results/runs/nEp0fm_comb_f3_s*' D='results/runs/nEp0fmD_comb_f3_s*' PD='results/runs/nEp0fmPD_comb_f3_s*'
#   (and --group single for the Single block)

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
DATA="${DATA:-norman}"
SPLIT="${SPLIT:-additive}"
ARMS="${ARMS:-P D PD}"
SEEDS="${SEEDS:-0 1 2}"
FOLD="${FOLD:-3}"
CELLEVAL_PY="${CELLEVAL_PY:-.env-ce-scdfm/bin/python}"
THREADS="${THREADS:-16}"
PW="${PW:-10}"
DW="${DW:-10}"

if [ ! -x "$CELLEVAL_PY" ]; then
  echo "no scoring interpreter at $CELLEVAL_PY (python -m venv .env-ce-scdfm && .env-ce-scdfm/bin/pip install -r requirements-celleval-scdfm.txt)" >&2
  exit 2
fi

blocks="all"
tail=""
case "$DATA" in
  norman)
    case "$SPLIT" in
      additive) ;;
      combinations) tail="_comb"; blocks="single double" ;;
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
    )
    ;;
  *) echo "unknown DATA '$DATA' (norman or combo)" >&2; exit 2 ;;
esac
if [ ! -f "$cache" ]; then
  echo "missing $cache - the reference runs were trained on it; build it with the script that made them" >&2
  exit 2
fi

for arm in $ARMS; do
  case "$arm" in
    P)  WEIGHTS=("train.pearson_weight=${PW}" train.discrim_weight=0) ;;
    D)  WEIGHTS=(train.pearson_weight=0 "train.discrim_weight=${DW}") ;;
    PD) WEIGHTS=("train.pearson_weight=${PW}" "train.discrim_weight=${DW}") ;;
    *) echo "unknown arm '$arm' (P D PD)" >&2; exit 2 ;;
  esac
  suffix=""
  case "$arm" in *P*) [ "$PW" = 10 ] || suffix="${suffix}p${PW}" ;; esac
  case "$arm" in *D*) [ "$DW" = 10 ] || suffix="${suffix}d${DW}" ;; esac
  for seed in $SEEDS; do
    if [ "$DATA" = norman ]; then tag="nEp0fm${arm}${suffix}${tail}_f${FOLD}_s${seed}"; else tag="cDfm${arm}${suffix}_s${seed}"; fi
    echo ""
    echo "=== ${tag}  ->  results/runs/${tag}   [${DATA} ${SPLIT}, ${WEIGHTS[*]}] ==="
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
      "${WEIGHTS[@]}"
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
