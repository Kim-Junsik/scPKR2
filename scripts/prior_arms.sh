#!/usr/bin/env bash
# The soft-prior experiments of docs/DESIGN_PRIOR.md, sections 2-4.
#
#   bash scripts/prior_arms.sh val                          H0: lambda grid x {KH, PH},
#                                                           validation split, seed 0
#   LAMBDA=0.1 bash scripts/prior_arms.sh KH PH             folds 0 1 2 4 x seeds 0 1 2
#   LAMBDA=0.1 FOLDS="0 1" bash scripts/prior_arms.sh KH PH   part of it, for a second pane
#   LAMBDA=0.1 bash scripts/prior_arms.sh RH                folds 0 1 2 4, seed 0 only (H3)
#
# Arms, every one with a dense residual paying LAMBDA * ||A_W||^2:
#   KH  KEGG map-link backbone             the model
#   PH  map-link backbone, shuffled         same shape, wrong backbone   (H2)
#   RH  correlation backbone, same edges    data alone                    (H3)
#
# Needs assets/kegg/kgml/ (python scripts/download_kegg.py --kgml). Every setting is
# written here, and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 4}"
LAMBDA="${LAMBDA:-}"
LAMBDAS="${LAMBDAS:-0.001 0.01 0.1 1 10}"

# D's settings, whitening off (a mask on whitened coordinates is not the KEGG graph), the
# map-link backbone and its residual.
SHARED=(
  data.raw_h5ad=data/norman/norman.h5ad
  split.source=reference
  split.method=additive
  "train.epochs=${EPOCHS}"
  "train.device=${DEVICE}"
  train.gene_endpoint_weight=10
  train.fm_weight=0
  train.endpoint_weight=0
  model.observable_whiten=none
  model.response=generator
  model.generator_mask_source=maplink
  model.generator_residual=true
)
VALIDATION=(split.fold=0 split.validation=true
            data.cache_h5ad=assets/norman_scanpy5000_additive_nval_fold0.h5ad)

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/prior_arms.sh val | KH PH RH   (KH/PH/RH need LAMBDA)" >&2
  exit 2
fi
[ -d assets/kegg/kgml ] || { echo "assets/kegg/kgml missing: python scripts/download_kegg.py --kgml" >&2; exit 2; }
for arm in "$@"; do
  case "$arm" in
    val) ;;
    KH|PH|RH)
      [ -n "$LAMBDA" ] || { echo "arm $arm needs LAMBDA, the value H0 chose" >&2; exit 2; }
      for fold in $FOLDS; do
        cache="assets/norman_scanpy5000_additive_fold${fold}.h5ad"
        [ -f "$cache" ] || { echo "missing $cache" >&2; exit 2; }
      done ;;
    *) echo "unknown arm '$arm' (val KH PH RH)" >&2; exit 2 ;;
  esac
done

tags=()
train() {  # tag, then overrides
  local tag="$1"; shift
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [$*] ==="
  python scripts/train.py --tag "$tag" --force --set "${SHARED[@]}" "$@"
  tags+=("$tag")
}

kind_of() {
  case "$1" in KH) echo kegg ;; PH) echo permuted ;; RH) echo correlation ;; esac
}

for arm in "$@"; do
  case "$arm" in
    val)
      for lambda in $LAMBDAS; do
        for pair in "K kegg" "P permuted"; do
          set -- $pair
          train "h$1_val_l${lambda}" "${VALIDATION[@]}" train.seed=0 \
            "model.generator_mask=$2" "model.generator_residual_penalty=${lambda}"
        done
      done ;;
    KH|PH|RH)
      kind=$(kind_of "$arm")
      seeds="$SEEDS"
      [ "$arm" = RH ] && seeds=0          # H3 is descriptive: one seed per fold
      for fold in $FOLDS; do
        for seed in $seeds; do
          # PH's shuffle follows the training seed: three seeds, three wrong backbones.
          train "h${arm:0:1}_f${fold}_s${seed}" "split.fold=${fold}" \
            "data.cache_h5ad=assets/norman_scanpy5000_additive_fold${fold}.h5ad" \
            "train.seed=${seed}" "model.generator_mask=${kind}" \
            "model.generator_residual_penalty=${LAMBDA}"
        done
      done ;;
  esac
done

echo ""
echo "=== what each run was actually trained with ==="
for tag in "${tags[@]}"; do
  python - "$tag" <<'PY'
import sys, torch
tag = sys.argv[1]
c = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
               weights_only=False)["config"]
m, t, s = c["model"], c["train"], c["split"]
print(f"  {tag:16s} fold={s['fold']} val={s.get('validation', False)}  "
      f"mask={m.get('generator_mask')}/{m.get('generator_mask_source')}  "
      f"residual={m.get('generator_residual')} lambda={m.get('generator_residual_penalty')}  "
      f"whiten={m.get('observable_whiten')}  fm={t.get('fm_weight')}/{t.get('endpoint_weight')}  "
      f"seed={t['seed']}")
PY
done
