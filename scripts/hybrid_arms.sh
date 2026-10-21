#!/usr/bin/env bash
# The hybrid experiments of docs/DESIGN_HYBRID.md, sections 3-6.
#
#   bash scripts/hybrid_arms.sh val                           H0: LS at each sigma,
#                                                             validation split, seed 0
#   SIGMA=0.3 bash scripts/hybrid_arms.sh B S L LF LS         folds 0 1 2 4 x seeds 0 1 2
#   SIGMA=0.3 FOLDS="0 1" bash scripts/hybrid_arms.sh LS      part of it, for a second pane
#
# Arms (coordinates x FM):
#   D   KEGG          no FM            existing runs: s2fin_f*_s*, not rerun here
#   B   KEGG          deterministic    sigma 0
#   S   KEGG          stochastic       sigma SIGMA
#   L   KEGG + z      no FM
#   LF  KEGG + z      deterministic
#   LS  KEGG + z      stochastic       the model
#
# Every setting is written here and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 4}"
SIGMA="${SIGMA:-}"
SIGMAS="${SIGMAS:-0.1 0.3 1.0}"
LATENT=64

# Stage 2's settings - dense generator, ZCA on the KEGG block, gene-space mean loss 10 -
# with the observable-space losses per arm below.
SHARED=(
  data.raw_h5ad=data/norman/norman.h5ad
  split.source=reference
  split.method=additive
  "train.epochs=${EPOCHS}"
  "train.device=${DEVICE}"
  train.gene_endpoint_weight=10
  model.observable_whiten=zca
  model.response=generator
)
VALIDATION=(split.fold=0 split.validation=true
            data.cache_h5ad=assets/norman_scanpy5000_additive_nval_fold0.h5ad)

arm_settings() {  # arm, sigma
  local fm="train.fm_weight=1 train.endpoint_weight=3"
  local off="train.fm_weight=0 train.endpoint_weight=0"
  case "$1" in
    B)  echo "$fm train.fm_sigma=0 model.latent_dim=0" ;;
    S)  echo "$fm train.fm_sigma=$2 model.latent_dim=0" ;;
    L)  echo "$off train.fm_sigma=0 model.latent_dim=${LATENT}" ;;
    LF) echo "$fm train.fm_sigma=0 model.latent_dim=${LATENT}" ;;
    LS) echo "$fm train.fm_sigma=$2 model.latent_dim=${LATENT}" ;;
  esac
}

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/hybrid_arms.sh val | B S L LF LS   (S and LS need SIGMA)" >&2
  exit 2
fi
for arm in "$@"; do
  case "$arm" in
    val|B|L|LF) ;;
    S|LS) [ -n "$SIGMA" ] || { echo "arm $arm needs SIGMA, the value H0 chose" >&2; exit 2; } ;;
    *) echo "unknown arm '$arm' (val B S L LF LS)" >&2; exit 2 ;;
  esac
done

tags=()
train() {  # tag, then overrides (space-separated words; no value holds a space)
  local tag="$1"; shift
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [$*] ==="
  # shellcheck disable=SC2068
  python scripts/train.py --tag "$tag" --force --set "${SHARED[@]}" $@
  tags+=("$tag")
}

for arm in "$@"; do
  case "$arm" in
    val)
      for sigma in $SIGMAS; do
        train "yLS_val_sig${sigma}" "${VALIDATION[*]}" train.seed=0 \
          "$(arm_settings LS "$sigma")"
      done ;;
    *)
      for fold in $FOLDS; do
        for seed in $SEEDS; do
          train "y${arm}_f${fold}_s${seed}" "split.fold=${fold}" \
            "data.cache_h5ad=assets/norman_scanpy5000_additive_fold${fold}.h5ad" \
            "train.seed=${seed}" "$(arm_settings "$arm" "${SIGMA:-0}")"
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
print(f"  {tag:18s} fold={s['fold']} val={s.get('validation', False)}  "
      f"latent={m.get('latent_dim')}  fm={t.get('fm_weight')}/{t.get('endpoint_weight')}  "
      f"sigma={t.get('fm_sigma')}  whiten={m.get('observable_whiten')}  "
      f"epochs={t['epochs']}  seed={t['seed']}")
PY
done
