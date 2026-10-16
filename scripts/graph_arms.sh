#!/usr/bin/env bash
# The graph experiments of docs/DESIGN_GRAPH.md, sections 6 and 7.
#
#   bash scripts/graph_arms.sh val                 K0: arm K at m = 1, 5, 10 on the
#                                                  validation split, seed 0
#   MIN_SHARED=5 bash scripts/graph_arms.sh K P    fold 3, three seeds, at the m K0 chose
#   bash scripts/graph_arms.sh D0                  fold 3, three seeds (no m needed)
#   bash scripts/graph_arms.sh KM PM               fold 3, the maplink graph and its shuffle
#
# Arms on fold 3:
#   D0  no mask, no whitening        what turning whitening off costs
#   K   KEGG graph                   the model
#   P   KEGG graph, labels shuffled  same shape, wrong wiring             (K1)
#   R   correlation graph            as many edges, from the data alone   (K2)
#   KM  KEGG maplink graph           the maps' own links (DESIGN_GRAPH.md 10)
#   PM  maplink graph, shuffled      same shape, wrong wiring             (K1 for KM)
# KM and PM need assets/kegg/kgml/: python scripts/download_kegg.py --kgml
# D itself is not rerun: results/runs/s2D_s0..2 are D (FINDINGS 21).
#
# Every setting is written here - see stage2_arms.sh's header for what passing them
# through a shell variable cost once - and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1 2}"
MIN_SHARED="${MIN_SHARED:-}"

# D's settings - S3 removed the observable-space losses - with whitening OFF, because a
# mask on whitened coordinates is not the KEGG graph (DESIGN_GRAPH.md 2.4).
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
)
FOLD3=(split.fold=3 data.cache_h5ad=assets/norman_scanpy5000_additive_fold3.h5ad)
VALIDATION=(split.fold=0 split.validation=true
            data.cache_h5ad=assets/norman_scanpy5000_additive_nval_fold0.h5ad)

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/graph_arms.sh val | D0 K P R KM PM" >&2
  exit 2
fi

tags=()
for arm in "$@"; do
  case "$arm" in
    val|D0|KM|PM) ;;
    K|P|R)
      if [ -z "$MIN_SHARED" ]; then
        echo "arm $arm needs MIN_SHARED, the m the validation split chose (K0)" >&2
        exit 2
      fi ;;
    *) echo "unknown arm '$arm' (val D0 K P R KM PM)" >&2; exit 2 ;;
  esac
done

train() {  # tag, then overrides
  local tag="$1"; shift
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [$*] ==="
  python scripts/train.py --tag "$tag" --force --set "${SHARED[@]}" "$@"
  tags+=("$tag")
}

for arm in "$@"; do
  case "$arm" in
    val)
      for m in 1 5 10; do
        train "gK_val_m${m}" "${VALIDATION[@]}" train.seed=0 \
          model.generator_mask=kegg "model.generator_mask_min_shared=${m}"
      done ;;
    D0)
      for seed in $SEEDS; do
        train "gD0_s${seed}" "${FOLD3[@]}" "train.seed=${seed}" model.generator_mask=none
      done ;;
    K|P|R)
      case "$arm" in K) kind=kegg ;; P) kind=permuted ;; R) kind=correlation ;; esac
      for seed in $SEEDS; do
        # P's shuffle follows the training seed, so its three runs are three different
        # wrong graphs rather than one.
        train "g${arm}_m${MIN_SHARED}_s${seed}" "${FOLD3[@]}" "train.seed=${seed}" \
          "model.generator_mask=${kind}" "model.generator_mask_min_shared=${MIN_SHARED}"
      done ;;
    KM|PM)
      case "$arm" in KM) kind=kegg ;; PM) kind=permuted ;; esac
      for seed in $SEEDS; do
        train "g${arm}_s${seed}" "${FOLD3[@]}" "train.seed=${seed}" \
          "model.generator_mask=${kind}" model.generator_mask_source=maplink
      done ;;
  esac
done

echo ""
echo "=== what each run was actually trained with ==="
for tag in "${tags[@]}"; do
  python - "$tag" <<'PY'
import sys, torch
tag = sys.argv[1]
saved = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
                   weights_only=False)
c = saved["config"]
m, t, s = c["model"], c["train"], c["split"]
mask = saved["model"].get("modulation.graph_mask") if "model" in saved else None
edges = "?" if mask is None else int((mask[:-1, :-1] > 0).sum()) - (mask.shape[0] - 1)
print(f"  {tag:16s} fold={s['fold']} val={s.get('validation', False)}  "
      f"mask={m.get('generator_mask')}/{m.get('generator_mask_source', 'shared')} "
      f"m={m.get('generator_mask_min_shared')}  "
      f"edges={edges}  whiten={m.get('observable_whiten')}  "
      f"fm={t.get('fm_weight')}/{t.get('endpoint_weight')}  seed={t['seed']}")
PY
done
