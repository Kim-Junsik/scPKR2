#!/usr/bin/env bash
# The best model with flow matching switched on. Nothing else changes.
#
#   bash scripts/best_fm.sh                    all five folds x 3 seeds, 15 runs
#   FOLDS="0 1 2" bash scripts/best_fm.sh      part of it, one pane each
#   FOLDS="3 4" bash scripts/best_fm.sh
#
# THE BEST MODEL is nE (FINDINGS 28 and 30): stage 2's D settings, the STRING knowledge
# encoder, perturbation dropout 0.25, 250 epochs, seeds 0 1 2. Its runs are nE_s* on fold 3
# and nE_f{0,1,2,4}_s* on the others, and their Table 1 L2 over the five folds is 1.5176.
#
# THE ONE DIFFERENCE: train.fm_weight=1.0, the config default, where nE had 0. Every other
# setting is written out below exactly as encoder_arms.sh writes it for nE. That includes
# train.endpoint_weight=0: the endpoint is a separate observable-space loss, and turning it
# on as well would be a second change. The flow matching is the deterministic kind stage 2's
# arm B trained with - fm_sigma 0, minibatch UOT coupling - and both of those are the
# defaults the nE runs had too, written out here only so the saved config shows them.
#
# Scored beside nE with the same protocol as Table 1's L2 (soft gate, scDFM's 1,000 genes,
# 4,096 cells, cell mean bounded). hybrid_eval.py labels each run from its saved config and
# does not look at the knowledge encoder, so in its output B means "nE with FM" and D means
# "nE". Its "deciding mean" leaves fold 3 out; the per-fold columns give all five.
#
#   python scripts/hybrid_eval.py results/runs/nEfm_f*_s* results/runs/nE_f*_s* results/runs/nE_s* --device cuda --bound-mean
#
# Every setting is written here and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# nE's length; overridable only for a local smoke test.
EPOCHS="${EPOCHS:-250}"

for fold in $FOLDS; do
  case "$fold" in
    0|1|2|3|4) ;;
    *) echo "unknown fold '$fold' (0 1 2 3 4)" >&2; exit 2 ;;
  esac
  cache="assets/norman_scanpy5000_additive_fold${fold}.h5ad"
  [ -f "$cache" ] || { echo "missing $cache" >&2; exit 2; }
done

tags=()
for fold in $FOLDS; do
  for seed in $SEEDS; do
    tag="nEfm_f${fold}_s${seed}"
    echo ""
    echo "=== ${tag}  ->  results/runs/${tag}   [nE + flow matching] ==="
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
      split.method=additive \
      "split.fold=${fold}" \
      "data.cache_h5ad=assets/norman_scanpy5000_additive_fold${fold}.h5ad" \
      "train.seed=${seed}" \
      "train.epochs=${EPOCHS}" \
      model.perturbation_dropout=0.25 \
      model.perturbation_knowledge=string
    tags+=("$tag")
  done
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
k = saved["model"].get("modulation.operators.knowledge")
covered = "-" if k is None else int((k.abs().sum(1) > 0).sum())
print(f"  {tag:16s} fold={s['fold']} seed={t['seed']} epochs={t['epochs']}  "
      f"fm={t['fm_weight']} endpoint={t['endpoint_weight']} sigma={t.get('fm_sigma')} "
      f"coupling={t['coupling']} gene={t['gene_endpoint_weight']}  "
      f"whiten={m.get('observable_whiten')} response={m.get('response')}  "
      f"knowledge={m.get('perturbation_knowledge')} covered={covered} "
      f"p={m.get('perturbation_dropout')}")
PY
done
