#!/usr/bin/env bash
# Stage 2 on all five norman additive folds - docs/DESIGN_STAGE2.md section 8, step 5 (S6).
#
#   bash scripts/stage2_folds.sh 0 1        in one pane
#   bash scripts/stage2_folds.sh 2 4        in another
#
# Fold 3 is not rerun: arm D of the ablation (results/runs/s2D_s0..2) was trained with
# exactly these settings, and the config printout at the end of stage2_arms.sh showed it.
# Pass 3 as well if a run under this script's own tag is wanted.
#
# The settings are arm D, not arm B. S3 was decided on fold 3: removing the flow-matching
# and endpoint terms changed L2 by +0.0015 (B 1.6837, D 1.6852), under the 0.01 the design
# fixed in advance, so the rule written before the run removes them.
#
# Every setting is written here rather than passed through a shell variable - see the
# header of stage2_arms.sh for what that cost once. Each run's SAVED config is printed at
# the end so a run that trained something else is visible before anything is scored.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1 2}"

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/stage2_folds.sh 0 1 2 4 ..." >&2
  exit 2
fi
for fold in "$@"; do
  case "$fold" in
    0|1|2|3|4) ;;
    *) echo "unknown fold '$fold' (0 1 2 3 4)" >&2; exit 2 ;;
  esac
  cache="assets/norman_scanpy5000_additive_fold${fold}.h5ad"
  [ -f "$cache" ] || { echo "missing $cache" >&2; exit 2; }
done

for fold in "$@"; do
  for seed in $SEEDS; do
    tag="s2fin_f${fold}_s${seed}"
    echo ""
    echo "=== fold ${fold}  seed ${seed}  ->  results/runs/${tag} ==="
    python scripts/train.py --tag "$tag" --force --set \
      data.raw_h5ad=data/norman/norman.h5ad \
      split.source=reference \
      split.method=additive \
      "split.fold=${fold}" \
      "data.cache_h5ad=assets/norman_scanpy5000_additive_fold${fold}.h5ad" \
      "train.epochs=${EPOCHS}" \
      "train.device=${DEVICE}" \
      train.gene_endpoint_weight=10 \
      train.fm_weight=0 \
      train.endpoint_weight=0 \
      model.observable_whiten=zca \
      model.response=generator \
      "train.seed=${seed}"
  done
done

echo ""
echo "=== what each run was actually trained with ==="
for fold in "$@"; do
  for seed in $SEEDS; do
    tag="s2fin_f${fold}_s${seed}"
    python - "$tag" <<'PY'
import sys, torch
tag = sys.argv[1]
c = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
               weights_only=False)["config"]
m, t = c["model"], c["train"]
print(f"  {tag:14s} response={m.get('response')}  fold={c['split']['fold']}  "
      f"cache={c['data']['cache_h5ad']}  epochs={t['epochs']}  "
      f"fm={t.get('fm_weight')}/{t.get('endpoint_weight')}  obs={m.get('observables')}")
PY
  done
done
