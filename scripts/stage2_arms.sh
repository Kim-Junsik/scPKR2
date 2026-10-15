#!/usr/bin/env bash
# Stage 2's ablation arms on norman additive fold 3 - docs/DESIGN_STAGE2.md section 7.
#
#   bash scripts/stage2_arms.sh B C D E      in one pane
#   bash scripts/stage2_arms.sh F G H        in another
#
# Every setting is written here rather than passed in through a shell variable. The first
# attempt built its command line from $COMMON, the panes it ran in did not have it, and
# all 21 runs trained stage 1's MLP on the default data: ridge read 2.2583 instead of
# fold 3's 1.8592, and six arms that cannot be equal came out equal to four decimals.
#
# After the arms finish, each run's SAVED config is printed, so a run that trained
# something other than what was asked for is visible before anything is scored.
#
# DEVICE, EPOCHS and SEEDS may be overridden from the environment for a local smoke
# test; the defaults are the experiment.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1 2}"

# Identical to the runs S1 was decided on (s2_f3_s*), except the length, which the
# validation split chose.
COMMON=(
  data.raw_h5ad=data/norman/norman.h5ad
  split.source=reference
  split.method=additive
  split.fold=3
  data.cache_h5ad=assets/norman_scanpy5000_additive_fold3.h5ad
  "train.epochs=${EPOCHS}"
  "train.device=${DEVICE}"
  train.gene_endpoint_weight=10
  model.observable_whiten=zca
  model.response=generator
)

arm_overrides() {
  case "$1" in
    B) echo "" ;;                                              # the model itself
    C) echo "model.generator_linear=true" ;;                   # no exponential
    D) echo "train.fm_weight=0 train.endpoint_weight=0" ;;     # no observable losses
    E) echo "model.observables=random" ;;                      # random gene sets
    F) echo "model.generator_explicit_interaction=true" ;;     # learned kappa
    G) echo "model.generator_constant=false" ;;                # no constant coordinate
    H) echo "model.shared_rank=0 model.private_rank=36" ;;     # no shared modes
    # Measured against D, the settings S3 left standing, not against B: does knowing
    # WHICH gene was perturbed - the targets forced into the anchor block - carry weight?
    I) echo "train.fm_weight=0 train.endpoint_weight=0 model.anchor_genes=variance" ;;
    *) echo "unknown arm '$1' (B C D E F G H I)" >&2; return 2 ;;
  esac
}

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/stage2_arms.sh B C D E ..." >&2
  exit 2
fi
for arm in "$@"; do arm_overrides "$arm" > /dev/null; done   # reject typos up front

for arm in "$@"; do
  extra=$(arm_overrides "$arm")
  for seed in $SEEDS; do
    tag="s2${arm}_s${seed}"
    echo ""
    echo "=== arm ${arm}  seed ${seed}  ->  results/runs/${tag}   [${extra:-the model itself}] ==="
    # shellcheck disable=SC2086
    python scripts/train.py --tag "$tag" --force --set "${COMMON[@]}" "train.seed=${seed}" $extra
  done
done

echo ""
echo "=== what each run was actually trained with ==="
for arm in "$@"; do
  for seed in $SEEDS; do
    tag="s2${arm}_s${seed}"
    python - "$tag" <<'PY'
import sys, torch
tag = sys.argv[1]
c = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
               weights_only=False)["config"]
m, t = c["model"], c["train"]
print(f"  {tag:10s} response={m.get('response')}  fold={c['split']['fold']}  "
      f"epochs={t['epochs']}  linear={m.get('generator_linear')}  "
      f"constant={m.get('generator_constant')}  kappa={m.get('generator_explicit_interaction')}  "
      f"obs={m.get('observables')}  anchors={m.get('anchor_genes')}  "
      f"ranks={m.get('shared_rank')}/{m.get('private_rank')}  "
      f"fm={t.get('fm_weight')}/{t.get('endpoint_weight')}")
PY
  done
done
