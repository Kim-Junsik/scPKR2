#!/usr/bin/env bash
# The knowledge-encoder experiments of docs/DESIGN_ENCODER.md, sections 4-5 and 8.
#
#   bash scripts/encoder_arms.sh vcc-val                     H0 on VCC: dropout x length
#                                                            x {E, S}, validation genes
#   P=0.5 EPOCHS=250 bash scripts/encoder_arms.sh vcc-test   V1/V2: {E, S, N} x 3 seeds
#   bash scripts/encoder_arms.sh combo-val                   H0 on ComboSciPlex
#   P=0.5 EPOCHS=2500 bash scripts/encoder_arms.sh combo-test   C1: {D, M, MP} x 3 seeds
#   P=0.5 bash scripts/encoder_arms.sh norman                N1: E on norman fold 3
#   P=0.25 bash scripts/encoder_arms.sh norman-controls      N2: S and N on norman fold 3,
#                                                            the controls N1 lacked (5.2)
#   EPOCHS=2500 bash scripts/encoder_arms.sh combo-dose      secondary: {M, MP} at dropout
#                                                            0.5 on the test split (5.1)
#
# Arms:
#   E   STRING knowledge          (genes)        M   mechanism knowledge   (drugs)
#   S   STRING, shuffled          the control    MP  mechanism, shuffled   the control
#   N   no knowledge (k_a = 0)    the encoder with nothing to say
#   D   the lookup table          stage 2 as it was (ComboSciPlex only - it cannot
#                                 predict a perturbation it never saw)
#
# Every setting is written here and each run's SAVED config is printed at the end.
# VCC needs data/vcc/vcc2025_cache.h5ad: python scripts/vcc_prepare.py --root data/vcc/2025

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
P="${P:-}"
EPOCHS="${EPOCHS:-}"
VCC_CACHE="${VCC_CACHE:-data/vcc/vcc2025_cache.h5ad}"
# The H0 grids, and norman's length; overridable only for a local smoke test.
VCC_VAL_EPOCHS="${VCC_VAL_EPOCHS:-100 250}"
COMBO_VAL_EPOCHS="${COMBO_VAL_EPOCHS:-1000 2500}"
NORMAN_EPOCHS="${NORMAN_EPOCHS:-250}"

# Stage 2's D settings: dense generator, ZCA, no observable-space losses.
MODEL=(
  "train.device=${DEVICE}"
  train.gene_endpoint_weight=10
  train.fm_weight=0
  train.endpoint_weight=0
  model.observable_whiten=zca
  model.response=generator
)
VCC=(
  "data.cache_h5ad=${VCC_CACHE}"
  "data.raw_h5ad=${VCC_CACHE}"
  split.source=vcc
  split.method=additive
  split.fold=0
)
COMBO=(
  data.raw_h5ad=data/combosciplex/combosciplex.h5ad
  data.control_label=control
  data.normalise_from_counts=counts
  data.n_hvg=5000
  data.hvg_criterion=scanpy
  split.source=list
  split.method=additive
  split.fold=0
)
COMBO_TEST=(data.cache_h5ad=assets/combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad)
COMBO_VAL=(data.cache_h5ad=assets/combosciplex_scanpy5000_additive_scdfm7val_fold0.h5ad
           split.validation=true)
NORMAN=(
  data.raw_h5ad=data/norman/norman.h5ad
  split.source=reference
  split.method=additive
  split.fold=3
  data.cache_h5ad=assets/norman_scanpy5000_additive_fold3.h5ad
)

knowledge_of() {
  case "$1" in
    E) echo "model.perturbation_knowledge=string" ;;
    S) echo "model.perturbation_knowledge=string model.knowledge_shuffle=true" ;;
    N) echo "model.perturbation_knowledge=zero" ;;
    M) echo "model.perturbation_knowledge=moa" ;;
    MP) echo "model.perturbation_knowledge=moa model.knowledge_shuffle=true" ;;
    D) echo "model.perturbation_knowledge=none" ;;
  esac
}

if [ "$#" -eq 0 ]; then
  echo "usage: bash scripts/encoder_arms.sh vcc-val | vcc-test | combo-val | combo-test | norman | norman-controls" >&2
  exit 2
fi
for step in "$@"; do
  case "$step" in
    vcc-val|combo-val) ;;
    combo-dose)
      [ -n "$EPOCHS" ] || { echo "combo-dose needs EPOCHS, the length H0 chose" >&2; exit 2; } ;;
    vcc-test|combo-test)
      [ -n "$P" ] && [ -n "$EPOCHS" ] || { echo "$step needs P and EPOCHS, the values H0 chose" >&2; exit 2; } ;;
    norman|norman-controls)
      [ -n "$P" ] || { echo "$step needs P, the value VCC's H0 chose" >&2; exit 2; } ;;
    *) echo "unknown step '$step'" >&2; exit 2 ;;
  esac
  case "$step" in
    vcc-*) [ -f "$VCC_CACHE" ] || { echo "missing $VCC_CACHE - run scripts/vcc_prepare.py" >&2; exit 2; } ;;
  esac
done

tags=()
train() {  # tag, then overrides
  local tag="$1"; shift
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [$*] ==="
  # shellcheck disable=SC2068
  python scripts/train.py --tag "$tag" --force --set "${MODEL[@]}" $@
  tags+=("$tag")
}

for step in "$@"; do
  case "$step" in
    vcc-val)
      for p in 0.25 0.5 0.75; do
        for epochs in $VCC_VAL_EPOCHS; do
          for arm in E S; do
            train "v${arm}_val_p${p}_e${epochs}" "${VCC[*]}" split.validation=true \
              train.seed=0 "train.epochs=${epochs}" "model.perturbation_dropout=${p}" \
              "$(knowledge_of $arm)"
          done
        done
      done ;;
    vcc-test)
      for arm in E S N; do
        for seed in $SEEDS; do
          train "v${arm}_s${seed}" "${VCC[*]}" "train.seed=${seed}" \
            "train.epochs=${EPOCHS}" "model.perturbation_dropout=${P}" "$(knowledge_of $arm)"
        done
      done ;;
    combo-val)
      for p in 0.25 0.5 0.75; do
        for epochs in $COMBO_VAL_EPOCHS; do
          for arm in M MP; do
            train "c${arm}_val_p${p}_e${epochs}" "${COMBO[*]}" "${COMBO_VAL[*]}" \
              train.seed=0 "train.epochs=${epochs}" "model.perturbation_dropout=${p}" \
              "$(knowledge_of $arm)"
          done
        done
      done ;;
    combo-test)
      for arm in D M MP; do
        for seed in $SEEDS; do
          train "c${arm}_s${seed}" "${COMBO[*]}" "${COMBO_TEST[*]}" "train.seed=${seed}" \
            "train.epochs=${EPOCHS}" "model.perturbation_dropout=${P}" "$(knowledge_of $arm)"
        done
      done ;;
    combo-dose)
      for arm in M MP; do
        for seed in $SEEDS; do
          train "c${arm}_p0.5_s${seed}" "${COMBO[*]}" "${COMBO_TEST[*]}" "train.seed=${seed}" \
            "train.epochs=${EPOCHS}" model.perturbation_dropout=0.5 "$(knowledge_of $arm)"
        done
      done ;;
    norman)
      for seed in $SEEDS; do
        train "nE_s${seed}" "${NORMAN[*]}" "train.seed=${seed}" "train.epochs=${NORMAN_EPOCHS}" \
          "model.perturbation_dropout=${P}" "$(knowledge_of E)"
      done ;;
    norman-controls)
      # Same settings as nE, knowledge shuffled (S) or absent (N): is the gain N1 found the
      # knowledge, or the dropout every encoder run trains with?
      for arm in S N; do
        for seed in $SEEDS; do
          train "n${arm}_s${seed}" "${NORMAN[*]}" "train.seed=${seed}" \
            "train.epochs=${NORMAN_EPOCHS}" "model.perturbation_dropout=${P}" \
            "$(knowledge_of $arm)"
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
saved = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
                   weights_only=False)
c = saved["config"]
m, t, s = c["model"], c["train"], c["split"]
k = saved["model"].get("modulation.operators.knowledge")
seen = saved["model"].get("modulation.operators.seen")
covered = "-" if k is None else int((k.abs().sum(1) > 0).sum())
print(f"  {tag:22s} cache={c['data']['cache_h5ad'].split('/')[-1]}  val={s.get('validation', False)}  "
      f"knowledge={m.get('perturbation_knowledge')} shuffled={m.get('knowledge_shuffle')}  "
      f"covered={covered} seen={'-' if seen is None else int(seen.sum())}  "
      f"p={m.get('perturbation_dropout')}  epochs={t['epochs']}  seed={t['seed']}")
PY
done
