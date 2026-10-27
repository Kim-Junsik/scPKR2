#!/usr/bin/env bash
# Table 2 diagnosis: where does nEfm's single over-prediction come from? One round.
#
#   bash scripts/table2_arms.sh                      D and N, folds 0-4, seed 0: 10 runs
#   ARMS=D bash scripts/table2_arms.sh               one arm per pane
#   ARMS=N bash scripts/table2_arms.sh
#
# THE FINDING. On the combinations split nEfm loses to ridge on the Single block (1.669
# against 1.416). The losses are weak singles predicted strong - JUN, ZC3HAV1, MAP7D1,
# UBASH3A, PTPN9 - many trained only beside a strong partner, often a STRING neighbour
# (MAP2K3/MAP2K6, UBASH3A/UBASH3B, FOXA1/FOXA3, the CEBPs). Genes in no training condition
# beat ridge. The amount of training data does not explain it (rho -0.02).
#
# THE QUESTION. Is the leak in the core - the generators summed and exponentiated - or in
# the parts that pull perturbations toward each other?
#
#   nEfm  STRING + dropout 0.25     c_a = c_bar + E k_a + delta_a   (the reported model)
#   N     dropout 0.25, no STRING   c_a = c_bar + delta_a           shared mean, no neighbours
#   D     neither                   c_a free per perturbation       the lookup table
#
# Everything else is nEfm's: FM on, endpoint 0, gene endpoint 10, ZCA, 250 epochs.
# D cannot predict a gene that is in no training condition (2-4 per fold): its generator
# stays at initialisation, which is the control - what ridge predicts for them too.
#
# ONE SEED. Seed spread on this block is 0.01-0.05 per fold; the gap under test is 0.25.
#
# THE RULE, fixed before any of these runs is seen. Single-block L2, mean over the five
# folds, seed 0 (nEfm's seed-0 runs: 1.6426 1.6975 1.6333 1.7538 1.6760 -> 1.6806):
#
#   D <= 1.55                 the composition is not the cause.
#     N <= 1.55               the STRING neighbours are: use the knowledge only where a
#                             perturbation has no data of its own.
#     N >  1.55               the shared mean and the dropout are: train without dropout
#                             on perturbations that have data.
#   D >= 1.62                 the composition is the cause - the core itself.
#   in between                both contribute; reported as such, no further arms.
#
# Either way the next step is ONE fix, confirmed once on all three tables, and the model
# is then frozen. No third diagnostic round.
#
# SCORING, L2 only - no cell-eval needed for the rule. One command per arm, because the
# script's grand mean mixes arms when several are passed together:
#
#   python scripts/compare_to_ridge.py results/runs/nDfm_comb_f*_s0 --device cuda --group single --bound-mean --per-condition
#   python scripts/compare_to_ridge.py results/runs/nNfm_comb_f*_s0 --device cuda --group single --bound-mean --per-condition
#   python scripts/compare_to_ridge.py results/runs/nDfm_comb_f*_s0 --device cuda --group double --bound-mean
#   python scripts/compare_to_ridge.py results/runs/nNfm_comb_f*_s0 --device cuda --group double --bound-mean
#
# The caches are the ones scripts/best_fm_table2.sh built; any that is missing is built here
# with the same settings.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"
FOLDS="${FOLDS:-0 1 2 3 4}"
ARMS="${ARMS:-D N}"
# nEfm's length; EPOCHS and CACHE_DIR are overridable only for a local smoke test.
EPOCHS="${EPOCHS:-250}"
CACHE_DIR="${CACHE_DIR:-assets}"

knowledge_of() {
  case "$1" in
    D) echo none ;;
    N) echo zero ;;
    *) echo "unknown arm '$1' (D or N)" >&2; exit 2 ;;
  esac
}
for arm in $ARMS; do knowledge_of "$arm" >/dev/null; done
for fold in $FOLDS; do
  case "$fold" in
    0|1|2|3|4) ;;
    *) echo "unknown fold '$fold' (0 1 2 3 4)" >&2; exit 2 ;;
  esac
done

tags=()
for fold in $FOLDS; do
  cache="${CACHE_DIR}/norman_scanpy5000_combinations_fold${fold}.h5ad"
  if [ ! -f "$cache" ]; then
    echo ""
    echo "=== building ${cache} ==="
    python scripts/build_data.py --set \
      data.raw_h5ad=data/norman/norman.h5ad \
      data.n_hvg=5000 \
      data.hvg_criterion=scanpy \
      data.force_include_targets=true \
      data.exclude_test_from_hvg=true \
      split.source=reference \
      split.method=combinations \
      "split.fold=${fold}" \
      "data.cache_h5ad=${cache}"
  fi

  for arm in $ARMS; do
    for seed in $SEEDS; do
      tag="n${arm}fm_comb_f${fold}_s${seed}"
      echo ""
      echo "=== ${tag}  ->  results/runs/${tag}   [arm ${arm}, combinations split] ==="
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
        split.method=combinations \
        "split.fold=${fold}" \
        "data.cache_h5ad=${cache}" \
        "train.seed=${seed}" \
        "train.epochs=${EPOCHS}" \
        model.perturbation_dropout=0.25 \
        "model.perturbation_knowledge=$(knowledge_of "$arm")"
      tags+=("$tag")
    done
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
seen = saved["model"].get("modulation.operators.seen")
print(f"  {tag:20s} split={s['method']} fold={s['fold']} seed={t['seed']} "
      f"epochs={t['epochs']}  fm={t['fm_weight']} endpoint={t['endpoint_weight']} "
      f"gene={t['gene_endpoint_weight']}  knowledge={m.get('perturbation_knowledge')} "
      f"seen={'-' if seen is None else int(seen.sum())}/{'-' if seen is None else seen.numel()} "
      f"p={m.get('perturbation_dropout')}  cache={c['data']['cache_h5ad'].split('/')[-1]}")
PY
done
