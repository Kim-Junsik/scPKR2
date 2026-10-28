#!/usr/bin/env bash
# The one fix table2_arms.sh's rule prescribed: nEfm without perturbation dropout.
#
#   bash scripts/final_p0.sh                        Table 2 first: combinations, 5 folds x 3 seeds
#   TABLE=1 bash scripts/final_p0.sh                then Table 1: additive, 5 folds x 3 seeds
#   FOLDS="0 1 2" bash scripts/final_p0.sh          part of it, one pane each
#   FOLDS="3 4" bash scripts/final_p0.sh
#
# WHAT THE ARMS SHOWED (table2_arms.sh, combinations, seed 0, Single-block L2):
#
#   D  lookup table, no dropout              1.4218   (ridge 1.4163)
#   N  shared mean + dropout, no STRING      1.7109
#   nEfm  STRING + shared mean + dropout     1.6806
#
# D <= 1.55 and N > 1.55: the composition is not the cause; the shared mean and the dropout
# are. Dropout switches a perturbation's own parameters off a quarter of the time, so the
# shared path c_bar + E k_a is trained to explain every perturbation by itself, and a gene
# seen only beside a strong partner is pulled toward that shared, strong effect. N, which
# has the dropout and no STRING, over-predicts the most (JUN 3.48 against ridge 1.67).
#
# THE FIX: model.perturbation_dropout=0, nothing else. A perturbation with training data
# then always has its own parameters on, c_a = c_bar + E k_a + delta_a, the STRING encoder
# included. A perturbation with none is still predicted from c_bar + E k_a alone
# (models/generator.py: private=False when seen is False). Every other setting is nEfm's.
#
# ORDER, and the rule, fixed before any of these runs is seen:
#
#   1. Table 2. Single-block L2, mean of the 15 runs, compare_to_ridge.py.
#        <= 1.55   the fix holds. Go to 2.
#        >  1.55   STRING itself leaks once the dropout is gone. The fallback is the
#                  knowledge used ONLY for perturbations without data: D's parameters for
#                  the rest. Applied to the nDfm runs, so no further diagnostic round.
#   2. Table 1, the same model. Reported whatever it reads.
#   3. Table 3 is scripts/best_fm_table3.sh as it stands: drugs have no STRING, and its D
#      already trains without dropout.
#   Then the model is frozen.
#
# SCORING, Table 2:
#
#   python scripts/compare_to_ridge.py results/runs/nEp0fm_comb_f*_s* --device cuda --group single --bound-mean --per-condition
#   python scripts/compare_to_ridge.py results/runs/nEp0fm_comb_f*_s* --device cuda --group double --bound-mean
#   for r in results/runs/nEp0fm_comb_f*_s*; do for g in single double; do python scripts/run_celleval.py $r --export-only --gate soft --bound-mean --infer-top-gene 1000 --pred-cells 128 --group $g || break 2; done; done
#   python scripts/paper_table.py results/runs/nEp0fm_comb_f*_s* --group single --group-export --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#   python scripts/paper_table.py results/runs/nEp0fm_comb_f*_s* --group double --group-export --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#
# SCORING, Table 1:
#
#   python scripts/compare_to_ridge.py results/runs/nEp0fm_f*_s* --device cuda --group double --bound-mean
#   for r in results/runs/nEp0fm_f*_s*; do python scripts/run_celleval.py $r --export-only --gate soft --bound-mean --infer-top-gene 1000 --pred-cells 128 || break; done
#   python scripts/paper_table.py results/runs/nEp0fm_f*_s* --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#
# Every setting is written here and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
TABLE="${TABLE:-2}"
# nEfm's length; EPOCHS and CACHE_DIR are overridable only for a local smoke test.
EPOCHS="${EPOCHS:-250}"
CACHE_DIR="${CACHE_DIR:-assets}"

case "$TABLE" in
  1) method=additive; prefix=nEp0fm ;;
  2) method=combinations; prefix=nEp0fm_comb ;;
  *) echo "unknown TABLE '$TABLE' (1 or 2)" >&2; exit 2 ;;
esac
for fold in $FOLDS; do
  case "$fold" in
    0|1|2|3|4) ;;
    *) echo "unknown fold '$fold' (0 1 2 3 4)" >&2; exit 2 ;;
  esac
done

tags=()
for fold in $FOLDS; do
  cache="${CACHE_DIR}/norman_scanpy5000_${method}_fold${fold}.h5ad"
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
      "split.method=${method}" \
      "split.fold=${fold}" \
      "data.cache_h5ad=${cache}"
  fi

  for seed in $SEEDS; do
    tag="${prefix}_f${fold}_s${seed}"
    echo ""
    echo "=== ${tag}  ->  results/runs/${tag}   [nEfm without dropout, ${method} split] ==="
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
      "split.fold=${fold}" \
      "data.cache_h5ad=${cache}" \
      "train.seed=${seed}" \
      "train.epochs=${EPOCHS}" \
      model.perturbation_dropout=0 \
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
seen = saved["model"].get("modulation.operators.seen")
covered = "-" if k is None else int((k.abs().sum(1) > 0).sum())
print(f"  {tag:22s} split={s['method']} fold={s['fold']} seed={t['seed']} "
      f"epochs={t['epochs']}  fm={t['fm_weight']} endpoint={t['endpoint_weight']} "
      f"sigma={t.get('fm_sigma')} coupling={t['coupling']} gene={t['gene_endpoint_weight']}  "
      f"whiten={m.get('observable_whiten')} response={m.get('response')}  "
      f"knowledge={m.get('perturbation_knowledge')} covered={covered} "
      f"seen={'-' if seen is None else int(seen.sum())}/{'-' if seen is None else seen.numel()} "
      f"p={m.get('perturbation_dropout')}  cache={c['data']['cache_h5ad'].split('/')[-1]}")
PY
done
