#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-results_babel}"
BATCH_SIZE="${2:-48}"

run_eval() {
  local cfg_name="$1"
  local decoder="$2"
  local ckpt_path="${RESULTS_ROOT}/${cfg_name}/${cfg_name}_/models/checkpoint_best.pt"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] checkpoint not found: ${ckpt_path}"
    return 0
  fi

  echo "[eval] ${cfg_name} -> ${ckpt_path}"
  python eval.py \
    --cfg auto \
    --ckpt_path "${ckpt_path}" \
    --method rssm \
    --decoder "${decoder}" \
    --batch_size "${BATCH_SIZE}"
}

run_eval "rssm_moflow" "moflow_structured"
run_eval "rssm_mlp" "mlp"
