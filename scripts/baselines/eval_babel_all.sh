#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-results_babel}"
BATCH_SIZE="${2:-48}"

run_eval() {
  local cfg_name="$1"
  local method="$2"
  local variant="$3"
  local decoder="$4"
  local exp_name="${5:-$cfg_name}"
  local ckpt_path="${RESULTS_ROOT}/${cfg_name}/${exp_name}_/models/checkpoint_best.pt"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] checkpoint not found: ${ckpt_path}"
    return 0
  fi

  echo "[eval] ${cfg_name} -> ${ckpt_path}"
  if [[ -n "${variant}" ]]; then
    python eval.py \
      --cfg auto \
      --ckpt_path "${ckpt_path}" \
      --method "${method}" \
      --variant "${variant}" \
      --decoder "${decoder}" \
      --batch_size "${BATCH_SIZE}"
  else
    python eval.py \
      --cfg auto \
      --ckpt_path "${ckpt_path}" \
      --method "${method}" \
      --decoder "${decoder}" \
      --batch_size "${BATCH_SIZE}"
  fi
}

run_eval "latent_ar_gru_moflow" "latent_ar" "gru" "moflow_structured"
run_eval "latent_ar_gru_mlp" "latent_ar" "gru" "mlp"
run_eval "latent_ar_transformer_moflow" "latent_ar" "transformer" "moflow_structured"
run_eval "latent_ar_transformer_mlp" "latent_ar" "transformer" "mlp"
run_eval "rssm_moflow" "rssm" "" "moflow_structured"
run_eval "rssm_mlp" "rssm" "" "mlp"
