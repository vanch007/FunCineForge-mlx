#!/bin/bash
# MPS single-device inference for Apple Silicon Mac
# Usage: cd exps && bash infer_mps.sh

# exp dir
infer_config="decode.yaml"
lm_ckpt_path="funcineforge_zh_en/llm/ds-model.pt.best/mp_rank_00_model_states.pt"
fm_ckpt_path="funcineforge_zh_en/flow/ds-model.pt.best/mp_rank_00_model_states.pt"
voc_ckpt_path="funcineforge_zh_en/vocoder/ds-model.pt.best/avg_5_removewn.pt"

# input & output
test_data_jsonl="data/demo.jsonl"
output_dir="results"

ext_opt=""
random_seed="0"

. parse_options.sh || exit 1;

echo "=== FunCineForge MPS Inference ==="
echo "output dir: ${output_dir}"
mkdir -p ${output_dir}
current_time=$(date "+%Y-%m-%d_%H-%M")
log_file="${output_dir}/log_mps_${current_time}.txt"
echo "log_file: ${log_file}"

workspace=$(pwd)

# MPS environment tuning
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Allow full unified memory usage

python infer.py \
  --config-path "${workspace}/decode_conf" \
  --config-name "${infer_config}" \
  ++node_rank=0 \
  ++world_size=1 \
  ++num_gpus=1 \
  ++disable_pbar=false \
  ++random_seed="${random_seed}" \
  ++data_jsonl="${test_data_jsonl}" \
  ++output_dir="${output_dir}" \
  ++lm_ckpt_path="${lm_ckpt_path}" \
  ++fm_ckpt_path="${fm_ckpt_path}" \
  ++voc_ckpt_path="${voc_ckpt_path}" ${ext_opt} 2>&1 | tee ${log_file}
