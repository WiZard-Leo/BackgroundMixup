#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.2}"
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-$PWD/datasets}"

export PIDRAY_TEST_SPLIT="${PIDRAY_TEST_SPLIT:-positive}"

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PWD/evaluations/run_test_all_ckpts_$(date +%Y%m%d_%H%M%S)_$$}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

test_dir() {
    local config_file="$1"
    local ckpt_dir="$2"

    for ckpt_path in "${ckpt_dir}"/model_*.pth; do
        if [[ ! -f "${ckpt_path}" ]]; then
            continue
        fi

        local ckpt_name
        ckpt_name="$(basename "${ckpt_path}" .pth)"
        local output_dir="${EVAL_OUTPUT_ROOT}/$(basename "${ckpt_dir}")/eval_${ckpt_name}_${PIDRAY_TEST_SPLIT}"

        echo "Evaluating ${ckpt_path}"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        python projects/eval_net.py \
            --config-file "${config_file}" \
            --num-gpus "${NUM_GPUS}" \
            --eval-only \
            train.init_checkpoint="${ckpt_path}" \
            train.output_dir="${output_dir}" \
            dataloader.evaluator.output_dir="${output_dir}" \
            model.num_classes=12 \
            model.criterion.num_classes=12
    done
}

test_dir \
    projects/mr_detr_align/configs/eval_r50.py \
    output/mr_detr_align_r50_12ep

test_dir \
    projects/mr_detr_align/configs/eval_r50.py \
    output/mr_detr_align_r50_12ep_bgm
