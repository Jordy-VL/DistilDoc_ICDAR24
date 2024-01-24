#!/bin/bash
# set -x
gpu_id=${1:-0}
model_name=${2:-"llama-7b"}
dataset_name=${3:-"docvqa"}
prompt=${4:-"plain"}  # plain or alpaca
split_name=${5:-"val_test"}
comment=${6:-""}

run_name=${model_name}__Prompt_${prompt}
if [ -n "${comment}" ]; then
    run_name=${run_name}__${comment}
fi

run_name=${run_name}__${dataset_name}

export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ "${dataset_name}" = "docvqa" ]; then
    python examples/claude_docvqa.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset_name} \
        --output_dir "outputs" \
        --results_dir "results" \
        --datas_dir ${DATAS_DIR} \
        --wandb_project "Layout" \
        --run_name ${run_name} \
        --prompt ${prompt} \
        --split_name ${split_name} \
        --per_device_eval_batch_size 1
elif [ "${dataset_name}" = "docvqa_due_azure" ]; then
    python examples/claude_docvqa_due_azure.py \
        --model_name_or_path ${model_name} \
        --dataset_name ${dataset_name} \
        --output_dir "outputs" \
        --results_dir "results" \
        --datas_dir ${DATAS_DIR} \
        --wandb_project "Layout" \
        --run_name ${run_name} \
        --prompt ${prompt} \
        --split_name ${split_name} \
        --per_device_eval_batch_size 1
else
    echo "wrong dataset: "${dataset_name}
fi
