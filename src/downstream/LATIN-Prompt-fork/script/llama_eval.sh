#!/bin/bash
# set -x
gpu_id=${1:-0}
model_name=${2:-"llama-7b"} 
dataset_name=${3:-"docvqa"}
prompt=${4:-"plain"} 
comment=${5:-""}
ocr_version=${6:-""}


run_name=${model_name}__Prompt_${prompt}
if [ -n "${comment}" ]; then
    run_name=${run_name}__${comment}
    comment="--comment ${comment}"
fi
if [ -n "${ocr_version}" ]; then
    run_name=${run_name}__${ocr_version}
fi
run_name=${run_name}__${dataset_name}

export CUDA_VISIBLE_DEVICES=${gpu_id}


python examples/llama_docvqa_due_azure.py \
    --model_name_or_path ${model_name} \
    --dataset_name ${dataset_name} \
    --output_dir "outputs" \
    --results_dir "results" \
    --datas_dir ${DATAS_DIR} \
    --wandb_project "Layout" \
    --run_name ${run_name} \
    --prompt ${prompt} \
    --per_device_eval_batch_size 2 \
    ${comment}


# #can you turn this python command into a variable and echo first?
# command="python examples/llama_docvqa_due_azure.py \
#     --model_name_or_path ${model_name} \
#     --dataset_name ${dataset_name} \
#     --output_dir "outputs" \
#     --results_dir "results" \
#     --datas_dir ${DATAS_DIR} \
#     --wandb_project "Layout" \
#     --run_name ${run_name} \
#     --prompt ${prompt} \
#     --per_device_eval_batch_size 2 \
#     ${comment}"
# echo $command
# $command