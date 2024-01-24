# !/bin/bash
# This script is used to run the downstream scripts for the project.
MODELS=(
output/teacher/vitb_imagenet_doclaynet_tecaher
output/teacher/resnet101_imagenet_doclaynet_teacher
output/student/vitb_vitt_simkd_fpn_doclaynet
output/student/r101_r50_reviewkd_doclaynet
output/student/r101_r50_simkd_doclaynet
output/student/vitt_imagenet_doclaynet_tecaher  
output/student/r50_doclaynet
output/student/vitb_vitt_reviewkd_doclaynet
)

DATASETS=(
docvqa
infographics_vqa
#DUDE
)

for model in ${MODELS[@]}; do
    for dataset in ${DATASETS[@]}; do
        echo "Running downstream scripts for model: $model for dataset: $dataset"
        command="python3 DLA_to_BBOX.py --ocr aws_neurips_time/$dataset/dev --dla $model/inference/$dataset/instances_predictions.pth --origin $(basename $model) --ignore_tokens Text"
        echo $command ; $command
    done    
done