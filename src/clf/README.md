# Classification

## Installation

```bash
!pip3 install numpy
!pip3 install scipy
!pip3 install scikit-learn
!pip3 install matplotlib
!pip3 install pandas
!pip3 install transformers
!pip3 install huggingface_hub
!pip3 install datasets
!pip3 install evaluate
!pip3 install timm
!pip3 install wandb
!pip3 install torch
!pip3 install torchvision
```

## other installs

* ensure you have wandb setup: https://docs.wandb.ai/quickstart
* ensure you are logged into HuggingFace using the cli: https://huggingface.co/docs/huggingface_hub/main/quick-start#login 


## Running

Everything under `scripts` contains the necessary runs

### Test the implementations

`./test_distillation.sh`

IF this does not work, contact @ANON, send full stacktrace and error message

### Run the experiments

Specific experiments to reproduce:

#### train teachers
    
```bash
    #teachers
    python3 train_teacher.py --teacher_model vit --epochs 100 --dataset 'maveriq/tobacco3482' --expt_name 'vit-base_tobacco' 
    python3 train_teacher.py --teacher_model dit --epochs 100 --dataset 'maveriq/tobacco3482' --expt_name 'dit-base_tobacco' 
    python3 train_teacher.py --teacher_model swin --epochs 10 --dataset 'rvl_cdip' --expt_name 'swin-base_rvl-cdip' 
    python3 train_teacher.py --teacher_model cnn --epochs 100 --dataset 'maveriq/tobacco3482' --gradient_accumulation_steps 1 --batch_size 64 --expt_name 'resnet101-base_tobacco' 
    python3 train_teacher.py --teacher_model cnn --epochs 10 --dataset 'rvl_cdip'  --gradient_accumulation_steps 1 --batch_size 64 --expt_name 'resnet101_rvl-cdip' 

    #for tobacco we can have another more advanced teacher model [supervised pre-trained on rvl-cdip, then on tobacco]
    python3 train_teacher.py --teacher_model dit --epochs 100 --dataset 'maveriq/tobacco3482' --sup_teacher_weights 'microsoft/dit-base-finetuned-rvlcdip' --save_intermediate_teachers --expt_name 'dit-finetuned_rvl_tobacco_multi' 
    python3 train_teacher.py --teacher_model vit --epochs 100 --dataset 'maveriq/tobacco3482' --sup_teacher_weights 'jordyvl/vit-base_rvl-cdip' --save_intermediate_teachers --expt_name 'vit-base_rvl_tobacco_multi' 
```


#### response and feature-based NK1000 (ViT)
```bash
cd clf
dataset="rvl_cdip-NK1000"
TEACHERS=(jordyvl/vit-base_rvl-cdip microsoft/dit-base-finetuned-rvlcdip ) #vit-small_rvl_cdip vit-tiny_rvl_cdip)
STUDENT_ARCHITECTURES=(tiny small)

for teacher in ${TEACHERS[@]}; do
    for architecture in ${STUDENT_ARCHITECTURES[@]}; do
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill_loss CE --distill kd --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224""
        echo $command; $command
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill_loss CE+KD --distill kd --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224" --alpha 0.5 --temperature 2.5"
        echo $command; $command
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill_loss MSE --distill kd --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224""
        echo $command; $command
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill kd --distill_loss NKD --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224" --gamma 1.5 --temperature 1"
        echo $command; $command
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill hint --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224""
        echo $command; $command
        CUDA_VISIBLE_DEVICES=0 command="python3 train_student.py --epochs 50 --dataset $dataset --distill simkd --student_model "vit-$architecture" --teacher_model $teacher --student_weights "WinKawaks/vit-$architecture-patch16-224""
        echo $command; $command
        CUDA_VISIBLE_DEVICES=0 command="python3 train_student.py --epochs 50 --dataset $dataset --distill og_simkd --student_model "vit-$architecture" --teacher_model $teacher  --student_weights "WinKawaks/vit-$architecture-patch16-224" --batch_size 16"
        echo $command; $command
    done
done
```
#### response and feature-based NK1000 (ResNet)

```bash
#### response-based
python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'CE+KD' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --alpha 0.5 --temperature 2.5 --batch_size 64 --gradient_accumulation_steps 1
python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'MSE' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --batch_size 64 --gradient_accumulation_steps 1
python3 train_student.py --epochs 50 --dataset $dataset --distill 'kd' --distill_loss 'NKD' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --gamma 1.5 --temperature 1 --batch_size 64 --gradient_accumulation_steps 1

#### feature-based 
python3 train_student.py --epochs 50 --dataset $dataset --distill 'hint' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --batch_size 64 --gradient_accumulation_steps 1
python3 train_student.py --epochs 50 --dataset $dataset --distill 'simkd' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --batch_size 64 --gradient_accumulation_steps 1
python3 train_student.py --epochs 50 --dataset $dataset --distill 'og_simkd' --student_model 'cnn' --teacher_model $teacher  --student_weights 'microsoft/resnet-50' --batch_size 64 --gradient_accumulation_steps 1
```


#### temperature ablation (e.g., ResNet)
```bash
TEMPERATURES=(1.5 2.5 5)
ALPHAS=(0.5 0.7 0.9)

for temperature in ${TEMPERATURES[@]}; do

    for alpha in ${ALPHAS[@]}; do
        python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'CE+KD' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --student_weights 'microsoft/resnet-50' --alpha $alpha --temperature $temperature --batch_size 64 --gradient_accumulation_steps 1
    done
done
```

#### students KD without pretraining
```bash
cd ../clf
dataset="rvl_cdip-NK1000"

# rand ViT
TEACHERS=(jordyvl/vit-base_rvl-cdip microsoft/dit-base-finetuned-rvlcdip) 
STUDENT_ARCHITECTURES=(tiny small)

for teacher in ${TEACHERS[@]}; do
    for architecture in ${STUDENT_ARCHITECTURES[@]}; do
        command="python3 train_student.py --epochs 50 --dataset $dataset --distill_loss CE --distill kd --student_model "vit-$architecture" --teacher_model $teacher --expt_name rand "
        echo $command ; $command


# rand ResNet
## logit-based
python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'CE' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --alpha 0.5 --temperature 2.5 --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand
python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'CE+KD' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --alpha 0.5 --temperature 1 --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand
python3 train_student.py --epochs 50 --dataset $dataset --distill_loss 'MSE' --distill 'kd' --student_model 'cnn' --teacher_model $teacher --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand
python3 train_student.py --epochs 50 --dataset $dataset --distill 'kd' --distill_loss 'NKD' --student_model 'cnn' --teacher_model $teacher --gamma 1.5 --temperature 1 --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand

## feature-based 
python3 train_student.py --epochs 50 --dataset $dataset --distill 'hint' --student_model 'cnn' --teacher_model $teacher --batch_size 64 --gradient_accumulation_steps 1
python3 train_student.py --epochs 50 --dataset $dataset --distill 'simkd' --student_model 'cnn' --teacher_model $teacher --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand
python3 train_student.py --epochs 50 --dataset $dataset --distill 'og_simkd' --student_model 'cnn' --teacher_model $teacher  --batch_size 64 --gradient_accumulation_steps 1 --expt_name rand
```
