# Document Layout Analysis

## Running

The main files are:

1. train_vanilla_teacher_net.py
2. train_distill_net.py

These requires you to pass a config from ./configs with additional defaultparser arguments for Detectron2

For example, 

```bash
 python -u train_distill_net.py --config-file ./configs/Distillation/MaskRCNN_ViTb_ViTt_3x_SimKD_FPN.yaml --num-gpus 1
```

The script inference_DLA.py is used to run inference on a trained model on an arbitrary (document) images dataset. It requires a config file and a model checkpoint.