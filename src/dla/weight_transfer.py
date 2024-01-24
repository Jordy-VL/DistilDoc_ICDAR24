import torch
import numpy as np
import os

weights_old = '/home/abanerjee/.torch/iopath_cache/detectron2/ImageNetPretrained/MAE/mae_pretrain_vit_base.pth'
weights_new = './vit_needed.pth'

wt = torch.load(weights_old, map_location='cpu')['model']

wt_new = {str('backbone.bottom_up.' + k): v for k,v in wt.items()}

torch.save(wt_new, weights_new)