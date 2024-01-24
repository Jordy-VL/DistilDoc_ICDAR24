import torch
import torch.nn as nn
import torchvision
from detectron2.modeling.backbone.resnet import BasicBlock, BottleneckBlock

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style='detectron2') -> None:
        super().__init__()

        if style == 'detectron2':
            self.block = BasicBlock(in_channels=in_channels, out_channels=out_channels)
        elif style == 'torchvision':
            self.block = torchvision.models.resnet.BasicBlock(inplanes=in_channels, planes=out_channels)
        else:
            print('Unknown option. Falling back to Detectron2 BasicBlock')
            self.block = BasicBlock(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x):
        return self.block(x)