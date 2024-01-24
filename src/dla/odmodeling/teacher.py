import os
from detectron2.config.config import CfgNode
from detectron2.config import get_cfg
import torch
from torch import nn
from torch.nn import functional as F
import detectron2.model_zoo

import torch.nn.functional as F
# from .utils import *
from .build import build_distill_configs
# from teacher_zoo import *

from detectron2.utils.registry import Registry
from detectron2.structures import ImageList
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model


TEACHER_REGISTRY = Registry("TEACHER")  # noqa F401 isort:skip
TEACHER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_teacher(cfg, parent):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.DISTILLER.TEACHER
    model = TEACHER_REGISTRY.get(meta_arch)(cfg, parent)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


def detach_wrapper(x, module):
    set_requires_grad(module, False)
    y = module(x)
    set_requires_grad(module, True)
    return y


def set_requires_grad(m, flag):
    for p in m.parameters():
        p.requires_grad = flag


def get_tea_cfg(cfg):
    cfg_ = get_cfg()
    cfg_.MODEL.DISTILLER = CfgNode()
    cfg_ = build_distill_configs(cfg_)
    cfg_.merge_from_file(os.path.join(
        'configs', cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG), allow_unsafe=True)
    return cfg_

def add_defaults_config(cfg):
    import model_zoo
    cfg = model_zoo.add_vitdet_defaults(cfg)
    cfg = model_zoo.add_dit_defaults(cfg)
    cfg = model_zoo.add_vanilla_vit_defaults(cfg)
    return cfg


@TEACHER_REGISTRY.register()
class ModelTeacher(nn.Module):
    def __init__(self, cfg, parent=None):
        super().__init__()
        self.cfg = cfg
        # self.dummy_params = nn.Parameter(torch.rand(1,1), requires_grad=True)               # to avoid zero parameter in teacher optimizer

        if cfg.MODEL.DISTILLER.MODEL_LOAD_OFFICIAL:
            cfg_ = detectron2.model_zoo.get_config(
                cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG, trained=True)

            if cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG.endswith('.py'):
                cfg_.model.backbone.bottom_up.stem.norm = \
                    cfg_.model.backbone.bottom_up.stages.norm = \
                    cfg_.model.backbone.norm = "FrozenBN"

                cfg_.model.roi_heads.box_head.conv_norm = \
                    cfg_.model.roi_heads.mask_head.conv_norm = "FrozenBN"

            device = cfg.MODEL.DEVICE
            if device is not None and isinstance(cfg_, CfgNode):
                cfg_.MODEL.DEVICE = device

            # print(device)

            if isinstance(cfg_, CfgNode):
                model = build_model(cfg_)
                DetectionCheckpointer(model).load(cfg_.MODEL.WEIGHTS)
            else:
                from detectron2.config import instantiate
                model = instantiate(cfg_.model)
                if device is not None:
                    model = model.to(device)
                if "train" in cfg_ and "init_checkpoint" in cfg_.train:
                    DetectionCheckpointer(model).load(
                        cfg_.train.init_checkpoint)

            pretrained_model = model
        # TO CHECK: cfg freeze and default setup
        elif cfg.MODEL.DISTILLER.MODEL_LOAD_FILE:
            cfg_ = get_cfg()
            cfg_ = add_defaults_config(cfg_)
            cfg_.merge_from_file(os.path.join('configs', cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG), allow_unsafe=True)
            cfg_.MODEL.DEVICE = device = cfg.MODEL.DEVICE

            pretrained_model = build_model(cfg_)
            DetectionCheckpointer(pretrained_model).load(cfg_.MODEL.WEIGHTS)
            print('Teacher Weights Loaded! FROM : ', str(cfg_.MODEL.WEIGHTS))
        else:
            cfg_ = get_tea_cfg(cfg)
            cfg_.MODEL.DEVICE = device = cfg.MODEL.DEVICE

            pretrained_model = build_model(cfg_)
            DetectionCheckpointer(pretrained_model).load(cfg_.MODEL.WEIGHTS)

        # we only leave backbone for distillation
        # we do not record the parameters
        for p in pretrained_model.parameters():
            p.requires_grad = False

        self.pretrained_model = [pretrained_model]
        self.model = [pretrained_model.backbone.bottom_up]
        # NOTE: pretrained model do not have backbone !
        pretrained_model.backbone.bottom_up = nn.Sequential()
        self.fpn = [pretrained_model.backbone]

        # pretrained_model.pixel_mean #
        self.pixel_mean = torch.tensor(
            cfg_.MODEL.PIXEL_MEAN, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_mean.requires_grad = False
        # pretrained_model.pixel_std #torch.tensor(cfg_.MODEL.PIXEL_STD, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_std = torch.tensor(
            cfg_.MODEL.PIXEL_STD, device=self.cfg.MODEL.DEVICE).view(-1, 1, 1)
        self.pixel_std.requires_grad = False

    def forward(self, batched_inputs, images, raw_outputs, fpn_outputs):
        # NOTE: Maybe add support for JIT
        # TODO: Modify for logit level distillation - Backbone + FPN implemented, Need to add + Head
        with torch.no_grad():
            images = [x["image"].to(self.cfg.MODEL.DEVICE)
                      for x in batched_inputs]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]

            images = ImageList.from_tensors(
                images, self.fpn[0].size_divisibility)

            r_feat = self.model[0](images.tensor)
        with torch.no_grad():
            feat = self.fpn[0](r_feat)
        
        teacher_loss_dict = {}      # Keeping it balnk in the wrapper

        return teacher_loss_dict, {'fpn_feat': feat, 'backbone_feat': r_feat, 'images': images}
