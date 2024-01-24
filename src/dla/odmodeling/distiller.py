import torch
from torch import nn
from torch.nn import functional as F

import torch.nn.functional as F
# from .utils import *

# from distiller_zoo import *

from detectron2.utils.registry import Registry

DISTILLER_REGISTRY = Registry("DISTILLER")  # noqa F401 isort:skip
DISTILLER_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_distiller(cfg, name, student, teacher):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    model = DISTILLER_REGISTRY.get(name)(cfg, student, teacher)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

@DISTILLER_REGISTRY.register()
class KDWrapper(nn.Module):
    """
    A wrapper class to implement different distillation 
    """
    def __init__(self, cfg, student, teacher) -> None:
        super().__init__()
        """
        student: student model
        teacher: teacher with wrapper
        """
        self.cfg = cfg
        self.student = [student]
        self.teacher = [teacher]

    def forward(self, features_dict, features_dict_tea):
        loss_dict = {}
        return loss_dict
    