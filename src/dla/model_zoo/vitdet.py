from functools import partial
import torch
import torch.nn as nn

from detectron2.config import CfgNode as CN
from detectron2.modeling.backbone import ViT, SimpleFeaturePyramid, get_vit_lr_decay_rate
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone import BACKBONE_REGISTRY


# class PyramidTransformer(SimpleFeaturePyramid):
#     def __init__(self, net, in_feature, out_channels, scale_factors, top_block=None, norm="LN", square_pad=0):
#         super(PyramidTransformer, self).__init__(net=net, 
#                                                  in_feature=in_feature, 
#                                                  out_channels=out_channels, 
#                                                  scale_factors=scale_factors, 
#                                                  top_block=top_block, 
#                                                  norm=norm, 
#                                                  square_pad=square_pad)
#         self.bottom_up = self.net

@BACKBONE_REGISTRY.register()
def build_vitdet_backbone(cfg, input_shape):
    """
    Create a ViTDet instance from config.

    Returns:
        ViT: a :class:`ViT` instance.
    """
    assert input_shape.height == input_shape.width, "Need Square Images"
    in_channel = input_shape.channels
    image_size = 1024 # input_shape.height

    arch = cfg.MODEL.VIT.ARCH
    assert arch in ['base', 'large', 'tiny', 'small', 'huge'], "Invalid ViT Architecture"
    patch_size          = cfg.MODEL.VIT.PATCH_SIZE
    mlp_ratio           = cfg.MODEL.VIT.MLP_RATIO
    qkv_bias            = cfg.MODEL.VIT.QKV_BIAS
    # drop_path_rate      = cfg.MODEL.VIT.DROP_PATH_RATE

    # use_abs_pos = True
    # re_pos_zero_init = True
    # residual_block_indexes = ()
    # use_act_checkpoint = False
    # pretrain_img_size = 224
    # pretrain_use_cls_token = True
    use_rel_pos = True
    window_size = 14
    out_feature = cfg.MODEL.VIT.OUT_FEATURES
    
    if cfg.MODEL.VIT.NORM == 'ln':
        # norm = nn.LayerNorm
        norm = partial(nn.LayerNorm, eps=1e-6)
    elif cfg.MODEL.VIT.NORM == 'bn':
        norm = nn.BatchNorm1d           # TODO: Need to check, may end up in error
    else:
        print('cfg.MODEL.VIT.NORM type not implemented')
        raise NotImplementedError
    
    if cfg.MODEL.VIT.ACT == 'relu':
        act = nn.ReLU
    elif cfg.MODEL.VIT.ACT == 'gelu':
        act = nn.GELU
    else:
        print('cfg.MODEL.VIT.ACT type not implemented')
        raise NotImplementedError

    if arch == 'base':
        embed_dim = 768
        depth = 12
        assert cfg.MODEL.VIT.DEPTH == depth, 'ViT base variant depth mismatch'
        num_heads = 12
        drop_path_rate = 0.1
        window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10,] # 2, 5, 8 11 for global attention
    elif arch == 'large':
        embed_dim = 1024
        depth = 24
        assert cfg.MODEL.VIT.DEPTH == depth, 'ViT large variant depth mismatch'
        num_heads = 16
        drop_path_rate = 0.4
        window_block_indexes = list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23)) # 5, 11, 17, 23 for global attention
    elif arch == 'huge':
        embed_dim == 1280
        depth = 32
        assert cfg.MODEL.VIT.DEPTH == depth, 'ViT huge variant depth mismatch'
        num_heads = 16
        drop_path_rate = 0.5
        window_block_indexes = list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
    elif arch == 'tiny':
        embed_dim == 192
        depth = 12
        assert cfg.MODEL.VIT.DEPTH == depth, 'ViT tiny variant depth mismatch'
        num_heads = 3
        drop_path_rate = 0.1
        window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10,]
    elif arch == 'small':
        embed_dim == 384
        depth = 12
        assert cfg.MODEL.VIT.DEPTH == depth, 'ViT small variant depth mismatch'
        num_heads = 8
        drop_path_rate = 0.1
        window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10,]
    else:
        print("Custom ViT models not supported yet!")

    backbone = ViT(img_size=image_size,
        patch_size=patch_size,
        in_chans=in_channel,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        norm_layer=norm,
        act_layer=act,
        use_rel_pos=use_rel_pos,
        window_size=window_size,
        window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        out_feature=out_feature)
    
    vit_pyramid = SimpleFeaturePyramid(net=backbone, 
                                        in_feature=out_feature,
                                        out_channels=256,
                                        scale_factors=(4.0, 2.0, 1.0, 0.5),
                                        top_block=LastLevelMaxPool(),
                                        norm="LN",
                                        square_pad=1024)
    
    vit_pyramid.bottom_up = vit_pyramid.net

    return vit_pyramid

def add_vitdet_defaults(cfg):
    cfg.MODEL.VIT = CN()
    cfg.MODEL.VIT.ARCH = 'base'
    cfg.MODEL.VIT.DEPTH = 12
    cfg.MODEL.VIT.PATCH_SIZE = 16
    cfg.MODEL.VIT.MLP_RATIO = 4.0
    cfg.MODEL.VIT.QKV_BIAS = True
    cfg.MODEL.VIT.NORM = 'ln'
    cfg.MODEL.VIT.ACT = 'gelu'
    cfg.MODEL.VIT.OUT_FEATURES = "last_feat"
    cfg.SOLVER.OPTIMIZER = 'SGD'
    return cfg