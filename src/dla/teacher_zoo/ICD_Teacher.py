import torch
import torch.nn as nn
import torch.nn.functional as F

from odmodeling.teacher import TEACHER_REGISTRY, ModelTeacher
from arch_utils.ICD_modules import *


@TEACHER_REGISTRY.register()
class ICDTeacher(ModelTeacher):
    """
    Instance Identification with model teacher
    use the pretrained model as teacher, train a feature extractor by identification task.
    """

    def __init__(self, cfg, parent=None):
        super().__init__(cfg, parent)
        hidden_dim = cfg.MODEL.DISTILLER.KD.HIDDEN_DIM
        self.parent_buffer = [parent]
        self.pos_embedding = PositionEmbeddingSine(
            hidden_dim//2, normalize=True)
        self.attention_module = build_decoder_module(
            cfg)  # DecoderWrapper(cfg)
        self.feat_keys = cfg.MODEL.DISTILLER.KD.INPUT_FEATS

        self.ins_encoder = build_instance_encoder(cfg)

        self.reconst_w = cfg.MODEL.DISTILLER.KD.VALUE_RECONST
        if self.reconst_w > 0:
            self.reconst_projector = nn.ModuleList([
                nn.Linear(hidden_dim, 256) for i in range(max(cfg.MODEL.DISTILLER.KD.ATT_LAYERS, 1))
            ])

    def concate_multiscale_reps(self, feat, pos_emb, mask):
        # permute and concate features form multiscale to a tensor in transformer definition
        keys = self.feat_keys

        feat = torch.cat([feat[k].flatten(2).permute(2, 0, 1)
                          for k in keys], 0)  # S, N, C
        pos_emb = torch.cat([pos_emb[k].flatten(2).permute(
            2, 0, 1) for k in keys], 0)  # S, N, C
        mask = torch.cat([mask[k].flatten(2).squeeze(1)
                          for k in keys], 1)  # N, S
        return feat, pos_emb, mask

    def forward(self, batched_inputs, images, raw_outputs, fpn_outputs):
        # get raw features
        _, tea_feat_dict = super().forward(
            batched_inputs, images, raw_outputs, fpn_outputs)

        images = tea_feat_dict['images']
        tea_raw_feat, tea_fpn_feat = tea_feat_dict['backbone_feat'], tea_feat_dict['fpn_feat']

        # mask_out: zero for foreground, one for bg: BoolTensor(N, 1, H, W)
        mask_out = mask_out_padding(tea_fpn_feat, images)

        pos_embs = {k: self.pos_embedding(
            tea_fpn_feat[k], mask_out[k]) for k in self.feat_keys}

        # feat, pos: [S, N, C]; mask: [N, S], Note that mask has not normalized by softmax
        feat_k, pos_embs, mask_padding = self.concate_multiscale_reps(
            tea_fpn_feat, pos_embs, mask_out)

        feat_v = feat_k

        # instance encoding: [K, N, C], ins_mask: bool[K, N], instance_gt: (0-1)[K, N]
        # NOTE: (0 for Fake Instance) in ins_mask
        ins_feat, ins_mask, ins_mask_gt = self.ins_encoder(
            batched_inputs, pro_feats=tea_fpn_feat)

        decoded_feat_list, att_mask_list, value_list = self.attention_module(
            ins_feat, feat_k, feat_v, query_mask=ins_mask, key_padding_mask=mask_padding, pos_embedding=pos_embs)

        loss_dict = self.loss(decoded_feat_list, ins_mask, ins_mask_gt)

        aux_feat = {
            'mask_out': mask_out,
            'pos_embs': pos_embs,
            'mask_padding': mask_padding,
            'encoded_ins': (ins_feat, ins_mask, ins_mask_gt),
            'decoded_feat': decoded_feat_list,
            'decoded_mask': att_mask_list,
            'decoded_value': value_list,
        }

        if self.reconst_w > 0:
            loss_dict['loss_reconst'] = self.loss_reconst(feat_v, value_list)

        return loss_dict, {'fpn_feat': tea_fpn_feat, 'backbone_feat': tea_raw_feat, 'aux_feat': aux_feat}

    def loss_reconst(self, feat_v, value_list):
        # This is an option motivated by Information Bottleneck, which minimizes reconstruction loss
        # feat_v : [seq_len, bsz, hidden_dim]
        feat_v = feat_v.detach()
        loss = 0.0
        for i, value in enumerate(value_list):
            # value : [seq_len, bsz, num_heads, head_dim]
            value = value.flatten(2)
            value = self.reconst_projector[i](value)
            loss += F.mse_loss(value, feat_v)

        return loss / (i + 1)

    def loss(self, feat_list, ins_mask, ins_mask_gt):
        # this is the identification loss that identifies a given instance is real or fake

        loss_dict = {}
        for i, dfeat in enumerate(feat_list):
            loss = self.ins_encoder.loss(dfeat, ins_mask_gt, ins_mask)
            loss = {'tea.%s.%s' % (i, k): v for k, v in loss.items()}
            loss_dict.update(loss)

        return loss_dict