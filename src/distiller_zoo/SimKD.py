import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_utils import *

from odmodeling.distiller import DISTILLER_REGISTRY, KDWrapper

# @DISTILLER_REGISTRY.register()
# class SimKD_FPN(KDWrapper):
#     """
#     Distillation reused teacher classifier, Simple KD, CVPR 2022
#     """

#     def __init__(self, cfg, student, teacher) -> None:
#         super().__init__(cfg, student, teacher)
#         # self.cfg = cfg
#         # self.student = [student]

#         # self.cfg = cfg
#         self.crit = nn.MSELoss()
#         # self.projector = nn.Identity()      # FPN has same number of channels TODO: Need to recheck for feature map height and width

#         self.proj_keys = list(cfg.MODEL.DISTILLER.KD.INPUT_FEATS)
#         if cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'multiple':
#             assert type(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT) == list
#             assert type(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT) == list
#             assert len(list(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT)) == len(self.proj_keys)
#             assert len(list(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT)) == len(self.proj_keys)
#             in_channel_dim = {k: list(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT)[ind] for ind,k in enumerate(self.proj_keys)} # cfg.MODEL.DISTILLER.KD.PROJ_INFEAT
#             out_channel_dim = {k: list(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT)[ind] for ind,k in enumerate(self.proj_keys)} # cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT
#         elif cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'single':
#             assert type(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT) == int
#             assert type(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT) == int
#             in_channel_dim = {k: cfg.MODEL.DISTILLER.KD.PROJ_INFEAT for k in self.proj_keys} # [cfg.MODEL.DISTILLER.KD.PROJ_INFEAT]
#             out_channel_dim = {k: cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT for k in self.proj_keys} # [cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT]
        
#         self.projector = nn.ModuleDict({}) # {}
#         # for inc,outc in zip(in_channel_dim,out_channel_dim):
#         if cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'multiple':
#             for k in self.proj_keys:
#                 inc = in_channel_dim[k]
#                 outc = out_channel_dim[k]
#                 if cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'conv':
#                     # self.projector[k] = ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2') # .to(torch.device(cfg.MODEL.DEVICE))
#                     self.projector.update({k: ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2')})
#                 elif cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'none':
#                     # self.projector[k] = nn.Identity() # .to(torch.device(cfg.MODEL.DEVICE))
#                     self.projector.update({k: nn.Identity()}) # .to(torch.device(cfg.MODEL.DEVICE))
#                 else:
#                     print('Not Implemented Yet!')
#                     assert False
#         elif cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'single':
#             if cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'conv':
#                 uniform_proj = ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2')
#             elif cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'none':
#                 uniform_proj = nn.Identity()
#             else:
#                 print('Not Implemented Yet!')
#                 assert False
#             for k in self.proj_keys:
#                 self.projector.update({k: uniform_proj})

#         self.student_reshape = cfg.MODEL.DISTILLER.KD.RESHAPE
#         self.teacher_reshape = cfg.MODEL.DISTILLER.KD.T_RESHAPE

#         # load teacher detection head to the student
#         origin_dict = student.state_dict()
#         ckpt = {k: v for k, v in teacher.pretrained_model[0].state_dict(
#         ).items() if ('backbone' not in k) and ('pixel_' not in k) and (k in origin_dict)}
#         origin_dict.update(ckpt)
#         mskeys, unexp_keys = student.load_state_dict(origin_dict)
#         print('Loading Pretrained Teacher Detection Head to Student Finished !!')
#         print('Warning!!')
#         print('Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
#         print('Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))
#         # if comm.is_main_process():
#         #     logger.info('Loading Pretrained Detection Head Finished !!')
#         #     logger.warning(
#         #         'Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
#         #     logger.warning(
#         #         'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

#     def forward(self, features_dict, features_dict_tea):
        
#         target = features_dict_tea['fpn_feat']
#         source = features_dict['fpn_feat']

#         loss_dict = {}
#         loss_value = 0

#         # for idx, proj in enumerate(self.projector):
#         for idx, proj in self.projector.items():
#             s = source[idx]
#             t = target[idx]
#             assert s.shape == t.shape
#             B, C, h, w = s.shape[0], s.shape[1], s.shape[2], s.shape[3]

#             if self.student_reshape is not None:
#                 if self.student_reshape == 'C2T':
#                     s = torch.permute(torch.flatten(s, start_dim=-2), (0,2,1))
#                 elif self.student_reshape == 'T2C':
#                     p = torch.sqrt(int(s.size()[-1]))
#                     ss = s.size()[:-1]
#                     s = torch.reshape(torch.permute(s, (0,2,1)), (*ss,int(p),int(p))) if p%1==0 else torch.reshape(torch.permute(s[:,1:,:], (0,2,1)), (*ss,int(p),int(p)))
#                 else:
#                     print('Invalid option')
            
#             if self.teacher_reshape is not None:
#                 if self.teacher_reshape == 'C2T':
#                     t = torch.permute(torch.flatten(t, start_dim=-2), (0,2,1))
#                 elif self.teacher_reshape == 'T2C':
#                     p = torch.sqrt(int(t.size()[-1]))
#                     ts = t.size()[:-1]
#                     t = torch.reshape(torch.permute(t, (0,2,1)), (*ts,int(p),int(p))) if p%1==0 else torch.reshape(torch.permute(t[:,1:,:], (0,2,1)), (*ts,int(p),int(p)))
#                 else:
#                     print('Invalid option')

#             proj_s = proj(s)
            
#             loss_value += self.crit(proj_s.view(B*h*w, -1), t.view(B*h*w, -1))

#         loss_dict['SimKD_loss'] = loss_value
#         # print(loss_dict)
        
#         return loss_dict
    
@DISTILLER_REGISTRY.register()
class SimKD(KDWrapper):
    """
    Distillation reused teacher classifier, Simple KD, CVPR 2022
    """

    def __init__(self, cfg, student, teacher) -> None:
        super().__init__(cfg, student, teacher)
        # self.cfg = cfg
        # self.student = [student]

        # self.cfg = cfg
        self.projector = nn.Identity()      # TODO: Need to recheck for feature map size
        self.crit = nn.MSELoss()

        # load teacher FPN to the student
        ckpt = {k: v for k, v in teacher.fpn[0].state_dict().items() if 'bottom_up' not in k}
        origin_dict = student.backbone.state_dict()
        origin_dict.update(ckpt)
        mskeys, unexp_keys = student.backbone.load_state_dict(origin_dict)
        print('Loading Pretrained Teacher FPN Module to Student Finished !!')
        print('Warning!!')
        print('Incompatible Keys: <%s>' % [k for k in teacher.fpn[0].state_dict() if k not in origin_dict])
        print('Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))
        # if comm.is_main_process():
        #     logger.info('Loading Pretrained FPN Module Finished !!')
        #     logger.warning(
        #         'Incompatible Keys: <%s>' % [k for k in teacher.fpn[0].state_dict() if k not in origin_dict])
        #     logger.warning(
        #         'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

        # load teacher detection head to the student
        origin_dict = student.state_dict()
        ckpt = {k: v for k, v in teacher.pretrained_model[0].state_dict(
        ).items() if ('backbone' not in k) and ('pixel_' not in k) and (k in origin_dict)}
        origin_dict.update(ckpt)
        mskeys, unexp_keys = student.load_state_dict(origin_dict)
        print('Loading Pretrained Teacher Detection Head to Student Finished !!')
        print('Warning!!')
        print('Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
        print('Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))
        # if comm.is_main_process():
        #     logger.info('Loading Pretrained Detection Head Finished !!')
        #     logger.warning(
        #         'Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
        #     logger.warning(
        #         'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

    def forward(self, features_dict, features_dict_tea):
        
        target = features_dict_tea['backbone_feat']
        source = features_dict['backbone_feat']

        loss_dict = {}
        loss_value = 0

        for (s, t) in zip(source, target):
            loss_value += self.crit(source, target)

        loss_dict['SimKD_loss'] = loss_value
        
        return loss_dict
    

# OverRide
@DISTILLER_REGISTRY.register()
class SimKD_FPN(KDWrapper):
    """
    Distillation reused teacher classifier, Simple KD, CVPR 2022
    """

    def __init__(self, cfg, student, teacher) -> None:
        super().__init__(cfg, student, teacher)
        # self.cfg = cfg
        # self.student = [student]

        # self.cfg = cfg
        self.crit = nn.MSELoss()
        # self.projector = nn.Identity()      # FPN has same number of channels TODO: Need to recheck for feature map height and width

        # self.proj_keys = list(cfg.MODEL.DISTILLER.KD.INPUT_FEATS)
        # if cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'multiple':
        #     assert type(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT) == list
        #     assert type(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT) == list
        #     assert len(list(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT)) == len(self.proj_keys)
        #     assert len(list(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT)) == len(self.proj_keys)
        #     in_channel_dim = {k: list(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT)[ind] for ind,k in enumerate(self.proj_keys)} # cfg.MODEL.DISTILLER.KD.PROJ_INFEAT
        #     out_channel_dim = {k: list(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT)[ind] for ind,k in enumerate(self.proj_keys)} # cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT
        # elif cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'single':
        #     assert type(cfg.MODEL.DISTILLER.KD.PROJ_INFEAT) == int
        #     assert type(cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT) == int
        #     in_channel_dim = {k: cfg.MODEL.DISTILLER.KD.PROJ_INFEAT for k in self.proj_keys} # [cfg.MODEL.DISTILLER.KD.PROJ_INFEAT]
        #     out_channel_dim = {k: cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT for k in self.proj_keys} # [cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT]
        
        # self.projector = nn.ModuleDict({}) # {}
        self.dummy = nn.Parameter(torch.randn(1), requires_grad=True)

        # for inc,outc in zip(in_channel_dim,out_channel_dim):
        # if cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'multiple':
        #     for k in self.proj_keys:
        #         inc = in_channel_dim[k]
        #         outc = out_channel_dim[k]
        #         if cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'conv':
        #             # self.projector[k] = ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2') # .to(torch.device(cfg.MODEL.DEVICE))
        #             self.projector.update({k: ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2')})
        #         elif cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'none':
        #             # self.projector[k] = nn.Identity() # .to(torch.device(cfg.MODEL.DEVICE))
        #             self.projector.update({k: nn.Identity()}) # .to(torch.device(cfg.MODEL.DEVICE))
        #         else:
        #             print('Not Implemented Yet!')
        #             assert False
        # elif cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE == 'single':
        #     if cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'conv':
        #         uniform_proj = ResNetBlock(in_channels=inc,out_channels=outc,style='detectron2')
        #     elif cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE == 'none':
        #         uniform_proj = nn.Identity()
        #     else:
        #         print('Not Implemented Yet!')
        #         assert False
        #     for k in self.proj_keys:
        #         self.projector.update({k: uniform_proj})

        self.student_reshape = cfg.MODEL.DISTILLER.KD.RESHAPE
        self.teacher_reshape = cfg.MODEL.DISTILLER.KD.T_RESHAPE

        # load teacher detection head to the student
        origin_dict = student.state_dict()
        ckpt = {k: v for k, v in teacher.pretrained_model[0].state_dict(
        ).items() if ('backbone' not in k) and ('pixel_' not in k) and (k in origin_dict)}
        origin_dict.update(ckpt)
        mskeys, unexp_keys = student.load_state_dict(origin_dict)
        print('Loading Pretrained Teacher Detection Head to Student Finished !!')
        print('Warning!!')
        print('Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
        print('Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))
        # if comm.is_main_process():
        #     logger.info('Loading Pretrained Detection Head Finished !!')
        #     logger.warning(
        #         'Incompatible Keys: <%s>' % [k for k in teacher.pretrained_model[0].state_dict() if k not in origin_dict])
        #     logger.warning(
        #         'Incompatible: <%s> \n Unexpected: <%s>' % (mskeys, unexp_keys))

    def forward(self, features_dict, features_dict_tea):
        
        target = features_dict_tea['fpn_feat']
        source = features_dict['fpn_feat']

        loss_dict = {}
        loss_value = 0

        # for idx, proj in enumerate(self.projector):
        # for idx, proj in self.projector.items():
        for idx in source.keys():
            s = source[idx]
            t = target[idx]
            assert s.shape == t.shape
            B, C, h, w = s.shape[0], s.shape[1], s.shape[2], s.shape[3]

            if self.student_reshape is not None:
                if self.student_reshape == 'C2T':
                    s = torch.permute(torch.flatten(s, start_dim=-2), (0,2,1))
                elif self.student_reshape == 'T2C':
                    p = torch.sqrt(int(s.size()[-1]))
                    ss = s.size()[:-1]
                    s = torch.reshape(torch.permute(s, (0,2,1)), (*ss,int(p),int(p))) if p%1==0 else torch.reshape(torch.permute(s[:,1:,:], (0,2,1)), (*ss,int(p),int(p)))
                else:
                    print('Invalid option')
            
            if self.teacher_reshape is not None:
                if self.teacher_reshape == 'C2T':
                    t = torch.permute(torch.flatten(t, start_dim=-2), (0,2,1))
                elif self.teacher_reshape == 'T2C':
                    p = torch.sqrt(int(t.size()[-1]))
                    ts = t.size()[:-1]
                    t = torch.reshape(torch.permute(t, (0,2,1)), (*ts,int(p),int(p))) if p%1==0 else torch.reshape(torch.permute(t[:,1:,:], (0,2,1)), (*ts,int(p),int(p)))
                else:
                    print('Invalid option')

            # proj_s = proj(s)
            proj_s = s
            
            loss_value += self.crit(proj_s.view(B*h*w, -1), t.view(B*h*w, -1))

        loss_dict['SimKD_loss'] = loss_value
        # print(loss_dict)
        
        return loss_dict
