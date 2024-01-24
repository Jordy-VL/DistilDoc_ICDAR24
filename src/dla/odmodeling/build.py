from detectron2.config import CfgNode as CN

import model_zoo

def build_distill_configs(cfg):
    cfg.EXTRA_CFG = []
    # load extra configs for convenience

    cfg = model_zoo.add_vitdet_defaults(cfg)
    cfg = model_zoo.add_dit_defaults(cfg)
    cfg = model_zoo.add_vanilla_vit_defaults(cfg)

    cfg.SOLVER.OPTIMIZER = 'SGD'
    cfg.MODEL.DISTILLER = CN()

    cfg.MODEL.DISTILLER.PRETRAIN_TEACHER_ITERS = 0
    # pretrain teacher for extra iterations (with maximum learning rate)
    # remember to set teacher warmup to zero !
    cfg.MODEL.DISTILLER.BYPASS_DISTILL = 1000
    # when to start distillation
    cfg.MODEL.DISTILLER.BYPASS_DISTILL_AFTER = 99999999
    # when to disable distillation
    cfg.MODEL.DISTILLER.STOP_JOINT = 9999999
    # when to stop joint training (set teachers loss to 0)

    cfg.MODEL.DISTILLER.FIX_BACKBONE_BEFORE = 0
    cfg.MODEL.DISTILLER.FIX_HEAD_BEFORE = 99999999

    cfg.MODEL.DISTILLER.TEACHER = 'ModelTeacher' #'ICDTeacher'
    # transconv, trans, small, meanteacher ..
    cfg.MODEL.DISTILLER.TYPES = ['SimKD_FPN']
    # options: ICD, SimKD_FPN, SimKD

    cfg.MODEL.DISTILLER.PRELOAD_TEACHER = ''
    cfg.MODEL.DISTILLER.PRELOAD_TYPE = 'teacher_only'
    # 'all', 'teacher_only'

    # bag of tricks
    cfg.MODEL.DISTILLER.PRELOAD_FPN = False
    cfg.MODEL.DISTILLER.PRELOAD_HEAD = True

    cfg.MODEL.DISTILLER.IGNORE_DISTILL = ['']

    # model distiller
    cfg.MODEL.DISTILLER.MODEL_DISTILLER_CONFIG = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml' # 'COCO-Detection/retinanet_R_101_FPN_3x.yaml'
    cfg.MODEL.DISTILLER.MODEL_LOAD_OFFICIAL = False
    cfg.MODEL.DISTILLER.MODEL_LOAD_FILE = True
    # load model from MODEL_ZOO or configs.weights

    cfg.MODEL.DISTILLER.BACKBONE_NAME = "build_resnet_fpn_backbone"

    _C = cfg.MODEL.DISTILLER
    build_tea_optimizer(_C)
    cfg.MODEL.DISTILLER.KD = CN()
    cfg.MODEL.DISTILLER.DISTILL_FLAG = 'distill_only'
    # options : 'distill_only', 'supervision_only', 'distill_and_supervision'

    ############################ INSTANCE CONDITIONAL DISTILLATION (ICD) ################################
    cfg.MODEL.DISTILLER.KD.TASK_NAME = 'InstanceEncoder'

    cfg.MODEL.DISTILLER.KD.DECODER = 'DecoderWrapper'
    cfg.MODEL.DISTILLER.KD.DECODER_POSEMB_ON_V = False
    cfg.MODEL.DISTILLER.KD.DISTILL_WITH_POS = False

    cfg.MODEL.DISTILLER.KD.ADD_SCALE_INDICATOR = True

    cfg.MODEL.DISTILLER.KD.DISTILL_NORM = 'ln'

    cfg.MODEL.DISTILLER.KD.NUM_CLASSES = 80
    cfg.MODEL.DISTILLER.KD.UNIFORM_SAMPLE_CLASS = False
    cfg.MODEL.DISTILLER.KD.UNIFORM_SAMPLE_BOX = False

    cfg.MODEL.DISTILLER.KD.INPUT_FEATS = ['p3', 'p4', 'p5', 'p6', 'p7']
    # NOTE: This is fundermental

    cfg.MODEL.DISTILLER.KD.ATT_LAYERS = 1
    # 0 for multi-head att, > 0 for cascade of N layers of transformer decoder

    cfg.MODEL.DISTILLER.KD.PROJECT_POS = True
    # project position embeddings

    cfg.MODEL.DISTILLER.KD.SAMPLE_RULE = 'relative'
    # negative sampling rule: {fixed, relative}
    cfg.MODEL.DISTILLER.KD.NUM_NEG_POS_RATIO = 5.0
    cfg.MODEL.DISTILLER.KD.NUM_LABELS = 100
    cfg.MODEL.DISTILLER.KD.MAX_LABELS = 300

    cfg.MODEL.DISTILLER.KD.DROPOUT = 0.0

    cfg.MODEL.DISTILLER.KD.CONFIDENCE_LOSS_WEIGHT = 1.0

    cfg.MODEL.DISTILLER.KD.USE_POS_EMBEDDING = True

    cfg.MODEL.DISTILLER.KD.ATT_HEADS = 8
    cfg.MODEL.DISTILLER.KD.HIDDEN_DIM = 256
    # following common practice

    cfg.MODEL.DISTILLER.KD.VALUE_RECONST = -1.0

    cfg.MODEL.DISTILLER.KD.INS_ATT_MIMIC = CN({
        'WEIGHT_VALUE': 8.0,

        'TEMP_MASK': 1.0,
        'TEMP_VALUE': 1.0,
        'TEMP_DECAY': False,
        'TEMP_DECAY_TO': 1.0,

        'DISTILL_NEGATIVE': False,

        'LOSS_FORM': 'mse',
        'SAMPLE_RANGE': 0.3,
    })

    ############################ SIMPLE KNOWLEDGE DISTILLATION (SimKD/SimKD_FPN) ################################
    cfg.MODEL.DISTILLER.KD.PROJECTOR_TYPE = 'conv'
    # options : 'conv' - ResNet Block, 'mlp' - Transformer MLP, 'tencoder' - Transformer Encoder, 'none' - No Projector (Teacher and Student has same channel dim)
    cfg.MODEL.DISTILLER.KD.PROJECTOR_MODE = 'multiple'
    # options : 'single' - single proector for all the layers, 'multiple' - separate projectors for each of the layers
    cfg.MODEL.DISTILLER.KD.PROJ_INFEAT = [256, 256, 256, 256, 256]   # use an integer if single projector, use list with chennel dim for each lateral outputs
    cfg.MODEL.DISTILLER.KD.PROJ_OUTFEAT = [256, 256, 256, 256, 256]  # use an integer if single projector, use list with chennel dim for each lateral outputs
    cfg.MODEL.DISTILLER.KD.RESHAPE = None                       # modes: C2T/T2C : (h*w=p*p -> h,w,p)/(p=h=w -> p*p,p,p)
    cfg.MODEL.DISTILLER.KD.T_RESHAPE = None                     # modes: C2T/T2C : (h*w=p*p -> h,w,p)/(p=h=w -> p*p,p,p)

    cfg.MODEL.DISTILLER.KD.INPUT_FEATS = ['p2', 'p3', 'p4', 'p5', 'p6'] # override from ICD

    ######################## Loss ############################
    cfg.MODEL.DISTILLER.DISTILL_OFF = 0
    cfg.MODEL.DISTILLER.DISTILL_ON = 1
    # 0: no distill but update adapter head
    # 1: distill student
    # 2: distill teachers
    # others: two way backward

    return cfg


def build_tea_optimizer(_C):
    # This is copied from detectron2 and modified according to DETR

    _C.SOLVER = CN()
    # See detectron2/solver/build.py for LR scheduler options
    # Follows DETR settings
    _C.SOLVER.OPTIMIZER = 'ADAMW'
    # Step LR causes significant change in few iters, we use cosine for simplicity
    _C.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    # NOTE: MAX_ITER will to rewrite automatically
    _C.SOLVER.MAX_ITER = -1
    _C.SOLVER.BASE_LR = 1e-4
    _C.SOLVER.MOMENTUM = 0.9
    _C.SOLVER.NESTEROV = False

    _C.SOLVER.RESCALE_INTERVAL = False
    _C.SOLVER.NUM_DECAYS = 3
    _C.SOLVER.BASE_LR_END = 0.0

    _C.SOLVER.WEIGHT_DECAY = 0.0001
    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    _C.SOLVER.WEIGHT_DECAY_NORM = 0.0

    _C.SOLVER.GAMMA = 0.1
    # The iteration number to decrease learning rate by GAMMA.
    _C.SOLVER.STEPS = (30000,)

    # All follows DETR settings
    _C.SOLVER.WARMUP_FACTOR = 1.0
    _C.SOLVER.WARMUP_ITERS = 10
    _C.SOLVER.WARMUP_METHOD = "linear"

    # Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
    # biases. This is not useful (at least for recent models). You should avoid
    # changing these and they exist only to reproduce Detectron v1 training if
    # desired.
    _C.SOLVER.BIAS_LR_FACTOR = 1.0
    _C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

    # Gradient clipping
    _C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": True})
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    _C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    # Maximum absolute value used for clipping gradients
    _C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    _C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0