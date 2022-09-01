# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import os
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
# Logging frequency
_C.PRINT_FREQ = 20
# Checkpoint frequency
_C.CKPT_FREQ = 5000

# -----------------------------------------------------------------------------
# CUDNN
# -----------------------------------------------------------------------------
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = 'wholistic_segmentor'
# pretrained model (including decoder, head, etc) on other dataset
# need to do a net surgery to remove classifiers etc.
_C.MODEL.WEIGHTS = ''
_C.MODEL.BN_MOMENTUM = 0.1

# ---------------------------------------------------------------------------- 
# Backbone options
# ---------------------------------------------------------------------------- 
_C.MODEL.BACKBONE = CN()

# META could be
# resnet
# mobilenet_v2
# mnasnet
_C.MODEL.BACKBONE.META = 'resnet'

# NAME could be
# For resnet:
# 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
# For mobilenet_v2:
# 'mobilenet_v2'
# For mnasnet:
# 'mnasnet0_5', 'mnasnet0_75' (no official weight), 'mnasnet1_0', 'mnasnet1_3' (no official weight)
_C.MODEL.BACKBONE.NAME = "resnet50"
# Controls output stride
_C.MODEL.BACKBONE.DILATION = (False, False, True)
# pretrained backbone provided by official PyTorch modelzoo
_C.MODEL.BACKBONE.PRETRAINED = True
_C.MODEL.BACKBONE.WEIGHTS = ''

# Low-level feature key
# For resnet backbone:
# res2: 256
# res3: 512
# res4: 1024
# res5: 2048

# For mobilenet_v2 backbone:
# layer_4: 24
# layer_7: 32
# layer_14: 96
# layer_18: 320

# For mnasnet backbone:
# layer_9: 24 (0_5: 16)
# layer_10: 40 (0_5: 24)
# layer_12: 96 (0_5: 48)
# layer_14: 320 (0_5: 160)

# ---------------------------------------------------------------------------- 
# Decoder options
# ---------------------------------------------------------------------------- 
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.IN_CHANNELS = 2048
_C.MODEL.DECODER.FEATURE_KEY = 'res5'
_C.MODEL.DECODER.DECODER_CHANNELS = 256
_C.MODEL.DECODER.ATROUS_RATES = (6, 12, 18)

_C.MODEL.DECODER.CONV_TYPE = 'depthwise_separable_conv'
_C.MODEL.DECODER.CONV_KERNEL = 5
_C.MODEL.DECODER.CONV_PADDING = 2
_C.MODEL.DECODER.CONV_STACK = 1

# ---------------------------------------------------------------------------- #
# DeepLabV3+ options
# ---------------------------------------------------------------------------- #
_C.MODEL.DEEPLABV3PLUS = CN()
_C.MODEL.DEEPLABV3PLUS.LOW_LEVEL_CHANNELS = 256
_C.MODEL.DEEPLABV3PLUS.LOW_LEVEL_KEY = 'res2'
_C.MODEL.DEEPLABV3PLUS.LOW_LEVEL_CHANNELS_PROJECT = 48

# ---------------------------------------------------------------------------- 
# Wholistic-Segmentor options
# ---------------------------------------------------------------------------- 
_C.MODEL.WHOLISTIC_SEGMENTOR = CN()
_C.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_CHANNELS                 = (512, 256)
_C.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_KEY                      = ('res3', 'res2')
_C.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_CHANNELS_PROJECT         = (64, 32)
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT = CN()
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.ENABLE                     = False
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.LOW_LEVEL_CHANNELS_PROJECT = (32, 16)
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.DECODER_CHANNELS           = 128
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.HEAD_CHANNELS              = 128
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.ASPP_CHANNELS              = 256
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.NUM_CLASSES                = (1, 2)
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.CLASS_KEY                  = ('center', 'offset')
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.FOREGROUND_SEG             = False
_C.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.FOREGROUND_ARCH            = 'v1'

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = './datasets/coco'
_C.DATASET.DATASET = 'coco_known_unknown'
_C.DATASET.DATASET_SPLIT = 'manual'
_C.DATASET.DATASET_SPLIT_SPECIFIC = '20_60'
_C.DATASET.NUM_CLASSES = 171
_C.DATASET.TRAIN_SPLIT = 'train'
_C.DATASET.TEST_SPLIT = 'val'
_C.DATASET.CROP_SIZE = (513, 1025)
_C.DATASET.MIRROR = True
_C.DATASET.MIN_SCALE = 0.5
_C.DATASET.MAX_SCALE = 2.0
_C.DATASET.SCALE_STEP_SIZE = 0.1
_C.DATASET.MEAN = (0.485, 0.456, 0.406)
_C.DATASET.STD = (0.229, 0.224, 0.225)
_C.DATASET.SEMANTIC_ONLY = False
_C.DATASET.IGNORE_STUFF_IN_OFFSET = True

_C.DATASET.MIN_RESIZE_VALUE = -1
_C.DATASET.MAX_RESIZE_VALUE = -1
_C.DATASET.RESIZE_FACTOR = -1

_C.DATASET.SEGMENT_ONLY = False
_C.DATASET.SMALL_SEGMENT_AREA = 0
_C.DATASET.SMALL_SEGMENT_WEIGHT = 1

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0001
# Weight decay of norm layers.
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0
# Bias.
_C.SOLVER.BIAS_LR_FACTOR = 2.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.OPTIMIZER = 'sgd'
_C.SOLVER.ADAM_BETAS = (0.9, 0.999)
_C.SOLVER.ADAM_EPS = 1e-08

_C.SOLVER.LR_SCHEDULER_NAME = 'WarmupPolyLR'
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.GAMMA = 0.1

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.POLY_LR_POWER = 0.9
_C.SOLVER.POLY_LR_CONSTANT_ENDING = 0

_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.LOSS = CN()

_C.LOSS.SEMANTIC = CN()
_C.LOSS.SEMANTIC.NAME = 'cross_entropy'
# TODO: make `ignore` more consistent
_C.LOSS.SEMANTIC.IGNORE = 255
_C.LOSS.SEMANTIC.REDUCTION = 'mean'
_C.LOSS.SEMANTIC.THRESHOLD = 0.7
_C.LOSS.SEMANTIC.MIN_KEPT = 100000
_C.LOSS.SEMANTIC.TOP_K_PERCENT = 1.0
_C.LOSS.SEMANTIC.WEIGHT = 1.0

_C.LOSS.CENTER = CN()
_C.LOSS.CENTER.NAME = 'mse'
_C.LOSS.CENTER.REDUCTION = 'none'
_C.LOSS.CENTER.WEIGHT = 200.0

_C.LOSS.OFFSET = CN()
_C.LOSS.OFFSET.NAME = 'l1'
_C.LOSS.OFFSET.REDUCTION = 'none'
_C.LOSS.OFFSET.WEIGHT = 0.01

_C.LOSS.FOREGROUND = CN()
_C.LOSS.FOREGROUND.NAME = 'cross_entropy'
_C.LOSS.FOREGROUND.IGNORE = 255
_C.LOSS.FOREGROUND.REDUCTION = 'mean'
_C.LOSS.FOREGROUND.THRESHOLD = 0.7
_C.LOSS.FOREGROUND.MIN_KEPT = 100000
_C.LOSS.FOREGROUND.TOP_K_PERCENT = 1.0
_C.LOSS.FOREGROUND.WEIGHT = 1.0

_C.LOSS.BPD = CN()
_C.LOSS.BPD.NAME   = 'bpd_loss'
_C.LOSS.BPD.WEIGHT = 0.0001        # 0.0001

_C.LOSS.DFC = CN()
_C.LOSS.DFC.NAME   = 'dfc_loss'
_C.LOSS.DFC.WEIGHT = 0.0           # 10.0

_C.LOSS.CONTRASTIVE = CN()
_C.LOSS.CONTRASTIVE.NAME   = 'contrastive_loss'
_C.LOSS.CONTRASTIVE.WEIGHT = 0.0  

_C.LOSS.DML = CN()
_C.LOSS.DML.NAME   = 'dml_loss'
_C.LOSS.DML.WEIGHT = 0.0                


# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.IMS_PER_BATCH = 32
_C.TRAIN.MAX_ITER = 90000
_C.TRAIN.RESUME = False

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.SAMPLER_TRAIN = 'TrainingSampler'
_C.DATALOADER.TRAIN_SHUFFLE = True
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# DEBUG
# -----------------------------------------------------------------------------
_C.DEBUG = CN()
_C.DEBUG.DEBUG = True
_C.DEBUG.DEBUG_FREQ = 100
_C.DEBUG.TARGET_KEYS = ('semantic', 'center', 'offset', 'semantic_weights', 'center_weights', 'offset_weights')
_C.DEBUG.OUTPUT_KEYS = ('semantic', 'center', 'offset')
_C.DEBUG.KEEP_INTERVAL = 1000

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()

_C.TEST.GPUS = (0, )
_C.TEST.CROP_SIZE = (1025, 2049)

_C.TEST.SEMANTIC_FOLDER   = 'semantic'
_C.TEST.FOREGROUND_FOLDER = 'foreground'
_C.TEST.SEGMENT_FOLDER    = 'segment'
_C.TEST.WHOLISTIC_FOLDER  = 'wholistic'

_C.TEST.EVAL_FOREGROUND = False
_C.TEST.EVAL_WHOLISTIC  = False
_C.TEST.EVAL_SEMANTIC   = False
_C.TEST.EVAL_SEGMENT    = False
_C.TEST.EVAL_DFC        = False

_C.TEST.EVAL_SEMANTIC_N_PLUS_ONE = False
_C.TEST.EVAL_SEMANTIC_CONFIDENCE_ADJUSTMENT = False
_C.TEST.EVAL_SEMANTIC_ANOMALY = False

_C.TEST.EVAL_SEMANTIC_MSP = False
_C.TEST.EVAL_SEMANTIC_MAXLOGIT = False

_C.TEST.MODEL_FILE = ''
_C.TEST.TEST_TIME_AUGMENTATION = False
_C.TEST.FLIP_TEST = False
_C.TEST.SCALE_LIST = [1]

_C.TEST.DEBUG = False

_C.TEST.ORACLE_SEMANTIC = False
_C.TEST.ORACLE_FOREGROUND = False
_C.TEST.ORACLE_CENTER = False
_C.TEST.ORACLE_OFFSET = False

_C.TEST.INSTANCE_SCORE_TYPE = "semantic"
_C.TEST.CONFIDENCE_ADJUSTMENT_SCALE = 1.0
_C.TEST.MSP_PROBILITY_THRESHOLD = 0.1
_C.TEST.MAXLOGIT_PROBILITY_THRESHOLD = 0.1

# -----------------------------------------------------------------------------
# POST PROCESSING
# -----------------------------------------------------------------------------
_C.POST_PROCESSING = CN()
_C.POST_PROCESSING.CENTER_THRESHOLD = 0.1
_C.POST_PROCESSING.NMS_KERNEL = 7
_C.POST_PROCESSING.TOP_K_INSTANCE = 200
_C.POST_PROCESSING.STUFF_AREA = 2048


def update_config(cfg, args):
    cfg.defrost()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
