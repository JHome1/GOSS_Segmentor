# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import torch
from .backbone import resnet, mobilenet, mnasnet, hrnet, xception
from .meta_arch import WholisticSegmentor
from .loss      import RegularCE, OhemCE, DeepLabCE, L1Loss, MSELoss, CrossEntropyLoss, BPDLoss, PixelContrastLoss, VarLoss


def build_segmentation_model_from_cfg(config):
    """Builds segmentation model with specific configuration.
    Args:
        config: the configuration.

    Returns:
        A nn.Module segmentation model.
    """
    model_map = {
                 'wholistic_segmentor': WholisticSegmentor,
                 }

    model_cfg = {
                 'wholistic_segmentor': dict(
                                             replace_stride_with_dilation        = config.MODEL.BACKBONE.DILATION,
                                             in_channels                         = config.MODEL.DECODER.IN_CHANNELS,
                                             feature_key                         = config.MODEL.DECODER.FEATURE_KEY,

                                             low_level_channels                  = config.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_CHANNELS,
                                             low_level_key                       = config.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_KEY,
                                             low_level_channels_project          = config.MODEL.WHOLISTIC_SEGMENTOR.LOW_LEVEL_CHANNELS_PROJECT,

                                             decoder_channels                    = config.MODEL.DECODER.DECODER_CHANNELS,
                                             atrous_rates                        = config.MODEL.DECODER.ATROUS_RATES,
                                             num_classes                         = config.DATASET.NUM_CLASSES,
                                             has_segment                         = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.ENABLE,

                                             segment_low_level_channels_project  = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.LOW_LEVEL_CHANNELS_PROJECT,
                                             segment_decoder_channels            = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.DECODER_CHANNELS,
                                             segment_head_channels               = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.HEAD_CHANNELS,
                                             segment_aspp_channels               = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.ASPP_CHANNELS,
                                             segment_num_classes                 = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.NUM_CLASSES,
                                             segment_class_key                   = config.MODEL.WHOLISTIC_SEGMENTOR.SEGMENT.CLASS_KEY,

                                             semantic_loss                       = build_loss_from_cfg(config.LOSS.SEMANTIC),
                                             semantic_loss_weight                = config.LOSS.SEMANTIC.WEIGHT,
                                             bpd_loss                            = build_loss_from_cfg(config.LOSS.BPD),
                                             bpd_loss_weight                     = config.LOSS.BPD.WEIGHT,
                                             contrastive_loss                    = build_loss_from_cfg(config.LOSS.CONTRASTIVE),
                                             contrastive_loss_weight             = config.LOSS.CONTRASTIVE.WEIGHT,
                                             dml_loss                            = build_loss_from_cfg(config.LOSS.DML),
                                             dml_loss_weight                     = config.LOSS.DML.WEIGHT,
                                             dfc_loss                            = build_loss_from_cfg(config.LOSS.DFC),
                                             dfc_loss_weight                     = config.LOSS.DFC.WEIGHT,
                                             ),
                 }

    if config.MODEL.BACKBONE.META == 'resnet':
        backbone = resnet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            replace_stride_with_dilation=model_cfg[config.MODEL.META_ARCHITECTURE]['replace_stride_with_dilation']
        )
    elif config.MODEL.BACKBONE.META == 'mobilenet_v2':
        backbone = mobilenet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
        )
    elif config.MODEL.BACKBONE.META == 'mnasnet':
        backbone = mnasnet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
        )
    elif config.MODEL.BACKBONE.META == 'hrnet':
        backbone = hrnet.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
        )
    elif config.MODEL.BACKBONE.META == 'xception':
        backbone = xception.__dict__[config.MODEL.BACKBONE.NAME](
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            replace_stride_with_dilation=model_cfg[config.MODEL.META_ARCHITECTURE]['replace_stride_with_dilation']
        )
    else:
        raise ValueError('Unknown meta backbone {}, please first implement it.'.format(config.MODEL.BACKBONE.META))

    model = model_map[config.MODEL.META_ARCHITECTURE](backbone,
                                                      **model_cfg[config.MODEL.META_ARCHITECTURE])

    # set batchnorm momentum
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.momentum = config.MODEL.BN_MOMENTUM
    return model


def build_loss_from_cfg(config):
    """Builds loss function with specific configuration.
    Args:
        config: the configuration.

    Returns:
        A nn.Module loss.
    """
    if config.NAME == 'cross_entropy':
        # return CrossEntropyLoss(ignore_index=config.IGNORE, reduction='mean')
        return RegularCE(ignore_label=config.IGNORE)
    elif config.NAME == 'ohem':
        return OhemCE(ignore_label=config.IGNORE, threshold=config.THRESHOLD, min_kept=config.MIN_KEPT)
    elif config.NAME == 'hard_pixel_mining':
        return DeepLabCE(ignore_label=config.IGNORE, top_k_percent_pixels=config.TOP_K_PERCENT)
    elif config.NAME == 'mse':
        return MSELoss(reduction=config.REDUCTION)
    elif config.NAME == 'l1':
        return L1Loss(reduction=config.REDUCTION)
    elif config.NAME == 'bpd_loss':
        return BPDLoss()
    elif config.NAME == 'contrastive_loss':
        return PixelContrastLoss()
    elif config.NAME == 'dml_loss':
        return VarLoss()
    elif config.NAME == 'dfc_loss':
        return torch.nn.L1Loss(size_average = True)
    else:
        raise ValueError('Unknown loss type: {}'.format(config.NAME))
