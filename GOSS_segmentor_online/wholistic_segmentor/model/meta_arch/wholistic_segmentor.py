# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .base                             import BaseSegmentationModel
from wholistic_segmentor.model.decoder import WholisticSegmentorDecoder
from wholistic_segmentor.utils         import AverageMeter

__all__ = ["WholisticSegmentor"]


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        self.dim = dim_in

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class WholisticSegmentor(BaseSegmentationModel):
    def __init__(self, backbone, in_channels, feature_key, low_level_channels, low_level_key,
                 low_level_channels_project, decoder_channels, atrous_rates, num_classes,
                 semantic_loss, semantic_loss_weight, bpd_loss, bpd_loss_weight, contrastive_loss, contrastive_loss_weight, dml_loss, dml_loss_weight, dfc_loss, dfc_loss_weight, **kwargs):

        decoder = WholisticSegmentorDecoder(in_channels, feature_key, low_level_channels, low_level_key,
                                            low_level_channels_project, decoder_channels, atrous_rates, num_classes,
                                            **kwargs)

        super(WholisticSegmentor, self).__init__(backbone, decoder)

        self.num_classes = num_classes

        self.semantic_loss                    = semantic_loss
        self.semantic_loss_weight             = semantic_loss_weight
        self.loss_meter_dict                  = OrderedDict()
        self.loss_meter_dict['Loss']          = AverageMeter()
        self.loss_meter_dict['Semantic loss'] = AverageMeter()

        self.bpd_loss                    = bpd_loss
        self.loss_meter_dict['BPD loss'] = AverageMeter()
        self.bpd_loss_weight             = bpd_loss_weight

        self.contrastive_loss = contrastive_loss
        self.loss_meter_dict['Contrastive loss'] = AverageMeter()
        self.contrastive_loss_weight             = contrastive_loss_weight

        self.dml_loss = dml_loss
        self.loss_meter_dict['DML loss'] = AverageMeter()
        self.dml_loss_weight             = dml_loss_weight

        self.dfc_loss = dfc_loss
        self.loss_meter_dict['DFC loss'] = AverageMeter()
        self.dfc_loss_weight             = dfc_loss_weight

        # Initialize parameters.
        self._init_params()

        self.projection_head = ProjectionHead(dim_in=in_channels, proj_dim=256)
        self.feature_key = feature_key

    def _upsample_predictions(self, pred, input_shape):
        """Upsamples final prediction, with special handling to offset.
            Args:
                pred (dict): stores all output of the segmentation model.
                input_shape (tuple): spatial resolution of the desired shape.
            Returns:
                result (OrderedDict): upsampled dictionary.
            """
        # Override upsample method to correctly handle `offset`
        result = OrderedDict()
        for key in pred.keys():
            out = F.interpolate(pred[key], size=input_shape, mode='bilinear', align_corners=True)
            if 'offset' in key:
                scale = (input_shape[0] - 1) // (pred[key].shape[2] - 1)
                out *= scale
            result[key] = out
        return result

    def loss(self, results, targets=None, embedding=None):
        batch_size = results['semantic'].size(0)
        loss = 0

        if targets is not None:
            # semantic loss
            if 'semantic_weights' in targets.keys():
                semantic_loss = self.semantic_loss(results['semantic'], targets['semantic'], semantic_weights=targets['semantic_weights']) * self.semantic_loss_weight
            else:
                semantic_loss = self.semantic_loss(results['semantic'], targets['semantic']) * self.semantic_loss_weight
            self.loss_meter_dict['Semantic loss'].update(semantic_loss.detach().cpu().item(), batch_size)
            loss += semantic_loss

            # bpd loss
            if self.bpd_loss_weight != 0.0:
                bpd_loss = self.bpd_loss_weight * self.bpd_loss(results['flux'], targets['flux'], targets['bpb_weight_matrix'])
                self.loss_meter_dict['BPD loss'].update(bpd_loss.detach().cpu().item(), batch_size)
                loss += bpd_loss

            # dfc loss using BPD head
            if self.dfc_loss_weight != 0.0:
                unsupervised_targetZ = torch.zeros(results['flux'].shape[0],results['flux'].shape[1]-1,results['flux'].shape[2],results['flux'].shape[3]).cuda()
                unsupervised_targetY = torch.zeros(results['flux'].shape[0],results['flux'].shape[1],results['flux'].shape[2]-1,results['flux'].shape[3]).cuda()
                unsupervised_targetX = torch.zeros(results['flux'].shape[0],results['flux'].shape[1],results['flux'].shape[2],results['flux'].shape[3]-1).cuda()
                trans_resultZ = results['flux'][:, 1:, :, :] - results['flux'][:, 0:-1, :, :]
                trans_resultY = results['flux'][:, :, 1:, :] - results['flux'][:, :, 0:-1, :] 
                trans_resultX = results['flux'][:, :, :, 1:] - results['flux'][:, :, :, 0:-1] 

                lz = self.dfc_loss(unsupervised_targetZ.cuda(), trans_resultZ.cuda())
                ly = self.dfc_loss(unsupervised_targetY.cuda(), trans_resultY.cuda())
                lx = self.dfc_loss(unsupervised_targetX.cuda(), trans_resultX.cuda())

                dfc_loss = self.dfc_loss_weight * (lz+ly+lx)
                self.loss_meter_dict['DFC loss'].update(dfc_loss.detach().cpu().item(), batch_size)
                loss += dfc_loss

            # contrastive loss
            if self.contrastive_loss_weight != 0.0:
                _, pred_ = torch.max(results['semantic'], 1)
                contrastive_loss = self.contrastive_loss_weight * self.contrastive_loss(embedding, targets['semantic'], pred_)
                self.loss_meter_dict['Contrastive loss'].update(contrastive_loss.detach().cpu().item(), batch_size)
                loss += contrastive_loss

            # dml loss
            if self.dml_loss_weight != 0.0:
                logits = results['semantic']
                logits = logits.permute((0, 2, 3, 1))
                logits = torch.flatten(logits, start_dim=0, end_dim=2)

                dml_loss = self.dml_loss_weight * self.dml_loss(logits, torch.flatten(targets['semantic'], start_dim=0, end_dim=2), self.num_classes)
                self.loss_meter_dict['DML loss'].update(dml_loss.detach().cpu().item(), batch_size)
                loss += dml_loss

        # In distributed DataParallel, this is the loss on one machine, need to average the loss again
        # in train loop.
        results['loss'] = loss
        self.loss_meter_dict['Loss'].update(loss.detach().cpu().item(), batch_size)

        return results

    def forward(self, x, targets = None):
        input_shape = x.shape[-2:]

        # !!!targets type is int32, should be long
        # contract: features is a dict of tensors
        features  = self.backbone(x)
        embedding = self.projection_head(features[self.feature_key])

        pred = self.decoder(features)
        results = self._upsample_predictions(pred, input_shape)

        if targets is None:
            return results
        else:
            return self.loss(results, targets, embedding)
