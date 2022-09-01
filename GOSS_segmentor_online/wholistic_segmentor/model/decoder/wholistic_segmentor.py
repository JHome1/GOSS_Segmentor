# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
from collections import OrderedDict
from functools   import partial

import torch
from torch    import nn
from torch.nn import functional as F

from .aspp        import ASPP
from .conv_module import stacked_conv

__all__ = ["WholisticSegmentorDecoder"]


class BPDHead(nn.Module):
    def __init__(self,class_key):
        super(BPDHead, self,).__init__()
        self.class_key = class_key
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.d2conv_ReLU = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=2, dilation=2),
                                         nn.ReLU(inplace=True))
        self.d4conv_ReLU = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=4, dilation=4),
                                         nn.ReLU(inplace=True))
        self.d8conv_ReLU = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=8, dilation=8),
                                         nn.ReLU(inplace=True))
        self.d16conv_ReLU = nn.Sequential(nn.Conv2d(1024, 128, kernel_size=3, padding=16, dilation=16),
                                          nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(896, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(inplace=True))

        self.predict_layer = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 512, kernel_size=1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(512, 2, kernel_size=1))

    def forward(self, low_level_features):
        stage3 = low_level_features['res2']
        stage4 = low_level_features['res3']
        stage5 = low_level_features['res4']

        tmp_size = stage3.size()[2:]
        d2conv_ReLU = self.d2conv_ReLU(stage5)
        d4conv_ReLU = self.d4conv_ReLU(stage5)
        d8conv_ReLU = self.d8conv_ReLU(stage5)
        d16conv_ReLU = self.d16conv_ReLU(stage5)

        dilated_conv_concat = torch.cat((d2conv_ReLU, d4conv_ReLU, d8conv_ReLU, d16conv_ReLU), 1)
        sconv1 = self.conv1(dilated_conv_concat)
        sconv1 = F.interpolate(sconv1, size=tmp_size, mode='bilinear', align_corners=True)

        sconv2 = self.conv2(stage5)
        sconv2 = F.interpolate(sconv2, size=tmp_size, mode='bilinear', align_corners=True)

        sconv3 = self.conv3(stage4)
        sconv3 = F.interpolate(sconv3, size=tmp_size, mode='bilinear', align_corners=True)

        sconv4 = self.conv4(stage3)
        sconv4 = F.interpolate(sconv4, size=tmp_size, mode='bilinear', align_corners=True)

        sconcat = torch.cat((sconv1, sconv2, sconv3, sconv4), 1)

        pred_flux = self.predict_layer(sconcat)
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = pred_flux
        return pred


class SingleWholisticSegmentorDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None):
        super(SingleWholisticSegmentorDecoder, self).__init__()
        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp          = ASPP(in_channels, out_channels=aspp_channels, atrous_rates=atrous_rates)
        self.feature_key   = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(nn.Sequential(nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                                         nn.BatchNorm2d(low_level_channels_project[i]),
                                         nn.ReLU()
                                         )
                           )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(fuse_conv(fuse_in_channels,
                                  decoder_channels,
                                  )
                        )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):

        x = features[self.feature_key]
        x = self.aspp(x)

        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

        return x


class SingleWholisticSegmentorHead(nn.Module):
    def __init__(self, decoder_channels, head_channels, num_classes, class_key):
        super(SingleWholisticSegmentorHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                                                     fuse_conv(
                                                               decoder_channels,
                                                               head_channels,
                                                               ),
                                                     nn.Conv2d(head_channels, num_classes[i], 1)
                                                     )
        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def forward(self, x):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class WholisticSegmentorDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels, low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, num_classes, **kwargs):
        super(WholisticSegmentorDecoder, self).__init__()
        self.semantic_decoder = SingleWholisticSegmentorDecoder(in_channels, feature_key, low_level_channels,
                                                                low_level_key, low_level_channels_project,
                                                                decoder_channels, atrous_rates)
        self.semantic_head    = SingleWholisticSegmentorHead(decoder_channels, decoder_channels, [num_classes], ['semantic'])
        if kwargs.get('has_segment', False):
            self.segment_head = BPDHead(['flux'])

    def set_image_pooling(self, pool_size):
        self.semantic_decoder.set_image_pooling(pool_size)
        if self.segment_decoder is not None:
            self.segment_decoder.set_image_pooling(pool_size)

    def forward(self, features):
        pred = OrderedDict()

        # Semantic branch
        semantic = self.semantic_decoder(features)
        semantic = self.semantic_head(semantic)
        for key in semantic.keys():
            pred[key] = semantic[key]

        # Segment branch
        if self.segment_head is not None:
            flux = self.segment_head(features)
            for key in flux.keys():
                pred[key] = flux[key]

        return pred
