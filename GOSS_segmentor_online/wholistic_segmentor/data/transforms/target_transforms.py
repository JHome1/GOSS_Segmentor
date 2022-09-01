# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import numpy as np
import torch
import pycocotools._mask as mask_util
from ..pycocotools.coco import COCO
import cv2


class SemanticTargetGenerator(object):
    """
    Generates semantic training target only for Panoptic-DeepLab (no instance).
    Annotation is assumed to have Cityscapes format.
    Arguments:
        ignore_label: Integer, the ignore label for semantic segmentation.
        rgb2id: Function, panoptic label is encoded in a colored image, this function convert color to the
            corresponding panoptic label.
        thing_list: List, a list of thing classes
        sigma: the sigma for Gaussian kernel.
    """
    def __init__(self, ignore_label, rgb2id):
        self.ignore_label = ignore_label
        self.rgb2id = rgb2id

    def __call__(self, panoptic, segments):
        """Generates the training target.
        reference: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py
        reference: https://github.com/facebookresearch/detectron2/blob/master/datasets/prepare_panoptic_fpn.py#L18
        Args:
            panoptic: numpy.array, colored image encoding panoptic label.
            segments: List, a list of dictionary containing information of every segment, it has fields:
                - id: panoptic id, after decoding `panoptic`.
                - category_id: semantic class id.
                - area: segment area.
                - bbox: segment bounding box.
                - iscrowd: crowd region.
        Returns:
            A dictionary with fields:
                - semantic: Tensor, semantic label, shape=(H, W).
        """
        panoptic = self.rgb2id(panoptic)
        semantic = np.zeros_like(panoptic, dtype=np.uint8) + self.ignore_label

        for seg in segments:
            cat_id = seg["category_id"]
            semantic[panoptic == seg["id"]] = cat_id

        return dict(semantic=torch.as_tensor(semantic.astype('long')))


class WholisticTargetGenerator(object):
    def __init__(self, semantic_only, segment_only, unknown_class_list, small_segment_area=0, small_segment_weight=1, sigma=8):

        self.small_segment_area   = small_segment_area
        self.small_segment_weight = small_segment_weight

        self.semantic_only = semantic_only
        self.segment_only    = segment_only
        self.unknown_class_list = unknown_class_list

        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        self.coco=COCO()


    def __call__(self, semantic, segments):
        # 1. assign semantic label
        semantic = semantic

        ## set for semantic branch
        foreground       = np.zeros_like(semantic, dtype=np.uint8)
        semantic_weights = np.ones_like(semantic, dtype=np.uint8)

        ## set for segment branch
        border_foreground = np.zeros((semantic.shape[0]+2,semantic.shape[1]+2),dtype=np.uint8)
        border_semantic = cv2.copyMakeBorder(semantic,1,1,1,1,cv2.BORDER_CONSTANT,value=0)

        flux = np.zeros((2,semantic.shape[0]+2,semantic.shape[1]+2),dtype=np.float32)
        bpd_weights = np.zeros((semantic.shape[0]+2,semantic.shape[1]+2), dtype=np.float32)

        for seg in segments:
            seg["segmentation"][0] = list(map(int, seg["segmentation"][0]))

            # 2. assign foreground label
            if self.segment_only: 
                foreground[semantic == seg["category_id"]] = 1
            else: 
                if seg['category_id'] in self.unknown_class_list:
                    foreground[semantic == seg["category_id"]] = 1
            mask = self.coco.annToMask(seg)
            mask_index = np.where(mask == 1)
            if len(mask_index[0]) == 0:
                # the segment is completely cropped
                continue
            # 3. assign semantic_weights
            # Find instance area
            seg_area = len(mask_index[0])
            if seg_area < self.small_segment_area:
                semantic_weights[semantic == seg["category_id"]] = self.small_segment_weight

            # 4. creat flux map and weight matrix for flux map
            border_foreground= (border_semantic == seg["category_id"]).astype(np.uint8)
            bpd_weights[border_foreground > 0] = 1. / (np.sqrt(border_foreground.sum())+0.00001)

            _, labels = cv2.distanceTransformWithLabels(border_foreground, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
                                                        labelType=cv2.DIST_LABEL_PIXEL)

            index = np.copy(labels)
            index[border_foreground > 0] = 0
            place = np.argwhere(index > 0)
            if(place.shape[0]==0) : continue
            nearCord = place[labels - 1, :]
            x = nearCord[:, :, 0]
            y = nearCord[:, :, 1]
            nearPixel = np.zeros((2, semantic.shape[0]+2, semantic.shape[1]+2))
            nearPixel[0, :, :] = x
            nearPixel[1, :, :] = y
            grid = np.indices(border_foreground.shape)
            grid = grid.astype(float)
            diff = grid - nearPixel
            flux[:, border_foreground > 0] = diff[:, border_foreground > 0]

        bpd_weights = bpd_weights[1:-1, 1:-1]
        flux = flux[:, 1:-1, 1:-1]
        return dict(
                    semantic          = torch.as_tensor(semantic.astype('long')),
                    foreground        = torch.as_tensor(foreground.astype('long')),
                    flux              = torch.as_tensor(flux.astype(np.float32)),
                    semantic_weights  = torch.as_tensor(semantic_weights.astype(np.float32)),
                    bpb_weight_matrix = torch.as_tensor(bpd_weights.astype(np.float32))
                    )


class WholisticIDGenerator(object):
    def __init__(self):

        self.coco=COCO()

    def __call__(self, semantic, segments):

        semantic     = semantic
        wholistic_id = np.zeros_like(semantic)

        for seg in segments:
 
            mask = self.coco.annToMask(seg)
            mask_index = np.where(mask == 1)
            if len(mask_index[0]) == 0:
                # the segment has been completely cropped
                continue

            wholistic_id[mask_index] = seg['id']

        return dict(
                    semantic     = torch.as_tensor(semantic.astype('long')),
                    wholistic_id = torch.as_tensor(wholistic_id.astype('long'))
                    )
