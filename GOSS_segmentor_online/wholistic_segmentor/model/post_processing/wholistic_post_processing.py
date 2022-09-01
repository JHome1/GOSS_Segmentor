# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import torch
import torch.nn.functional as F

from .semantic_post_processing import get_semantic_segmentation

__all__ = ['find_instance_center', 'get_segment_segmentation', 'get_wholistic_segmentation']


def find_instance_center(ctr_hmp, threshold=0.1, nms_kernel=3, top_k=None):
    """
    Find the center points from the center heatmap.
    Arguments:
        ctr_hmp: A Tensor of shape [N, 1, H, W] of raw center heatmap output, where N is the batch size,
            for consistent, we only support N=1.
        threshold: A Float, threshold applied to center heatmap score.
        nms_kernel: An Integer, NMS max pooling kernel size.
        top_k: An Integer, top k centers to keep.
    Returns:
        A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
    """
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1)

    # NMS
    nms_padding = (nms_kernel - 1) // 2
    ctr_hmp_max_pooled = F.max_pool2d(ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding)
    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, 'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    if top_k is None:
        return ctr_all
    elif ctr_all.size(0) < top_k:
        return ctr_all
    else:
        # find top k centers.
        top_k_scores, _ = torch.topk(torch.flatten(ctr_all), top_k)
        return torch.nonzero(ctr_hmp > top_k_scores[-1])


def group_pixels(ctr, offsets):
    """
    Gives each pixel in the image an instance id.
    Arguments:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).
        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size,
            for consistent, we only support N=1. The order of second dim is (offset_y, offset_x).
    Returns:
        A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).
    """
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(height, dtype=offsets.dtype, device=offsets.device).repeat(1, width, 1).transpose(1, 2)
    x_coord = torch.arange(width, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    # distance: [K, H*W]
    distance = torch.norm(ctr - ctr_loc, dim=-1)

    # finds center with minimum distance at each location, offset by 1, to reserve id=0 for stuff
    instance_id = torch.argmin(distance, dim=0).reshape((1, height, width)) + 1
    return instance_id


def get_segment_segmentation(sem_seg, ctr_hmp, offsets, unknown_class_list, threshold=0.1, nms_kernel=3, top_k=None,
                             unknown_class_seg=None):

    if unknown_class_seg is None:
        # gets foreground segmentation
        unknown_class_seg = torch.zeros_like(sem_seg)
        for unknown_class_class in unknown_class_list:
            unknown_class_seg[sem_seg == unknown_class_class] = 1

    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel, top_k=top_k)
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)
    seg = group_pixels(ctr, offsets)

    return unknown_class_seg * seg, ctr.unsqueeze(0)


def merge_semantic_and_segment(sem_seg, seg, label_divisor, unknown_class_list, stuff_area, void_label):

    # In case unknown class mask does not align with semantic prediction
    pan_seg                    = torch.zeros_like(sem_seg) + void_label
    unknown_class_seg          = seg > 0
    semantic_unknown_class_seg = torch.zeros_like(sem_seg)
    for unknown_class in unknown_class_list:
        semantic_unknown_class_seg[sem_seg == unknown_class] = 1

    # keep track of segment id for each class
    class_id_tracker = {}

    # paste unknown_class region by majority voting
    segment_ids = torch.unique(seg)
    for seg_id in segment_ids:
        if seg_id == 0:
            continue
        # Make sure only do majority voting within semantic_unknown_class_seg
        unknown_class_mask = (seg == seg_id) & (semantic_unknown_class_seg == 1)
        if torch.nonzero(unknown_class_mask).size(0) == 0:
            continue
        class_id, _ = torch.mode(sem_seg[unknown_class_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_seg_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_seg_id = 1
        class_id_tracker[class_id.item()] += 1
        pan_seg[unknown_class_mask] = class_id * label_divisor + new_seg_id

    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in unknown_class_list:
            # unknown class
            continue
        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~unknown_class_seg)
        area = torch.nonzero(stuff_mask).size(0)
        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


def get_wholistic_segmentation(sem, ctr_hmp, offsets, unknown_class_list, label_divisor, stuff_area, void_label,
                               threshold=0.1, nms_kernel=3, top_k=None, foreground_mask=None):

    if sem.dim() != 4 and sem.dim() != 3:
        raise ValueError('Semantic prediction with un-supported dimension: {}.'.format(sem.dim()))
    if sem.dim() == 4 and sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if foreground_mask is not None:
        if foreground_mask.dim() != 4 and foreground_mask.dim() != 3:
            raise ValueError('Foreground prediction with un-supported dimension: {}.'.format(sem.dim()))

    if sem.dim() == 4:
        semantic = get_semantic_segmentation(sem)
    else:
        semantic = sem

    if foreground_mask is not None:
        if foreground_mask.dim() == 4:
            unknown_class_seg = get_semantic_segmentation(foreground_mask)
        else:
            unknown_class_seg = foreground_mask
    else:
        unknown_class_seg = None

    segment, center = get_segment_segmentation(semantic, ctr_hmp, offsets, unknown_class_list,
                                               threshold=threshold, nms_kernel=nms_kernel, top_k=top_k,
                                               unknown_class_seg=unknown_class_seg)
    wholistic = merge_semantic_and_segment(semantic, segment, label_divisor, unknown_class_list, stuff_area, void_label)

    return wholistic, center
