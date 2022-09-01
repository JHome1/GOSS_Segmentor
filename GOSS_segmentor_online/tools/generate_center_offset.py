# -----------------------------------------
# Project: 'GOSS Segmentor' 
# Written by Jie Hong (jie.hong@anu.edu.au)
# -----------------------------------------
import torch
import numpy as np


def generate_center_offset_from_super_bpd(super_bpd, sigma=8): 
    size = 6 * sigma + 3
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0, y0 = 3 * sigma + 1, 3 * sigma + 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    height, width = super_bpd.shape[0], super_bpd.shape[1]
    center        = np.zeros((1, height, width), dtype=np.float32)
    offset        = np.zeros((2, height, width), dtype=np.float32)

    y_coord = np.ones_like(super_bpd, dtype=np.float32)
    x_coord = np.ones_like(super_bpd, dtype=np.float32)
    y_coord = np.cumsum(y_coord, axis=0) - 1
    x_coord = np.cumsum(x_coord, axis=1) - 1

    for mask in np.unique(super_bpd):

        mask_index = np.where(super_bpd == mask)

        # filter out small segments
        if mask_index[0].shape[0] < 2048: continue

        # assign center_points
        center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])

        # assign center&offset
        # generate center heatmap
        y, x = int(center_y), int(center_x)
        # outside image boundary
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        # upper left
        ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
        # bottom right
        br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))
               
        c, d   = max(0, -ul[0]), min(br[0], width) - ul[0]
        a, b   = max(0, -ul[1]), min(br[1], height) - ul[1]

        cc, dd = max(0, ul[0]), min(br[0], width)
        aa, bb = max(0, ul[1]), min(br[1], height)
        center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], g[a:b, c:d])
                
        # generate offset (2, h, w) -> (y-dir, x-dir)
        offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
        offset_x_index = (np.ones_like(mask_index[0]),  mask_index[0], mask_index[1])
               
        offset[offset_y_index] = center_y - y_coord[mask_index]
        offset[offset_x_index] = center_x - x_coord[mask_index]

    return torch.as_tensor(center.astype(np.float32)), torch.as_tensor(offset.astype(np.float32))
