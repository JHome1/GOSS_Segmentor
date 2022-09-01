# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import logging
from collections import OrderedDict

import os
import torch
import torch.nn as nn
import numpy as np

from fvcore.common.file_io     import PathManager
from wholistic_segmentor.utils import save_annotation


class SemanticEvaluator:

    def __init__(self, num_classes, ignore_label=255, output_dir=None, train_id_to_eval_id=None, 
                 colormap=None):

        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            PathManager.mkdirs(self._output_dir+'_color')
            PathManager.mkdirs(self._output_dir+'_confidence')
        self._num_classes  = num_classes
        self._ignore_label = ignore_label
        self._N = num_classes + 1  # store ignore label in the last class
        self._train_id_to_eval_id = train_id_to_eval_id
        self.colormap = colormap

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)
        self._logger = logging.getLogger(__name__)


    @staticmethod
    def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
        converted_prediction = prediction.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_prediction[prediction == train_id] = eval_id

        return converted_prediction


    def _convert_eval_id_to_train_id(self, gt_label, train_id_to_eval_id):
        converted_get_label = gt_label.copy()
        for train_id, eval_id in enumerate(train_id_to_eval_id):
            converted_get_label[gt_label == eval_id] = train_id

        return converted_get_label


    def update(self, pred, pred_conf, label, gt, image_filename=None):
        pred = pred.astype(np.int)
        gt   = gt.astype(np.int)
        gt[gt == self._ignore_label] = self._num_classes

        # comment it if 'TEST_SPLIT' is 'train2017' (for GAN model)
        if self._train_id_to_eval_id is not None:          
            gt    = self._convert_eval_id_to_train_id(gt, self._train_id_to_eval_id)
            label = self._convert_eval_id_to_train_id(label, self._train_id_to_eval_id)

        self._conf_matrix += np.bincount(self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2).reshape(self._N, self._N)

        if self._output_dir:
            # save colorized semantic prediction
            save_annotation(pred, self._output_dir+'_color', image_filename, add_colormap=True, colormap=self.colormap)

            # save semantic prediction 
            if self._train_id_to_eval_id is not None:
                pred = self._convert_train_id_to_eval_id(pred, self._train_id_to_eval_id)
            if image_filename is None:
                raise ValueError('Need to provide filename to save.')
            save_annotation(pred, self._output_dir, image_filename, add_colormap=False)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        acc = np.zeros(self._num_classes, dtype=np.float)
        iou = np.zeros(self._num_classes, dtype=np.float)
        tp  = self._conf_matrix.diagonal()[:-1].astype(np.float)

        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)

        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]

        macc = np.sum(acc) / np.sum(acc_valid)
        miou = np.sum(iou) / np.sum(iou_valid)
        fiou = np.sum(iou * class_weights)
        pacc = np.sum(tp) / np.sum(pos_gt)

        # check IoU for each specific class
        print(iou)

        res = {}
        res["mIoU"]  = 100 * miou
        res["fwIoU"] = 100 * fiou
        res["mACC"]  = 100 * macc
        res["pACC"]  = 100 * pacc

        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results
