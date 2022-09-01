# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import logging
from collections import OrderedDict
import os
import json

import numpy as np
import cv2
import PIL.Image as img
from PIL import ImageFilter
from tabulate import tabulate

from fvcore.common.file_io      import PathManager
from wholistic_segmentor.utils  import save_annotation
from .wholisticapi_evaluation   import pq_compute

from ..data.transforms import WholisticIDGenerator

logger = logging.getLogger(__name__)


class COCOWholisticEvaluator:

    def __init__(self, output_dir=None, train_id_to_eval_id=None, label_divisor=256, void_label=65280, num_classes=121,
                 gt_dir='./datasets/coco', split='val2017', data_split='manual', data_split_specific='20_60',
                 small_segment_area=None, small_segment_weight=None, colormap=None):

        if output_dir is None:
            raise ValueError('Must provide a output directory.')
        self._output_dir = output_dir
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            PathManager.mkdirs(self._output_dir+'_cluster')
            PathManager.mkdirs(self._output_dir+'_color')

        self._wholistic_dir = os.path.join(self._output_dir, 'predictions')
        if self._wholistic_dir:
            PathManager.mkdirs(self._wholistic_dir)

        self._predictions      = []
        self._predictions_json = os.path.join(output_dir, 'predictions.json')

        self._train_id_to_eval_id = train_id_to_eval_id
        self._label_divisor       = label_divisor
        self._void_label          = void_label
        self._num_classes         = num_classes

        self._logger = logging.getLogger(__name__)

        self._gt_json_file   = os.path.join(gt_dir+'_stuff_'+data_split, 'annotations', 'segments_'+data_split+data_split_specific+'_{}.json'.format(split))    
        self._gt_folder      = os.path.join(gt_dir+'_stuff_'+data_split, data_split+data_split_specific, data_split+data_split_specific+'_{}'.format(split)) 

        self._pred_json_file = os.path.join(output_dir, 'predictions.json')
        self._pred_folder    = self._wholistic_dir

        self.target_transform = WholisticIDGenerator()
        self.colormap = colormap


    # def update(self, wholistic, semantic_pred, image_filename=None, image_id=None):
    def update(self, wholistic, semantic_pred, image_filename=None, image_id=None, image_itself=None):
        from panopticapi.utils import id2rgb

        if image_filename is None:
            raise ValueError('Need to provide image_filename.')
        if image_id is None:
            raise ValueError('Need to provide image_id.')

        # Change void region.
        wholistic[wholistic == self._void_label] = 0

        segments_info = []
        for who_lab in np.unique(wholistic):

            pred_class = who_lab // self._label_divisor
            if self._train_id_to_eval_id is not None:
                pred_class = self._train_id_to_eval_id[pred_class]

            segments_info.append(
                                 {
                                  'id':          int(who_lab),
                                  'category_id': int(pred_class),
                                  }
                                 )

        # save wholistic prediction
        save_annotation(id2rgb(wholistic), self._wholistic_dir, image_filename, add_colormap=False)

        # save clustering prediction   
        edge = img.fromarray(id2rgb(wholistic).astype(dtype=np.uint8))
        edge = edge.filter(ImageFilter.FIND_EDGES)
        edge = edge.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edge = np.array(edge)
        edge[np.where(edge>0)] = 255

        edge_grey = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        edge_contours, _ = cv2.findContours(edge_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_countours = np.zeros(edge.shape)

        cv2.drawContours(img_countours, edge_contours, -1, (255, 255, 255), 2)
        save_annotation(img_countours, self._output_dir+'_cluster', image_filename, add_colormap=False)

        # save colorized wholistic prediction
        save_annotation(semantic_pred, self._output_dir+'_color', image_filename, add_colormap=True, image=img_countours, image_itself=image_itself, colormap=self.colormap)

        self._predictions.append(
                                 {
                                  'image_id':      int(image_id),
                                  'file_name':     image_filename + '.png',
                                  'segments_info': segments_info,
                                  }
                                 )


    def evaluate(self, data_loader, config):
       
        gt_json_file   = self._gt_json_file
        gt_folder      = self._gt_folder
        pred_json_file = self._pred_json_file
        pred_folder    = self._pred_folder

        with open(gt_json_file, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions
        with PathManager.open(self._predictions_json, "w") as f:
            f.write(json.dumps(json_data))

        pq_res = pq_compute(gt_json_file, pred_json_file, gt_folder, pred_folder, 
                            self.target_transform, data_loader, config)

        res               = {}
        res["PQ"]         = 100 * pq_res["All"]["pq"]
        res["SQ"]         = 100 * pq_res["All"]["sq"]
        res["RQ"]         = 100 * pq_res["All"]["rq"]
        res["PQ_known"]   = 100 * pq_res["Known Classes"]["pq"]
        res["SQ_known"]   = 100 * pq_res["Known Classes"]["sq"]
        res["RQ_known"]   = 100 * pq_res["Known Classes"]["rq"]
        res["PQ_unknown"] = 100 * pq_res["UnKnown Class"]["pq"]
        res["SQ_unknown"] = 100 * pq_res["UnKnown Class"]["sq"]
        res["RQ_unknown"] = 100 * pq_res["UnKnown Class"]["rq"]
        res["WQ"]         = 0.5*res["PQ_known"] + 0.5*res["PQ_unknown"]

        results = OrderedDict({"wholistic_seg": res})
        self._logger.info(results)
        _print_wholistic_results(pq_res, res["WQ"])

        return results


def _print_wholistic_results(pq_res, wq):
    headers = ["", "PQ", "#categories"]
    data = []

    for name in ["All", "Known Classes", "UnKnown Class"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)

    table = tabulate(data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
    print("WQ:", wq)
    logger.info("wholistic Evaluation Results:\n" + table)
