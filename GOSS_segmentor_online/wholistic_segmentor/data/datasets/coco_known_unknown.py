# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import json
import os

import numpy as np

from .base_dataset import BaseDataset
from .utils        import DatasetDescriptor
from ..transforms  import build_transforms, Resize, WholisticTargetGenerator
from .build_information import build_data_split_information


class COCOKnown_unknown(BaseDataset):
    def __init__(self,
                 root,
                 split,
                 data_split,
                 data_split_specific,
                 min_resize_value=641,
                 max_resize_value=641,
                 resize_factor=32,
                 is_train=True,
                 crop_size=(641, 641),
                 mirror=True,
                 min_scale=0.5,
                 max_scale=2.,
                 scale_step_size=0.25,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 semantic_only=False,
                 segment_only =False,
                 small_segment_area=0,
                 small_segment_weight=1,
                 **kwargs):

        super(COCOKnown_unknown, self).__init__(root, split, is_train, crop_size, mirror, min_scale, max_scale, scale_step_size, mean, std)

        # Configure '_COCO_KNOWN_UNKNOWN_' based on which dataset split is used
        self._COCO_STUFF_KNOWN_UNKNOWN_INFORMATION, self._COCO_STUFF_UNKNOWN_CLASS_LIST, self._COCO_STUFF_KNOWN_UNKNOWN_TRAIN_ID_TO_EVAL_ID, self._COCO_STUFF_KNOWN_UNKNOWN_CATEGORIES = build_data_split_information(data_split, data_split_specific)

        assert split in self._COCO_STUFF_KNOWN_UNKNOWN_INFORMATION.splits_to_sizes.keys()

        if semantic_only==True and segment_only==False:
           self.has_segment = False
        else: self.has_segment = True

        self.num_classes     = self._COCO_STUFF_KNOWN_UNKNOWN_INFORMATION.num_classes
        self.ignore_label    = self._COCO_STUFF_KNOWN_UNKNOWN_INFORMATION.ignore_label
        self.label_pad_value = (0, 0, 0)

        self.data_split = data_split
        self.data_split_specific = data_split_specific

        self.label_divisor      = 256
        self.label_dtype        = np.float32
        self.unknown_class_list = self._COCO_STUFF_UNKNOWN_CLASS_LIST

        # Get image and annotation list.
        self.img_list                  = []
        self.img_pixelmap_list         = []
        self.img_pixelmap_anomaly_list = []
        self.seg_list                  = []

        if 'val' in self.split:
            json_filename = os.path.join(self.root+'_stuff_'+self.data_split, 'annotations', 'segments_'+self.data_split+self.data_split_specific+'_{}.json'.format(self.split))
        else:
            json_filename = os.path.join(self.root+'_stuff_'+self.data_split, 'annotations', 'segments_'+self.data_split+self.data_split_specific+'_{}.json'.format(self.split))

        dataset = json.load(open(json_filename))
     
        # First sort by image id.
        images          = sorted(dataset['images'],          key=lambda i: i['id'])
        images_pixelmap = sorted(dataset['images_pixelmap'], key=lambda i: i['id'])
        annotations     = sorted(dataset['annotations'],     key=lambda i: i['image_id'])
            
        for img in images:
            img_file_name = img['file_name']
            self.img_list.append(os.path.join(self.root, self.split, img_file_name))
                                         
        for img in images_pixelmap:
            img_file_name = img['file_name']

            if 'val' in self.split: 
                self.img_pixelmap_list.append(os.path.join(self.root+'_stuff_'+self.data_split, 
                                                           self.data_split+self.data_split_specific, 
                                                           self.data_split+self.data_split_specific+'_'+self.split, 
                                                           img_file_name))
                self.img_pixelmap_anomaly_list.append(os.path.join(self.root+'_stuff_'+self.data_split, 
                                                                   self.data_split+self.data_split_specific, 
                                                                   self.data_split+self.data_split_specific+'_anomaly_'+self.split, 
                                                                   img_file_name))
            else:
                self.img_pixelmap_list.append(os.path.join(self.root+'_stuff_'+self.data_split, 
                                                           self.data_split+self.data_split_specific, 
                                                           self.data_split+self.data_split_specific+'_'+self.split, 
                                                           img_file_name))
                self.img_pixelmap_anomaly_list = None
            
        for ann in annotations:
            self.seg_list.append(ann['segments_info'])

        assert len(self.img_list) == self._COCO_STUFF_KNOWN_UNKNOWN_INFORMATION.splits_to_sizes[self.split]

        # Define transform operations.
        self.pre_augmentation_transform = Resize(min_resize_value, max_resize_value, resize_factor)
        self.transform = build_transforms(self, is_train)
        if semantic_only and not segment_only:
            self.target_transform = None

        elif not semantic_only and segment_only:
            self.target_transform = WholisticTargetGenerator(semantic_only=False, segment_only=True,
                                                             unknown_class_list   = self._COCO_STUFF_UNKNOWN_CLASS_LIST,
                                                             small_segment_area   = small_segment_area,
                                                             small_segment_weight = small_segment_weight,
                                                             )
        else:
            self.target_transform = WholisticTargetGenerator(semantic_only=False, segment_only=False,
                                                             unknown_class_list   = self._COCO_STUFF_UNKNOWN_CLASS_LIST,
                                                             small_segment_area   = small_segment_area,
                                                             small_segment_weight = small_segment_weight,
                                                             )
        
        # Generates semantic label for evaluation.
        self.raw_label_transform = WholisticTargetGenerator(semantic_only=True, segment_only=False,
                                                            unknown_class_list   = self._COCO_STUFF_UNKNOWN_CLASS_LIST,
                                                            small_segment_area   = small_segment_area,
                                                            small_segment_weight = small_segment_weight,
                                                            )


    # @staticmethod
    def train_id_to_eval_id(self):
        return self._COCO_STUFF_KNOWN_UNKNOWN_TRAIN_ID_TO_EVAL_ID


    # @staticmethod
    def create_label_colormap(self):
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i, color in enumerate(self._COCO_STUFF_KNOWN_UNKNOWN_CATEGORIES):
            colormap[i] = color['color']
        return colormap
