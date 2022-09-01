# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import sys
sys.dont_write_bytecode = True
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import argparse
import cv2
import pprint
import logging
import time
import math
import scipy.io as sio
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import PIL.Image as img

import _init_paths
from fvcore.common.file_io     import PathManager
from wholistic_segmentor.config       import config, update_config
from wholistic_segmentor.utils.logger import setup_logger
from wholistic_segmentor.model        import build_segmentation_model_from_cfg
from wholistic_segmentor.data         import build_train_loader_from_cfg, build_test_loader_from_cfg
from wholistic_segmentor.utils        import AverageMeter
from wholistic_segmentor.model.post_processing import (get_semantic_segmentation, get_wholistic_segmentation,
                                                       get_semantic_calibration_segmentation, get_semantic_anomaly_segmentation, 
                                                       get_semantic_msp_segmentation, get_semantic_maxlogit_segmentation)
from wholistic_segmentor.evaluation            import (SemanticEvaluator,
                                                       COCOWholisticEvaluator)
from wholistic_segmentor.utils.test_utils      import multi_scale_inference

# import demo
import superBPD_cuda
import generate_center_offset
import anom_utils


def eval_ood_measure(conf, seg_label, out_labels, mask=None):
    if mask is not None:
        seg_label = seg_label[mask]

    out_label = seg_label == out_labels[0]
    for label in out_labels:
        out_label = np.logical_or(out_label, seg_label == label)

    in_scores  = - conf[np.logical_not(out_label)]
    out_scores = - conf[out_label]

    if (len(out_scores) != 0) and (len(in_scores) != 0):
        auroc, aupr, fpr = anom_utils.get_and_print_results(out_scores, in_scores)
        return auroc, aupr, fpr
    else:
        print("This image does not contain any OOD pixels or is only OOD.")
        return 0.0, 0.0, 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network with single process')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    args = parse_args()

    logger = logging.getLogger('segmentation_test')
    if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called
        setup_logger(output=config.OUTPUT_DIR, name='segmentation_test')

    logger.info(pprint.pformat(args))
    logger.info(config)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.TEST.GPUS)
    if len(gpus) > 1:
        raise ValueError('Test only supports single core.')
    device = torch.device('cuda:{}'.format(gpus[0]))

    # build model
    model = build_segmentation_model_from_cfg(config)
    bpd_cuda_python = superBPD_cuda.bpd_cuda_python().cuda()

    # Change ASPP image pooling
    output_stride = 2 ** (5 - sum(config.MODEL.BACKBONE.DILATION))
    train_crop_h, train_crop_w = config.TEST.CROP_SIZE
    scale = 1. / output_stride
    pool_h = int((float(train_crop_h) - 1.0) * scale + 1.0)
    pool_w = int((float(train_crop_w) - 1.0) * scale + 1.0)
    logger.info("Model:\n{}".format(model))
    model = model.to(device)

    # build data_loader
    data_loader = build_test_loader_from_cfg(config)

    # load model
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(config.OUTPUT_DIR, 'final_state.pth')

    if os.path.isfile(model_state_file):
        model_weights = torch.load(model_state_file)
        if 'state_dict' in model_weights.keys():
            model_weights = model_weights['state_dict']
            logger.info('Evaluating a intermediate checkpoint.')
        model.load_state_dict(model_weights, strict=True)
        logger.info('Test model loaded from {}'.format(model_state_file))
    else:
        if not config.DEBUG.DEBUG:
            raise ValueError('Cannot find test model.')

    data_time = AverageMeter()
    net_time  = AverageMeter()
    post_time = AverageMeter()
    timing_warmup_iter = 10

    # open_set_semantic_metric
    auroc_sum = 0.0
    aupr_sum  = 0.0
    fpr_sum   = 0.0
    open_num  = 1

    # semantic_metric
    semantic_metric = None
    if config.TEST.EVAL_SEMANTIC:
        semantic_metric = SemanticEvaluator(num_classes=data_loader.dataset.num_classes,
                                            ignore_label=data_loader.dataset.ignore_label,
                                            output_dir=os.path.join(config.OUTPUT_DIR, config.TEST.SEMANTIC_FOLDER),
                                            train_id_to_eval_id=data_loader.dataset.train_id_to_eval_id(),
                                            colormap = data_loader.dataset.create_label_colormap())

    # wholistic_metric
    wholistic_metric = None
    if config.TEST.EVAL_WHOLISTIC:
        wholistic_metric =COCOWholisticEvaluator(output_dir          =os.path.join(config.OUTPUT_DIR, config.TEST.WHOLISTIC_FOLDER),
                                                 train_id_to_eval_id =data_loader.dataset.train_id_to_eval_id(),
                                                 label_divisor       =data_loader.dataset.label_divisor,
                                                 void_label          =data_loader.dataset.label_divisor * data_loader.dataset.ignore_label,
                                                 num_classes         =config.DATASET.NUM_CLASSES,
                                                 gt_dir              =config.DATASET.ROOT,
                                                 split               =config.DATASET.TEST_SPLIT,
                                                 data_split          =config.DATASET.DATASET_SPLIT,
                                                 data_split_specific =config.DATASET.DATASET_SPLIT_SPECIFIC,
                                                 small_segment_area  =config.DATASET.SMALL_SEGMENT_AREA,
                                                 small_segment_weight=config.DATASET.SMALL_SEGMENT_WEIGHT,
                                                 colormap            =data_loader.dataset.create_label_colormap())


    # image_filename_list = [os.path.splitext(os.path.basename(ann))[0] for ann in data_loader.dataset.ann_list]
    image_filename_list = [os.path.splitext(os.path.basename(ann))[0] for ann in data_loader.dataset.img_list]

    # Train loop.
    try:
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                 
                if i == timing_warmup_iter:
                    data_time.reset()
                    net_time.reset()
                    post_time.reset()

                # data
                start_time = time.time()
                for key in data.keys():
                    try:
                        data[key] = data[key].to(device)
                    except:
                        pass

                image = data.pop('image')
                torch.cuda.synchronize(device)
                data_time.update(time.time() - start_time)

                start_time = time.time()
                if config.TEST.TEST_TIME_AUGMENTATION:
                    raw_image = data['raw_image'][0].cpu().numpy()
                    out_dict = multi_scale_inference(config, model, raw_image, device)
                else:
                    out_dict = model(image)

                torch.cuda.synchronize(device)
                net_time.update(time.time() - start_time)
                start_time = time.time()

                # choose which method in getting predicted semantic map
                semantic_conf = out_dict['semantic']
                if config.TEST.EVAL_SEMANTIC_CONFIDENCE_ADJUSTMENT:
                    semantic_pred, semantic_conf_max = get_semantic_calibration_segmentation(out_dict['semantic'], config.TEST.CONFIDENCE_ADJUSTMENT_SCALE)
                elif config.TEST.EVAL_SEGMENT:
                    semantic_pred, _ = get_semantic_segmentation(out_dict['semantic'])
                    semantic_pred = torch.ones_like(semantic_pred)*data_loader.dataset.unknown_class_list[0] 
                elif config.TEST.EVAL_SEMANTIC_MSP:   
                    semantic_pred, semantic_conf_max = get_semantic_msp_segmentation(out_dict['semantic'], data_loader.dataset.unknown_class_list[0], config.TEST.MSP_PROBILITY_THRESHOLD)
                elif config.TEST.EVAL_SEMANTIC_MAXLOGIT:   
                    semantic_pred, semantic_conf_max = get_semantic_maxlogit_segmentation(out_dict['semantic'], data_loader.dataset.unknown_class_list[0], config.TEST.MAXLOGIT_PROBILITY_THRESHOLD)
                else:
                    semantic_pred, semantic_conf_max = get_semantic_segmentation(out_dict['semantic'])            

                if 'foreground' in out_dict:
                    foreground_pred, _ = get_semantic_segmentation(out_dict['foreground'])
                else:
                    foreground_pred = None

                # code added for super bpd here
                flux = out_dict["flux"]
                if config.TEST.EVAL_DFC:
                    flux = flux[0]
                    angles = torch.atan2(flux[0,...], flux[1,...])
                    angles[angles < 0] += 2*math.pi
                    height, width = angles.shape

                    height_resize = 64
                    width_resize  = 128
                    angles_resize = F.interpolate(angles.unsqueeze(0).unsqueeze(0), size=[height_resize, width_resize], mode='bilinear', align_corners=False)
                    angles_resize = angles_resize.squeeze(0).squeeze(0).cuda()

                    root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = bpd_cuda_python.forward(angles_resize, height_resize, width_resize, 45, 116, 68, 5)
                    super_BPDs = F.interpolate(super_BPDs.float().unsqueeze(0).unsqueeze(0), size=[height, width], mode='nearest')
                    super_BPDs = super_BPDs.int().squeeze(0).squeeze(0)

                else: 
                    flux = flux[0]
                    angles = torch.atan2(flux[1,...], flux[0,...])
                    angles[angles < 0] += 2*math.pi
                    height, width = angles.shape

                    height_resize = 64
                    width_resize  = 128
                    angles_resize = F.interpolate(angles.unsqueeze(0).unsqueeze(0), size=[height_resize, width_resize], mode='bilinear', align_corners=False)
                    angles_resize = angles_resize.squeeze(0).squeeze(0).cuda()

                    root_points, super_BPDs_before_dilation, super_BPDs_after_dilation, super_BPDs = bpd_cuda_python.forward(angles_resize, height_resize, width_resize, 45, 116, 68, 5)
                    super_BPDs = F.interpolate(super_BPDs.float().unsqueeze(0).unsqueeze(0), size=[height, width], mode='nearest')
                    super_BPDs = super_BPDs.int().squeeze(0).squeeze(0)

                out_dict['center'], out_dict['offset'] = generate_center_offset.generate_center_offset_from_super_bpd(super_BPDs.cpu().numpy())              
                out_dict['center'] = out_dict['center'].unsqueeze(0).cuda()
                out_dict['offset'] = out_dict['offset'].unsqueeze(0).cuda()


                if config.TEST.EVAL_WHOLISTIC:
                    wholistic_pred, _ = get_wholistic_segmentation(semantic_pred,
                                                                   out_dict['center'],
                                                                   out_dict['offset'],
                                                                   unknown_class_list=data_loader.dataset.unknown_class_list,
                                                                   # unknown_class_list=[255],
                                                                   label_divisor     =data_loader.dataset.label_divisor,
                                                                   stuff_area        =config.POST_PROCESSING.STUFF_AREA,
                                                                   void_label        =(data_loader.dataset.label_divisor *
                                                                                       data_loader.dataset.ignore_label),
                                                                   threshold         =config.POST_PROCESSING.CENTER_THRESHOLD,
                                                                   nms_kernel        =config.POST_PROCESSING.NMS_KERNEL,
                                                                   top_k             =config.POST_PROCESSING.TOP_K_INSTANCE,
                                                                   foreground_mask   =foreground_pred)
                else:
                    wholistic_pred = None 
                  

                torch.cuda.synchronize(device)
                post_time.update(time.time() - start_time)
                logger.info('[{}/{}]\t'
                            'Data Time: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t'
                            'Network Time: {net_time.val:.3f}s ({net_time.avg:.3f}s)\t'
                            'Post-processing Time: {post_time.val:.3f}s ({post_time.avg:.3f}s)\t'.format(
                             i, len(data_loader), data_time=data_time, net_time=net_time, post_time=post_time))

                semantic_pred = semantic_pred.squeeze(0).cpu().numpy()
 
                if wholistic_pred is not None:
                    wholistic_pred = wholistic_pred.squeeze(0).cpu().numpy()

                # Crop padded regions.
                image_size = data['size'].squeeze(0).cpu().numpy()
                semantic_pred = semantic_pred[:image_size[0], :image_size[1]]
                if wholistic_pred is not None:
                    wholistic_pred = wholistic_pred[:image_size[0], :image_size[1]]

                # Resize back to the raw image size.
                raw_image_size = data['raw_size'].squeeze(0).cpu().numpy()
                if raw_image_size[0] != image_size[0] or raw_image_size[1] != image_size[1]:
                    semantic_pred = cv2.resize(semantic_pred.astype(np.float64), (raw_image_size[1], raw_image_size[0]),
                                               interpolation=cv2.INTER_NEAREST).astype(np.int32)
                    if wholistic_pred is not None:
                        wholistic_pred = cv2.resize(wholistic_pred.astype(np.float64),
                                                    (raw_image_size[1], raw_image_size[0]),
                                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
                
                # Evaluate open-set semantic segmentation
                if  config.TEST.EVAL_SEMANTIC_MSP or config.TEST.EVAL_SEMANTIC_MAXLOGIT or config.TEST.EVAL_SEMANTIC_N_PLUS_ONE or config.TEST.EVAL_SEMANTIC_CONFIDENCE_ADJUSTMENT:

                    if not config.TEST.EVAL_SEGMENT:
                        semantic_conf_max = semantic_conf_max.squeeze(0).cpu().numpy()
                        label = data['semantic'].squeeze(0).cpu().numpy()
                        label = semantic_metric._convert_eval_id_to_train_id(label, semantic_metric._train_id_to_eval_id)
                        label[label==255] = data_loader.dataset.unknown_class_list[0]
                        out_labels        = data_loader.dataset.unknown_class_list
                        auroc, aupr, fpr = eval_ood_measure(semantic_conf_max, label, out_labels, mask=None)        

                        if auroc != 0.0:
                            auroc_sum = auroc_sum + auroc
                            aupr_sum  = aupr_sum  + aupr
                            fpr_sum   = fpr_sum   + fpr
                            open_num  = open_num  + 1

                # Evaluates semantic segmentation.
                if semantic_metric is not None:
                    semantic_metric.update(semantic_pred,
                                           semantic_conf,
                                           data['semantic'].squeeze(0).cpu().numpy(),
                                           data['raw_label'].squeeze(0).cpu().numpy(),
                                           image_filename_list[i])

                # Optional: evaluates wholistic segmentation.
                if wholistic_metric is not None:
                    image_id = '_'.join(image_filename_list[i].split('_')[:3])
                    wholistic_metric.update(wholistic_pred,
                                            semantic_pred,
                                            image_filename=image_filename_list[i],
                                            image_id=image_id,
                                            image_itself=data['raw_image'][0].cpu().numpy())


    except Exception:
        logger.exception("Exception during testing:")
        raise

    finally:
        logger.info("Inference finished.")
        logger.info("Mean AUROC, AUPR and FPR are: {:.3f}/{:.3f}/{:.3f}".format(auroc_sum/open_num, aupr_sum/open_num, fpr_sum/open_num))
        if semantic_metric is not None:
            semantic_results = semantic_metric.evaluate()
            logger.info(semantic_results)
        if wholistic_metric is not None:
            wholistic_results = wholistic_metric.evaluate(data_loader, config)
            logger.info(wholistic_metric)


if __name__ == '__main__':
    main()
