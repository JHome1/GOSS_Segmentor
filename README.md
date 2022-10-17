# GOSS_Segmentor
This repository is for GOSS Segmentor (GST) introduced in the following paper:  
**GOSS: Towards Generalized Open-set Semantic Segmentation**

## Paper and Citation
The paper can be downloaded from [arXiv](https://arxiv.org/abs/2203.12116).  
If you find our paper/code is useful, please cite:

        @article{hong2022goss,
                 title={Goss: Towards generalized open-set semantic segmentation},
                 author={Hong, Jie and Li, Weihao and Han, Junlin and Zheng, Jiyang and Fang, Pengfei and Harandi, Mehrtash and Petersson, Lars},
                 journal={arXiv preprint arXiv:2203.12116},
                 year={2022}
                 }
                 
## Task Definition
<p align="center">
  <img width="900" src="https://github.com/JHome1/GOSS_Segmentor/blob/main/Figure1.png">
</p>

## Environments
* Install packages in ```README_pkgs.md```
* If you had the error of "undefined symbol" as importing ```pycocotools```, use the following steps:
```
pip uninstall pycocotools
git clone https://github.com/pdollar/coco.git
cd ./coco/PythonAPI
CC=gcc python3 setup.py build_ext install
```

## Datasets
We expect the directory structure to be the following:
```
datasets/coco
  train2017/    # train images
  val2017/      # val images
  
datasets/coco_stuff_voc
  annotations/
    segments_voc_20_60_train2017.json
    segments_voc_20_60_val2017.json
  voc_20_60/
    voc_20_60_train2017/
    voc_20_60_val2017/
    
 datasets/coco_stuff_manual
  annotations/
    segments_manual_20_60_train2017.json
    segments_manual_20_60_val2017.json
  manual_20_60/
    manual_20_60_train2017/
    manual_20_60_val2017/
    
 datasets/coco_stuff_random
  annotations/
    segments_random_20_60_train2017.json
    segments_random_20_60_val2017.json
  random_20_60/
    random_20_60_train2017/
    random_20_60_val2017/
```
