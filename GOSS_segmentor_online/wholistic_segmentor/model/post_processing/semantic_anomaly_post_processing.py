# -----------------------------------------
# Project: 'GOSS Segmentor' 
# Written by Jie Hong (jie.hong@anu.edu.au)
# -----------------------------------------
import torch
import torch.nn as nn
import numpy as np

__all__ = ['get_semantic_anomaly_segmentation']


def get_semantic_anomaly_segmentation(sem, sem_anomaly, num_classes):
    sem = sem + sem_anomaly
    sem = sem.cpu().numpy()
    sem = np.where(sem > (num_classes-1), num_classes-1, sem)
   
    return torch.from_numpy(sem).cuda()
