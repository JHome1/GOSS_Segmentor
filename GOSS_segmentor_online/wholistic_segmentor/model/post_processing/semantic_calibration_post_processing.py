# -----------------------------------------
# Project: 'GOSS Segmentor' 
# Written by Jie Hong (jie.hong@anu.edu.au)
# -----------------------------------------
import torch
import torch.nn as nn

__all__ = ['get_semantic_calibration_segmentation', 'get_semantic_msp_segmentation', 'get_semantic_maxlogit_segmentation']

softmax = nn.Softmax(dim=0)


class MinMaxScaler(nn.Module):
    def __init__(self):
        super(MinMaxScaler, self).__init__()
  
    def forward(self, x):
        x_max, _ = torch.max(x, 1)
        x_min, _ = torch.min(x, 1)

        x_max = torch.unsqueeze(x_max, 1)
        x_min = torch.unsqueeze(x_min, 1)

        x = (x-x_min)/(x_max-x_min)
        return x


def get_semantic_calibration_segmentation(sem, scale=1.0):
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)

    num_classes = sem.shape[0]
    sem         = softmax(sem)
    sem_known   = torch.split(sem, num_classes-1)[0]
    sem_unknown = scale*torch.split(sem, num_classes-1)[1]
    sem         = torch.cat((sem_known, sem_unknown), 0) 

    sem_max, sem_index  = torch.max(sem, dim=0, keepdims=True)
    # sem_index = torch.argmax(sem, dim=0, keepdim=True)

    return sem_index, sem_max


def get_semantic_msp_segmentation(sem, num_classes, prob_threshold=0.1):
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)

    sem = softmax(sem)
    sem_max, sem_index = torch.max(sem, dim=0, keepdims=True)
    sem_index[sem_max<prob_threshold] = num_classes

    return sem_index, sem_max

 
def get_semantic_maxlogit_segmentation(sem, num_classes, prob_threshold=0.1):
    normal  = MinMaxScaler()

    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    sem = sem.squeeze(0)

    sem_max, sem_index = torch.max(sem, dim=0, keepdims=True)
    sem_normal         = normal(sem_max)
    sem_index[sem_normal<prob_threshold] = num_classes 

    return sem_index, sem_normal
