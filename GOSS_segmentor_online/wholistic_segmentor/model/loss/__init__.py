from torch import nn
from .criterion import RegularCE, OhemCE, DeepLabCE, BPDLoss, PixelContrastLoss, VarLoss

L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
CrossEntropyLoss = nn.CrossEntropyLoss

BPDLoss = BPDLoss
PixelContrastLoss = PixelContrastLoss
VarLoss = VarLoss
