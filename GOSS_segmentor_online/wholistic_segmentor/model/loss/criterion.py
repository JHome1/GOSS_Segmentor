# ---------------------------------------------------------
# Modified from 'panoptic-deeplab' 
# Reference: https://github.com/bowenc0221/panoptic-deeplab
# ---------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class RegularCE(nn.Module):
    """
    Regular cross entropy loss for semantic segmentation, support pixel-wise loss weight.
    Arguments:
        ignore_label: Integer, label to ignore.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, weight=None):
        super(RegularCE, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        labels =labels.long()
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        pixel_losses = pixel_losses[mask]
        return pixel_losses.mean()


class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is widely used in PyTorch semantic segmentation frameworks.
    Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/1b3ae72f6025bde4ea404305d502abea3c2f5266/lib/core/criterion.py#L29
    Arguments:
        ignore_label: Integer, label to ignore.
        threshold: Float, threshold for softmax score (of gt class), only predictions with softmax score
            below this threshold will be kept.
        min_kept: Integer, minimum number of pixels to be kept, it is used to adjust the
            threshold value to avoid number of examples being too small.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, threshold=0.7,
                 min_kept=100000, weight=None):
        super(OhemCE, self).__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        predictions = F.softmax(logits, dim=1)
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        mask = labels.contiguous().view(-1) != self.ignore_label

        tmp_labels = labels.clone()
        tmp_labels[tmp_labels == self.ignore_label] = 0
        # Get the score for gt class at each pixel location.
        predictions = predictions.gather(1, tmp_labels.unsqueeze(1))
        predictions, indices = predictions.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = predictions[min(self.min_kept, predictions.numel() - 1)]
        threshold = max(min_value, self.threshold)

        pixel_losses = pixel_losses[mask][indices]
        pixel_losses = pixel_losses[predictions < threshold]
        return pixel_losses.mean()


class DeepLabCE(nn.Module):
    """
    Hard pixel mining mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its value < 1.0, only compute the loss for
            the top k percent pixels (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """
    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, logits, labels, **kwargs):
        if 'semantic_weights' in kwargs:
            pixel_losses = self.criterion(logits, labels) * kwargs['semantic_weights']
            pixel_losses = pixel_losses.contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class BPDLoss(nn.Module):
    def __init__(self):
        super(BPDLoss, self).__init__()

    def forward(self, pred_flux, gt_flux, weight_matrix):
        device_id = pred_flux.device
        weight_matrix = weight_matrix.cuda(device_id)
        gt_flux = gt_flux.cuda(device_id)

        gt_flux = 0.999999 * gt_flux / ((gt_flux.norm(p=2, dim=1)).unsqueeze(1) + 1e-9)

        # norm loss
        norm_loss = weight_matrix.unsqueeze(1) * (pred_flux - gt_flux) ** 2
        norm_loss = norm_loss.sum()

        # angle loss
        pred_flux  = 0.999999 * pred_flux / ((pred_flux.norm(p=2, dim=1)).unsqueeze(1) + 1e-9)
        angle_loss = weight_matrix.unsqueeze(1) * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        angle_loss = angle_loss.sum()

        total_loss = norm_loss + angle_loss
        return total_loss


class PixelContrastLoss(nn.Module):
    def __init__(self, ignore_label=-1):
        super(PixelContrastLoss, self).__init__()

        self.temperature = 0.07
        self.base_temperature = 0.07

        self.ignore_label = ignore_label

        self.max_samples = 1024
        self.max_views = 10
        self.loss_weight = 1.0

    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0

        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y)

            this_classes = [x for x in this_classes if x != self.ignore_label]

            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]
            if len(this_classes) == 0:
                this_classes = [x for x in this_classes if x != self.ignore_label]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        #### Predict
        predict = predict.unsqueeze(1).float().clone()
        predict = torch.nn.functional.interpolate(predict,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        predict = predict.squeeze(1).long()
        assert predict.shape[-1] == feats.shape[-1], '{} {}'.format(predict.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)

        return loss * self.loss_weight


class VarLoss(nn.Module):              
      def __init__(self):
          super(VarLoss, self).__init__()
          self.mse = nn.MSELoss()
         
      def forward(self, logits, targets, num_classes):
          features = logits
          label    = targets.cpu().data.numpy()
        
          loss = 0.0
          instances, counts = np.unique(label, False, False, True)
          total_size = int(np.sum(counts))

          for instance in instances:
              if instance == num_classes or instance == 255: continue

              locations = torch.LongTensor(np.where(label == instance)[0]).cuda()
              vectors   = torch.index_select(features, dim=0, index=locations)

              # get instance mean and distances to mean of all points in an instance
              vectors_proto = torch.zeros_like(vectors)              
              vectors_proto[:, int(instance)] = vectors[:, int(instance)]     

              loss = self.mse(vectors, vectors_proto)

          return torch.tensor(loss)
