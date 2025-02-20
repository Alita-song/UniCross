"""
Author: xxx
Date: October 22, 2024
"""
from __future__ import print_function

import torch
import torch.nn as nn


class WeightSupConLoss(nn.Module):
    """
    Weight Supervised Contrastive Learning
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(WeightSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, weight=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, dim].
            labels: ground truth of shape [bsz].
            weight: weight of mask
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        # create mask:
        #     mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        #     has the same class as sample i.
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size, batch_size]

        if weight is not None:
            mask = mask * weight

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz * n_views, dim]

        # compute logits (similarity matrix)： [bsz * n_views, bsz * n_views]
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask: [batch_size, batch_size] ---> [batch_size * n_views, batch_size * n_views]
        mask = mask.repeat(contrast_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        loss = loss / 2

        return loss
