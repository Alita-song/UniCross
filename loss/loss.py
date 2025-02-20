# -*-coding:utf-8-*-
import torch
import torch.nn.functional as F


def calculate_my_loss(mri_feature, pet_feature, labels, weight_supcon_criterion, weight):
    """
    2 view

    Args:
        mri_feature:  (batch_size, feature_dim)
        pet_feature:  (batch_size, feature_dim)
        labels:  (batch_size)
        weight_supcon_criterion: weight_supcon_criterion
        weight: weight matrix
    """
    mri_feature_norm = F.normalize(mri_feature, p=2, dim=1)
    pet_feature_norm = F.normalize(pet_feature, p=2, dim=1)

    # (batch_size, 2, feature_dim)
    features = torch.cat([mri_feature_norm.unsqueeze(1), pet_feature_norm.unsqueeze(1)], dim=1)

    loss_contrastive = weight_supcon_criterion(features, labels, weight)

    return loss_contrastive


if __name__ == '__main__':
    batch_size = 8
    feature_dim = 768
    feature1 = torch.randn(batch_size, feature_dim)
    feature2 = torch.randn(batch_size, feature_dim)
    label = torch.randint(0, 2, (batch_size,))
    # criterion = DualModalitySupConLoss(temperature=0.07)
    # loss = calculate_supconloss(feature1, feature2, labels, supcon_criterion)
    # loss = calculate_my_supconloss(feature1, feature2, label, temperature = 0.07)
    # loss_c = calculate_dual_modality_supconloss(feature1, feature2, label, criterion)
    # print(f"Supervised Contrastive Loss: {loss_c.item()}")
