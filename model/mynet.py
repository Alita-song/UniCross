# -*-coding:utf-8-*-
import torch
from torch import nn
from torchinfo import summary

from model.vit3d import vit_b16_backbone
from model.fusion_modules import SumFusion, ConcatFusion, GatedFusion, FiLM, CrossAttention


def create_vit_backbone(pretrained=True):
    """
    vit backbone. output feature dim  is 768
    """
    model = vit_b16_backbone()
    if pretrained:
        pre_trained_model_path = 'pre_train_model/ViT_B_pretrained_noaug_mae75_BRATS2023_IXI_OASIS3.pth.tar'
        checkpoint = torch.load(pre_trained_model_path, map_location='cpu', weights_only=True)
        print("Loaded pre-trained checkpoint from: %s" % pre_trained_model_path)
        checkpoint_model = checkpoint['net']
        model.load_state_dict(checkpoint_model, strict=False)

    return model


class MyNet(nn.Module):
    """
    traditional joint multimodal learning
    """

    def __init__(self, mri_model, pet_model, fusion_method, out_feature_dim, class_num):
        super(MyNet, self).__init__()
        self.mri_model = mri_model
        self.pet_model = pet_model

        if fusion_method == 'sum':
            self.fusion_module = SumFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif fusion_method == 'concat':
            self.fusion_module = ConcatFusion(input_dim=out_feature_dim, output_dim=class_num)
        elif fusion_method == 'film':
            self.fusion_module = FiLM(input_dim=out_feature_dim, output_dim=class_num, x_film=True)
        elif fusion_method == 'gated':
            self.fusion_module = GatedFusion(input_dim=out_feature_dim, output_dim=class_num, x_gate=True)
        elif fusion_method == 'CrossAttention':
            self.fusion_module = CrossAttention(input_dim=out_feature_dim, output_dim=class_num)
        else:
            raise NotImplementedError('Incorrect fusion method: {}!'.format(fusion_method))

    def forward(self, mri, pet):
        mri_feature = self.mri_model(mri)
        pet_feature = self.pet_model(pet)

        m, p, output = self.fusion_module(mri_feature, pet_feature)

        return m, p, output



class MriClassifier(nn.Module):
    """
    Mri modality classifier
    """

    def __init__(self, mri_model, out_feature_dim, class_num):
        super(MriClassifier, self).__init__()
        self.net = mri_model
        self.classifier = nn.Linear(in_features=out_feature_dim, out_features=class_num)

    def forward(self, x):
        m = self.net(x)
        output = self.classifier(m)
        return m, output


class PetClassifier(nn.Module):
    """
    Pet modality classifier
    """

    def __init__(self, pet_model, out_feature_dim, class_num):
        super(PetClassifier, self).__init__()
        self.net = pet_model
        self.classifier = nn.Linear(in_features=out_feature_dim, out_features=class_num)

    def forward(self, x):
        p = self.net(x)
        output = self.classifier(p)
        return p, output


if __name__ == '__main__':
    # test_model = MyNet()
    # summary(test_model, [(1,1,128,128,128),(1,1,128,128,128)])
    create_vit_backbone(True)
    print("finish")
