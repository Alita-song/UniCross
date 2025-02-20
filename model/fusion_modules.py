import torch
from torch import nn
import torch.nn.functional as F


class SelfAttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=8)

    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(0)


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_size):
        super(CrossAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=8)

    def forward(self, x1, x2):
        # x1 Q
        # x2 K V
        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)

        attn_output, _ = self.attention(query=x1, key=x2, value=x2)

        return attn_output.squeeze(0)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # Hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits


class SumFusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=768, dim=768, output_dim=2, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=768, dim=768, output_dim=2, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class CrossAttention(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(CrossAttention, self).__init__()
        self.cross_attention = CrossAttentionModule(input_dim)
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y):
        mri_pet = self.cross_attention(x, y)
        pet_mri = self.cross_attention(y, x)
        output = torch.cat((mri_pet, pet_mri), dim=1)
        output = self.fc_out(output)
        return x, y, output

class ClinicalGuideCrossAttention(nn.Module):
    def __init__(self, input_dim=768, output_dim=2):
        super(ClinicalGuideCrossAttention, self).__init__()
        self.cross_attention1 = CrossAttentionModule(input_dim)
        self.cross_attention2 = CrossAttentionModule(input_dim)
        self.clinical_encoder = nn.Linear(in_features=4, out_features=768)
        self.fc_out = nn.Linear(input_dim * 2, output_dim)

    def forward(self, x, y, z):
        z = self.clinical_encoder(z)
        mri = self.cross_attention1(z, x)
        pet = self.cross_attention2(z, y)
        output = torch.cat((mri, pet), dim=1)
        output = self.fc_out(output)
        return x, y, output


