import torch
import torch.nn.functional as F


def calculate_entropy(output):
    probabilities = F.softmax(output, dim=0)
    # probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities)
    return entropy


def calculate_gating_weights2(encoder_output_1, encoder_output_2):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)

    max_entropy = max(entropy_1, entropy_2,)

    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)

    sum_weights = gating_weight_1 + gating_weight_2

    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights

    return gating_weight_1, gating_weight_2


def calculate_gating_weights3(encoder_output_1, encoder_output_2, encoder_output_3):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    entropy_3 = calculate_entropy(encoder_output_3)

    max_entropy = max(entropy_1, entropy_2, entropy_3)

    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    gating_weight_3 = torch.exp(max_entropy - entropy_3)

    sum_weights = gating_weight_1 + gating_weight_2 + gating_weight_3

    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    gating_weight_3 /= sum_weights

    return gating_weight_1, gating_weight_2, gating_weight_3
