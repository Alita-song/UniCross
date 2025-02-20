import torch
import torch.nn.functional as F

def certainty_aware_fusion(out1, out2, out3, temperature=1):

    prediction1 = F.softmax(out1, dim=1)  # (B, n_classes)
    prediction2 = F.softmax(out2, dim=1)  # (B, n_classes)
    prediction3 = F.softmax(out3, dim=1)  # (B, n_classes)

    max_prob1 = prediction1.max(1)[0]  # (B,)
    max_prob2 = prediction2.max(1)[0]  # (B,)
    max_prob3 = prediction3.max(1)[0]  # (B,)

    w = torch.stack([max_prob1, max_prob2, max_prob3], dim=0)  # (3, B)
    w = F.softmax(w / temperature, dim=0)  # (3, B)

    out = (
        out1 * w[0].unsqueeze(1) +
        out2 * w[1].unsqueeze(1) +
        out3 * w[2].unsqueeze(1)
    )  # (B, n_classes)

    return out

