"""Define your architecture here."""

import torch
import torch.nn as nn
from torchvision import models


class BonusMobileNetV3Small(nn.Module):
    """
    Small, efficient binary classifier for Deepfakes detection.
    Outputs a single logit (use BCEWithLogitsLoss).
    """
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Replace classifier head:
        # mobilenet_v3_small.classifier is [Linear, Hardswish, Dropout, Linear]
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 1)  # 1 logit

        # Optionally adjust dropout (keep it small)
        for m in backbone.classifier:
            if isinstance(m, nn.Dropout):
                m.p = dropout

        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (B, 1) logits
        return self.net(x)

def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = BonusMobileNetV3Small()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model


