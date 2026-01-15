"""Define your architecture here."""

import torch
import torch.nn as nn
from torchvision import models


class BonusMobileNetV3Small(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, 2)  # <-- 2 logits (real/fake)

        self.net = backbone

    def forward(self, x):
        return self.net(x)


def my_bonus_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    model = BonusMobileNetV3Small()
    model.load_state_dict(torch.load('checkpoints/bonus_model.pt')['model'])
    return model
