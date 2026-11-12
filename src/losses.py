"""Gradient-modulated contrastive objective."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientModulatedContrastive(nn.Module):
    def __init__(self, temperature: float = 0.07, modulation: float = 0.25):
        super().__init__()
        self.temperature = temperature
        self.modulation = modulation

    def forward(self, image_embed, text_embed):
        logits = image_embed @ text_embed.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        # damp the gradient of hard negatives so early training stays stable
        grad_scale = 1.0 + self.modulation * torch.abs(logits).mean()
        return (loss_i2t + loss_t2i) / 2.0 / grad_scale
