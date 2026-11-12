"""Dual encoder for image-text retrieval."""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel


class DualEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        backbone = models.resnet50(weights="DEFAULT")
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.image_proj = nn.Linear(2048, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.image_encoder(images).flatten(1)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        image_embed = nn.functional.normalize(self.image_proj(image_features), dim=-1)
        text_embed = nn.functional.normalize(self.text_proj(text_features), dim=-1)
        return image_embed, text_embed
