"""Training loop for cross-modal retrieval."""

import argparse
import torch

from src.model import DualEncoder
from src.losses import GradientModulatedContrastive


def main(epochs: int):
    model = DualEncoder()
    objective = GradientModulatedContrastive()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for images, input_ids, attention_mask in build_loader():
            image_embed, text_embed = model(images, input_ids, attention_mask)
            loss = objective(image_embed, text_embed)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    main(parser.parse_args().epochs)
