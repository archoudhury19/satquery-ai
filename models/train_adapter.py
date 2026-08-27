from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BigEarthNetAdapterDataset(Dataset):
    """
    Dataset for fine-tuning the RS-VLM adapter using BigEarthNet / RSVQA pairs.
    Each sample contains an image embedding, text embedding, and target class label.
    """

    def __init__(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        self.img_emb = image_embeddings
        self.txt_emb = text_embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.img_emb[idx], self.txt_emb[idx], self.labels[idx]


def train_adapter(
    checkpoint_dir: Path,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
) -> None:
    """
    Train / Fine-tune the RSVQA adapter on remote-sensing multimodal representations.
    """
    from models.rs_vlm import RSVQAAdapter

    vocab_file = checkpoint_dir / "answer_vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    num_classes = len(vocab.get("answer_to_id", {}))
    print(f"[Train] Loaded vocabulary with {num_classes} classes.")

    model = RSVQAAdapter(
        input_dim=1024,
        hidden_dim=512,
        num_classes=num_classes,
    ).to(device)

    # If checkpoint exists, load weights for continued fine-tuning
    adapter_weights = checkpoint_dir / "adapter.pt"
    if adapter_weights.exists():
        state_dict = torch.load(adapter_weights, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Train] Resumed from checkpoint: {adapter_weights}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print(f"[Train] Adapter ready. Training configuration: epochs={epochs}, lr={lr}, device={device}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SatQuery RS-VLM Adapter on BigEarthNet / RSVQA")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints/satquery_rs_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_adapter(Path(args.checkpoint_dir), epochs=args.epochs, lr=args.lr)
