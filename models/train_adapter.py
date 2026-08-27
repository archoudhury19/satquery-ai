from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class BigEarthNetAdapterDataset(Dataset):
    """
    Dataset for fine-tuning the RS-VLM adapter using BigEarthNet / RSVQA pairs.
    Each sample contains an image embedding (512-D), text embedding (512-D), and target class ID.
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


def build_training_samples(vlm: Any, demo_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Collect comprehensive QA pairs from external open-source benchmarks (BigEarthNet.txt, VRSBench, RSVQA).
    Returns list of (image_path, question, answer_string).
    """
    samples: List[Tuple[str, str, str]] = []

    # 1. External dataset manifest (BigEarthNet.txt arXiv:2603.29630 + VRSBench schema)
    from models.dataset_fetcher import build_augmented_external_dataset, EXTERNAL_DATA_DIR
    manifest_file = EXTERNAL_DATA_DIR / "external_dataset_manifest.json"
    if not manifest_file.exists():
        build_augmented_external_dataset(EXTERNAL_DATA_DIR)

    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                ext_data = json.load(f)
                for item in ext_data.get("samples", []):
                    img_path = item.get("image_path")
                    for qa in item.get("qa_pairs", []):
                        samples.append((img_path, qa["question"], qa["answer"].strip().lower()))
            print(f"[Train] Ingested {len(samples)} QA pairs from external dataset manifest.")
        except Exception as exc:
            print(f"[Train] Note: Error reading external manifest: {exc}")

    # 2. VRSBench annotations
    vrsbench_json = demo_dir / "vrsbench" / "annotations.json"
    if vrsbench_json.exists():
        with open(vrsbench_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("samples", []):
                img_name = item.get("image")
                img_path = str(demo_dir / "vrsbench" / img_name)
                for qa in item.get("vqa", []):
                    samples.append((img_path, qa["question"], qa["answer"].strip().lower()))

    # 3. CDVQA change questions
    cdvqa_json = demo_dir / "cdvqa" / "annotations.json"
    if cdvqa_json.exists():
        with open(cdvqa_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("samples", []):
                img_t1 = str(demo_dir / "cdvqa" / item.get("image_t1"))
                img_t2 = str(demo_dir / "cdvqa" / item.get("image_t2"))
                for qa in item.get("vqa", []):
                    samples.append((img_t2, qa["question"], qa["answer"].strip().lower()))

    return samples


def train_adapter(
    checkpoint_dir: Path,
    demo_dir: Path = Path("demo_data"),
    epochs: int = 35,
    lr: float = 3e-4,
    batch_size: int = 16,
    device: str = "cpu",
) -> None:
    """
    Train and fit the RSVQA adapter on remote-sensing multimodal representations.
    """
    from models.rs_vlm import RSVQAAdapter, RemoteSensingVLM

    vocab_file = checkpoint_dir / "answer_vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    ans_to_id = vocab.get("answer_to_id", {})
    num_classes = len(ans_to_id)
    print(f"[Train] Loaded vocabulary with {num_classes} classes.")

    print("[Train] Initializing GeoRSCLIP feature extractor...")
    vlm = RemoteSensingVLM(model_dir=str(checkpoint_dir))
    if not vlm.available:
        raise RuntimeError("Failed to load base GeoRSCLIP model.")

    raw_samples = build_training_samples(vlm, demo_dir)
    print(f"[Train] Collected {len(raw_samples)} QA training samples.")

    # Encode all samples into feature tensors
    img_embs: List[torch.Tensor] = []
    txt_embs: List[torch.Tensor] = []
    label_ids: List[int] = []

    for img_path, question, answer in raw_samples:
        p = Path(img_path)
        if not p.exists():
            continue

        label_id = ans_to_id.get(answer, ans_to_id.get("<other>", 49))

        try:
            pil_img = vlm._load_image(p)
            img_tensor = vlm.preprocess(pil_img).unsqueeze(0).to(device)
            txt_tensor = vlm.tokenizer([question]).to(device)

            with torch.no_grad():
                img_feat = vlm.model.encode_image(img_tensor).float()
                txt_feat = vlm.model.encode_text(txt_tensor).float()

            img_embs.append(img_feat.squeeze(0).cpu())
            txt_embs.append(txt_feat.squeeze(0).cpu())
            label_ids.append(label_id)
        except Exception as e:
            print(f"[Train] Warning: skipping sample {img_path}: {e}")

    if not img_embs:
        raise ValueError("No valid training samples could be extracted.")

    dataset = BigEarthNetAdapterDataset(
        image_embeddings=torch.stack(img_embs),
        text_embeddings=torch.stack(txt_embs),
        labels=torch.tensor(label_ids, dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    adapter = RSVQAAdapter(
        input_dim=1024,
        hidden_dim=512,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[Train] Starting Adapter Training ({epochs} epochs, {len(dataset)} samples)...")
    adapter.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_img, batch_txt, batch_labels in dataloader:
            batch_img = batch_img.to(device)
            batch_txt = batch_txt.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = adapter(batch_img, batch_txt)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

        avg_loss = total_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}]  Loss: {avg_loss:.4f}  Accuracy: {acc:.1f}%")

    adapter_save_path = checkpoint_dir / "adapter.pt"
    torch.save(adapter.state_dict(), adapter_save_path)
    print(f"\n[Train] Fine-tuned adapter weights successfully saved to {adapter_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SatQuery RS-VLM Adapter on BigEarthNet / RSVQA")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints/satquery_rs_model")
    parser.add_argument("--demo-dir", type=str, default="demo_data")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    train_adapter(Path(args.checkpoint_dir), demo_dir=Path(args.demo_dir), epochs=args.epochs, lr=args.lr)
