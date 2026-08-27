from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "checkpoints"
    / "satquery_rs_model"
)

GENERATED_DIR = BASE_DIR / "generated"

GENERATED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR = Path(
    os.getenv(
        "SATQUERY_RS_MODEL_DIR",
        str(DEFAULT_MODEL_DIR),
    )
)


# ============================================================
# TIFF NORMALIZATION
# ============================================================

def normalize_band(
    band: np.ndarray,
) -> np.ndarray:
    """
    Robust percentile normalization for raster display.
    """

    band = band.astype(
        np.float32
    )

    finite = np.isfinite(
        band
    )

    if not finite.any():
        return np.zeros_like(
            band,
            dtype=np.uint8,
        )

    lo, hi = np.percentile(
        band[finite],
        [2, 98],
    )

    if hi <= lo:
        lo = float(
            np.min(
                band[finite]
            )
        )

        hi = float(
            np.max(
                band[finite]
            )
        )

    if hi <= lo:
        return np.zeros_like(
            band,
            dtype=np.uint8,
        )

    band = np.clip(
        (band - lo)
        / (hi - lo),
        0.0,
        1.0,
    )

    return (
        band * 255
    ).astype(
        np.uint8
    )


# ============================================================
# RSVQA ADAPTER
# ============================================================

class RSVQAAdapter(
    torch.nn.Module
):

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 50,
    ) -> None:

        super().__init__()

        self.classifier = (
            torch.nn.Sequential(
                torch.nn.LayerNorm(
                    input_dim
                ),
                torch.nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                torch.nn.GELU(),
                torch.nn.Dropout(
                    0.1
                ),
                torch.nn.Linear(
                    hidden_dim,
                    num_classes,
                ),
            )
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:

        image_features = (
            image_features
            / (
                image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        text_features = (
            text_features
            / (
                text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-8
            )
        )

        combined = torch.cat(
            [
                image_features,
                text_features,
            ],
            dim=-1,
        )

        return self.classifier(
            combined
        )


# ============================================================
# REMOTE SENSING VLM
# ============================================================

class RemoteSensingVLM:
    """
    Local GeoRSCLIP + trained RSVQA adapter.

    Base model:
        GeoRSCLIP ViT-B/32

    Adaptation:
        RSVQAAdapter trained on RSVQA-LR-2k

    Local files:

        models/checkpoints/satquery_rs_model/
            adapter.pt
            answer_vocab.json
            config.json
    """

    def __init__(
        self,
        model_dir: Optional[
            str | Path
        ] = None,
    ) -> None:

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_dir = Path(
            model_dir
            or MODEL_DIR
        )

        self.adapter_path = (
            self.model_dir
            / "adapter.pt"
        )

        self.vocab_path = (
            self.model_dir
            / "answer_vocab.json"
        )

        self.config_path = (
            self.model_dir
            / "config.json"
        )

        self.base_model_id = (
            "BiliSakura/GeoRSCLIP-ViT-B-32"
        )

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.adapter = None

        self.id_to_answer: Dict[
            int,
            str,
        ] = {}

        self.config: Dict[
            str,
            Any,
        ] = {}

        self._available = False

        try:

            self._load()

            self._available = True

        except Exception:

            print(
                "\n[RS-VLM] FAILED TO LOAD MODEL"
            )

            print(
                "[RS-VLM] Model directory:",
                self.model_dir,
            )

            traceback.print_exc()

    # ========================================================
    # STATUS
    # ========================================================

    @property
    def available(
        self,
    ) -> bool:

        return self._available

    # ========================================================
    # LOADING
    # ========================================================

    def _load(
        self,
    ) -> None:

        required = [
            self.adapter_path,
            self.vocab_path,
            self.config_path,
        ]

        missing = [
            str(path)
            for path in required
            if not path.exists()
        ]

        if missing:

            raise FileNotFoundError(
                "Missing RS model files:\n"
                + "\n".join(
                    missing
                )
            )

        print(
            "[RS-VLM] Model directory:",
            self.model_dir,
        )

        print(
            "[RS-VLM] Device:",
            self.device,
        )

        # ----------------------------------------------------
        # Config
        # ----------------------------------------------------

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.config = json.load(
                f
            )

        num_classes = int(
            self.config.get(
                "num_classes",
                50,
            )
        )

        image_dim = int(
            self.config.get(
                "image_embedding_dim",
                512,
            )
        )

        text_dim = int(
            self.config.get(
                "text_embedding_dim",
                512,
            )
        )

        hidden_dim = int(
            self.config.get(
                "hidden_dim",
                512,
            )
        )

        input_dim = int(
            self.config.get(
                "input_dim",
                image_dim + text_dim,
            )
        )

        # ----------------------------------------------------
        # Download/load GeoRSCLIP base checkpoint
        # ----------------------------------------------------

        print(
            "[RS-VLM] Downloading/loading GeoRSCLIP..."
        )

        geo_checkpoint = hf_hub_download(
            repo_id=self.base_model_id,
            filename="GeoRSCLIP-ViT-B-32.pt",
        )

        # Important:
        # create architecture only; load GeoRSCLIP
        # checkpoint ourselves.
        (
            self.model,
            _,
            self.preprocess,
        ) = open_clip.create_model_and_transforms(
            "ViT-B/32",
            pretrained=None,
            quick_gelu=True,
        )

        checkpoint = torch.load(
            geo_checkpoint,
            map_location="cpu",
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            if "state_dict" in checkpoint:

                checkpoint = checkpoint[
                    "state_dict"
                ]

            elif "model" in checkpoint:

                checkpoint = checkpoint[
                    "model"
                ]

        load_result = (
            self.model.load_state_dict(
                checkpoint,
                strict=True,
            )
        )

        print(
            "[RS-VLM] Base model loaded"
        )

        print(
            "[RS-VLM] Missing keys:",
            len(
                load_result.missing_keys
            ),
        )

        print(
            "[RS-VLM] Unexpected keys:",
            len(
                load_result.unexpected_keys
            ),
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        for param in (
            self.model.parameters()
        ):

            param.requires_grad = False

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = (
            open_clip.get_tokenizer(
                "ViT-B/32"
            )
        )

        # ----------------------------------------------------
        # Answer vocabulary
        # ----------------------------------------------------

        with self.vocab_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            vocab = json.load(
                f
            )

        raw_vocab = vocab.get(
            "id_to_answer",
            {},
        )

        self.id_to_answer = {
            int(key): str(value)
            for key, value
            in raw_vocab.items()
        }

        if not self.id_to_answer:

            raise ValueError(
                "answer_vocab.json contains "
                "no id_to_answer entries."
            )

        actual_classes = len(
            self.id_to_answer
        )

        if (
            actual_classes
            != num_classes
        ):

            raise ValueError(
                "Adapter/config vocabulary "
                "mismatch: "
                f"config says {num_classes}, "
                f"vocab contains "
                f"{actual_classes}."
            )

        # ----------------------------------------------------
        # Adapter
        # ----------------------------------------------------

        self.adapter = RSVQAAdapter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        adapter_state = torch.load(
            self.adapter_path,
            map_location="cpu",
        )

        self.adapter.load_state_dict(
            adapter_state,
            strict=True,
        )

        self.adapter = self.adapter.to(
            self.device
        )

        self.adapter.eval()

        for param in (
            self.adapter.parameters()
        ):

            param.requires_grad = False

        print(
            "[RS-VLM] Adapter loaded"
        )

        print(
            "[RS-VLM] Classes:",
            num_classes,
        )

        print(
            "[RS-VLM] [OK] MODEL READY"
        )

    # ========================================================
    # IMAGE LOADING
    # ========================================================

    def _load_image(
        self,
        path: Path,
    ) -> Image.Image:

        suffix = (
            path.suffix.lower()
        )

        # ----------------------------------------------------
        # PNG/JPEG
        # ----------------------------------------------------

        if suffix in {
            ".png",
            ".jpg",
            ".jpeg",
        }:

            return (
                Image.open(
                    path
                ).convert("RGB")
            )

        # ----------------------------------------------------
        # TIFF / GeoTIFF
        # ----------------------------------------------------

        if suffix in {
            ".tif",
            ".tiff",
        }:

            import rasterio

            with rasterio.open(
                path
            ) as src:

                if src.count >= 3:

                    channels = []

                    for band_index in [
                        1,
                        2,
                        3,
                    ]:

                        band = (
                            src.read(
                                band_index
                            )
                            .astype(
                                np.float32
                            )
                        )

                        channels.append(
                            normalize_band(
                                band
                            )
                        )

                    rgb = np.stack(
                        channels,
                        axis=-1,
                    )

                else:

                    band = normalize_band(
                        src.read(1).astype(
                            np.float32
                        )
                    )

                    rgb = np.stack(
                        [
                            band,
                            band,
                            band,
                        ],
                        axis=-1,
                    )

            return Image.fromarray(
                rgb
            ).convert("RGB")

        raise ValueError(
            f"Unsupported image format: "
            f"{suffix}"
        )

    # ========================================================
    # VQA
    # ========================================================

    def analyze(
        self,
        image_path: str | Path,
        question: str,
    ) -> Dict[str, Any]:

        if not self.available:

            raise RuntimeError(
                "Remote-sensing model is unavailable."
            )

        question = str(
            question
        ).strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        image = self._load_image(
            image_path
        )

        image_input = (
            self.preprocess(
                image
            )
            .unsqueeze(0)
            .to(self.device)
        )

        text_input = (
            self.tokenizer(
                [question]
            ).to(self.device)
        )

        with torch.inference_mode():

            image_features = (
                self.model.encode_image(
                    image_input
                ).float()
            )

            text_features = (
                self.model.encode_text(
                    text_input
                ).float()
            )

            logits = self.adapter(
                image_features,
                text_features,
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

            predicted_id = int(
                probabilities.argmax().item()
            )

            confidence = float(
                probabilities[
                    predicted_id
                ].item()
            )

        answer = (
            self.id_to_answer.get(
                predicted_id,
                "<unknown>",
            )
        )

        # ----------------------------------------------------
        # Top-5 predictions
        # ----------------------------------------------------

        top_k = min(
            5,
            probabilities.numel(),
        )

        top_probs, top_ids = torch.topk(
            probabilities,
            k=top_k,
        )

        top_answers = []

        for probability, index in zip(
            top_probs.tolist(),
            top_ids.tolist(),
        ):

            top_answers.append(
                {
                    "answer": self.id_to_answer.get(
                        int(index),
                        "<unknown>",
                    ),
                    "confidence": float(
                        probability
                    ),
                }
            )

        return {
            "answer": answer,
            "confidence": confidence,
            "model": (
                "GeoRSCLIP + RSVQA Adapter"
            ),
            "top_answers": top_answers,
            "question": question,
        }

    # ========================================================
    # TEXT-GUIDED GROUNDING
    # ========================================================

    def ground(
        self,
        image_path: str | Path,
        text: str,
    ) -> Dict[str, Any]:
        """
        Lightweight text-guided grounding.

        The image is divided into a 4x4 grid. GeoRSCLIP
        image/text similarity is calculated for every tile.
        The highest-scoring tile is returned as a proposed
        bounding box.

        This is an MVP spatial-grounding method, not a
        pixel-accurate segmentation model.
        """

        if not self.available:

            raise RuntimeError(
                "Remote-sensing model is unavailable."
            )

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        text = str(
            text
        ).strip()

        if not text:

            raise ValueError(
                "Grounding text cannot be empty."
            )

        image = self._load_image(
            image_path
        )

        image_width, image_height = (
            image.size
        )

        # ----------------------------------------------------
        # Generate 4x4 image tiles
        # ----------------------------------------------------

        grid = 4

        crops = []
        boxes = []

        for row in range(grid):

            for col in range(grid):

                x1 = int(
                    col
                    * image_width
                    / grid
                )

                y1 = int(
                    row
                    * image_height
                    / grid
                )

                x2 = int(
                    (col + 1)
                    * image_width
                    / grid
                )

                y2 = int(
                    (row + 1)
                    * image_height
                    / grid
                )

                crop = image.crop(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

                crops.append(
                    crop
                )

                boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

        # ----------------------------------------------------
        # Text embedding
        # ----------------------------------------------------

        text_input = self.tokenizer(
            [text]
        ).to(self.device)

        with torch.inference_mode():

            text_features = (
                self.model.encode_text(
                    text_input
                ).float()
            )

            text_features /= (
                text_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-8
            )

            # ------------------------------------------------
            # Image tile embeddings
            # ------------------------------------------------

            image_inputs = torch.stack(
                [
                    self.preprocess(
                        crop
                    )
                    for crop in crops
                ]
            ).to(
                self.device
            )

            image_features = (
                self.model.encode_image(
                    image_inputs
                ).float()
            )

            image_features /= (
                image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                + 1e-8
            )

            scores = (
                image_features
                @ text_features.T
            ).squeeze(
                -1
            )

        # ----------------------------------------------------
        # Tile probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            scores * 10.0,
            dim=0,
        )

        best_index = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[
                best_index
            ].item()
        )

        x1, y1, x2, y2 = boxes[
            best_index
        ]

        # ----------------------------------------------------
        # Location
        # ----------------------------------------------------

        cx = (
            x1 + x2
        ) / 2.0

        cy = (
            y1 + y2
        ) / 2.0

        if cx < image_width / 3:
            horizontal = "west"

        elif cx > (
            2 * image_width / 3
        ):
            horizontal = "east"

        else:
            horizontal = "central"

        if cy < image_height / 3:
            vertical = "north"

        elif cy > (
            2 * image_height / 3
        ):
            vertical = "south"

        else:
            vertical = "central"

        if (
            horizontal == "central"
            and vertical == "central"
        ):
            location = "central"

        elif horizontal == "central":
            location = vertical

        elif vertical == "central":
            location = horizontal

        else:
            location = (
                f"{vertical}-{horizontal}"
            )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        rgb = np.array(
            image
        ).copy()

        overlay = rgb.copy()

        alpha = 0.40

        overlay[
            y1:y2,
            x1:x2,
            0
        ] = 255

        overlay[
            y1:y2,
            x1:x2,
            1
        ] = 0

        overlay[
            y1:y2,
            x1:x2,
            2
        ] = 0

        blended = (
            (
                1 - alpha
            )
            * rgb
            + alpha
            * overlay
        ).astype(
            np.uint8
        )

        cv2.rectangle(
            blended,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            3,
        )

        output_path = (
            GENERATED_DIR
            / (
                "grounding_"
                f"{uuid.uuid4().hex[:8]}"
                ".png"
            )
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                blended,
                cv2.COLOR_RGB2BGR,
            ),
        )

        # ----------------------------------------------------
        # Top grounding regions
        # ----------------------------------------------------

        top_k = min(
            5,
            probabilities.numel(),
        )

        top_probs, top_indices = (
            torch.topk(
                probabilities,
                k=top_k,
            )
        )

        top_regions = []

        for prob, idx in zip(
            top_probs.tolist(),
            top_indices.tolist(),
        ):

            bx1, by1, bx2, by2 = (
                boxes[int(idx)]
            )

            top_regions.append(
                {
                    "confidence": float(
                        prob
                    ),
                    "bounding_box": {
                        "x1": bx1,
                        "y1": by1,
                        "x2": bx2,
                        "y2": by2,
                    },
                }
            )

        return {
            "answer": (
                f"Detected the requested "
                f"region in the {location} "
                "part of the image."
            ),
            "confidence": confidence,
            "model": (
                "GeoRSCLIP "
                "Text-Guided Grounding"
            ),
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "location": location,
            "overlay": output_path.name,
            "text": text,
            "top_regions": top_regions,
        }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Loading SatQuery remote-sensing model..."
    )

    model = RemoteSensingVLM()

    print(
        "MODEL_AVAILABLE =",
        model.available,
    )