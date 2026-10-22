"""Bildvorverarbeitung wie im Inferenz-Code des Teams.

Graustufen-BEV-Bild -> 3 Kanäle -> Resize auf (train_width, train_height)
-> [0, 1] -> ImageNet-Normalisierung -> Tensor (1, 3, H, W).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_gray(path: str | Path) -> np.ndarray:
    """Lädt ein Bild als uint8-Graustufenarray (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_model_rgb(gray: np.ndarray, width: int, height: int) -> np.ndarray:
    """Graustufen -> RGB (uint8) in Modellauflösung. Grundlage für Overlays."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) in Modellauflösung -> normalisierter Tensor (1, 3, H, W)."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def preprocess(gray: np.ndarray, width: int, height: int) -> tuple[torch.Tensor, np.ndarray]:
    """Gibt (Eingabetensor, RGB-Bild in Modellauflösung) zurück."""
    rgb = to_model_rgb(gray, width, height)
    return to_tensor(rgb), rgb
