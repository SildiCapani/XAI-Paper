"""Grad-CAM und HiResCAM (pytorch-grad-cam) für die UFLDv2-Zeilenanker.

pytorch-grad-cam erwartet ein Modell mit einem Tensor als Ausgabe; das Modell
des Teams liefert ein Dict. LocRowOnly gibt daher nur loc_row weiter.
"""

import numpy as np
import torch
from pytorch_grad_cam import GradCAM, HiResCAM

METHODS = {"gradcam": GradCAM, "hirescam": HiResCAM}


class LocRowOnly(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)["loc_row"]


class RowBinTarget:
    """Summe der Logits fester Bins: out[bins[i], anchors[i], lane].

    Mit den Gewinner-Bins des unverdeckten Bildes fragt das: „Welche Bildbereiche
    halten die Linie an ihrer ursprünglichen Position?“
    """

    def __init__(self, lane: int, anchors: np.ndarray, bins: np.ndarray):
        self.lane = lane
        self.anchors = torch.as_tensor(anchors)
        self.bins = torch.as_tensor(bins)

    def __call__(self, out: torch.Tensor) -> torch.Tensor:  # out: (grid, num_row, num_lanes)
        return out[self.bins, self.anchors, self.lane].sum()


def compute_cam(net, x: torch.Tensor, target: RowBinTarget, method: str = "hirescam", layer: str = "layer4") -> np.ndarray:
    """CAM in Eingabeauflösung (H, W), auf [0, 1] skaliert."""
    wrapped = LocRowOnly(net).eval()
    target_layer = getattr(net.model, layer)
    with METHODS[method](model=wrapped, target_layers=[target_layer]) as cam:
        return cam(input_tensor=x, targets=[target])[0]


def energy_share(cam: np.ndarray, mask: np.ndarray) -> float:
    """Anteil der CAM-Masse innerhalb der Maske."""
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else float("nan")
