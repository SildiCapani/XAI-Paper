"""Dekodiert die Zeilenanker-Ausgabe (loc_row/exist_row) in Bildpunkte.

Gleiche Rechnung wie pred2coords im model_wrapper des Teams (Softmax über
argmax ± local_width, +0.5, / (grid - 1)), aber in Modellauflösung
(train_width × train_height) statt in Originalauflösung. Die Spaltenanker
(loc_col) werden hier noch nicht verwendet: im BEV-Bild sind bei den Testbildern
fast immer alle 55 Zeilenanker aktiv, aber nur wenige Spaltenanker.
"""

import numpy as np
import torch


def exist_prob(pred: dict) -> np.ndarray:
    """P(Spur existiert) pro Zeilenanker und Spur, Form (num_row, num_lanes)."""
    return pred["exist_row"][0].detach().softmax(0)[1].numpy()


def decode_row_lanes(pred: dict, cfg, local_width: int = 1) -> tuple[list[np.ndarray], np.ndarray]:
    """Gibt (Punkte je Spur, Gewinner-Bins) zurück.

    Punkte: Liste mit einem Array (num_row, 2) je Spur, Spalten (x, y);
    NaN, wo der Anker laut exist_row keine Spur hat.
    Gewinner-Bins: argmax über die Gitterzellen, Form (num_row, num_lanes).
    """
    loc = pred["loc_row"][0].detach()  # (grid, num_row, num_lanes)
    exist = pred["exist_row"][0].detach().argmax(0)  # (num_row, num_lanes)
    grid, num_row, num_lanes = loc.shape
    best = loc.argmax(0)
    ys = cfg.row_anchor * cfg.train_height

    lanes = []
    for lane in range(num_lanes):
        pts = np.full((num_row, 2), np.nan)
        for k in range(num_row):
            if not exist[k, lane]:
                continue
            b = int(best[k, lane])
            idx = torch.arange(max(0, b - local_width), min(grid - 1, b + local_width) + 1)
            x = (loc[idx, k, lane].softmax(0) * idx.float()).sum() + 0.5
            pts[k] = (float(x) / (grid - 1) * cfg.train_width, ys[k])
        lanes.append(pts)
    return lanes, best.numpy()
