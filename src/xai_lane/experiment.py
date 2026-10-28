"""Verdeckungsexperiment und CAM-Auswertung für ein Bild und ein Linienpaar (T, N).

Bedingungen (siehe experiments/01_first_look.py):
  0 Referenz, A Teilstück von T, B A + N ganz, C nur N, D A + Kontrollstreifen.
Wird von 01_first_look.py (ein Bild, mit Abbildung) und 02_batch.py (viele
Bilder) benutzt, damit beide exakt dasselbe messen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from xai_lane.cam import RowBinTarget, compute_cam, energy_share
from xai_lane.decode import decode_row_lanes, exist_prob
from xai_lane.occlusion import fill, polyline_mask
from xai_lane.preprocess import to_tensor

CONDITIONS = ("0", "A", "B", "C", "D")
CAM_METHODS = ("gradcam", "hirescam")


@dataclass
class FrameResult:
    table: pd.DataFrame  # Index: Bedingung
    cam_table: pd.DataFrame  # Index: (Bild, Methode)
    hidden: np.ndarray  # verdeckte Zeilenanker von T
    lanes0: list[np.ndarray]  # Referenzpunkte je Spur
    preds: dict  # Bedingung -> Punkte je Spur
    imgs: dict  # Bedingung -> verdecktes RGB-Bild
    cams: dict  # (Bild, Methode) -> CAM
    masks: dict  # Name -> Maske


class SkipFrame(Exception):
    """T hat im gewählten Bereich keine Punkte oder N wird nicht erkannt."""


def predict(net, rgb: np.ndarray) -> dict:
    with torch.no_grad():
        return net(to_tensor(rgb))


def default_thickness(cfg) -> int:
    """Maskenbreite: 14 px bei 512 px Bildbreite, proportional skaliert."""
    return max(3, round(14 * cfg.train_width / 512))


def default_layer(cfg) -> str:
    """Tiefste ResNet-Stufe, deren Feature-Map noch mindestens 8 Zellen breit ist
    (512 px -> layer4 mit 12×16, 128 px -> layer3 mit 6×8)."""
    for layer, stride in (("layer4", 32), ("layer3", 16)):
        if cfg.train_width // stride >= 8:
            return layer
    return "layer2"


def run_frame(net, cfg, rgb: np.ndarray, T: int, N: int, span=(0.35, 0.65), fill_method="median",
              thickness=None, layer=None, cam_images=("0", "A"), conditions=CONDITIONS) -> FrameResult:
    shape = rgb.shape[:2]
    thickness = thickness or default_thickness(cfg)
    layer = layer or default_layer(cfg)

    # Referenz
    pred0 = predict(net, rgb)
    lanes0, bins0 = decode_row_lanes(pred0, cfg)
    k0, k1 = (int(s * cfg.num_row) for s in span)
    hidden = np.arange(k0, k1)
    hidden = hidden[~np.isnan(lanes0[T][hidden, 0])]
    if len(hidden) == 0 or np.isnan(lanes0[N][:, 0]).all():
        raise SkipFrame

    # Masken
    seg = np.full_like(lanes0[T], np.nan)
    seg[hidden] = lanes0[T][hidden]
    m_hidden = polyline_mask(seg, shape, thickness)
    m_t = polyline_mask(lanes0[T], shape, thickness)
    m_n = polyline_mask(lanes0[N], shape, thickness)
    m_ctrl = polyline_mask((lanes0[T] + lanes0[N]) / 2, shape, thickness) & ~m_t & ~m_n
    masks = {"T_verdeckt": m_hidden, "T_sichtbar": m_t & ~m_hidden, "N": m_n, "Kontrolle": m_ctrl}

    all_conditions = {
        "0": np.zeros(shape, bool),
        "A": m_hidden,
        "B": m_hidden | m_n,
        "C": m_n,
        "D": m_hidden | m_ctrl,
    }
    # nur die angefragten Bedingungen rechnen (Video braucht z. B. nur "0" und eine)
    conditions = {c: all_conditions[c] for c in conditions}

    rows, preds, imgs = [], {}, {}
    for name, mask in conditions.items():
        img = fill(rgb, mask, fill_method, ring=thickness + 1)
        pred = pred0 if name == "0" else predict(net, img)
        lanes, _ = decode_row_lanes(pred, cfg)
        dx = np.abs(lanes[T][hidden, 0] - lanes0[T][hidden, 0])
        rows.append({
            "Bedingung": name,
            "dx_mittel_px": np.nanmean(dx) if not np.isnan(dx).all() else np.nan,
            "dx_max_px": np.nanmax(dx) if not np.isnan(dx).all() else np.nan,
            # in % der Bildbreite, damit Modelle mit verschiedener Eingabegröße vergleichbar sind
            "dx_mittel_pct": 100 * np.nanmean(dx) / cfg.train_width if not np.isnan(dx).all() else np.nan,
            "Anker_verloren": int(np.isnan(dx).sum()),
            "P_exist_T": exist_prob(pred)[hidden, T].mean(),
        })
        preds[name], imgs[name] = lanes, img
    table = pd.DataFrame(rows).set_index("Bedingung")

    # CAMs für die verdeckten Anker von T, Ziel = Referenz-Bins
    target = RowBinTarget(T, hidden, bins0[hidden, T])
    regions = {r: masks[r] for r in ("N", "T_sichtbar", "T_verdeckt")}
    cams, cam_rows = {}, []
    for img_name in cam_images:
        x = to_tensor(imgs[img_name])
        for method in CAM_METHODS:
            cam = compute_cam(net, x, target, method, layer)
            cams[(img_name, method)] = cam
            row = {"Bild": img_name, "Methode": method}
            for r, m in regions.items():
                # Verhältnis > 1: mehr CAM-Masse als die Fläche allein erwarten lässt
                row[f"{r}_Anteil"] = energy_share(cam, m)
                row[f"{r}_vs_Flaeche"] = energy_share(cam, m) / m.mean()
            cam_rows.append(row)
    cam_table = pd.DataFrame(cam_rows).set_index(["Bild", "Methode"]) if cam_rows else pd.DataFrame()

    return FrameResult(table, cam_table, hidden, lanes0, preds, imgs, cams, masks)
