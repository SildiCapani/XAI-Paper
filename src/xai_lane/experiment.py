"""Verdeckungsexperiment und Heatmap-Auswertung für ein Bild und ein Linienpaar (T, N).

Frage: Fehlt die Linie T ganz, orientiert sich das Modell dann an der Nachbarlinie N?

Bedingungen:
  0  Referenz, nichts verdeckt
  E  T ganz weg
  F  T ganz weg und N ganz weg
Gemessen wird nur an den mittleren Zeilenankern (span, 35–65 % = Anker 19–34),
weil die Vorhersage am oberen und unteren Bildrand ohnehin unsicher ist.
Alle Auswertungsskripte rufen run_frame auf, damit alle exakt dasselbe messen.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from xai_lane.cam import RowBinTarget, compute_cam, energy_share
from xai_lane.decode import decode_row_lanes, exist_prob
from xai_lane.occlusion import fill, polyline_mask
from xai_lane.preprocess import to_tensor

CONDITIONS = ("0", "E", "F")
CAM_METHODS = ("hirescam",)  # Grad-CAM entfernt, Begründung im README


@dataclass
class FrameResult:
    table: pd.DataFrame  # Index: Bedingung
    cam_table: pd.DataFrame  # Index: (Bild, Methode)
    anker: np.ndarray  # Messanker: Zeilenanker von T im Bereich span
    lanes0: list[np.ndarray]  # Referenzpunkte je Spur
    preds: dict  # Bedingung -> Punkte je Spur
    imgs: dict  # Bedingung -> verdecktes RGB-Bild
    cams: dict  # (Bild, Methode) -> Heatmap
    masks: dict  # "T", "N" -> Maske


class SkipFrame(Exception):
    """T hat im gewählten Bereich keine Punkte oder N wird nicht erkannt."""


def predict(net, rgb: np.ndarray) -> dict:
    """Ein Bild durchs Netz schicken. Gibt das Dict mit loc_row, exist_row, loc_col, exist_col zurück."""
    with torch.no_grad():  # keine Gradienten nötig, spart Zeit und Speicher
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


def run_frame(net, cfg, rgb: np.ndarray, T: int, N: int, span=(0.35, 0.65), fill_method="inpaint",
              thickness=None, layer=None, cam_images=("E", "F"), conditions=CONDITIONS) -> FrameResult:
    """Das ganze Experiment für ein Bild und ein Linienpaar (T = Ziellinie, N = Nachbar).

    1. Referenz: Vorhersage auf dem Originalbild, Messanker von T bestimmen
    2. Masken bauen: ganz T, ganz N
    3. Je Bedingung: Masken übermalen, neu vorhersagen, Verschiebung von T messen
    4. Heatmaps (HiResCAM) auf den Bildern in cam_images, Anteil auf N und T messen

    Wirft SkipFrame, wenn T im Bereich span oder N gar nicht erkannt wird.
    """
    shape = rgb.shape[:2]
    thickness = thickness or default_thickness(cfg)
    layer = layer or default_layer(cfg)

    # 1. Referenz
    pred0 = predict(net, rgb)
    lanes0, bins0 = decode_row_lanes(pred0, cfg)
    # span (0.35, 0.65) -> Anker-Nummern, bei 55 Ankern 19 bis 34. Davon nur die, an denen T erkannt wurde.
    k0, k1 = (int(s * cfg.num_row) for s in span)
    anker = np.arange(k0, k1)
    anker = anker[~np.isnan(lanes0[T][anker, 0])]
    if len(anker) == 0 or np.isnan(lanes0[N][:, 0]).all():
        raise SkipFrame

    # 2. Masken entlang der vorhergesagten Linien
    m_t = polyline_mask(lanes0[T], shape, thickness)
    m_n = polyline_mask(lanes0[N], shape, thickness)
    masks = {"T": m_t, "N": m_n}
    all_conditions = {
        "0": np.zeros(shape, bool),
        "E": m_t,
        "F": m_t | m_n,
    }
    conditions = {c: all_conditions[c] for c in conditions}

    # 3. Je Bedingung übermalen, neu vorhersagen, messen
    rows, preds, imgs = [], {}, {}
    for name, mask in conditions.items():
        img = fill(rgb, mask, fill_method, ring=thickness + 1)
        pred = pred0 if name == "0" else predict(net, img)
        lanes, _ = decode_row_lanes(pred, cfg)
        # Verschiebung = wie weit T an jedem Messanker von seiner Originalposition abweicht.
        # NaN heißt: dort wird T gar nicht mehr erkannt (zählt als "Anker_verloren").
        dx = np.abs(lanes[T][anker, 0] - lanes0[T][anker, 0])
        rows.append({
            "Bedingung": name,
            "dx_mittel_px": np.nanmean(dx) if not np.isnan(dx).all() else np.nan,
            "dx_max_px": np.nanmax(dx) if not np.isnan(dx).all() else np.nan,
            # in % der Bildbreite, damit Modelle mit verschiedener Eingabegröße vergleichbar sind
            "dx_mittel_pct": 100 * np.nanmean(dx) / cfg.train_width if not np.isnan(dx).all() else np.nan,
            "Anker_verloren": int(np.isnan(dx).sum()),
            "P_exist_T": exist_prob(pred)[anker, T].mean(),
        })
        preds[name], imgs[name] = lanes, img
    table = pd.DataFrame(rows).set_index("Bedingung")

    # 4. Heatmaps für die Messanker von T, Ziel = Kästchen aus dem Originalbild:
    # "Was hält die Linie an ihrer alten Stelle, obwohl sie weg ist?"
    target = RowBinTarget(T, anker, bins0[anker, T])
    cams, cam_rows = {}, []
    for img_name in cam_images:
        x = to_tensor(imgs[img_name])
        for method in CAM_METHODS:
            cam = compute_cam(net, x, target, method, layer)
            cams[(img_name, method)] = cam
            row = {"Bild": img_name, "Methode": method}
            for r, m in masks.items():
                # Verhältnis > 1: mehr Heatmap auf der Maske, als ihre Fläche allein erwarten lässt
                row[f"{r}_Anteil"] = energy_share(cam, m)
                row[f"{r}_vs_Flaeche"] = energy_share(cam, m) / m.mean()
            cam_rows.append(row)
    cam_table = pd.DataFrame(cam_rows).set_index(["Bild", "Methode"]) if cam_rows else pd.DataFrame()

    return FrameResult(table, cam_table, anker, lanes0, preds, imgs, cams, masks)
