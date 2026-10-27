"""Masken entlang vorhergesagter Linien und Füllverfahren für die Verdeckung.

Alles arbeitet auf dem RGB-Bild in Modellauflösung (uint8), also vor der
Normalisierung. Die Masken sind bool-Arrays (H, W).
"""

import cv2
import numpy as np


def polyline_mask(points: np.ndarray, shape: tuple[int, int], thickness: int = 14) -> np.ndarray:
    """Maske entlang einer Punktfolge (N, 2); NaN-Punkte werden übersprungen."""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = points[~np.isnan(points).any(axis=1)]
    if len(pts) >= 2:
        cv2.polylines(mask, [np.round(pts).astype(np.int32)], False, 1, thickness)
    return mask.astype(bool)


def fill(rgb: np.ndarray, mask: np.ndarray, method: str = "median", ring: int = 15) -> np.ndarray:
    """Ersetzt die maskierten Pixel durch Straße.

    median  -- Median der Grauwerte in einem Ring (Breite ring px) um die Maske (schwarze
               Bildränder ausgenommen), ergibt eine flache Fläche.
    inpaint -- cv2.inpaint (Telea), setzt die Umgebung glatt fort.
    blendung -- heller, weich auslaufender Lichtfleck über der Maske, wie Sonne
               oder Scheinwerfer: Linie und Straße werden gleich hell, die Linie
               verschwindet, die Umgebung bleibt erkennbar.
    """
    out = rgb.copy()
    if not mask.any():
        return out
    m = mask.astype(np.uint8)
    if method == "inpaint":
        return cv2.inpaint(out, m, max(2, ring // 3), cv2.INPAINT_TELEA)
    if method == "blendung":
        # weiche Maske: Kern = Maske, Rand läuft über etwa ring px aus
        soft = cv2.GaussianBlur(cv2.dilate(m, np.ones((ring, ring), np.uint8)).astype(np.float32), (0, 0), ring / 2)
        soft = np.clip(soft / soft.max() * 1.5, 0, 1)[..., None]
        return (out * (1 - soft) + 235 * soft).astype(np.uint8)
    if method != "median":
        raise ValueError(f"Unbekanntes Füllverfahren: {method}")
    ring = cv2.dilate(m, np.ones((ring, ring), np.uint8)).astype(bool) & ~mask
    gray = rgb[..., 0]
    ring &= gray > 5
    out[mask] = np.median(gray[ring]) if ring.any() else np.median(gray)
    return out
