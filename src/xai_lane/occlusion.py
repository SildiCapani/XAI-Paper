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


def fill(rgb: np.ndarray, mask: np.ndarray, method: str = "median") -> np.ndarray:
    """Ersetzt die maskierten Pixel durch Straße.

    median  -- Median der Grauwerte in einem Ring um die Maske (schwarze
               Bildränder ausgenommen), ergibt eine flache Fläche.
    inpaint -- cv2.inpaint (Telea), setzt die Umgebung glatt fort.
    """
    out = rgb.copy()
    if not mask.any():
        return out
    m = mask.astype(np.uint8)
    if method == "inpaint":
        return cv2.inpaint(out, m, 5, cv2.INPAINT_TELEA)
    if method != "median":
        raise ValueError(f"Unbekanntes Füllverfahren: {method}")
    ring = cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(bool) & ~mask
    gray = rgb[..., 0]
    ring &= gray > 5
    out[mask] = np.median(gray[ring]) if ring.any() else np.median(gray)
    return out
