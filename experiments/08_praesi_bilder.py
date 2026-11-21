"""Bilder für die Folien: ein Bild pro Schritt der Versuchsanordnung.

Schritt "Verdecken" (occlusion.py):
  1. Vorhersage auf dem Originalbild (decode)
  2. Maske entlang der Ziellinie T (polyline_mask)
  3. Maske gefüllt mit inpaint (fill), so wie im Hauptergebnis
  4. Zum Vergleich: median, dort bleibt ein flacher Streifen sichtbar

Aufruf:  .venv/bin/python experiments/08_praesi_bilder.py [--frame 0] [--lane 1] [--model bev_2026]
Ausgabe: docs/bilder/praesi/schritt_verdecken_ganz.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from xai_lane import paths
from xai_lane.decode import decode_row_lanes
from xai_lane.model_loader import load_model
from xai_lane.occlusion import fill, polyline_mask
from xai_lane.preprocess import load_gray, to_model_rgb, to_tensor

OUT = paths.PROJECT_ROOT / "docs" / "bilder" / "praesi"
LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def predict_lanes(net, cfg, rgb):
    """Bild durchs Netz schicken und die Linienpunkte (55 Anker, NaN = nicht erkannt) zurückgeben."""
    with torch.no_grad():
        pred = net(to_tensor(rgb))
    lanes, _ = decode_row_lanes(pred, cfg)
    return lanes


def overlay(rgb, mask, color=(235, 60, 50), alpha=0.6):
    """Maske halbdurchsichtig rot über das Bild legen."""
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, np.float32)
    return out.astype(np.uint8)


def schritt_verdecken(net, cfg, rgb, T, out):
    """Vier Panels: Vorhersage -> Maske -> inpaint -> median (Vergleich)."""
    lanes = predict_lanes(net, cfg, rgb)
    thickness = max(3, round(14 * cfg.train_width / 512))
    mask = polyline_mask(lanes[T], rgb.shape[:2], thickness)  # ganze Linie T, wie Bedingung E

    panels = [
        (rgb, f"1  Vorhersage des Modells\n(decode: 55 Punkte je Linie)"),
        (overlay(rgb, mask), f"2  Maske: {thickness} px breit entlang\nder Linie {LANE_NAMES[T]} (polyline_mask)"),
        (fill(rgb, mask, "inpaint"), "3  Gefüllt mit inpaint\n(sieht aus wie Straße)"),
        (fill(rgb, mask, "median"), "Vergleich: median\n(flacher Streifen bleibt sichtbar)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    for i, (ax, (img, title)) in enumerate(zip(axes, panels)):
        ax.imshow(img)
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b" if i < 3 else "#52514e")
        ax.axis("off")
    for k, lane in enumerate(lanes):  # Punkte nur im ersten Panel
        ok = ~np.isnan(lane).any(axis=1)
        axes[0].scatter(lane[ok, 0], lane[ok, 1], s=7, color=LANE_COLORS[k],
                        label=LANE_NAMES[k] + (" (= T)" if k == T else ""))
    axes[0].legend(loc="lower left", fontsize=8, frameon=True, markerscale=2)
    for spine in axes[3].spines.values():
        spine.set_visible(True)
    fig.suptitle("Verdecken: Wir nehmen die ganze Linie aus dem Bild", x=0.01, ha="left",
                 fontsize=14, color="#0b0b0b")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--lane", type=int, default=1, help="Ziellinie T: 0 links, 1 Mitte, 2 rechts")
    p.add_argument("--model", default=paths.DEFAULT_MODEL)
    args = p.parse_args()

    net, cfg = load_model(args.model)
    image = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[args.frame]
    rgb = to_model_rgb(load_gray(image), cfg.train_width, cfg.train_height)
    OUT.mkdir(parents=True, exist_ok=True)
    schritt_verdecken(net, cfg, rgb, args.lane, OUT / "schritt_verdecken_ganz.png")


if __name__ == "__main__":
    main()
