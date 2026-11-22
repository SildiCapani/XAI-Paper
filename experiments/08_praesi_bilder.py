"""Bilder für die Folien: ein Bild pro Schritt der Versuchsanordnung.

Schritt "Verdecken" (occlusion.py):
  1. Vorhersage auf dem Originalbild (decode)
  2. Maske entlang der Ziellinie T (polyline_mask)
  3. Maske gefüllt mit inpaint (fill), so wie im Hauptergebnis
  4. Zum Vergleich: median, dort bleibt ein flacher Streifen sichtbar

Schritt "Viele Bilder" (02_batch.py, muss vorher gelaufen sein):
  links:  Verschiebung von T, E (nur T weg) gegen F (T und N weg), je Linienpaar
  rechts: Heatmap auf N, Original gegen E (T weg), je Linienpaar

Aufruf:  .venv/bin/python experiments/08_praesi_bilder.py [--frame 0] [--lane 1] [--model bev_2026]
Ausgabe: docs/bilder/praesi/schritt_verdecken_ganz.png, schritt_viele_bilder.png
"""

import argparse

import pandas as pd

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


def _bars(ax, s, left, right, labels, colors, ylabel, title):
    """Zwei Balken je Linienpaar (gepaarter Vergleich), Wert über jeden Balken geschrieben."""
    groups = [f"{LANE_NAMES[t]} ← {LANE_NAMES[n]}" for t, n in zip(s["T"], s.N)]
    w = 0.38
    for off, col, lab, color in ((-w / 2, left, labels[0], colors[0]), (w / 2, right, labels[1], colors[1])):
        x = np.arange(len(s)) + off
        ax.bar(x, s[col], w, color=color, label=lab, edgecolor="#fcfcfb", linewidth=2)
        for xi, v in zip(x, s[col]):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color="#52514e")
    ax.set_ylim(0, 1.3 * s[[left, right]].max().max())  # Platz für die Legende über den Balken
    ax.set_xticks(range(len(s)), groups)
    ax.set_xlabel("Linie T ← Nachbarlinie N", color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, loc="left", color="#0b0b0b")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    ax.set_facecolor("#fcfcfb")


def schritt_viele_bilder(model, out):
    """Ergebnis über alle Bilder aus 02_batch/zusammenfassung.csv."""
    csv = paths.results_dir(model) / "02_batch" / "zusammenfassung.csv"
    if not csv.exists():
        print(f"  übersprungen, erst 02_batch.py laufen lassen ({csv} fehlt)")
        return
    s = pd.read_csv(csv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    _bars(a1, s, "dx_E_median_pct", "dx_F_median_pct", ["E: T weg", "F: T und N weg"], ["#2a78d6", "#eb6834"],
          "Verschiebung von T\n(Median, % der Bildbreite)", "Ohne Nachbarlinie liegt T weiter daneben")
    _bars(a2, s, "cam_N_0_median", "cam_N_E_median", ["Original", "E: T weg"], ["#a8a7a2", "#2a78d6"],
          "Heatmap auf N\n(Median, × erwartet nach Fläche)", "Fehlt T, schaut das Modell mehr auf N")
    fig.suptitle(f"{s.n_Bilder.min()}–{s.n_Bilder.max()} Bilder je Linienpaar, Modell {model}",
                 x=0.01, ha="left", fontsize=11, color="#52514e")
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
    schritt_viele_bilder(args.model, OUT / "schritt_viele_bilder.png")


if __name__ == "__main__":
    main()
