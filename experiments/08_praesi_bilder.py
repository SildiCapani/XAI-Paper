"""Bilder für die Folien: ein Bild pro Schritt der Versuchsanordnung.

Schritt "Verdecken" (occlusion.py):
  1. Vorhersage auf dem Originalbild (decode)
  2. Maske entlang der Ziellinie T (polyline_mask)
  3. Maske gefüllt mit inpaint (fill), so wie im Hauptergebnis
  4. Zum Vergleich: median, dort bleibt ein flacher Streifen sichtbar

Schritt "Ein Beispiel" (run_frame wie 01_first_look.py, nur größer beschriftet für die Folie):
  Original gegen T weg, oben mit Vorhersage, unten die Heatmaps

Schritt "Heatmap pro Linie": für jede der drei Linien oben die Heatmap im Original
  (wo sucht das Modell diese Linie?), unten dieselbe Linie weggenommen (wohin weicht es aus?)

Schritt "Sanity Check": Heatmap trainiert gegen schrittweise zufällige Gewichte (wie 07_sanity_check.py)

Schritt "Viele Bilder" (02_batch.py, muss vorher gelaufen sein):
  Heatmap auf N, Original gegen T weg, je Linienpaar

Aufruf:  .venv/bin/python experiments/08_praesi_bilder.py [--frame 0] [--lane 1] [--model bev_2026]
Ausgabe: docs/bilder/praesi/schritt_verdecken_ganz.png, schritt_ein_bild.png, schritt_pro_linie.png,
         schritt_sanity_heatmaps.png, schritt_viele_bilder.png
"""

import argparse
import copy
import importlib

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from xai_lane import paths
from xai_lane.decode import decode_row_lanes
from xai_lane.cam import compute_cam
from xai_lane.experiment import default_layer, run_frame
from xai_lane.model_loader import load_model
from xai_lane.occlusion import fill, polyline_mask
from xai_lane.preprocess import load_gray, to_model_rgb, to_tensor

OUT = paths.PROJECT_ROOT / "docs" / "bilder" / "praesi"
LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"font.size": 12})  # Folien: Schrift muss auch aus der letzten Reihe lesbar sein


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
        (rgb, "1  Vorhersage\n(55 Punkte je Linie)"),
        (overlay(rgb, mask), f"2  Maske\n({thickness} px entlang der Linie)"),
        (fill(rgb, mask, "inpaint"), "3  Inpainting\n(sieht aus wie Straße)"),
        (fill(rgb, mask, "median"), "Vergleich: flache Füllung\n(Streifen bleibt sichtbar)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))
    for i, (ax, (img, title)) in enumerate(zip(axes, panels)):
        ax.imshow(img)
        ax.set_title(title, loc="left", fontsize=13, color="#0b0b0b" if i < 3 else "#52514e")
        ax.axis("off")
    for k, lane in enumerate(lanes):  # Punkte nur im ersten Panel
        ok = ~np.isnan(lane).any(axis=1)
        axes[0].scatter(lane[ok, 0], lane[ok, 1], s=7, color=LANE_COLORS[k],
                        label=LANE_NAMES[k] + (" (= T)" if k == T else ""))
    axes[0].legend(loc="lower left", fontsize=10, frameon=True, markerscale=2)
    for spine in axes[3].spines.values():
        spine.set_visible(True)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def schritt_ein_bild(net, cfg, rgb, T, N, out):
    """Ein Bild, Original gegen T weg: oben Vorhersage, unten Heatmap (wie 01_first_look.py).

    Bedingung F (zwei Linien weg) kommt auf den Folien nicht als Bild vor, nur als Gegenprobe in Zahlen.
    """
    res = run_frame(net, cfg, rgb, T, N, conditions=("0", "E"), cam_images=("0", "E"))
    titles = {"0": "Original", "E": f"Linie {LANE_NAMES[T]} weg"}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6))
    for ax, (name, img) in zip(axes[0], res.imgs.items()):
        ax.imshow(img)
        for k, pts in enumerate(res.preds[name]):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=3, color=LANE_COLORS[k])
        ax.set_title(f"{titles[name]}\nVerschiebung {res.table.loc[name, 'dx_mittel_px']:.0f} px", fontsize=14)
    for ax, ((img_name, _), cam) in zip(axes[1], res.cams.items()):
        ax.imshow(res.imgs[img_name])
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(f"Heatmap: {titles[img_name]}", fontsize=14)
    for ax in axes.flat:
        ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def schritt_pro_linie(net, cfg, rgb, out):
    """Pro Linie eine Spalte: oben Heatmap im Original, unten Heatmap mit dieser Linie weggenommen.

    Ziel ist hier die ganze Linie (span 0–1), damit man sieht, wo das Modell die Linie überall sucht.
    """
    neighbor = {0: 1, 1: 2, 2: 1}  # N wird nur für die Prüfung in run_frame gebraucht
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for T in range(3):
        res = run_frame(net, cfg, rgb, T, neighbor[T], span=(0, 1), cam_images=("0", "E"))
        for row, name in enumerate(("0", "E")):
            ax = axes[row, T]
            ax.imshow(res.imgs[name])
            ax.imshow(res.cams[(name, "hirescam")], cmap="jet", alpha=0.45)
            ax.set_title(f"Linie {LANE_NAMES[T]}" + (" gesucht" if name == "0" else " weg"), fontsize=14,
                         color=LANE_COLORS[T] if name == "0" else "#0b0b0b")
            ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def schritt_sanity_heatmaps(net, cfg, rgb, T, out):
    """2 × 2 Heatmaps: trainiert, letzte Schicht zufällig, bis layer4 zufällig, alles zufällig."""
    sanity = importlib.import_module("07_sanity_check")  # experiments/ liegt beim Aufruf im Suchpfad
    target, _ = sanity.targets_for(net, cfg, rgb)[T]
    x = to_tensor(rgb)
    layer = default_layer(cfg)
    show = {"trainiert": "trainiert", "cls.3": "+ letzte Schicht zufällig",
            "layer4": "+ bis layer4 zufällig", "conv1": "alles zufällig"}
    cams = {"trainiert": compute_cam(net, x, target, "hirescam", layer)}
    torch.manual_seed(0)
    rnd = copy.deepcopy(net)  # Kopie, das trainierte Netz bleibt heil
    for stage, modules in sanity.STAGES[1:]:
        sanity.randomize(rnd, modules)
        if stage in show:
            cams[stage] = compute_cam(rnd, x, target, "hirescam", layer)
    fig, axes = plt.subplots(2, 2, figsize=(7, 5.6))
    for ax, (stage, cam) in zip(axes.flat, cams.items()):
        ax.imshow(rgb)
        ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title(show[stage], fontsize=14)
        ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def _bars(ax, s, left, right, labels, colors, ylabel, title):
    """Zwei Balken je Linienpaar (gepaarter Vergleich), Wert über jeden Balken geschrieben."""
    groups = [f"{LANE_NAMES[t]}\n← {LANE_NAMES[n]}" for t, n in zip(s["T"], s.N)]
    w = 0.38
    for off, col, lab, color in ((-w / 2, left, labels[0], colors[0]), (w / 2, right, labels[1], colors[1])):
        x = np.arange(len(s)) + off
        ax.bar(x, s[col], w, color=color, label=lab, edgecolor="#fcfcfb", linewidth=2)
        for xi, v in zip(x, s[col]):
            ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=11, color="#52514e")
    ax.set_ylim(0, 1.45 * s[[left, right]].max().max())  # Platz für die Legende über den Balken
    ax.set_xticks(range(len(s)), groups)
    ax.set_xlabel("Linie T ← Nachbarlinie N", color="#52514e")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.set_title(title, loc="left", color="#0b0b0b", fontsize=13)
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
    # nur die Heatmap: F (zwei Linien weg) ist nur eine Gegenprobe und kommt als Zahl auf die Folie
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    _bars(ax, s, "cam_N_0_median", "cam_N_E_median", ["Original", "Linie T weg"], ["#a8a7a2", "#2a78d6"],
          "Heatmap auf N\n(Median, × erwartet nach Fläche)", "Fehlt T, schaut das Modell mehr auf N")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.relative_to(paths.PROJECT_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--lane", type=int, default=1, help="Ziellinie T: 0 links, 1 Mitte, 2 rechts")
    p.add_argument("--beispiel", type=int, nargs=2, default=(1, 0), metavar=("T", "N"),
                   help="Linienpaar für die Beispielfolie, Standard: Mitte weg, Nachbar links")
    p.add_argument("--model", default=paths.DEFAULT_MODEL)
    args = p.parse_args()

    net, cfg = load_model(args.model)
    image = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[args.frame]
    rgb = to_model_rgb(load_gray(image), cfg.train_width, cfg.train_height)
    OUT.mkdir(parents=True, exist_ok=True)
    schritt_verdecken(net, cfg, rgb, args.lane, OUT / "schritt_verdecken_ganz.png")
    schritt_ein_bild(net, cfg, rgb, *args.beispiel, OUT / "schritt_ein_bild.png")
    schritt_pro_linie(net, cfg, rgb, OUT / "schritt_pro_linie.png")
    schritt_sanity_heatmaps(net, cfg, rgb, args.lane, OUT / "schritt_sanity_heatmaps.png")
    schritt_viele_bilder(args.model, OUT / "schritt_viele_bilder.png")


if __name__ == "__main__":
    main()
