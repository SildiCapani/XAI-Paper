"""Erklärt das Verdeckungsexperiment Schritt für Schritt in einer Abbildung.

  1  Zeilenanker: feste Bildzeilen, in denen das Modell je Linie eine x-Position sucht
  2  Vorhersage: ein Punkt pro Anker und Linie; markiert sind die Anker im Bereich --span
  3  Maske: Streifen entlang dieser Punkte (Ausschnitt, vergrößert)
  4  Übermalt: derselbe Ausschnitt nach dem Füllen
  5  Neue Vorhersage auf dem übermalten Bild im Vergleich zur alten

Aufruf:  .venv/bin/python experiments/03_schritte.py --frame 0 --lane 1 [--fill blendung] [--model bev_2026]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from xai_lane import paths
from xai_lane.experiment import SkipFrame, default_thickness, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = ["tab:blue", "tab:green", "tab:red"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--lane", type=int, default=1, help="Linie, deren Stück verdeckt wird")
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--fill", default="inpaint", choices=["median", "inpaint", "blendung"])
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    args = parser.parse_args()
    T = args.lane
    N = 1 if T != 1 else 2  # Nachbar wird hier nur für run_frame gebraucht

    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))
    rgb = to_model_rgb(load_gray(images[args.frame]), cfg.train_width, cfg.train_height)
    try:
        res = run_frame(net, cfg, rgb, T, N, args.span, args.fill, cam_images=())
    except SkipFrame:
        raise SystemExit("Die Linie wird im gewählten Bereich nicht erkannt.")
    hidden, lanes0, mask = res.hidden, res.lanes0, res.masks["T_verdeckt"]
    ys = cfg.row_anchor * cfg.train_height
    H, W = rgb.shape[:2]

    # Ausschnitt um das verdeckte Stück
    pts = lanes0[T][hidden]
    pad = int(0.12 * W)
    x0, x1 = max(0, int(np.nanmin(pts[:, 0])) - pad), min(W, int(np.nanmax(pts[:, 0])) + pad)
    y0, y1 = max(0, int(pts[:, 1].min()) - pad), min(H, int(pts[:, 1].max()) + pad)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.6))

    ax = axes[0]
    ax.imshow(rgb)
    for k, y in enumerate(ys):
        sel = k in hidden
        ax.axhline(y, color="magenta" if sel else "white", lw=1.2 if sel else 0.4, alpha=1 if sel else 0.5)
    ax.text(W + 2, ys[0], "Anker 0", fontsize=8, va="center")
    ax.text(W + 2, ys[-1], f"Anker {cfg.num_row - 1}", fontsize=8, va="center")
    ax.text(W + 2, ys[hidden].mean(), f"Anker {hidden[0]}–{hidden[-1]}\n(Bereich {args.span[0]:.0%}–{args.span[1]:.0%})",
            fontsize=8, va="center", color="magenta")
    ax.set_title(f"1) {cfg.num_row} Zeilenanker: in jeder dieser\nZeilen sucht das Modell jede Linie")

    ax = axes[1]
    ax.imshow(rgb)
    for lane, p in enumerate(lanes0):
        ax.plot(p[:, 0], p[:, 1], ".", ms=4, color=LANE_COLORS[lane])
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=7, mfc="none", mec="magenta", mew=1.5)
    ax.set_title(f"2) Vorhersage: 1 Punkt pro Anker und Linie\nmagenta: Punkte der Linie „{LANE_NAMES[T]}“ im Bereich")

    ax = axes[2]
    ax.imshow(rgb[y0:y1, x0:x1], interpolation="nearest", extent=(x0, x1, y1, y0))
    overlay = np.zeros((H, W, 4))
    overlay[mask] = (1, 0, 1, 0.45)
    ax.imshow(overlay[y0:y1, x0:x1], interpolation="nearest", extent=(x0, x1, y1, y0))
    ax.plot(pts[:, 0], pts[:, 1], ".", ms=6, color="magenta")
    ax.set_title(f"3) Maske (Ausschnitt): {default_thickness(cfg)} px breiter Streifen\nentlang der magenta Punkte")

    ax = axes[3]
    ax.imshow(res.imgs["A"][y0:y1, x0:x1], interpolation="nearest", extent=(x0, x1, y1, y0))
    ax.contour(np.arange(x0, x1), np.arange(y0, y1), mask[y0:y1, x0:x1], levels=[0.5], colors="magenta", linewidths=0.8)
    ax.set_title(f"4) Pixel unter der Maske übermalt\n(Füllung: {args.fill})")

    ax = axes[4]
    ax.imshow(res.imgs["A"])
    new = res.preds["A"][T]
    ax.plot(lanes0[T][:, 0], lanes0[T][:, 1], "x", ms=4, color="white", label="vorher (Originalbild)")
    ax.plot(new[:, 0], new[:, 1], ".", ms=5, color=LANE_COLORS[T], label="nachher (übermaltes Bild)")
    ax.legend(loc="lower right", fontsize=8)
    r = res.table.loc["A"]
    ax.set_title(f"5) Neue Vorhersage: im Bereich im Mittel\n{r.dx_mittel_px:.1f} px verschoben, P(Linie da) = {r.P_exist_T:.2f}")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{args.model} ({W}×{H}): {images[args.frame].name}", y=1.0)
    fig.tight_layout()
    out_dir = paths.results_dir(args.model) / "03_schritte"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"frame{args.frame:05d}_T{T}_{args.fill}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Gespeichert: {out}")


if __name__ == "__main__":
    main()
