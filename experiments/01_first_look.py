"""Erster Durchlauf auf einem Testbild: ganze Linie wegnehmen + HiResCAM.

Für Ziellinie T und Nachbarlinie N (Indizes: 0 links, 1 Mitte, 2 rechts):
  0  Referenz, nichts verdeckt
  E  T ganz weg
  F  T ganz weg und N ganz weg
Gemessen wird an den Messankern von T (mittlerer Bildbereich, --span): mittlere
x-Abweichung zur Referenz (px, Modellauflösung) und P(Linie da).
Referenz ist die Vorhersage auf dem unverdeckten Bild (es gibt keine Labels).

Danach HiResCAM für genau diese Anker (Ziel: Punktzahl der Referenz-Kästchen)
auf den Bildern 0, E und F, mit dem Anteil der Heatmap auf N und auf T.

Aufruf:  .venv/bin/python experiments/01_first_look.py --frame 0 --lane 1 --neighbor 2 [--model bev_2026]
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from xai_lane import paths
from xai_lane.experiment import SkipFrame, default_layer, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = ["tab:blue", "tab:green", "tab:red"]
TITLES = {"0": "0: Original", "E": "E: T ganz weg", "F": "F: T und N weg"}


def main() -> None:
    """Ein Bild, Bedingungen 0, E, F: Tabelle im Terminal und eine Abbildung.

    Ablauf: Argumente lesen → Modell und Bild laden → run_frame rechnet alles
    (Masken, Übermalen, neue Vorhersagen, Heatmaps) → Tabellen ausgeben → Abbildung speichern.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=0, help="Index in der sortierten Testbildliste")
    parser.add_argument("--lane", type=int, default=1, help="Ziellinie T")
    parser.add_argument("--neighbor", type=int, default=2, help="Nachbarlinie N")
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65),
                        help="Messbereich als Anteil der Zeilenanker (oben=0, unten=1)")
    parser.add_argument("--fill", default="inpaint", choices=["median", "inpaint", "blendung"])
    parser.add_argument("--thickness", type=int, help="Maskenbreite in px (Modellauflösung), Standard skaliert mit der Bildbreite")
    parser.add_argument("--layer", choices=["layer2", "layer3", "layer4"],
                        help="CAM-Schicht, Standard: layer4 bei 512 px, layer3 bei 128 px")
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    args = parser.parse_args()
    T, N = args.lane, args.neighbor

    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))
    # Bild laden (grau) und auf die Eingabegröße des Modells bringen, z. B. 512×384 mit 3 Kanälen
    rgb = to_model_rgb(load_gray(images[args.frame]), cfg.train_width, cfg.train_height)
    try:
        # Die ganze Arbeit passiert in src/xai_lane/experiment.py, damit alle Skripte
        # exakt dasselbe messen. cam_images: auf welchen Bildern eine Heatmap gerechnet wird.
        res = run_frame(net, cfg, rgb, T, N, args.span, args.fill, args.thickness, args.layer,
                        cam_images=("0", "E", "F"))
    except SkipFrame:
        raise SystemExit("T hat im gewählten Bereich keine Punkte oder N wird nicht erkannt.")
    args.layer = args.layer or default_layer(cfg)
    table, cam_table, anker, lanes0 = res.table, res.cam_table, res.anker, res.lanes0

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(f"Modell {args.model} ({cfg.train_width}×{cfg.train_height}), Bild {args.frame} ({images[args.frame].name}), "
          f"T={LANE_NAMES[T]}, N={LANE_NAMES[N]}, Messanker {anker[0]}–{anker[-1]} ({len(anker)}), Füllung {args.fill}")
    print("\nVerschiebung von T an den Messankern gegenüber der Referenz:")
    print(table)
    print(f"\nHeatmap-Anteile ({args.layer}) für die Messanker von T:")
    print(cam_table)

    # Abbildung: oben die Bilder mit Vorhersage, unten die Heatmaps
    out_dir = paths.results_dir(args.model) / "01_first_look"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame{args.frame:05d}_T{T}_N{N}_{args.fill}_{args.layer}"
    if tuple(args.span) != (0.35, 0.65):  # nur abweichende Bereiche im Namen, Standardnamen bleiben gleich
        stem += f"_span{args.span[0] * 100:.0f}-{args.span[1] * 100:.0f}"
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, (name, img) in zip(axes[0], res.imgs.items()):
        ax.imshow(img)
        for lane, pts in enumerate(res.preds[name]):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=3, color=LANE_COLORS[lane])
        ax.plot(lanes0[T][anker, 0], lanes0[T][anker, 1], "w--", lw=1)  # wo T ursprünglich lag
        r = table.loc[name]
        ax.set_title(f"{TITLES[name]}\ndx = {r.dx_mittel_px:.1f} px, P(T da) = {r.P_exist_T:.2f}")
    for ax, ((img_name, method), cam) in zip(axes[1], res.cams.items()):
        ax.imshow(res.imgs[img_name])
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.contour(res.masks["N"], levels=[0.5], colors="white", linewidths=0.7)
        ax.contour(res.masks["T"], levels=[0.5], colors="magenta", linewidths=0.7)
        s = cam_table.loc[(img_name, method)]
        ax.set_title(f"Heatmap auf {img_name}\nauf N: {s.N_vs_Flaeche:.2f}×, auf T: {s.T_vs_Flaeche:.2f}× (vs. Fläche)")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{args.model}: {images[args.frame].name} – T={LANE_NAMES[T]}, N={LANE_NAMES[N]}   "
                 "(weiß gestrichelt: alte Position von T, weiß: Maske N, magenta: Maske T)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=110)
    table.to_csv(out_dir / f"{stem}_verdeckung.csv")
    cam_table.to_csv(out_dir / f"{stem}_cam.csv")
    print(f"\nAbbildung und Tabellen in {out_dir}")


if __name__ == "__main__":
    main()
