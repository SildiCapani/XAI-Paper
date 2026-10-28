"""Erster Durchlauf auf einem Testbild: Verdeckung + Grad-CAM/HiResCAM.

Für Ziellinie T und Nachbarlinie N (Indizes: 0 links, 1 Mitte, 2 rechts):
  0  Referenz, nichts verdeckt
  A  mittleres Teilstück von T verdeckt
  B  A + N vollständig verdeckt
  C  nur N verdeckt
  D  A + Kontrollstreifen gleicher Form auf freier Straße (zwischen T und N)
Gemessen wird an den Zeilenankern des verdeckten Teilstücks von T: mittlere
x-Abweichung zur Referenz (px, Modellauflösung) und P(Spur existiert).
Referenz ist die Vorhersage auf dem unverdeckten Bild (es gibt keine Labels).

Danach Grad-CAM und HiResCAM für genau diese Anker (Ziel: Logits der Referenz-Bins)
auf dem Referenzbild und auf Bild A, mit dem Anteil der CAM-Masse auf N, auf dem
sichtbaren und auf dem verdeckten Teil von T.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=0, help="Index in der sortierten Testbildliste")
    parser.add_argument("--lane", type=int, default=1, help="Ziellinie T")
    parser.add_argument("--neighbor", type=int, default=2, help="Nachbarlinie N")
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65),
                        help="verdeckter Bereich von T als Anteil der Zeilenanker (oben=0, unten=1)")
    parser.add_argument("--fill", default="median", choices=["median", "inpaint", "blendung"])
    parser.add_argument("--thickness", type=int, help="Maskenbreite in px (Modellauflösung), Standard skaliert mit der Bildbreite")
    parser.add_argument("--layer", choices=["layer2", "layer3", "layer4"],
                        help="CAM-Schicht, Standard: layer4 bei 512 px, layer3 bei 128 px")
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    args = parser.parse_args()
    T, N = args.lane, args.neighbor

    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))
    rgb = to_model_rgb(load_gray(images[args.frame]), cfg.train_width, cfg.train_height)
    try:
        res = run_frame(net, cfg, rgb, T, N, args.span, args.fill, args.thickness, args.layer)
    except SkipFrame:
        raise SystemExit("T hat im gewählten Bereich keine Punkte oder N wird nicht erkannt.")
    args.layer = args.layer or default_layer(cfg)
    table, cam_table, hidden, lanes0 = res.table, res.cam_table, res.hidden, res.lanes0
    m_n, m_hidden = res.masks["N"], res.masks["T_verdeckt"]

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(f"Modell {args.model} ({cfg.train_width}×{cfg.train_height}), Bild {args.frame} ({images[args.frame].name}), T={LANE_NAMES[T]}, N={LANE_NAMES[N]}, "
          f"verdeckte Anker {hidden[0]}–{hidden[-1]} ({len(hidden)}), Füllung {args.fill}")
    print("\nVerdeckung (Abweichung von T an den verdeckten Ankern zur Referenz):")
    print(table)
    print(f"\nCAM-Masse ({args.layer}) für die verdeckten Anker von T:")
    print(cam_table)

    # Abbildung
    out_dir = paths.results_dir(args.model) / "01_first_look"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame{args.frame:05d}_T{T}_N{N}_{args.fill}_{args.layer}"
    if tuple(args.span) != (0.35, 0.65):  # nur abweichende Bereiche im Namen, Standardnamen bleiben gleich
        stem += f"_span{args.span[0] * 100:.0f}-{args.span[1] * 100:.0f}"
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for ax, (name, img) in zip(axes[0], res.imgs.items()):
        ax.imshow(img)
        for lane, pts in enumerate(res.preds[name]):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=3, color=LANE_COLORS[lane])
        ax.plot(lanes0[T][hidden, 0], lanes0[T][hidden, 1], "w--", lw=1)
        r = table.loc[name]
        ax.set_title(f"{name}: dx={r.dx_mittel_px:.1f}px, P={r.P_exist_T:.2f}")
    for ax, ((img_name, method), cam) in zip(axes[1], res.cams.items()):
        ax.imshow(res.imgs[img_name])
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.contour(m_n, levels=[0.5], colors="white", linewidths=0.7)
        ax.contour(m_hidden, levels=[0.5], colors="magenta", linewidths=0.7)
        s = cam_table.loc[(img_name, method)]
        ax.set_title(f"{method} auf {img_name}: N={s.N_Anteil:.2f}, T_vis={s.T_sichtbar_Anteil:.2f}")
    axes[1, 4].axis("off")
    axes[1, 4].text(0, 0.5, "weiß: Maske N\nmagenta: verdecktes Stück von T\n"
                    "weiß gestrichelt: Referenzposition T", fontsize=11, va="center")
    for ax in axes[0]:
        ax.axis("off")
    for ax in axes[1, :4]:
        ax.axis("off")
    fig.suptitle(f"{args.model}: {images[args.frame].name} – T={LANE_NAMES[T]}, N={LANE_NAMES[N]}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=110)
    table.to_csv(out_dir / f"{stem}_verdeckung.csv")
    cam_table.to_csv(out_dir / f"{stem}_cam.csv")
    print(f"\nAbbildung und Tabellen in {out_dir}")


if __name__ == "__main__":
    main()
