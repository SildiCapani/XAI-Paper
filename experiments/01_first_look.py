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

Aufruf:  .venv/bin/python experiments/01_first_look.py --frame 0 --lane 1 --neighbor 2
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from xai_lane import paths
from xai_lane.cam import RowBinTarget, compute_cam, energy_share
from xai_lane.decode import decode_row_lanes, exist_prob
from xai_lane.model_loader import load_model
from xai_lane.occlusion import fill, polyline_mask
from xai_lane.preprocess import load_gray, to_model_rgb, to_tensor

LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = ["tab:blue", "tab:green", "tab:red"]


def predict(net, rgb: np.ndarray) -> dict:
    with torch.no_grad():
        return net(to_tensor(rgb))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frame", type=int, default=0, help="Index in der sortierten Testbildliste")
    parser.add_argument("--lane", type=int, default=1, help="Ziellinie T")
    parser.add_argument("--neighbor", type=int, default=2, help="Nachbarlinie N")
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65),
                        help="verdeckter Bereich von T als Anteil der Zeilenanker (oben=0, unten=1)")
    parser.add_argument("--fill", default="median", choices=["median", "inpaint"])
    parser.add_argument("--thickness", type=int, default=14, help="Maskenbreite in px (Modellauflösung)")
    parser.add_argument("--layer", default="layer4", choices=["layer2", "layer3", "layer4"])
    args = parser.parse_args()
    T, N = args.lane, args.neighbor

    net, cfg = load_model()
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))
    rgb = to_model_rgb(load_gray(images[args.frame]), cfg.train_width, cfg.train_height)
    shape = rgb.shape[:2]

    # Referenz
    pred0 = predict(net, rgb)
    lanes0, bins0 = decode_row_lanes(pred0, cfg)
    k0, k1 = (int(s * cfg.num_row) for s in args.span)
    hidden = np.arange(k0, k1)
    hidden = hidden[~np.isnan(lanes0[T][hidden, 0])]
    if len(hidden) == 0 or np.isnan(lanes0[N][:, 0]).all():
        raise SystemExit("T hat im gewählten Bereich keine Punkte oder N wird nicht erkannt.")

    # Masken
    seg = np.full_like(lanes0[T], np.nan)
    seg[hidden] = lanes0[T][hidden]
    m_hidden = polyline_mask(seg, shape, args.thickness)
    m_t = polyline_mask(lanes0[T], shape, args.thickness)
    m_t_visible = m_t & ~m_hidden
    m_n = polyline_mask(lanes0[N], shape, args.thickness)
    m_ctrl = polyline_mask((lanes0[T] + lanes0[N]) / 2, shape, args.thickness) & ~m_t & ~m_n

    conditions = {
        "0": np.zeros(shape, bool),
        "A": m_hidden,
        "B": m_hidden | m_n,
        "C": m_n,
        "D": m_hidden | m_ctrl,
    }

    rows, preds, imgs = [], {}, {}
    for name, mask in conditions.items():
        img = fill(rgb, mask, args.fill)
        pred = pred0 if name == "0" else predict(net, img)
        lanes, _ = decode_row_lanes(pred, cfg)
        dx = np.abs(lanes[T][hidden, 0] - lanes0[T][hidden, 0])
        rows.append({
            "Bedingung": name,
            "dx_mittel_px": np.nanmean(dx) if not np.isnan(dx).all() else np.nan,
            "dx_max_px": np.nanmax(dx) if not np.isnan(dx).all() else np.nan,
            "Anker_verloren": int(np.isnan(dx).sum()),
            "P_exist_T": exist_prob(pred)[hidden, T].mean(),
        })
        preds[name], imgs[name] = lanes, img
    table = pd.DataFrame(rows).set_index("Bedingung")

    # CAMs für die verdeckten Anker von T, Ziel = Referenz-Bins
    target = RowBinTarget(T, hidden, bins0[hidden, T])
    regions = {"N": m_n, "T_sichtbar": m_t_visible, "T_verdeckt": m_hidden}
    cams, cam_rows = {}, []
    for img_name in ("0", "A"):
        x = to_tensor(imgs[img_name])
        for method in ("gradcam", "hirescam"):
            cam = compute_cam(net, x, target, method, args.layer)
            cams[(img_name, method)] = cam
            row = {"Bild": img_name, "Methode": method}
            for r, m in regions.items():
                # Verhältnis > 1: mehr CAM-Masse als die Fläche allein erwarten lässt
                row[f"{r}_Anteil"] = energy_share(cam, m)
                row[f"{r}_vs_Flaeche"] = energy_share(cam, m) / m.mean()
            cam_rows.append(row)
    cam_table = pd.DataFrame(cam_rows).set_index(["Bild", "Methode"])

    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 3)
    print(f"Bild {args.frame} ({images[args.frame].name}), T={LANE_NAMES[T]}, N={LANE_NAMES[N]}, "
          f"verdeckte Anker {hidden[0]}–{hidden[-1]} ({len(hidden)}), Füllung {args.fill}")
    print("\nVerdeckung (Abweichung von T an den verdeckten Ankern zur Referenz):")
    print(table)
    print(f"\nCAM-Masse ({args.layer}) für die verdeckten Anker von T:")
    print(cam_table)

    # Abbildung
    out_dir = paths.RESULTS_DIR / "01_first_look"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"frame{args.frame:05d}_T{T}_N{N}_{args.fill}_{args.layer}"
    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5))
    for ax, (name, img) in zip(axes[0], imgs.items()):
        ax.imshow(img)
        for lane, pts in enumerate(preds[name]):
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=3, color=LANE_COLORS[lane])
        ax.plot(lanes0[T][hidden, 0], lanes0[T][hidden, 1], "w--", lw=1)
        r = table.loc[name]
        ax.set_title(f"{name}: dx={r.dx_mittel_px:.1f}px, P={r.P_exist_T:.2f}")
    for ax, ((img_name, method), cam) in zip(axes[1], cams.items()):
        ax.imshow(imgs[img_name])
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
    fig.suptitle(f"{images[args.frame].name} – T={LANE_NAMES[T]}, N={LANE_NAMES[N]}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=110)
    table.to_csv(out_dir / f"{stem}_verdeckung.csv")
    cam_table.to_csv(out_dir / f"{stem}_cam.csv")
    print(f"\nAbbildung und Tabellen in {out_dir}")


if __name__ == "__main__":
    main()
