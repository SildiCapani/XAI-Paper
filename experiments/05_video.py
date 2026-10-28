"""Verdeckt eine Linie in allen Bildern einer Fahrt und schreibt das Ergebnis als Video.

Für jedes Bild läuft run_frame (dieselben Masken wie in 01/02), nur ohne CAM.
Links das Originalbild mit der Referenzvorhersage, rechts das verdeckte Bild
(Bedingung --condition) mit der neuen Vorhersage; weiße Kreuze = Referenz von T.
Oben rechts: Verschiebung von T im verdeckten Stück und P(Linie da).

Aufruf:  .venv/bin/python experiments/05_video.py --model bev_2026 --lane 1 --neighbor 2 --condition B
         .venv/bin/python experiments/05_video.py --start 0 --end 300 --span 0.0 0.65 --fill blendung
Ergebnis: results/<modell>/05_video/T{T}_N{N}_{Bedingung}_{fill}[_span..].mp4 (+ CSV je Bild)
"""

import argparse

import cv2
import numpy as np
import pandas as pd

from xai_lane import paths
from xai_lane.experiment import SkipFrame, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

LANE_NAMES = ["links", "Mitte", "rechts"]
LANE_COLORS = [(180, 119, 31), (44, 160, 44), (40, 39, 214)]  # BGR: tab:blue, tab:green, tab:red
OUT_WIDTH = 768  # jede Hälfte wird auf diese Breite hochskaliert


def draw(rgb, lanes, scale, mask=None, ref=None, T=None) -> np.ndarray:
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    if mask is not None:
        big = cv2.resize(mask.astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 255), 1)
    if ref is not None:
        for x, y in ref[T]:
            if not np.isnan(x):
                cv2.drawMarker(img, (int(x * scale), int(y * scale)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 7, 1)
    for lane, pts in enumerate(lanes):
        for x, y in pts:
            if not np.isnan(x):
                cv2.circle(img, (int(x * scale), int(y * scale)), 3, LANE_COLORS[lane], -1)
    return img


def label(img, lines) -> None:
    for i, text in enumerate(lines):
        y = 22 + 22 * i
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lane", type=int, default=1, help="T: Linie, deren Stück verdeckt wird")
    parser.add_argument("--neighbor", type=int, default=2, help="N: Nachbarlinie")
    parser.add_argument("--condition", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--fill", default="inpaint", choices=["median", "inpaint", "blendung"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="letztes Bild (exklusiv), Standard: alle")
    parser.add_argument("--step", type=int, default=1, help="jedes step-te Bild")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    args = parser.parse_args()
    T, N, cond = args.lane, args.neighbor, args.condition

    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[args.start:args.end:args.step]
    scale = OUT_WIDTH / cfg.train_width
    size = (2 * OUT_WIDTH, int(cfg.train_height * scale))

    out_dir = paths.results_dir(args.model) / "05_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    span = "" if tuple(args.span) == (0.35, 0.65) else f"_span{args.span[0] * 100:.0f}-{args.span[1] * 100:.0f}"
    stem = f"T{T}_N{N}_{cond}_{args.fill}{span}"
    writer = cv2.VideoWriter(str(out_dir / f"{stem}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)

    rows = []
    for i, path in enumerate(images):
        rgb = to_model_rgb(load_gray(path), cfg.train_width, cfg.train_height)
        try:
            res = run_frame(net, cfg, rgb, T, N, args.span, args.fill, cam_images=(), conditions=("0", cond))
        except SkipFrame:
            left = right = draw(rgb, [], scale)
            label(right, [path.name, f"{LANE_NAMES[T]} oder {LANE_NAMES[N]} nicht erkannt"])
        else:
            r = res.table.loc[cond]
            rows.append({"Bild": path.name, **r.to_dict()})
            mask = res.masks["T_verdeckt"] if cond != "C" else res.masks["N"]
            if cond == "B":
                mask = mask | res.masks["N"]
            elif cond == "D":
                mask = mask | res.masks["Kontrolle"]
            left = draw(rgb, res.lanes0, scale)
            label(left, [path.name, "Original + Vorhersage"])
            right = draw(res.imgs[cond], res.preds[cond], scale, mask, res.lanes0, T)
            label(right, [f"Bedingung {cond}: T = {LANE_NAMES[T]}, N = {LANE_NAMES[N]}",
                          f"T verschoben {r.dx_mittel_px:.1f} px "
                          f"({r.dx_mittel_pct:.2f} % Breite), P = {r.P_exist_T:.2f}"])
        writer.write(np.hstack([left, right]))
        if i % 50 == 0:
            print(f"[{i + 1}/{len(images)}] {path.name}", flush=True)
    writer.release()

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    print(f"{len(table)} von {len(images)} Bildern ausgewertet, Median-Verschiebung "
          f"{table.dx_mittel_px.median():.2f} px ({table.dx_mittel_pct.median():.2f} % Breite)")
    print(f"Video: {out_dir / stem}.mp4")


if __name__ == "__main__":
    main()
