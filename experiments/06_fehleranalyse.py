"""Fehlerarten bei Verdeckung über viele Bilder (ohne CAM, deshalb schneller als 02_batch).

Für jedes step-te Bild und die Paare Mitte ← rechts und links ← Mitte laufen alle
Bedingungen A–D (wie in 01/02). Zusätzlich zu dx wird je Bedingung gemessen:
  sprung_anteil  Anteil der verdeckten Anker, an denen T näher an einer anderen
                 Referenzlinie liegt als an der eigenen (Linien "rücken nach")
  zickzack       mittlere 2. Differenz von x im verdeckten Stück (Referenz: zickzack_ref)
  N_dx           mittlere Verschiebung der Nachbarlinie N
  P_exist_N      Sicherheit, dass N existiert, obwohl N übermalt ist (nur B, C)

Aufruf:  .venv/bin/python experiments/06_fehleranalyse.py [--model bev_2026] [--step 2] [--end 700]
Ergebnis: results/<modell>/06_fehleranalyse/fehler.csv und zusammenfassung.csv
"""

import argparse
import time

import numpy as np
import pandas as pd

from xai_lane import paths
from xai_lane.decode import exist_prob
from xai_lane.experiment import SkipFrame, predict, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

PAIRS = [(1, 2), (0, 1)]  # (T, N)


def frame_rows(net, cfg, rgb, name, fill_method):
    """Rechnet für ein Bild die Bedingungen A–D und misst zusätzlich die Fehlerarten.

    Gibt eine Liste von Zeilen zurück (eine pro Linienpaar × Bedingung). Neben dx aus run_frame:
    Rückt T auf eine andere Linie? Wird T zackig? Wandert N mit? Glaubt das Modell noch an N?
    """
    rows = []
    for T, N in PAIRS:
        try:
            r = run_frame(net, cfg, rgb, T, N, (0.35, 0.65), fill_method, cam_images=())
        except SkipFrame:
            continue
        h, ref = r.hidden, r.lanes0
        others = [k for k in range(len(ref)) if k != T]
        for c in "ABCD":
            lanes = r.preds[c]
            xt = lanes[T][h, 0]
            # Abstand jedes neuen T-Punkts zur alten Position von T und zur nächsten anderen Linie.
            # Liegt er näher an einer anderen Linie, ist T dorthin "nachgerückt".
            d_self = np.abs(xt - ref[T][h, 0])
            d_other = np.nanmin(np.stack([np.abs(xt - ref[k][h, 0]) for k in others]), axis=0)
            dn = np.abs(lanes[N][:, 0] - ref[N][:, 0])
            row = {"Bild": name, "T": T, "N": N, "Bedingung": c, **r.table.loc[c].to_dict(),
                   "sprung_anteil": np.mean(d_other < d_self) if not np.isnan(xt).all() else np.nan,
                   # 2. Differenz: wie stark die Richtung von Punkt zu Punkt wechselt (gerade Linie ≈ 0)
                   "zickzack": np.nanmean(np.abs(np.diff(xt, 2))) if len(h) > 2 else np.nan,
                   "zickzack_ref": np.nanmean(np.abs(np.diff(ref[T][h, 0], 2))) if len(h) > 2 else np.nan,
                   "N_dx": np.nanmean(dn) if not np.isnan(dn).all() else np.nan}
            if c in "BC":
                known = ~np.isnan(ref[N][:, 0])
                row["P_exist_N"] = exist_prob(predict(net, r.imgs[c]))[known, N].mean()
            rows.append(row)
    return rows


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Fasst alle Zeilen pro Linienpaar und Bedingung zusammen: Mediane und Anteile der Fehlerarten."""
    # Ja/Nein-Spalten: Mittelwert davon = Anteil der Bilder, in denen der Fehler auftritt
    df = df.assign(gross=df.dx_mittel_pct > 3, sehr_gross=df.dx_mittel_pct > 10,
                   sprung=df.sprung_anteil > 0.5, verloren=df.Anker_verloren > 0)
    g = df.groupby(["T", "N", "Bedingung"])
    return pd.DataFrame({
        "n": g.size(),
        "dx_median_pct": g.dx_mittel_pct.median(),
        "dx_p90_pct": g.dx_mittel_pct.quantile(0.9),
        "gross_>3%": g.gross.mean(),
        "sehr_gross_>10%": g.sehr_gross.mean(),
        "Linie_verloren": g.verloren.mean(),
        "nachgerueckt": g.sprung.mean(),
        "zickzack": g.zickzack.median(),
        "zickzack_ref": g.zickzack_ref.median(),
        "P_T": g.P_exist_T.mean(),
        "P_N_obwohl_weg": g.P_exist_N.mean(),
    }).round(3)


def main() -> None:
    """Läuft frame_rows über alle gewählten Bilder, speichert zwischendurch und am Ende die Statistik."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    parser.add_argument("--end", type=int, help="letztes Bild (exklusiv), Standard: alle")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--fill", default="inpaint", choices=["median", "inpaint", "blendung"])
    args = parser.parse_args()

    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[: args.end : args.step]
    out_dir = paths.results_dir(args.model) / "06_fehleranalyse"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, path in enumerate(images):
        rgb = to_model_rgb(load_gray(path), cfg.train_width, cfg.train_height)
        rows += frame_rows(net, cfg, rgb, path.name, args.fill)
        if i % 25 == 0:
            pd.DataFrame(rows).to_csv(out_dir / "fehler.csv", index=False)
            print(f"[{i + 1}/{len(images)}] {path.name}  ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "fehler.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "zusammenfassung.csv")
    pd.set_option("display.width", 250)
    print(summary.to_string())
    print(f"\nCSV-Dateien in {out_dir}")


if __name__ == "__main__":
    main()
