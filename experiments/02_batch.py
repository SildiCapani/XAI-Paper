"""Hauptexperiment über viele Testbilder und alle Linienpaare, mit Statistik.

Läuft run_frame (wie 01_first_look.py) für jedes step-te Bild und jedes Paar (T, N)
benachbarter Linien. Aufeinanderfolgende Frames sind fast gleich, deshalb wird ausgedünnt.
Ergebnisse:
  results/<modell>/02_batch/verdeckung.csv  eine Zeile je Bild × Paar × Bedingung (0, E, F)
  results/<modell>/02_batch/cam.csv         eine Zeile je Bild × Paar × Heatmap-Bild (0, E)
  results/<modell>/02_batch/zusammenfassung.csv  Statistik je Paar (auch im Terminal)

Statistik (gepaart über die Bilder, einseitiger Wilcoxon-Vorzeichen-Rang-Test):
  F − E > 0 ?               Fehlt T ganz, wird es noch schlechter, wenn auch N fehlt?
  Heatmap auf N: E > 0 ?    Springt die Heatmap auf N, wenn T fehlt?
  Spearman(Heatmap auf N in E, F − E): Passt die Erklärung zum Experiment?

Aufruf:  .venv/bin/python experiments/02_batch.py --step 20 [--model bev_2026]
         .venv/bin/python experiments/02_batch.py --step 20 --resume   (abgebrochenen Lauf fortsetzen)
         .venv/bin/python experiments/02_batch.py --summary-only
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from xai_lane import paths
from xai_lane.experiment import SkipFrame, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]  # (T, N): nur direkt benachbarte Linien
LANE_NAMES = ["links", "Mitte", "rechts"]


def run(args, out_dir) -> None:
    """Rechnet das Experiment für viele Bilder und speichert die Rohdaten als CSV.

    Für jedes Bild × Linienpaar wird run_frame aufgerufen. Jede Bedingung ergibt
    eine Zeile in verdeckung.csv, jede Heatmap eine Zeile in cam.csv. Bilder, in denen
    T oder N nicht erkannt wird, werden übersprungen (SkipFrame).
    """
    net, cfg = load_model(args.model)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[:: args.step]
    occ_rows, cam_rows, skipped = [], [], 0
    out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and (out_dir / "verdeckung.csv").exists():
        # gespeichert wird nur nach vollständigen Bildern, also sind diese komplett
        occ_rows = pd.read_csv(out_dir / "verdeckung.csv").to_dict("records")
        cam_rows = pd.read_csv(out_dir / "cam.csv").to_dict("records")
        done = {r["Bild"] for r in occ_rows}
        print(f"Fortsetzen: {len(done)} Bilder schon fertig")
    t0 = time.time()
    for i, path in enumerate(images):
        if path.name in done:
            continue
        rgb = to_model_rgb(load_gray(path), cfg.train_width, cfg.train_height)
        for T, N in PAIRS:
            key = {"Bild": path.name, "T": T, "N": N}
            try:
                # Heatmap auf 0 und E: reicht für "springt die Heatmap auf N, wenn T fehlt?"
                res = run_frame(net, cfg, rgb, T, N, args.span, args.fill, args.thickness, args.layer,
                                cam_images=("0", "E"))
            except SkipFrame:
                skipped += 1
                continue
            occ_rows += [{**key, "Bedingung": c, **r} for c, r in res.table.to_dict("index").items()]
            cam_rows += [{**key, "CAM_Bild": b, "Methode": m, **r}
                         for (b, m), r in res.cam_table.to_dict("index").items()]
        # nach jedem Bild speichern, damit ein abgebrochener Lauf nicht alles verliert
        pd.DataFrame(occ_rows).to_csv(out_dir / "verdeckung.csv", index=False)
        pd.DataFrame(cam_rows).to_csv(out_dir / "cam.csv", index=False)
        print(f"[{i + 1}/{len(images)}] {path.name}  ({time.time() - t0:.0f} s)", flush=True)

    print(f"{len(images)} Bilder, {skipped} Kombinationen übersprungen (Linie nicht erkannt)")


def one_sided_p(v: pd.Series) -> float:
    """Wilcoxon-Test: Sind die Differenzen über alle Bilder systematisch größer als 0?
    Er vergleicht nur Ränge, deshalb stören einzelne Ausreißer kaum."""
    return wilcoxon(v, alternative="greater").pvalue if (v != 0).any() else np.nan


def summarize(out_dir) -> pd.DataFrame:
    """Macht aus den Rohdaten eine Statistik: eine Zeile pro Linienpaar."""
    occ = pd.read_csv(out_dir / "verdeckung.csv")
    cam = pd.read_csv(out_dir / "cam.csv")
    cam = cam[cam.Methode == "hirescam"]
    # pivot_table dreht die Tabelle: eine Zeile pro Bild, eine Spalte pro Bedingung (0, E, F).
    # So kann man F − E Bild für Bild ausrechnen (gepaarter Vergleich).
    idx = ["T", "N", "Bild"]
    dx = occ.pivot_table(index=idx, columns="Bedingung", values="dx_mittel_px")
    pct = occ.pivot_table(index=idx, columns="Bedingung", values="dx_mittel_pct")
    p_exist = occ.pivot_table(index=idx, columns="Bedingung", values="P_exist_T")
    cam_n = cam.pivot_table(index=idx, columns="CAM_Bild", values="N_vs_Flaeche")
    cam_n_anteil = cam.pivot_table(index=idx, columns="CAM_Bild", values="N_Anteil")

    rows = []
    for (T, N), g in dx.groupby(level=["T", "N"]):
        row = {"T": T, "N": N, "n_Bilder": len(g)}
        # 1. Verschiebung: Fehlt T, wie weit liegt die Vorhersage daneben? Mit und ohne N?
        for c in ("E", "F"):
            row[f"dx_{c}_median"] = g[c].median()
            row[f"dx_{c}_median_pct"] = pct.loc[(T, N), c].median()
        v = (g["F"] - g["E"]).dropna()
        row["F_minus_E_median"] = v.median()
        row["F_minus_E_anteil_pos"] = (v > 0).mean()
        row["F_minus_E_p"] = one_sided_p(v)
        # 2. Sicherheit: Meldet das Modell T noch als vorhanden, obwohl T weg ist?
        for c in ("E", "F"):
            row[f"P_exist_{c}_mittel"] = p_exist.loc[(T, N), c].mean()
            row[f"P_exist_{c}_median"] = p_exist.loc[(T, N), c].median()
        # 3. Heatmap: Springt sie auf N, wenn T fehlt? (Bild E gegen Original 0)
        h = cam_n.loc[(T, N)].dropna()
        row["cam_N_0_median"] = h["0"].median()
        row["cam_N_E_median"] = h["E"].median()
        d = h["E"] - h["0"]
        row["cam_N_E_groesser_anteil"] = (d > 0).mean()
        row["cam_N_E_groesser_p"] = one_sided_p(d)
        # 4. Passt die Heatmap zum Experiment? Je mehr Heatmap in E auf N, desto größer F − E?
        # Spearman: +1 = gleiche Rangfolge (Erklärung passt), 0 = kein Zusammenhang
        both = pd.concat([cam_n_anteil.loc[(T, N), "E"], v.droplevel(["T", "N"])], axis=1, join="inner").dropna()
        row["cam_N_E_spearman_F_minus_E"] = spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "zusammenfassung.csv", index=False)
    return summary


def main() -> None:
    """Liest die Argumente, startet run (außer bei --summary-only) und gibt die Statistik lesbar aus."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=20, help="jedes step-te Testbild verwenden")
    parser.add_argument("--fill", default="inpaint", choices=["median", "inpaint", "blendung"])
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--thickness", type=int, help="Maskenbreite in px, Standard skaliert mit der Bildbreite")
    parser.add_argument("--layer", choices=["layer2", "layer3", "layer4"],
                        help="CAM-Schicht, Standard: layer4 bei 512 px, layer3 bei 128 px")
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    parser.add_argument("--out", default="02_batch", help="Unterordner in results/<modell>/")
    parser.add_argument("--summary-only", action="store_true", help="nur vorhandene CSVs auswerten")
    parser.add_argument("--resume", action="store_true", help="bereits gerechnete Bilder überspringen")
    args = parser.parse_args()

    out_dir = paths.results_dir(args.model) / args.out
    if not args.summary_only:
        run(args, out_dir)
    summary = summarize(out_dir)
    for _, r in summary.iterrows():
        print(f"\n== T={LANE_NAMES[int(r['T'])]}, N={LANE_NAMES[int(r.N)]}, {int(r.n_Bilder)} Bilder ==")
        print(f"  dx-Median [% Breite]: E (T weg) {r.dx_E_median_pct:.2f} | F (+ N weg) {r.dx_F_median_pct:.2f} | "
              f"F−E bei {r.F_minus_E_anteil_pos:.0%} der Bilder > 0, p = {r.F_minus_E_p:.3g}")
        print(f"  P(T da): E {r.P_exist_E_mittel:.3f} (Median {r.P_exist_E_median:.3f}) | "
              f"F {r.P_exist_F_mittel:.3f} (Median {r.P_exist_F_median:.3f})")
        print(f"  Heatmap auf N (vs. Fläche): Original {r.cam_N_0_median:.2f} -> T weg {r.cam_N_E_median:.2f}, "
              f"bei {r.cam_N_E_groesser_anteil:.0%} der Bilder gestiegen, p = {r.cam_N_E_groesser_p:.3g} | "
              f"Spearman mit F−E: {r.cam_N_E_spearman_F_minus_E:+.2f}")
    print(f"\nCSV-Dateien in {out_dir}")


if __name__ == "__main__":
    main()
