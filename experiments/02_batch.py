"""Verdeckungsexperiment über viele Testbilder und alle Linienpaare, mit Statistik.

Läuft run_frame (wie 01_first_look.py) für jedes step-te Bild, jedes Paar (T, N)
benachbarter Linien und jedes Füllverfahren. Aufeinanderfolgende Frames sind fast
gleich, deshalb wird ausgedünnt. Ergebnisse:
  results/<modell>/02_batch/verdeckung.csv  eine Zeile je Bild × Paar × Füllung × Bedingung
  results/<modell>/02_batch/cam.csv         eine Zeile je Bild × Paar × Füllung × CAM-Bild × Methode
  results/<modell>/02_batch/zusammenfassung.csv  Statistik je Paar × Füllung (auch im Terminal)

Statistik (gepaart über die Bilder, Wilcoxon-Vorzeichen-Rang-Test):
  B − A > 0 ?   Fehlt N, wird die Lücke in T schlechter geschätzt?
  B − D > 0 ?   ... stärker als bei gleich großer Störung auf freier Straße?
  Spearman(CAM-Anteil auf N in Bild A, B − A): Passt die Erklärung zum Experiment?
Mit --ef zusätzlich (T ganz weg):
  F − E > 0 ?   Fehlt T ganz, wird es noch schlechter, wenn auch N fehlt?
  CAM auf N in E > in 0 ?   Springt die Heatmap auf N, wenn T fehlt?
  Spearman(CAM-Anteil auf N in Bild E, F − E): Passt die Erklärung zum Experiment?

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
from xai_lane.experiment import ALL_CONDITIONS, CONDITIONS, SkipFrame, run_frame
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, to_model_rgb

PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1)]  # (T, N): nur direkt benachbarte Linien


def run(args, out_dir) -> None:
    """Rechnet das Experiment für viele Bilder und speichert die Rohdaten als CSV.

    Für jedes Bild × Linienpaar × Füllung wird run_frame aufgerufen. Jede Bedingung ergibt
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
            for fill_method in args.fill:
                key = {"Bild": path.name, "T": T, "N": N, "Fuellung": fill_method}
                try:
                    res = run_frame(net, cfg, rgb, T, N, args.span, fill_method,
                                    args.thickness, args.layer,
                                    cam_images=("0", "A", "E") if args.ef else ("A",),
                                    conditions=ALL_CONDITIONS if args.ef else CONDITIONS)
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


def summarize(out_dir) -> pd.DataFrame:
    """Macht aus den Rohdaten eine Statistik: eine Zeile pro Linienpaar und Füllung.

    Enthält Mediane der Verschiebung je Bedingung, die Tests „B schlechter als A / D?“
    (und mit --ef „F schlechter als E?“) und den Abgleich Heatmap ↔ Experiment (Spearman).
    """
    occ = pd.read_csv(out_dir / "verdeckung.csv")
    cam = pd.read_csv(out_dir / "cam.csv")
    # pivot_table dreht die Tabelle: eine Zeile pro Bild, eine Spalte pro Bedingung (A, B, C, D, ...).
    # So kann man B − A Bild für Bild ausrechnen (gepaarter Vergleich).
    dx = occ.pivot_table(index=["Fuellung", "T", "N", "Bild"], columns="Bedingung", values="dx_mittel_px")
    occ_pct = occ.pivot_table(index=["Fuellung", "T", "N", "Bild"], columns="Bedingung", values="dx_mittel_pct")
    p_exist = occ.pivot_table(index=["Fuellung", "T", "N", "Bild"], columns="Bedingung", values="P_exist_T")
    dx["B_minus_A"] = dx["B"] - dx["A"]
    dx["B_minus_D"] = dx["B"] - dx["D"]

    rows = []
    for (fill_method, T, N), g in dx.groupby(level=["Fuellung", "T", "N"]):
        row = {"Fuellung": fill_method, "T": T, "N": N, "n_Bilder": len(g)}
        for c in ("A", "B", "C", "D"):
            row[f"dx_{c}_median"] = g[c].median()
        pct = occ_pct.loc[(fill_method, T, N)]
        for c in ("A", "B", "C", "D"):
            row[f"dx_{c}_median_pct"] = pct[c].median()
        for d in ("B_minus_A", "B_minus_D"):
            v = g[d].dropna()
            row[f"{d}_median"] = v.median()
            row[f"{d}_anteil_pos"] = (v > 0).mean()
            # Wilcoxon-Test: Sind die Differenzen über alle Bilder systematisch größer als 0?
            # Er vergleicht nur Ränge, deshalb stören einzelne Ausreißer kaum.
            # einseitig: Hypothese "ohne N wird es schlechter"
            row[f"{d}_p"] = wilcoxon(v, alternative="greater").pvalue if (v != 0).any() else np.nan
        pe = p_exist.loc[(fill_method, T, N)]
        row["P_exist_A_mittel"] = pe["A"].mean()
        row["P_exist_B_mittel"] = pe["B"].mean()
        for method in ("hirescam",):
            c = cam[(cam.Fuellung == fill_method) & (cam["T"] == T) & (cam.N == N)
                    & (cam.CAM_Bild == "A") & (cam.Methode == method)].set_index("Bild")
            row[f"{method}_N_vs_Flaeche"] = c["N_vs_Flaeche"].median()
            # Spearman: Haben Bilder mit viel Heatmap auf N auch einen großen Effekt B − A?
            # +1 = gleiche Rangfolge (Erklärung passt), 0 = kein Zusammenhang
            effect = g["B_minus_A"].droplevel(["Fuellung", "T", "N"])
            both = pd.concat([c["N_Anteil"], effect], axis=1, join="inner").dropna()
            row[f"{method}_spearman"] = spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
        if "E" in g.columns:
            # Fehlt T ganz: Wird T schlechter, wenn zusätzlich N fehlt?
            row["dx_E_median_pct"] = pct["E"].median()
            row["dx_F_median_pct"] = pct["F"].median()
            v = (g["F"] - g["E"]).dropna()
            row["F_minus_E_median"] = v.median()
            row["F_minus_E_anteil_pos"] = (v > 0).mean()
            row["F_minus_E_p"] = wilcoxon(v, alternative="greater").pvalue if (v != 0).any() else np.nan
            # Springt die Heatmap auf N, wenn T fehlt? (Bild E gegen Original 0)
            c = cam[(cam.Fuellung == fill_method) & (cam["T"] == T) & (cam.N == N) & (cam.Methode == "hirescam")]
            n0 = c[c.CAM_Bild == "0"].set_index("Bild")["N_vs_Flaeche"]
            nE = c[c.CAM_Bild == "E"].set_index("Bild")["N_vs_Flaeche"]
            both = pd.concat([n0, nE], axis=1, join="inner", keys=["0", "E"]).dropna()
            row["cam_N_0_median"] = both["0"].median()
            row["cam_N_E_median"] = both["E"].median()
            d = both["E"] - both["0"]
            row["cam_N_E_groesser_anteil"] = (d > 0).mean()
            row["cam_N_E_groesser_p"] = wilcoxon(d, alternative="greater").pvalue if (d != 0).any() else np.nan
            # Passt die Heatmap zum Experiment? Je mehr Karte in E auf N, desto größer F − E?
            effect = v.droplevel(["Fuellung", "T", "N"])
            nE_anteil = c[c.CAM_Bild == "E"].set_index("Bild")["N_Anteil"]
            both = pd.concat([nE_anteil, effect], axis=1, join="inner").dropna()
            row["cam_N_E_spearman_F_minus_E"] = spearmanr(both.iloc[:, 0], both.iloc[:, 1]).statistic
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "zusammenfassung.csv", index=False)
    return summary


def main() -> None:
    """Liest die Argumente, startet run (außer bei --summary-only) und gibt die Statistik lesbar aus."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=int, default=20, help="jedes step-te Testbild verwenden")
    parser.add_argument("--fill", nargs="+", default=["median", "inpaint"], choices=["median", "inpaint", "blendung"])
    parser.add_argument("--span", type=float, nargs=2, default=(0.35, 0.65))
    parser.add_argument("--thickness", type=int, help="Maskenbreite in px, Standard skaliert mit der Bildbreite")
    parser.add_argument("--layer", choices=["layer2", "layer3", "layer4"],
                        help="CAM-Schicht, Standard: layer4 bei 512 px, layer3 bei 128 px")
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=list(paths.MODELS))
    parser.add_argument("--ef", action="store_true",
                        help="zusätzlich E (T ganz weg) und F (T + N ganz weg), CAM auf 0, A, E; "
                             "Ergebnisse in 02_batch_EF")
    parser.add_argument("--summary-only", action="store_true", help="nur vorhandene CSVs auswerten")
    parser.add_argument("--resume", action="store_true", help="bereits gerechnete Bilder überspringen")
    args = parser.parse_args()

    out_dir = paths.results_dir(args.model) / ("02_batch_EF" if args.ef else "02_batch")
    if not args.summary_only:
        run(args, out_dir)
    summary = summarize(out_dir)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.precision", 3)
    names = ["links", "Mitte", "rechts"]
    for _, r in summary.iterrows():
        print(f"\n== T={names[r['T']]}, N={names[r.N]}, Füllung {r.Fuellung}, {r.n_Bilder} Bilder ==")
        print(f"  dx-Median [px]   A {r.dx_A_median:.2f} | B {r.dx_B_median:.2f} | "
              f"C {r.dx_C_median:.2f} | D {r.dx_D_median:.2f}")
        print(f"  dx-Median [% Breite] A {r.dx_A_median_pct:.2f} | B {r.dx_B_median_pct:.2f} | "
              f"C {r.dx_C_median_pct:.2f} | D {r.dx_D_median_pct:.2f}")
        print(f"  B−A: Median {r.B_minus_A_median:+.2f} px, bei {r.B_minus_A_anteil_pos:.0%} der Bilder > 0, "
              f"p = {r.B_minus_A_p:.3g}")
        print(f"  B−D: Median {r.B_minus_D_median:+.2f} px, bei {r.B_minus_D_anteil_pos:.0%} der Bilder > 0, "
              f"p = {r.B_minus_D_p:.3g}")
        print(f"  P(exist) A {r.P_exist_A_mittel:.3f} | B {r.P_exist_B_mittel:.3f}")
        print(f"  HiResCAM auf N (vs. Fläche): {r.hirescam_N_vs_Flaeche:.2f}"
              f" | Spearman mit B−A: {r.hirescam_spearman:+.2f}")
        if "F_minus_E_p" in summary.columns:
            print(f"  T ganz weg [% Breite]: E {r.dx_E_median_pct:.2f} | F (+ N weg) {r.dx_F_median_pct:.2f} | "
                  f"F−E bei {r.F_minus_E_anteil_pos:.0%} der Bilder > 0, p = {r.F_minus_E_p:.3g}")
            print(f"  HiResCAM auf N (vs. Fläche): Original {r.cam_N_0_median:.2f} -> T weg {r.cam_N_E_median:.2f}, "
                  f"bei {r.cam_N_E_groesser_anteil:.0%} der Bilder gestiegen, p = {r.cam_N_E_groesser_p:.3g} | "
                  f"Spearman mit F−E: {r.cam_N_E_spearman_F_minus_E:+.2f}")
    print(f"\nCSV-Dateien in {out_dir}")


if __name__ == "__main__":
    main()
