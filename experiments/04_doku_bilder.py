"""Erzeugt die Bilder für docs/ERKLAERUNG.md und kopiert sie nach docs/bilder/.

results/ ist nicht im git, die Doku-Bilder sollen es sein. Vorher müssen
01_first_look.py, 03_schritte.py und 02_batch.py (Modell bev) gelaufen sein;
das Skript ruft sie für die verwendeten Beispiele selbst auf.

Aufruf:  .venv/bin/python experiments/04_doku_bilder.py
"""

import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from xai_lane import paths

DOCS = paths.PROJECT_ROOT / "docs" / "bilder"
PY = sys.executable
EXP = paths.PROJECT_ROOT / "experiments"
LANE_NAMES = ["links", "Mitte", "rechts"]
# Kategoriale Farben in fester Reihenfolge (validiert, Kontrast-Warnung -> Tabelle in der Doku)
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#eda100"}
LABELS = {"A": "A: Stück von T", "B": "B: + Nachbar weg", "C": "C: nur Nachbar weg", "D": "D: Kontrollstreifen"}

# (Skript, Argumente, erzeugte Datei relativ zu results/, Name in docs/bilder)
FIGURES = [
    ("03_schritte.py", ["--frame", "0", "--lane", "1", "--fill", "inpaint", "--model", "bev_2026"],
     "bev_2026/03_schritte/frame00000_T1_inpaint.png", "schritte_2026_inpaint.png"),
    ("03_schritte.py", ["--frame", "0", "--lane", "1", "--fill", "inpaint", "--model", "bev"],
     "bev/03_schritte/frame00000_T1_inpaint.png", "schritte_bev_inpaint.png"),
    ("03_schritte.py", ["--frame", "0", "--lane", "1", "--fill", "blendung", "--model", "bev"],
     "bev/03_schritte/frame00000_T1_blendung.png", "schritte_bev_blendung.png"),
    ("01_first_look.py", ["--frame", "0", "--lane", "1", "--neighbor", "2", "--fill", "inpaint", "--model", "bev_2026"],
     "bev_2026/01_first_look/frame00000_T1_N2_inpaint_layer4.png", "bedingungen_2026_frame0.png"),
    ("01_first_look.py", ["--frame", "600", "--lane", "0", "--neighbor", "1", "--fill", "inpaint", "--model", "bev"],
     "bev/01_first_look/frame00600_T0_N1_inpaint_layer3.png", "bedingungen_bev_frame600.png"),
]


def batch_chart(out) -> None:
    s = pd.read_csv(paths.results_dir("bev") / "02_batch" / "zusammenfassung.csv")
    s = s[s.Fuellung == "inpaint"].reset_index(drop=True)
    groups = [f"{LANE_NAMES[t]} ← {LANE_NAMES[n]}" for t, n in zip(s["T"], s.N)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.2
    for i, c in enumerate("ABCD"):
        x = [g + (i - 1.5) * (width + 0.01) for g in range(len(s))]
        ax.bar(x, s[f"dx_{c}_median_pct"], width, color=COLORS[c], label=LABELS[c], edgecolor="#fcfcfb", linewidth=2)
    ax.set_xticks(range(len(s)), [f"T = {g}" for g in groups])
    ax.set_ylabel("Verschiebung von T im gestörten Stück\n(Median, % der Bildbreite)", color="#52514e")
    ax.set_title("best_model_bev.pth, 60 Bilder, Füllung inpaint", color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for script, args, produced, name in FIGURES:
        subprocess.run([PY, str(EXP / script), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(paths.RESULTS_ROOT / produced, DOCS / name)
        print(f"  {name}")
    # Nahaufnahme der Masken in Bild 600 (einmalig erzeugt, siehe docs/ERKLAERUNG.md Abschnitt 7)
    masks = paths.results_dir("bev") / "erklaerung" / "frame00600_masken_links.png"
    if masks.exists():
        shutil.copy(masks, DOCS / "masken_frame600.png")
    batch_chart(DOCS / "batch_bev_inpaint.png")
    print(f"Bilder in {DOCS}")


if __name__ == "__main__":
    main()
