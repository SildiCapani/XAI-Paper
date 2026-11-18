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


MODEL_TITLES = {"bev": "altes Modell bev (128×96)", "bev_2026": "2026er-Modell bev_2026 (512×384)"}


def _batch_bars(ax, model) -> None:
    """Zeichnet das Balkendiagramm (A–D je Linienpaar) eines Modells in die Achse ax.

    Liest dafür die zusammenfassung.csv aus 02_batch. Wird von batch_chart und vergleich_chart benutzt.
    """
    s = pd.read_csv(paths.results_dir(model) / "02_batch" / "zusammenfassung.csv")
    s = s[s.Fuellung == "inpaint"].reset_index(drop=True)
    groups = [f"{LANE_NAMES[t]} ← {LANE_NAMES[n]}" for t, n in zip(s["T"], s.N)]
    width = 0.2
    for i, c in enumerate("ABCD"):
        x = [g + (i - 1.5) * (width + 0.01) for g in range(len(s))]
        ax.bar(x, s[f"dx_{c}_median_pct"], width, color=COLORS[c], label=LABELS[c], edgecolor="#fcfcfb", linewidth=2)
    ax.set_xticks(range(len(s)), groups)
    ax.set_xlabel("Ziellinie T ← weggenommener Nachbar N", color="#52514e")
    ax.set_title(f"{MODEL_TITLES[model]}, {s.n_Bilder.min()}–{s.n_Bilder.max()} Bilder je Paar",
                 color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    ax.set_facecolor("#fcfcfb")


def batch_chart(out, model="bev") -> None:
    """Balkendiagramm für ein einzelnes Modell, gespeichert unter out."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _batch_bars(ax, model)
    ax.set_ylabel("Verschiebung von T im gestörten Stück\n(Median, % der Bildbreite)", color="#52514e")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def vergleich_chart(out) -> None:
    """Beide Modelle nebeneinander mit gleicher y-Achse, damit die Größenordnung vergleichbar ist."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, model in zip(axes, ("bev", "bev_2026")):
        _batch_bars(ax, model)
    axes[0].set_ylabel("Verschiebung von T im gestörten Stück\n(Median, % der Bildbreite)", color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center")
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out, dpi=120)


def main() -> None:
    """Ruft die Experiment-Skripte für die Beispiele in FIGURES auf, kopiert deren Bilder nach
    docs/bilder/ und zeichnet die Balkendiagramme neu."""
    DOCS.mkdir(parents=True, exist_ok=True)
    for script, args, produced, name in FIGURES:
        subprocess.run([PY, str(EXP / script), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.copy(paths.RESULTS_ROOT / produced, DOCS / name)
        print(f"  {name}")
    # Nahaufnahme der Masken in Bild 600 (einmalig erzeugt, siehe docs/ERKLAERUNG.md Abschnitt 7)
    masks = paths.results_dir("bev") / "erklaerung" / "frame00600_masken_links.png"
    if masks.exists():
        shutil.copy(masks, DOCS / "masken_frame600.png")
    batch_chart(DOCS / "batch_bev_inpaint.png", "bev")
    batch_chart(DOCS / "batch_bev_2026_inpaint.png", "bev_2026")
    vergleich_chart(DOCS / "vergleich_modelle_inpaint.png")
    print(f"Bilder in {DOCS}")


if __name__ == "__main__":
    main()
