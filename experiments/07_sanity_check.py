"""Sanity Check für HiResCAM: Hängt die Heatmap überhaupt vom trainierten Modell ab?

Idee (Adebayo et al. 2018, „Sanity Checks for Saliency Maps“): Man ersetzt die gelernten
Gewichte Schicht für Schicht durch Zufallswerte, von oben (Kopf) nach unten (erste Faltung).
Zeigt die Heatmap danach immer noch dasselbe, erklärt sie nur das Bild (z. B. „helle Striche“)
und nicht das, was das Modell gelernt hat. Wird sie zu Unsinn, hängt sie wirklich am Modell.

Reihenfolge (kaskadierend, jede Stufe behält die vorherigen Zufallsschichten):
  cls.3 (letzte FC) → cls.1 (erste FC) → pool (1×1-Faltung) → layer4 → layer3 → layer2 → layer1 → conv1/bn1

Gemessen wird für jedes Bild und jede Linie T (Ziel wie in experiment.py: die Gewinner-Bins des
trainierten Modells in den Ankern 35–65 %):
  spearman      Rangkorrelation zur Heatmap des trainierten Modells (1 = gleich, 0 = nichts gemeinsam)
  T_vs_Flaeche  Anteil der Heatmap auf der Linie T geteilt durch ihren Flächenanteil
                (trainiert deutlich > 1; ohne Bezug zum Modell sollte er Richtung 1 fallen)

Aufruf:  .venv/bin/python experiments/07_sanity_check.py [--model bev_2026] [--step 120] [--seeds 3]
Ergebnis: results/<modell>/07_sanity_check/sanity.csv, zusammenfassung.csv, kurve.png, beispiel.png
"""

import argparse
import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from xai_lane import paths
from xai_lane.cam import RowBinTarget, compute_cam, energy_share
from xai_lane.decode import decode_row_lanes
from xai_lane.experiment import default_layer, default_thickness, predict
from xai_lane.model_loader import load_model
from xai_lane.occlusion import polyline_mask
from xai_lane.preprocess import load_gray, to_model_rgb, to_tensor

LANE_NAMES = ["links", "Mitte", "rechts"]
# Stufe -> Module, die in dieser Stufe zusätzlich zufällig werden
STAGES = [
    ("trainiert", []),
    ("cls.3", ["cls.3"]),
    ("cls.1", ["cls.1"]),
    ("pool", ["pool"]),
    ("layer4", ["model.layer4"]),
    ("layer3", ["model.layer3"]),
    ("layer2", ["model.layer2"]),
    ("layer1", ["model.layer1"]),
    ("conv1", ["model.conv1", "model.bn1"]),
]


def randomize(net, module_names: list[str]) -> None:
    """Setzt alle Faltungen, FC- und BatchNorm-Schichten unterhalb der Module auf ihre
    Start-Initialisierung zurück (Faltung/FC: zufällig; BatchNorm: Mittelwert 0, Varianz 1)."""
    for name in module_names:
        for m in net.get_submodule(name).modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()


def targets_for(net, cfg, rgb, span=(0.35, 0.65)):
    """Pro Linie T: Ziel (Gewinner-Bins des trainierten Modells) und Maske der ganzen Linie."""
    lanes0, bins0 = decode_row_lanes(predict(net, rgb), cfg)
    k0, k1 = (int(s * cfg.num_row) for s in span)
    out = {}
    for T in range(len(lanes0)):
        anker = np.arange(k0, k1)
        anker = anker[~np.isnan(lanes0[T][anker, 0])]
        if len(anker) == 0:
            continue
        mask = polyline_mask(lanes0[T], rgb.shape[:2], default_thickness(cfg))
        out[T] = (RowBinTarget(T, anker, bins0[anker, T]), mask)
    return out


def run(args, out_dir):
    """Für jedes Bild: Heatmap des trainierten Modells, dann Schicht für Schicht zufällig machen
    und jedes Mal die Heatmap neu rechnen und mit der trainierten vergleichen.

    Gibt die Rohdaten (auch als sanity.csv), die Karten für die Beispielabbildung, das erste Bild
    und die Config zurück.
    """
    trained, cfg = load_model(args.model)
    layer = default_layer(cfg)
    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))[:: args.step]
    rows, example = [], {}
    for i, path in enumerate(images):
        rgb = to_model_rgb(load_gray(path), cfg.train_width, cfg.train_height)
        x = to_tensor(rgb)
        targets = targets_for(trained, cfg, rgb)
        ref = {T: compute_cam(trained, x, tgt, "hirescam", layer) for T, (tgt, _) in targets.items()}

        def record(stage, seed, T, cam, mask):
            """Vergleicht eine Heatmap mit der trainierten und hängt das Ergebnis als Zeile an."""
            # Eine konstante Karte (alles 0) hat keine Rangfolge -> Korrelation 0
            rho = spearmanr(ref[T].ravel(), cam.ravel())[0] if cam.std() > 0 else 0.0
            rows.append({"Bild": path.name, "seed": seed, "Stufe": stage, "T": T,
                         "spearman": rho, "T_vs_Flaeche": energy_share(cam, mask) / mask.mean()})
            if i == 0 and seed == 0 and T == 1:
                example[stage] = cam

        for T, (_, mask) in targets.items():
            record("trainiert", 0, T, ref[T], mask)
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            net = copy.deepcopy(trained)  # Kopie, damit das trainierte Modell für das nächste Bild heil bleibt
            for stage, modules in STAGES[1:]:
                randomize(net, modules)
                for T, (tgt, mask) in targets.items():
                    record(stage, seed, T, compute_cam(net, x, tgt, "hirescam", layer), mask)
        print(f"{i + 1}/{len(images)} {path.name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sanity.csv", index=False)
    return df, example, images[0], cfg


def summarize(df: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Median und 25–75 %-Bereich je Stufe, in der Reihenfolge von STAGES."""
    order = [s for s, _ in STAGES]
    s = (df.groupby("Stufe")
         .agg(n=("spearman", "size"), spearman_median=("spearman", "median"),
              spearman_q25=("spearman", lambda v: v.quantile(0.25)), spearman_q75=("spearman", lambda v: v.quantile(0.75)),
              T_vs_Flaeche_median=("T_vs_Flaeche", "median"))
         .reindex(order))
    s.to_csv(out_dir / "zusammenfassung.csv")
    return s


def plot_curve(s: pd.DataFrame, out) -> None:
    """Die Kurve für die Folie: Ähnlichkeit (blau, linke Achse) und Fokus auf T (orange, rechte Achse) je Stufe."""
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    x = range(len(s))
    ax1.fill_between(x, s.spearman_q25, s.spearman_q75, color="#2a78d6", alpha=0.2, linewidth=0)
    ax1.plot(x, s.spearman_median, "o-", color="#2a78d6", label="Ähnlichkeit zur trainierten Karte (Spearman)")
    ax1.set_ylim(-0.2, 1.05)
    ax1.set_ylabel("Spearman (Median, Band = 25–75 %)", color="#52514e")
    ax2 = ax1.twinx()
    ax2.plot(x, s.T_vs_Flaeche_median, "s--", color="#eb6834", label="Heatmap auf der eigenen Linie (vs. Fläche)")
    ax2.axhline(1, color="#eb6834", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Anteil auf Linie T / Flächenanteil", color="#52514e")
    ax2.set_ylim(bottom=0)
    ax1.set_xticks(x, ["trainiert"] + [f"+ {st}" for st in s.index[1:]], rotation=30, ha="right")
    ax1.set_xlabel("Schichten von oben nach unten zufällig gemacht", color="#52514e")
    ax1.set_title("Sanity Check HiResCAM (Adebayo et al. 2018)", color="#0b0b0b", loc="left")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.tick_params(colors="#52514e", length=0)
    ax1.grid(axis="y", color="#e4e3df", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def plot_example(example: dict, image_path, cfg, out) -> None:
    """Die Heatmaps eines Bildes nebeneinander, von trainiert bis alles zufällig."""
    gray = to_model_rgb(load_gray(image_path), cfg.train_width, cfg.train_height)[..., 0]
    fig, axes = plt.subplots(1, len(example), figsize=(2.6 * len(example), 2.6))
    for ax, (stage, cam) in zip(axes, example.items()):
        ax.imshow(gray, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.set_title("trainiert" if stage == "trainiert" else f"+ {stage} zufällig", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{image_path.name}, Ziel: Mittellinie", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=120)


def main() -> None:
    """Argumente lesen, run starten, Statistik ausgeben und beide Abbildungen speichern."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=paths.DEFAULT_MODEL, choices=sorted(paths.MODELS))
    parser.add_argument("--step", type=int, default=120, help="jedes step-te Testbild")
    parser.add_argument("--seeds", type=int, default=3, help="Anzahl Zufallsinitialisierungen")
    args = parser.parse_args()
    out_dir = paths.results_dir(args.model) / "07_sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    df, example, first, cfg = run(args, out_dir)
    s = summarize(df, out_dir)
    print(s.round(3).to_string())
    plot_curve(s, out_dir / "kurve.png")
    plot_example(example, first, cfg, out_dir / "beispiel.png")
    print(f"Ergebnis in {out_dir}")


if __name__ == "__main__":
    main()
