# XAI-Paper: Nutzt die Spurerkennung benachbarte Markierungen?

Paper für das Explainable-AI-Modul. Untersucht wird die UFLDv2-Spurerkennung des
DHBW-Smart-Rollerz-Teams (BEV-Modell 2026, ResNet-„9“, Eingabe 512×384):
Nutzt das Modell benachbarte Fahrbahnmarkierungen, um eine teilweise verdeckte
Linie zu erkennen?

Methoden: gezielte Verdeckungsexperimente (Bedingungen A–F) und Grad-CAM vs.
HiResCAM, danach Abgleich der Erklärungen mit den Experimenten. Kein Neutraining.

## Struktur

| Ordner | Inhalt |
|---|---|
| `src/xai_lane/` | eigener Code (Loader, Vorverarbeitung, Verdeckung, CAM, Analyse) |
| `experiments/` | ausführbare Skripte/Notebooks, nummeriert nach Schritt |
| `results/` | erzeugte Ausgaben (nicht im git, jederzeit reproduzierbar) |
| `paper/` | LaTeX-Paper (Deutsch), `paper/figures/` für ausgewählte Abbildungen |

Modell, Config, Testbilder und Modellcode werden **nur lesend** per absolutem Pfad
aus `~/Documents/smarty_workspace/src/lane_detection_ai` verwendet
(siehe `src/xai_lane/paths.py`, überschreibbar mit `LANE_DETECTION_ROOT`).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python experiments/00_check_setup.py
```

## Offene Punkte (mit dem Team klären)

- Config nennt Originalgröße 1364×944, die Testbilder sind 800×640.
- `row_anchor` in `get_config` (`linspace(100, 1540, num_row) / 1550`) ist evtl.
  nicht die Formel aus dem Training – wichtig für die Zuordnung Anker-Zeile ↔ Bildzeile.
- Trainingscode (Lightning/Optuna) und Labels liegen nicht im Repo.
