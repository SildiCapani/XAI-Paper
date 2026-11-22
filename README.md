# XAI-Projekt: Nutzt die Spurerkennung benachbarte Markierungen?

Projekt für das Explainable-AI-Modul (Abgabe: Präsentation + Code). Untersucht wird die
UFLDv2-Spurerkennung des DHBW-Smart-Rollerz-Teams (ResNet-„9“, `best_model_bev_2026.pth`,
Eingabe 512×384, drei Linien: links, Mitte, rechts).

## Gruppe

| Name | Matr.-Nr. | E-Mail |
|---|---|---|
| Brisild Capani | 1962455 | inf24039@lehre.dhbw-stuttgart.de |

- GitHub: https://github.com/SildiCapani/XAI-Paper
- Datensatz: Testfahrt `eval_2025-03-06` (1192 BEV-Bilder) und Modellgewichte des Smart-Rollerz-Teams.
  Sie liegen auf dem NAS des Teams und sind zu groß für einen öffentlichen Link. Zugang bitte per
  E-Mail an inf24039@lehre.dhbw-stuttgart.de anfragen, dann stelle ich sie bereit.

**Frage:** Fehlt eine Fahrbahnlinie ganz, orientiert sich das Modell dann an der Nachbarlinie?

**Vorgehen:** Für eine Ziellinie T und eine Nachbarlinie N wird die Linie im Bild weggenommen
(Inpainting entlang der vorhergesagten Linie) und neu vorhergesagt:

| Bedingung | Was fehlt |
|---|---|
| 0 | nichts (Referenz) |
| E | T ganz |
| F | T und N ganz |

Gemessen wird an den mittleren Zeilenankern (35–65 %, beim 2026er-Modell Anker 19–34), wie weit
sich T verschiebt. Ist F deutlich schlechter als E, hat das Modell N benutzt. Dazu zeigt HiResCAM,
welche Bildstellen T an der alten Stelle halten, und es wird geprüft, ob die Heatmap pro Bild zum
gemessenen Effekt passt (Spearman). Grad-CAM wurde anfangs mitverglichen und verworfen: Es mittelt
den Gradienten über alle Stellen, beim vollverbundenen Kopf von UFLDv2 ist der Gradient aber je
Stelle verschieden. Kein Neutraining.

## Struktur

| Ordner | Inhalt |
|---|---|
| `src/xai_lane/` | eigener Code: Pfade, Modell laden, Vorverarbeitung, Dekodieren, Verdecken, HiResCAM, Experiment (`run_frame`) |
| `experiments/` | ausführbare Skripte, nummeriert nach Schritt |
| `docs/` | Folienbilder in `docs/bilder/praesi/` |
| `results/<modell>/` | erzeugte Ausgaben (nicht im git, jederzeit reproduzierbar) |

## Skripte

| Skript | Was es macht |
|---|---|
| `00_check_setup.py` | Modell lädt mit `strict=True`, Ausgabeformen, Gradienten an layer4 |
| `01_first_look.py` | ein Bild: Bedingungen 0, E, F und Heatmaps |
| `02_batch.py` | Hauptexperiment über viele Bilder und alle Linienpaare, Wilcoxon-Test und Spearman |
| `07_sanity_check.py` | Sanity Check nach Adebayo et al. 2018: Gewichte schrittweise zufällig, Heatmap muss zerfallen |
| `08_praesi_bilder.py` | Bilder für die Folien (braucht vorher einen Lauf von `02_batch.py`) |

Häufige Optionen: `--model bev_2026|bev`, `--lane` / `--neighbor` (0 links, 1 Mitte, 2 rechts),
`--fill inpaint|median|blendung`. `02_batch.py --step 20` dauert mit `bev_2026` auf der CPU etwa
eine Stunde und lässt sich mit `--resume` fortsetzen.

## Modell und Daten

Modell, Config, Testbilder und Modellcode gehören dem Team und sind **nicht** in diesem Repo. Sie
werden nur lesend aus `~/Documents/smarty_workspace/src/lane_detection_ai` verwendet
(siehe `src/xai_lane/paths.py`, überschreibbar mit `LANE_DETECTION_ROOT`).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
.venv/bin/python experiments/00_check_setup.py
```

## Grenzen

- Keine Labels: Die Referenz ist die Vorhersage auf dem Originalbild. Gemessen wird, wie stark sich
  das Modell ändert, nicht, wie richtig es ist.
- Die Masken folgen den Vorhersagen. Wo eine Linie nur geraten ist (oft links), wird keine echte
  Markierung übermalt.
- Die Formel der Zeilenanker (`linspace(100, 1540, num_row) / 1550`) stammt aus der Config und ist
  nicht mit dem Trainingscode abgeglichen.
