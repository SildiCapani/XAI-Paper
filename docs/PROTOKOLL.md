# Projektprotokoll

Hier stehen Entscheidungen, die man im fertigen Code nicht mehr sieht: was ausprobiert wurde,
warum es geändert wurde und welche Daten dahinterstehen.

---

## 2026-11-13: Grad-CAM entfernt, nur noch HiResCAM

### Ausgangslage

Am Anfang habe ich zwei Erklärungsverfahren nebeneinander gerechnet: **Grad-CAM** und
**HiResCAM**. Beide sollen zeigen, auf welche Bildstellen das Netz geschaut hat, als es die
Position einer Linie vorhergesagt hat. Beide benutzen dieselben zwei Zutaten aus der letzten
ResNet-Stufe (`layer4`, ein Raster aus 12 × 16 Feldern mit 512 Kanälen):

- **A (Aktivierung):** Wie stark ist ein Merkmal in einem Feld vorhanden?
- **G (Gradient):** Wie stark würde sich die Vorhersage ändern, wenn sich dieser Wert ändert?

Beide Verfahren habe ich zusätzlich selbst nachprogrammiert (`lernen/schritt7_gradcam.py`).

### Was ich festgestellt habe

**Grad-CAM gibt keine ortsgenaue Aussage, sondern eine allgemeine.** Es mittelt den Gradienten
eines Kanals über das ganze Bild und bekommt so *ein* Gewicht pro Kanal: „Wie wichtig ist dieses
Merkmal insgesamt?“ Dieses Gewicht wird dann überall dort eingesetzt, wo das Merkmal aktiv ist.
Ob die Stelle tatsächlich zur Vorhersage beigetragen hat, geht dabei verloren.

**HiResCAM** multipliziert Gradient und Aktivierung **in jedem Feld einzeln**. Es sagt also,
wie wichtig ein Merkmal *an genau dieser Stelle* war.

```
Grad-CAM:  Karte = Summe über Kanäle ( Mittelwert(G) × A )
HiResCAM:  Karte = Summe über Kanäle ( G × A )      (Feld für Feld)
```

### Warum das bei unserem Netz entscheidend ist

Grad-CAM wurde für Netze entwickelt, die am Ende über alle Bildstellen mitteln (Global Average
Pooling, z. B. Bildklassifikation). Dort ist der Gradient sowieso überall gleich und beide
Verfahren liefern dasselbe.

Unser UFLDv2-Kopf mittelt **nicht**: Er macht aus dem Raster einen flachen Vektor (1536 Zahlen)
und schickt ihn in voll verbundene Schichten. Diese haben für jedes Rasterfeld eigene Gewichte,
der Gradient ist also je nach Stelle verschieden. Grad-CAM verwischt genau diese Information.

Für unsere Forschungsfrage („Schaut das Netz auf die Nachbarlinie?“) ist das gefährlich:
Grad-CAM kann die Nachbarlinie markieren, nur weil dort dasselbe Merkmal (z. B. „heller Strich“)
aktiv ist, auch wenn sie die Vorhersage gar nicht beeinflusst hat. Die Frage *wo* ist bei uns
aber der ganze Punkt.

### Was die Daten gezeigt haben

**Sichtbar in den Heatmaps** (`docs/ERKLAERUNG.md`, Abschnitt 6): HiResCAM konzentriert sich auf
die Mittellinie, Grad-CAM verteilt sich über das halbe Bild.

**Batch-Lauf** (`experiments/02_batch.py`, gesichert in `docs/belege/gradcam_vs_hirescam_<modell>.csv`):
Spearman-Korrelation zwischen „CAM-Anteil auf der Nachbarlinie“ und dem gemessenen Effekt im
Verdeckungsexperiment (B − A). Höher heißt: Die Erklärung passt besser zu dem, was das Netz
im Experiment wirklich tut.

Modell `bev_2026` (Standardmodell), Füllung `inpaint`:

| T ← N | Bilder | Grad-CAM | HiResCAM |
|---|---|---|---|
| links ← Mitte | 40 | +0,05 | **+0,37** |
| Mitte ← links | 43 | +0,25 | **+0,42** |
| Mitte ← rechts | 56 | +0,22 | **+0,53** |
| rechts ← Mitte | 58 | −0,06 | **+0,28** |

Über beide Füllungen (`inpaint` und `median`) hinweg passt HiResCAM in **6 von 8** Fällen besser
zum Experiment. In den anderen 2 Fällen (Füllung `median`) liegen beide praktisch gleich auf
(0,14 gegen 0,15 und 0,28 gegen 0,29).

Beim älteren Modell `bev` zeigte Grad-CAM bei „links ← Mitte“ fast nichts auf der Mittellinie
(0,09-Fache des Flächenanteils), HiResCAM dagegen das 1,6-Fache, obwohl das Experiment dort den
stärksten Effekt misst.

### Ehrliche Einschränkung

Auch HiResCAM passt nur mäßig zum Experiment (Korrelationen um 0,3 bis 0,5, nicht 1). Es ist also
nicht „die Wahrheit“, sondern nur das verlässlichere der beiden Verfahren für diese Architektur.

### Entscheidung

Grad-CAM wird ab jetzt nicht mehr gerechnet. Alle Auswertungen nutzen nur noch HiResCAM.

### Was im Code geändert wurde

| Datei | Änderung |
|---|---|
| `src/xai_lane/cam.py` | `GradCAM` aus Import und `METHODS` entfernt |
| `src/xai_lane/experiment.py` | `CAM_METHODS = ("hirescam",)` |
| `experiments/01_first_look.py` | Beschreibung angepasst; die Abbildung zeigt nur noch HiResCAM |
| `experiments/02_batch.py` | Statistik und Ausgabe nur noch für HiResCAM |
| `README.md`, `docs/ERKLAERUNG.md` | Hinweis auf diese Entscheidung |

**Bewusst nicht geändert:**
- `lernen/schritt7_gradcam.py`: Dort bleiben beide Verfahren, weil der Vergleich der Grund für
  die Entscheidung ist und ich ihn erklären können muss.
- Die Bilder in `docs/bilder/`: Sie zeigen noch beide Verfahren nebeneinander.
- Beleg für die Zahlen oben: `docs/belege/gradcam_vs_hirescam_bev.csv` und `..._bev_2026.csv`.
  Sie wurden aus den Rohdaten des Batch-Laufs mit dem alten Code neu berechnet. `results/` ist
  nicht in Git und wird bei jedem neuen Lauf überschrieben, deshalb liegen die Belege hier.

### Quelle

R. L. Draelos, L. Carin: *Use HiResCAM instead of Grad-CAM for faithful explanations of
convolutional neural networks*, arXiv:2011.08891, 2020.

### Für die Präsentation (ein Satz)

> „Ich habe mit Grad-CAM angefangen, aber festgestellt, dass es den Einfluss eines Merkmals über
> das ganze Bild mittelt und damit nur eine allgemeine Aussage liefert, nicht *wo* das Netz
> hingeschaut hat. Weil unser Netz voll verbundene Schichten hat und unsere Frage genau das *Wo*
> ist, bin ich auf HiResCAM umgestiegen. Das passt auch in den Daten in 6 von 8 Fällen besser
> zum Verdeckungsexperiment.“

---

## 2026-11-16: Sanity Check für HiResCAM

### Warum

Bisher war gezeigt, dass HiResCAM besser zum Verdeckungsexperiment passt als Grad-CAM. Offen war
eine Grundsatzfrage: Zeigt die Heatmap überhaupt etwas, das das Modell *gelernt* hat, oder nur
auffällige Stellen im *Bild* (helle Striche)? Manche Erklärungsverfahren sehen fast gleich aus,
egal ob das Netz trainiert ist oder nicht. Sie funktionieren dann wie ein Kantenfinder
(Adebayo et al. 2018).

### Wie

`experiments/07_sanity_check.py`: Die Gewichte werden Schicht für Schicht durch Zufallswerte
ersetzt, von oben nach unten (`cls.3` → `cls.1` → `pool` → `layer4` → … → `conv1`). Jede Stufe
behält die vorherigen Zufallsschichten. Nach jeder Stufe wird die Heatmap für dasselbe Ziel neu
berechnet (Gewinner-Bins des trainierten Modells, Anker 35–65 %).
10 Testbilder (jedes 120.), alle erkannten Linien (26 Fälle), 3 Zufalls-Seeds.

Gemessen:
- **Spearman** zwischen neuer und trainierter Heatmap (1 = gleich, 0 = nichts gemeinsam)
- **Anteil auf der eigenen Linie T geteilt durch ihren Flächenanteil** (1 = nicht mehr als Zufall)

### Ergebnis (`bev_2026`)

| zufällig bis | Spearman (Median) | Heatmap auf T vs. Fläche |
|---|---|---|
| trainiert | 1,00 | **6,4** |
| + cls.3 (letzte FC-Schicht) | 0,34 | 1,9 |
| + cls.1 | 0,31 | 1,9 |
| + pool | 0,26 | 1,6 |
| + layer4 | 0,16 | 1,2 |
| + layer3 | 0,09 | 1,1 |
| + layer2 / layer1 | 0,07 / 0,04 | 1,0 |
| + conv1 (alles zufällig) | −0,01 | 0,9 |

Beleg: `docs/belege/sanity_check_bev_2026.csv`, Bilder in `docs/bilder/praesi/sanity_check_*.png`.

**HiResCAM besteht den Test.** Schon wenn nur die letzte Schicht zufällig ist, fällt der Fokus auf
die Linie von 6,4-fach auf 1,9-fach. Ab `layer4` hat die Karte kaum noch etwas mit dem Original
gemeinsam, und am Ende liegt sie nicht mehr auf der Linie als auf jeder anderen Stelle.

### Warum bleibt nach dem Kopf noch etwas Ähnlichkeit (0,26–0,34)?

HiResCAM ist Gradient × Aktivierung an `layer4`. Macht man nur den Kopf zufällig, ändert sich der
Gradient, die Aktivierungen von `layer4` sind aber noch die trainierten. Wo das Backbone „Strich“
erkennt, bleibt die Karte etwas hell. Erst wenn `layer4` selbst zufällig ist, verschwindet das.
Das ist erwartet und kein Fehler.

### Einschränkung

Nur 10 Bilder und ein Modell. Getestet wird nur, ob die Karte vom Modell *abhängt*, nicht ob sie
*richtig* ist. Dafür ist der Abgleich mit dem Verdeckungsexperiment da.

### Quelle

J. Adebayo et al.: *Sanity Checks for Saliency Maps*, NeurIPS 2018, arXiv:1810.03292.
