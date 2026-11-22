# Herkunft: Was ist von uns, was haben wir übernommen?

Drei Stufen:

- **Fremd**: nur benutzt, nicht von uns. Grob erklären können reicht.
- **Formel übernommen, Code selbst**: Die Idee ist fremd, die Umsetzung ist von uns.
- **Eigen**: von uns entworfen und geschrieben. Muss man im Detail erklären können.

---

## 1. Fremd: nur benutzt

| Was | Von wem | Wo es bei uns auftaucht |
|---|---|---|
| **UFLDv2-Verfahren**: Zeilen-/Spaltenanker, pro Anker „welches Kästchen?“ als Klassifikation, `exist`-Ausgabe, ResNet + 1×1-Faltung + vollverbundener Kopf | Qin, Zhang, Li: *Ultra Fast Deep Lane Detection with Hybrid Anchor Driven Ordinal Classification*, IEEE TPAMI 2022 ([Code](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)) | die ganze Modellausgabe (`loc_row`, `exist_row` …) |
| **Modellcode** `ParsingNet`, `get_model`, `get_config` | Smart-Rollerz-Team (`lane_detection_ai/model/`), basiert auf UFLDv2 | `model_loader.py` importiert ihn nur, ändert nichts |
| **ResNet-„9“** (BasicBlock, je 1 Block in layer1–4) | ResNet: He et al. 2016, Bausteine aus `torchvision`; die „9“-Variante ist vom Team | im Modell |
| **Trainierte Gewichte, Config, Testbilder** | Smart-Rollerz-Team (`models/`) | `paths.py` zeigt nur darauf, wir lesen sie nur |
| **ImageNet-Mittelwerte/-Streuung** | Standardwerte von ImageNet / torchvision | `preprocess.py` |
| **HiResCAM-Rechnung** | Draelos & Carin 2020 (Methode), Bibliothek `pytorch-grad-cam` (Code) | `cam.py` ruft sie nur auf |
| **Inpainting (Telea)**, Linien zeichnen, Weichzeichnen | OpenCV (`cv2.inpaint`, `cv2.polylines`, `cv2.dilate`, `cv2.GaussianBlur`) | `occlusion.py` |
| **Statistik-Tests** Wilcoxon, Spearman | SciPy | `02_batch.py`, `07_sanity_check.py` |
| **Idee Sanity Check** (Schichten zufällig machen) | Adebayo et al. 2018 | `07_sanity_check.py` setzt sie selbst um |

**Satz für den Prof:** „Das Modell und seine Ausgabeform sind UFLDv2 vom Team. HiResCAM kommt aus `pytorch-grad-cam`. Wir haben nichts davon verändert, nur benutzt.“

---

## 2. Formel übernommen, Code selbst

| Datei | Was übernommen ist | Was wir anders machen und warum |
|---|---|---|
| `src/xai_lane/decode.py` | Dekodier-Formel aus UFLDv2 (`pred2coords`): Gewinner-Kästchen ± 1, Softmax, gewichteter Mittelwert, + 0.5, / (grid − 1) | **keine Rundung** auf ganze Pixel (sonst wäre ein Effekt von 0,7 px unsichtbar); **NaN statt Weglassen**, immer 55 Zeilen (sonst kann man Zeile k vorher/nachher nicht vergleichen); gibt zusätzlich die **Gewinner-Kästchen** zurück (Ziel für HiResCAM); Modellauflösung statt Originalgröße; nur Zeilenanker |
| `src/xai_lane/model_loader.py` | Ablauf „Config lesen → Netz bauen → Gewichte laden“ wie im `model_wrapper.py` des Teams | **`strict=True`**: passt ein Gewicht nicht, bricht es ab. Der Team-Wrapper lädt mit `strict=False`, dort würde ein Fehler unbemerkt bleiben. Liest beide Checkpoint-Formate (alt und Lightning) |
| `src/xai_lane/preprocess.py` | Standard-Vorverarbeitung: Größe ändern, durch 255 teilen, ImageNet-Normalisierung | selbst geschrieben, Graustufe → 3 Kanäle für die BEV-Bilder |

---

## 3. Eigen

| Datei | Was es ist |
|---|---|
| `src/xai_lane/paths.py` | Wo liegt was, Auswahl mehrerer Modelle per `--model` |
| `src/xai_lane/occlusion.py` | Maske entlang einer vorhergesagten Linie, drei Füllarten (median, inpaint, blendung) |
| `src/xai_lane/cam.py` | **`RowBinTarget`**: *was* die Heatmap erklärt (Punktzahl der Gewinner-Kästchen in den verdeckten Ankern); `LocRowOnly`; `energy_share` (Anteil der Heatmap in einer Maske) |
| `src/xai_lane/experiment.py` | **Die Versuchsanordnung**: Bedingungen 0, E, F, Verschiebung dx, Auswahl der Anker 35–65 % |
| `experiments/` | Alle Auswertungen, Statistik, Folienbilder, Sanity Check |

**Satz für den Prof:** „Unser Beitrag ist die Versuchsanordnung: was verdeckt wird, wie gemessen wird und wie wir die Heatmap mit dem Messergebnis abgleichen. Modell und Erklärverfahren haben wir nicht erfunden.“
