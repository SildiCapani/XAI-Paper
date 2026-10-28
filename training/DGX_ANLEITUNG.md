# DGX-Ablauf: Fine-Tuning mit Linien-Verdeckung

Ziel: Das 2026er-Modell (`best_model_bev_2026.pth`) soll Linien auch dann finden, wenn
sie verdeckt oder überblendet sind. Dafür wird es auf dem DGX nachtrainiert. In jedem
zweiten Trainingsbild wird eine Linie gezielt übermalt oder mit Blendung überdeckt, das
Label bleibt aber stehen. Das Modell muss die Linie dann aus dem Rest des Bildes erschließen.

Damit wir am Ende wissen, ob es *an der Verdeckung* liegt, gibt es zwei Läufe mit
identischen Einstellungen:

| Lauf | Augmentation | Zweck |
|---|---|---|
| `kontrolle` | nur die bisherige des Teams | Wie viel bringt reines Weitertrainieren? |
| `verdeckung` | bisherige + Linien-Verdeckung (50 % der Bilder) | Wie viel bringt die Verdeckung zusätzlich? |

Danach werden **Basis, Kontrolle und Verdeckung** auf den Validierungsbildern verglichen,
sauber und verdeckt, gegen die echten Labels.

**Platzhalter:** `<DGX-USER>` ist dein Login auf dem DGX (z. B. `inf23xxx`), `<DGX-HOST>` die
Adresse des Servers. Alle Befehle mit `$USER` laufen auf dem DGX, dort ist das automatisch
dein Login.

---

## Übersicht

| Schritt | Wo | Dauer |
|---|---|---|
| 1. Dateien auf den DGX kopieren | Laptop | 2 min |
| 2. Repo klonen und Patch anwenden | DGX | 5 min |
| 3. Datensatz finden und prüfen | DGX | 10 min |
| 4. Container bereitstellen | DGX | 0–60 min |
| 5. Vorschau: Liegen Labels und Verdeckung richtig? | DGX → Laptop | 10 min |
| 6. Basis messen | DGX | ~15 min |
| 7. Zwei Trainingsläufe | DGX | je einige Stunden |
| 8. Auswertung | DGX → Laptop | ~30 min |
| 9. XAI-Experimente mit dem neuen Modell | Laptop | wie bisher |

---

## 1. Dateien auf den DGX kopieren (Laptop)

```bash
scp ~/Documents/xai_lane_paper/training/linien_verdeckung.patch <DGX-USER>@<DGX-HOST>:~/
scp ~/Documents/smarty_workspace/src/lane_detection_ai/models/best_model_bev_2026.pth <DGX-USER>@<DGX-HOST>:~/
```

## 2. Repo klonen und Patch anwenden (DGX)

```bash
ssh <DGX-USER>@<DGX-HOST>
git clone -b training https://github.com/DHBW-Smart-Rollerz/lane_detection_ai.git ~/lane_detection_ai_training
cd ~/lane_detection_ai_training
git checkout -b xai-linien-verdeckung
git apply ~/linien_verdeckung.patch
mkdir -p configs/pretrained_models slurm-outs
mv ~/best_model_bev_2026.pth configs/pretrained_models/
git status --short
```

`git status` muss 4 geänderte (`M`) und 6 neue (`??`) Dateien zeigen. Fragt `git clone` nach
einem Passwort, ist das Repo privat. Dann einen GitHub-Token benutzen oder den Ordner per `scp`
vom Laptop kopieren.

Die Warnung `trailing whitespace` beim `git apply` kannst du ignorieren. Sie kommt aus der
übernommenen Team-Config.

Bricht `git apply` mit `patch does not apply` ab, hat sich der Branch `training` seit dem
11.06.2026 geändert. Dann zuerst `git apply --3way ~/linien_verdeckung.patch` versuchen und mir
sonst die Fehlermeldung schicken.

## 3. Datensatz finden und prüfen (DGX)

Die Team-Skripte erwarten den Datensatz im Projektordner des Team-Accounts. Wahrscheinlich liegt er hier:

```bash
ls /raid/homes/inf23174/jan/lane_detection_ai_training/dataset/smartrollerz_bev/labels/
```

Kommt `Permission denied`, muss Jan (Account `inf23174`) dich freischalten. Dafür trägt er in
`add_user_folder.slurm` `TARGET_USER="<DGX-USER>"` ein und startet das Skript. Oder er sagt dir,
wo eine Kopie liegt.

Liegt der Datensatz, merk dir den Pfad und prüfe die drei nötigen Dateien:

```bash
export DATA_DIR=/raid/homes/inf23174/jan/lane_detection_ai_training/dataset/smartrollerz_bev   # anpassen
ls -la $DATA_DIR/labels/train_gt.txt $DATA_DIR/labels/eval_gt.txt $DATA_DIR/labels/smartrollerz_anno_cache.json
wc -l $DATA_DIR/labels/train_gt.txt $DATA_DIR/labels/eval_gt.txt
head -2 $DATA_DIR/labels/eval_gt.txt
```

- **`eval_gt.txt` ist Pflicht.** Ohne Validierung speichert das Training kein `best_weights.pth`,
  und es gibt nichts zu vergleichen.
- Fehlt die Datei, frag im Team, welche Aufteilung das 2026er-Modell benutzt hat. Teile
  `train_gt.txt` nicht selbst auf: Das Basismodell hat diese Bilder schon gesehen, und der
  Vergleich wäre unfair.
- Die Bilder sollten 1364×944 groß sein (`original_image_width/height` in der Config).

`DATA_DIR` musst du in jeder neuen SSH-Sitzung neu setzen. Oder du trägst es oben in
`linien_verdeckung.slurm` als Standardwert ein.

## 4. Container bereitstellen (DGX)

Der Code wird beim Start in den Container eingehängt. Der Container muss also **nicht** neu gebaut
werden, wenn sich nur Python-Dateien ändern. Er braucht nur die Bibliotheken (PyTorch, DALI,
`my_interp`).

**Variante A: den Container des Teams mitbenutzen**, falls du Lesezugriff hast:

```bash
ls -la /raid/homes/inf23174/jan/containers/lane_detection_ai_training.sqsh   # Pfad ggf. bei Jan erfragen
export CONTAINER_IMAGE=/raid/homes/inf23174/jan/containers/lane_detection_ai_training.sqsh
```

**Variante B: selbst bauen** (etwa 30–60 min). Das Team-Skript setzt `USER="jan"` und
benutzt dann `$HOME/jan/...`. Mit leerem `USER` stimmen die Pfade für deinen Klon:

```bash
cd ~/lane_detection_ai_training
sed -i 's/^USER="jan"/USER=""/' docker_build_train.slurm
mkdir -p ~/containers
sbatch --job-name=${USER}_image_build docker_build_train.slurm
```

Danach liegt der Container unter `~/containers/lane_detection_ai_training.sqsh`. Das ist der
Standard in `linien_verdeckung.slurm`. Baue ihn, **bevor** Daten im Projektordner liegen, sonst
kopiert Docker sie mit ins Image.

## 5. Vorschau: Liegen Labels und Verdeckung richtig? (DGX → Laptop)

Bevor du stundenlang trainierst, prüfe in 30 Trainingsbildern, ob die Labels auf den Markierungen
liegen und die Verdeckung die Linie trifft.

```bash
cd ~/lane_detection_ai_training
sbatch --job-name=${USER}_vorschau --export=ALL,MODE=vorschau linien_verdeckung.slurm
squeue -u $USER                          # läuft der Job? (R = running, PD = wartet)
tail -f slurm-outs/slurm-<JOBID>.out     # Ausgabe live, Strg+C beendet nur das Anschauen
```

Auf dem Laptop holst du die Bilder:

```bash
scp -r <DGX-USER>@<DGX-HOST>:~/lane_detection_ai_training/results/eval_verdeckung/vorschau ~/Documents/xai_lane_paper/results/dgx_vorschau
```

Jedes Bild hat drei Teile: **Original mit Labels** (weiße Kreuze), **verdeckt mit Labels** und
**verdeckt ohne Labels**.

- ✅ Die Kreuze liegen auf den Markierungen, und die Verdeckung lässt die Linie verschwinden.
- ❌ Die Kreuze liegen daneben, sind verschoben oder gespiegelt: **Stopp.** Schick mir zwei
  der Bilder. Dann stimmt die Umrechnung der Label-Koordinaten nicht, und das Training wäre wertlos.

## 6. Basis messen (DGX)

Vor dem Training läuft die Auswertung nur mit dem 2026er-Modell. Das liefert die Ausgangswerte
und testet die Kette Container → GPU → Modell → Labels:

```bash
sbatch --job-name=${USER}_eval --export=ALL,MODE=eval linien_verdeckung.slurm
```

Im Log muss `basis: 78/78 Tensoren aus configs/pretrained_models/best_model_bev_2026.pth` stehen.
Am Ende kommt eine Tabelle, siehe Schritt 8. Die Ergebnisse überschreibt die spätere Auswertung,
sichere sie also:

```bash
cp -r results/eval_verdeckung results/eval_verdeckung_nur_basis
```

## 7. Zwei Trainingsläufe (DGX)

```bash
sbatch --job-name=${USER}_kontrolle  --export=ALL,MODE=kontrolle  linien_verdeckung.slurm
sbatch --job-name=${USER}_verdeckung --export=ALL,MODE=verdeckung linien_verdeckung.slurm
```

Hast du nur eine GPU, startet der zweite Job, sobald der erste fertig ist. Das ist in Ordnung.

**In den ersten Minuten im Log prüfen:**

```bash
grep -E "Finetune|Tensoren|original_image_size|Error|error" slurm-outs/slurm-<JOBID>.out
```

1. `[Lightning] Finetune: 78/78 Tensoren geladen, 0 unerwartet`. **Das ist entscheidend.**
   Steht dort etwas anderes oder ein Fehler, sofort `scancel <JOBID>`. Ohne korrigierten Lader
   würde das Modell unbemerkt von null trainieren.
2. `[TrainCollect] dataset=Smartrollerz original_image_size=1364x944`.
3. **Die Dauer der ersten Epoche** steht im Fortschrittsbalken. Bei 40 Epochen und 8 h Zeitlimit
   darf eine Epoche höchstens etwa 11 min dauern.
   - Ist sie länger, beide Jobs abbrechen und in **beiden** Configs gleich viele Epochen
     einstellen, z. B. `epoch = 20` und `steps = [12, 17]`.
   - Gespeichert werden keine Zwischenstände, nur `best_weights.pth` bei jeder Verbesserung.
     Ein Abbruch durch das Zeitlimit lässt also das bis dahin beste Modell übrig.
   - Der Lauf `verdeckung` darf etwas langsamer sein, weil die Bilder zusätzlich bearbeitet werden.
4. `val/local_f1` sollte schon nach der ersten Epoche ungefähr so hoch sein wie beim 2026er-Modell.
   Startet der Wert nahe 0, wurden die Gewichte nicht geladen.

Die Läufe landen in `results/<datum>_lr_3e-04_b_32_ft_kontrolle/` bzw. `..._ft_verdeckung/`, das
Modell jeweils als `best_weights.pth`.

## 8. Auswertung (DGX → Laptop)

```bash
sbatch --job-name=${USER}_eval --export=ALL,MODE=eval linien_verdeckung.slurm
```

Das Skript nimmt automatisch den neuesten Lauf jeder Art. Einen anderen Lauf wählst du mit
`--export=ALL,MODE=eval,VERDECKUNG_RUN=<ordnername>`.

Ergebnisse auf den Laptop holen, inklusive der beiden Modelle für Schritt 9:

```bash
mkdir -p ~/Documents/xai_lane_paper/models
scp -r <DGX-USER>@<DGX-HOST>:~/lane_detection_ai_training/results/eval_verdeckung ~/Documents/xai_lane_paper/results/dgx_eval
scp <DGX-USER>@<DGX-HOST>:'~/lane_detection_ai_training/results/*_ft_kontrolle/best_weights.pth' ~/Documents/xai_lane_paper/models/bev_2026_ft_kontrolle.pth
scp <DGX-USER>@<DGX-HOST>:'~/lane_detection_ai_training/results/*_ft_verdeckung/best_weights.pth' ~/Documents/xai_lane_paper/models/bev_2026_ft_verdeckung.pth
```

Gibt es mehrere Läufe einer Art, passt das `*` auf mehrere Ordner. Dann den Ordnernamen ausschreiben.

### Die Tabelle lesen (`zusammenfassung.csv`)

Je Bedingung und Modell eine Zeile:

| Bedingung | Was wurde verdeckt? |
|---|---|
| `sauber` | nichts |
| `A_stueck` | mittleres Drittel einer Linie (Füllung aus dem Training) |
| `A_stueck_inpaint` | dasselbe, mit einer Füllung, **die im Training nicht vorkam** |
| `A_stueck_blendung` | dasselbe, mit Blendung |
| `B_stueck_nachbar` | mittleres Drittel einer Linie + ganze Nachbarlinie |
| `C_linie` | eine ganze Linie |

| Spalte | Bedeutung | gut ist |
|---|---|---|
| `treffer_%` | Anker, an denen die Linie höchstens 20 px (wie Team-F1) neben dem Label liegt | hoch |
| `verloren_%` | Anker, an denen das Modell keine Linie meldet | niedrig |
| `verwechselt_%` | Vorhersage liegt näher an einer *anderen* Linie (das Nachrücken aus der Fehleranalyse) | niedrig |
| `fehler_px_median` | typischer Abstand zum Label in Originalpixeln | niedrig |
| `P_exist` | Sicherheit des Modells | — |
| `falsch_positiv_%` | nur `sauber`: Linie vorhergesagt, wo kein Label ist | niedrig |

**Erfolg heißt:**
- `verdeckung` ist in A, B und C deutlich besser als `kontrolle`, vor allem bei `treffer_%` und
  `verwechselt_%`.
- Bei `sauber` ist es **nicht schlechter**, höchstens etwa 1 Prozentpunkt.
- Ist es auch bei `A_stueck_inpaint` besser, hat das Modell wirklich gelernt, Linien zu
  erschließen, und nicht nur die Trainingsfüllung zu erkennen.

Die Bilder `bild000_<modell>.jpg` zeigen für die ersten Validierungsbilder alle Bedingungen
nebeneinander. Weiße Kreuze sind das Label, farbige Punkte die Vorhersage, magenta der verdeckte Bereich.

## 9. XAI-Experimente mit dem neuen Modell (Laptop)

Die Modelle sind im XAI-Projekt schon eingetragen und funktionieren, sobald die `.pth`-Dateien in
`~/Documents/xai_lane_paper/models/` liegen:

```bash
cd ~/Documents/xai_lane_paper
.venv/bin/python experiments/05_video.py --model bev_2026_verdeckung --lane 1 --neighbor 2 --condition B --end 300
.venv/bin/python experiments/02_batch.py --model bev_2026_verdeckung --step 20 --fill inpaint
.venv/bin/python experiments/02_batch.py --model bev_2026_kontrolle  --step 20 --fill inpaint
```

Für das Paper ist das der Schluss: **XAI hat die Schwäche gefunden, eine gezielte Augmentation
behebt sie, und XAI zeigt, ob sich auch die Erklärung ändert.** Nutzt das Modell den Nachbarn
jetzt mehr oder weniger, und schaut HiResCAM anders hin?

---

## Fehlerbehebung

| Meldung | Ursache und Lösung |
|---|---|
| `[FEHLER] fehlt: …` | Pfad stimmt nicht. `DATA_DIR`, `CONTAINER_IMAGE` oder `PROJECT_DIR` per `export` setzen oder oben in `linien_verdeckung.slurm` eintragen. |
| `Error(s) in loading state_dict` | Config passt nicht zu den Gewichten. Nur die beiden `smartrollerz_bev2026_ft_*.py` benutzen, nicht `smartrollerz_res18_bev.py`. |
| `No eval_gt split detected` | `labels/eval_gt.txt` fehlt, siehe Schritt 3. |
| `KeyError: 'data/…_bev.jpg'` | Ein Bild aus der Liste fehlt im Label-Cache. Liste und Cache passen nicht zusammen, im Team nachfragen. |
| `CUDA out of memory` | In beiden Configs `batch_size = 16` setzen, gleich für beide Läufe. |
| Job bleibt bei `PD` | GPU belegt, warten. `squeue` zeigt den Grund in der letzten Spalte. |
| Epoche bei `verdeckung` mehr als doppelt so lang wie bei `kontrolle` | Die Bildbearbeitung bremst. In `data/dali_data.py` `max_workers=8` auf `16` erhöhen. |

## Was der Patch ändert

| Datei | Änderung |
|---|---|
| `lightning/module.py`, `utils/weights.py` | **Fehlerkorrektur:** Fine-Tuning lud aus `best_model_bev_2026.pth` 0 von 78 Gewichten, ohne Fehlermeldung. Jetzt werden alle drei Speicherformate erkannt, und Fehlendes führt zum Abbruch (`strict`). |
| `data/lane_occlusion.py` | **neu:** gezielte Verdeckung entlang der Label-Linien (Stück / ganze Linie / Stück + Nachbar; verwischen / grau / Blendung). |
| `data/dali_data.py`, `utils/common.py` | Im Trainings-Loader wird mit Wahrscheinlichkeit `aug_lane_occlusion_p` verdeckt, parallel in 8 Threads. Validierung bleibt unverändert. |
| `train_lightning.py` | `seed` aus der Config wird gesetzt, damit beide Läufe vergleichbar sind. |
| `configs/smartrollerz_bev2026_ft_*.py` | **neu:** 2026er-Config + Fine-Tuning: 40 Epochen, Lernrate 3e-4, Backbone ×0,3, Seed 42. Einziger Unterschied: `aug_lane_occlusion_p`. |
| `eval_verdeckung.py` | **neu:** Vergleich gegen echte Labels (Bedingungen wie im XAI-Experiment) und Vorschau. |
| `linien_verdeckung.slurm` | **neu:** ein SLURM-Skript für alle Schritte (`MODE=…`). |

**Lokal getestet:**
- Gewichte laden: 78/78, strict.
- Verdeckung: Darstellung geprüft, etwa 30 ms pro 1364×944-Bild, 16 Bilder in 8 Threads in etwa 0,25 s.
- Auswertungsskript und Vorschau komplett auf der CPU, mit Ersatz-Labels.
- Der Patch lässt sich auf den aktuellen `origin/training` (d7efd3e) anwenden.

**Nicht getestet**, weil hier keine GPU und kein DALI vorhanden sind: der Trainingslauf selbst.
Dafür sind die Prüfungen in Schritt 5–7 da.

**Hinweis zur Team-Metrik:** Der Validierungs-Loader des Teams benutzt dieselbe Pipeline wie das
Training, inklusive zufälliger Verschiebung, Drehung und Rechtecke. `val/local_f1` schwankt deshalb
von Epoche zu Epoche etwas. Für den Vergleich im Paper ist `eval_verdeckung.py` maßgeblich: feste
Bilder, feste Verdeckung, echte Labels.
