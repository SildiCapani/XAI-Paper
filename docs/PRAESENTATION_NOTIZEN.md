# Notizen für die Präsentation

Sammlung aller Erkenntnisse, Formulierungen und Stolperfallen für Folien und mündliche Prüfung.
Wird laufend ergänzt. Entscheidungen mit Begründung stehen ausführlicher in [PROTOKOLL.md](PROTOKOLL.md).

**Aufbau jedes Abschnitts:**
- 🗣 **Satz für die Folie / Prüfung**: so kann man es sagen
- 📊 **Beleg**: Zahlen und Bilder
- ⚠️ **Vorsicht**: Einschränkungen, die man selbst nennen sollte, bevor der Prof fragt
- ❓ **Mögliche Prof-Frage**

Bilder für die Folien liegen in [bilder/praesi/](bilder/praesi/).

---

## Vorschlag für den roten Faden

1. **Frage:** Nutzt die Spurerkennung benachbarte Linien, um eine verdeckte Linie zu finden?
2. **Modell:** UFLDv2 mit ResNet-9, 3 Linien, voll verbundener Kopf (→ Abschnitt 1)
3. **Methode:** Verdecken + HiResCAM. Grad-CAM wurde verworfen (→ Abschnitt 2, 3)
4. **Ergebnis 1:** Fehlt nur ein Stück, nutzt die KI fast nur die Linie selbst (→ Abschnitt 4)
5. **Ergebnis 2:** Fehlt die ganze Linie, nutzt sie die Nachbarn als „Lineal“ (→ Abschnitt 5)
6. **Das ist gelernt, nicht programmiert:** gut und gefährlich zugleich (→ Abschnitt 6)
7. **Realistischer Fall: Blendung** (→ Abschnitt 7)
8. **Ausblick:** Training mit Verdeckung, XAI prüft das Ergebnis (→ Abschnitt 8)

**Umfang (Einschätzung 2026-11-16):** Inhaltlich reicht das. Es ist eher zu viel als zu wenig.
Abschnitt 9 (RANSAC) gehört nicht in den Hauptvortrag, höchstens als Backup-Folie für Fragen.
Lieber weniger Folien und jede erklären können.

---

## 1. Das Modell kurz erklärt

🗣 „Das ResNet macht aus dem Bild ein Raster aus 12 × 16 Feldern mit 512 Merkmalen pro Feld. Eine
1×1-Faltung reduziert das auf 8 Merkmale pro Feld, also 1536 Zahlen. Eine voll verbundene Schicht
rechnet daraus alle Linienpositionen auf einmal aus.“

🗣 „Weil der Kopf voll verbunden ist, *kann* die Vorhersage jeder Linie vom ganzen Bild abhängen,
auch von den Nachbarlinien. Ob sie das *tut*, steht nirgends im Code. Das muss man messen.“

📊 Formen: Bild (1, 3, 384, 512) → layer4 (1, 512, 12, 16) → pool (1, 8, 12, 16) → flach (1, 1536)
→ Linear 1536 → 512 → 29070 → aufgeteilt in `loc_row` (100 × 55 × 3), `loc_col`, `exist_row`, `exist_col`.
29070 = 16500 + 12000 + 330 + 240.

❓ *„Warum heißt das Backbone ResNet-9?“* 1 Faltung am Anfang + 4 Stufen mit je 1 BasicBlock aus
2 Faltungen = 9 Faltungsschichten.

---

## 2. Was eine Heatmap eigentlich zeigt

🗣 „Eine Heatmap zeigt nicht, wohin die KI *allgemein* schaut, sondern welche Bildstellen *eine
bestimmte Antwort* beeinflusst haben. Wir fragen: Was hält die Linie T im verdeckten Bereich an
ihrer ursprünglichen Position?“

🗣 Präzise statt „Die KI schaut nur auf die Mitte“: „**Für die Vorhersage der Mittellinie** stützt
sich das Netz vor allem auf die Mittellinie selbst.“

📊 Pro Linie eine eigene Karte (Frage: „Wo ist links / Mitte / rechts?“): Jede Linie stützt sich
auf sich selbst. Bild: [heatmap_pro_linie_bild0.png](bilder/praesi/heatmap_pro_linie_bild0.png)

⚠️ Eine **Gesamtkarte** (alle Linien zusammen) sieht schön aus, verwischt aber, *welche* Linie
*worauf* schaut. Für die Forschungsfrage sind die Einzelkarten nötig.

---

## 3. Warum HiResCAM statt Grad-CAM

🗣 „Ich habe mit Grad-CAM angefangen. Es gewichtet jedes Merkmal mit seinem über das ganze Bild
**gemittelten** Gradienten. Es zeigt also, wo ein Merkmal aktiv ist, aber nicht, ob es *an dieser
Stelle* für die Vorhersage wichtig war. Unser Netz hat einen voll verbundenen Kopf, da ist der
Gradient je nach Stelle verschieden. Deshalb nutze ich HiResCAM, das Gradient und Aktivierung
Stelle für Stelle verrechnet.“

📊
- Bild 0, Mitte: Grad-CAM verteilt sich in Flecken über das ganze Bild, HiResCAM liegt klar auf der
  Mittellinie. Anteil auf dem verdeckten Stück: Grad-CAM 2–4 %, HiResCAM 7–13 %.
  Bild: [gradcam_vs_hirescam_bild0.png](bilder/praesi/gradcam_vs_hirescam_bild0.png)
- Batch (`bev_2026`): Die Heatmap passt bei HiResCAM in 6 von 8 Fällen besser zum
  Verdeckungsexperiment (Spearman), bei `inpaint` in 4 von 4.
  Daten: [belege/gradcam_vs_hirescam_bev_2026.csv](belege/gradcam_vs_hirescam_bev_2026.csv)
- Quelle: Draelos & Carin 2020, *Use HiResCAM instead of Grad-CAM …*

⚠️
- **Hauptargument ist die Architektur**, die Zahlen sind nur die Bestätigung. Beim alten Modell
  `bev` ist es gemischt.
- Nicht sagen „Grad-CAM zeigt keine Stellen“. Grad-CAM liefert eine Karte, nur das *Gewicht* ist
  über das Bild gemittelt.
- Auch HiResCAM passt nur mäßig (Spearman ~0,3–0,5). Es ist das verlässlichere Verfahren, nicht die
  Wahrheit.

❓ *„Woher wissen Sie, dass die Heatmap überhaupt etwas über das Modell aussagt?“* Zwei Antworten:
(1) Sie wurde mit dem Eingriff geprüft: Wo die Heatmap auf N springt, schadet es auch, N
wegzunehmen (Abschnitt 5, Spearman bis +0,70). (2) Sanity Check nach Adebayo et al. 2018 (Gewichte
zufällig machen → Heatmap muss sich stark ändern). Gemacht und bestanden, siehe unten.

🗣 Sanity Check: „Um zu prüfen, ob die Heatmap wirklich das Modell erklärt und nicht nur helle
Striche im Bild zeigt, habe ich die Gewichte Schicht für Schicht durch Zufallszahlen ersetzt. Die
Heatmap zerfällt dabei. Trainiert liegt sie 6-mal stärker auf der Linie, als es ihrer Fläche
entspricht. Ist nur die letzte Schicht zufällig, ist es noch knapp 2-mal, am Ende 1-mal, also
Zufall. HiResCAM hängt also am Gelernten.“

📊 10 Bilder, 26 Linien, 3 Seeds (`experiments/07_sanity_check.py`): Spearman zur trainierten
Karte 1,00 → 0,34 (nur letzte FC zufällig) → 0,16 (bis layer4) → −0,01 (alles zufällig).
Bilder: [sanity_check_kurve.png](bilder/praesi/sanity_check_kurve.png),
[sanity_check_beispiel.png](bilder/praesi/sanity_check_beispiel.png).
Daten: [belege/sanity_check_bev_2026.csv](belege/sanity_check_bev_2026.csv). Details in [PROTOKOLL.md](PROTOKOLL.md).

⚠️ Der Test zeigt nur, dass die Karte vom Modell *abhängt*, nicht, dass sie *richtig* ist. Die
Richtigkeit prüft der Abgleich mit dem Verdeckungsexperiment. Beides zusammen ist das Argument.

❓ *„Warum ist nach dem Zufälligmachen des Kopfes noch 0,3 Ähnlichkeit übrig?“* HiResCAM =
Gradient × Aktivierung an layer4. Die Aktivierungen sind dann noch trainiert, nur der Gradient
ist zufällig. Erst wenn layer4 selbst zufällig ist, verschwindet der Rest.

❓ *„Wann wären beide gleich?“* Bei Netzen mit Global Average Pooling am Ende (z. B.
Bildklassifikation). Da ist der Gradient überall gleich.

---

## 4. Ergebnis 1: Fehlt nur ein Stück, braucht die KI den Nachbarn kaum

🗣 „Fehlt nur ein Stück der Linie, ergänzt die KI die Lücke vor allem aus den sichtbaren Resten der
Linie selbst. Der Nachbar hilft messbar, aber nur sehr wenig.“

📊 Batch `bev_2026`, Füllung `inpaint`, ~40–58 Bilder pro Linienpaar: Fehlt zusätzlich N (B statt A),
wird T nur ~0,1 % der Bildbreite schlechter (unter 1 px), aber in 60–71 % der Bilder, signifikant bei
3 von 4 Paaren (p < 0,01; „rechts ← Mitte“ p = 0,055).
Daten: `results/bev_2026/02_batch/zusammenfassung.csv`

⚠️ Klein, aber regelmäßig. Signifikant heißt hier nicht „groß“.

🗣 Sonderfall linke Linie: „Die Mittellinie hängt nicht an der linken Linie. Nimmt man links weg,
ändert sich die Mitte praktisch nicht, und die Heatmap zeigt auch weniger auf links, als es ihrer
Fläche entspricht. Experiment und Erklärung sagen dasselbe.“

📊 „Mitte ← links“, `bev_2026`, `inpaint`: C = 0,06 % der Bildbreite, B nicht größer als A.
HiResCAM-Anteil auf links = 0,69-Fache des Flächenanteils (unter 1). Umgekehrt „links ← Mitte“:
3,1-Fache. Grund vermutlich: Links ist oft keine echte Markierung, sondern Bordstein oder Bildrand
(siehe Bild 600 in [ERKLAERUNG.md](ERKLAERUNG.md)).

⚠️ Nicht umdrehen: Die KI ignoriert die linke Linie nicht. Fehlt sie, erfindet sie trotzdem eine.

---

## 5. Ergebnis 2: Fehlt die ganze Linie, nutzt die KI die Nachbarn als „Lineal“

🗣 „Fehlt eine Linie komplett, sagt das Netz sie trotzdem an ungefähr der richtigen Stelle voraus,
nur ungenauer. Die Heatmap zeigt, dass es sich dann an den Nachbarlinien orientiert. Es nutzt sie
wie ein Lineal, um den Abstand abzuschätzen. Solange noch Reste der Linie selbst sichtbar sind,
braucht es den Nachbarn kaum.“

🗣 „Fehlt eine Linie, verlagert die KI ihre Aufmerksamkeit auf benachbarte Strukturen, meist die
Nachbarlinien, in einzelnen Fällen auch Zebrastreifen. In 8 von 8 getesteten Bildern stieg der
Heatmap-Anteil auf den Nachbarn um das 2- bis 6-fache.“

📊
- **8 Bilder, Mitte komplett weg** (`inpaint`): Anteil der Heatmap „Wo ist Mitte?“ auf den Nachbarn
  steigt von 3–7 % auf 7–18 %, auf der Mitte selbst sinkt er von 11–20 % auf 3–10 %.
  Die Mitte wandert 4–23 px.
  Bild: [mitte_weg_8_bilder.png](bilder/praesi/mitte_weg_8_bilder.png), Skript: `lernen/schritt7c_mehrere_bilder.py`
- **Ursächlicher Beleg (Bild 0, T = Mitte, N = links):** Mitte ganz weg (E) → 11,3 px daneben.
  Mitte **und** links weg (F) → 29,1 px, fast dreimal so viel.
- Bild 0, jede Linie einzeln weg: [linien_weg_bild0.png](bilder/praesi/linien_weg_bild0.png)
- **Batch über 60 Bilder, alle 4 Linienpaare** (`02_batch.py --ef --fill inpaint`, 40–58 Bilder pro
  Paar). E = T ganz weg, F = T und N ganz weg. Werte in % der Bildbreite (1 % ≈ 5 px):

  | T ← N | Bilder | E | F | F schlechter als E bei | p | Heatmap auf N (vs. Fläche): Original → T weg | gestiegen bei | Spearman Heatmap ↔ (F − E) |
  |---|---|---|---|---|---|---|---|---|
  | links ← Mitte | 40 | 2,10 | 3,25 | 75 % | 6·10⁻⁴ | 2,39 → 4,02 | 88 % | +0,32 |
  | Mitte ← links | 43 | 2,39 | 3,39 | 81 % | 3·10⁻⁶ | 0,56 → 1,10 | 91 % | +0,16 |
  | Mitte ← rechts | 56 | 2,94 | **5,54** | 84 % | 6·10⁻⁹ | 1,06 → 2,73 | **100 %** | **+0,70** |
  | rechts ← Mitte | 58 | 2,79 | 4,97 | 77 % | 3·10⁻⁵ | 1,17 → 2,62 | 97 % | +0,57 |

  Daten: [belege/ganze_linie_weg_EF_bev_2026.csv](belege/ganze_linie_weg_EF_bev_2026.csv)

  🗣 „Über 60 Bilder und alle vier Linienpaare: Fehlt eine Linie ganz, steigt der Heatmap-Anteil auf
  dem Nachbarn in 88–100 % der Bilder, im Median auf das 1,7- bis 2,6-Fache. Nimmt man den Nachbarn
  dann zusätzlich weg, wird die Linie in 75–84 % der Bilder deutlich schlechter, bis fast doppelt so
  weit daneben. Heatmap und Eingriff sagen dasselbe, alle Effekte sind hochsignifikant.“

  🗣 Vergleich zu Ergebnis 1: „Fehlt nur ein Stück, macht der Nachbar unter 1 px Unterschied. Fehlt
  die ganze Linie, sind es 5–13 px. Der Nachbar wird erst dann wirklich wichtig, wenn von der Linie
  selbst nichts mehr übrig ist.“

  🗣 Wie gut passt die Erklärung? „Am besten bei der rechten Linie als Nachbar (Spearman +0,70):
  Je stärker die Heatmap auf rechts springt, desto mehr schadet es, rechts wegzunehmen. Bei der
  linken Linie passt es kaum (+0,16). Dort ist oft keine echte Markierung, sondern nur ein Rand.“

⚠️
- **Nicht sagen „die KI springt auf die Nachbarlinie“.** Die *Vorhersage* bleibt an der richtigen
  Stelle (nur zackig/ungenauer). Was springt, ist die *Heatmap*, also woran sie sich orientiert.
- Bei Zebrastreifen (Bild 74, 111) nutzt sie auch den Zebrastreifen, nicht nur die Nachbarlinien.
- Mit der Übermalung `median` bleibt ein sichtbarer Streifen, dem die KI folgen kann. Für diese
  Aussage nur `inpaint` verwenden.

❓ *„Warum braucht man F, reicht die Heatmap aus E nicht?“* Die Heatmap zeigt nur, *wohin* die KI
schaut. Erst wenn man N zusätzlich wegnimmt und T dadurch schlechter wird, ist belegt, dass sie N
auch *benutzt*. Heatmap = Hinweis, Eingriff = Beweis.

---

## 6. Gelernt, nicht programmiert: gut und gefährlich

🗣 „Dieses Verhalten wurde nicht programmiert. Im Training gibt es keine Regel, die Linien
miteinander verbindet. Das Netz hat aus den Trainingsdaten gelernt, dass die Linien meistens gleich
weit auseinanderliegen. Sichtbar wird so etwas nur durch XAI, im Code würde man es nie finden.“

📊 Trainings-Config des 2026er-Modells: Losses nur pro Linie (Klassifikation, Existenz, Mean-Loss).
Die Form-Regeln `sim_loss_w` und `shp_loss_w` stehen auf 0.0.

🗣 „Das ist zweischneidig: **Gut**, weil das Auto bei abgefahrenen Markierungen trotzdem ungefähr
richtig fährt. **Gefährlich**, weil ein falscher Nachbar, etwa ein Lichtfleck, die Linie mitzieht.“

📊 Beobachtung aus [ERKLAERUNG.md](ERKLAERUNG.md): Ein Lichtfleck zwischen zwei Linien verschob
beim alten Modell eine Linie um fast 20 % der Bildbreite (Einzelbild).

❓ *„Ist das Verhalten gewollt?“* → beide Seiten nennen. Die Entscheidung liegt beim Team.

---

## 7. Realistischer Fall: Blendung

🗣 „Dass eine Linie komplett fehlt, ist im echten Betrieb selten. Realistischer ist Blendung: Licht
fällt auf die Straße und die Kamera übersteuert. Dann ist die Linie nicht nur für die KI weg,
sondern physikalisch: Die Pixel sind reinweiß, da steht keine Information mehr drin. Die KI muss
raten und stützt sich auf das, was noch sichtbar ist.“

📊
- 147 von 1192 Testbildern haben mehr als 5 % fast weiße Fläche.
- **Szene 374–385:** Sonnenlicht fällt in Streifen über die Fahrbahn. Die Punkte von Mitte und links
  springen in die hellen Streifen, die linke Linie geht immer wieder verloren.
  Bild: [blendung_szene_366-390.png](bilder/praesi/blendung_szene_366-390.png)
- **Heatmap Bild 380:** Nur 1–6 % der Karte liegen auf der weißen Fläche (die 9 % des Bildes
  ausmacht). Die KI schaut nicht ins Weiße, sondern auf die sichtbaren Reste. Bei der linken Linie
  sucht sie herum, auch an der **Kante des Lichtstreifens**.
  Bild: [blendung_heatmap_366_380.png](bilder/praesi/blendung_heatmap_366_380.png)
- **Kontrastverstärkung hilft nicht:** In der hellen Fläche von Bild 434 sind 84 % der Pixel genau
  255. Nach der Verstärkung sind die erkannten Zeilen praktisch gleich (0/37/55 → 0/38/55).
  Bild: [kontrast_verstaerken_434.png](bilder/praesi/kontrast_verstaerken_434.png)

🗣 Für das Team: „Gegen Übersteuerung hilft keine Bildverarbeitung danach, nur die Kamera
(Belichtung, HDR) oder ein Modell, das gelernt hat, mit fehlenden Linien umzugehen.“

⚠️
- Die simulierte Blendung `--fill blendung` ist **ungeeignet** als „Linie durch Licht verdeckt“: Sie
  legt einen hellen Streifen *entlang* der Linie, das sieht selbst wie eine Linie aus. Höchstens als
  Test „falsche Linie durch Lichtstreifen“ verwenden.
- Bild 434 hat zwar viel Weiß, aber **neben** der Fahrbahn. Für Blendung auf Linien die Szene
  374–385 verwenden.
- Szene 374–385 ist auch eine Kurve mit vielen Markierungen, nicht jeder Fehler kommt vom Licht.
  Bisher nur 2 Bilder mit Heatmap ausgewertet.

---

## 8. Ausblick: Training und XAI als Kreislauf

🗣 „XAI hat gezeigt, wie die KI mit fehlenden Linien umgeht. Im nächsten Schritt kann man gezielt
mit verdeckten Linien und Blendung trainieren und dann mit denselben XAI-Experimenten prüfen, ob
das neue Modell robuster ist und ob es den Nachbarn mehr oder weniger nutzt.“

📊 Vorbereitet, aber noch nicht trainiert: `training/linien_verdeckung.patch` +
`training/DGX_ANLEITUNG.md`. Zwei Läufe (Kontrolle / Verdeckung), damit klar ist, ob eine
Verbesserung von der Verdeckung kommt oder nur vom längeren Training. Die Modelle sind in
`src/xai_lane/paths.py` schon eingetragen.

⚠️ Training mit „ganze Linie weg, Label bleibt“ **verstärkt** wahrscheinlich das Lineal-Verhalten,
denn der Nachbar ist dann der einzige Weg zur richtigen Antwort. Ob das gewünscht ist, muss man
abwägen (siehe Abschnitt 6).

---

## 9. Nebenfrage: Wie stark schwankt die KI, und hilft RANSAC?

🗣 „Im Normalfall ist die Erkennung ruhig: Von Bild zu Bild springen die Linien im Median nur
1,5–2 px. Probleme gibt es in schwierigen Szenen wie bei Blendung, dort springen sie im Median
5–13 px. Eine robuste Glättung wie RANSAC hilft genau dort, macht den Normalfall aber leicht
schlechter und kann im Extremfall komplett danebenliegen. Sinnvoller ist eine Glättung, die nur
eingreift, wenn die KI unsicher ist.“

📊 Alle 1192 Testbilder, Polynom 2. Grades pro Linie (wie die ROS-Node des Teams):

| | normale Anpassung (Team) | RANSAC |
|---|---|---|
| Sprung von Bild zu Bild, alle Bilder (Median) | 1,4–1,8 px | 1,6–1,9 px |
| Sprünge > 10 px, alle Bilder | 3–7 % | 4–9 % (schlechter) |
| Sprung, Blendungsszene 366–390 (Median) | 5,5–13,2 px | 2,2–3,5 px (besser) |
| Sprünge > 10 px, Blendungsszene | 35–60 % | 19–21 % (besser) |

- In 14–22 % der Bilder liegt mindestens ein Punkt einer Linie mehr als 10 px neben der glatten Kurve
  (Ausreißer). Am meisten bei der linken Linie.
- **RANSAC-Ausfall:** Bild 382 (Blendung): Die RANSAC-Kurve für die Mitte liegt oben bei x = −124,
  also außerhalb des Bildes. RANSAC hat sich auf eine falsche Punktgruppe festgelegt.
- Die Linke Linie wird nur in 69 % der Bilder mit mindestens 10 Punkten erkannt (Mitte 96 %, rechts 100 %).

🗣 Empfehlung fürs Team (Ausblick, nicht Kern der XAI-Arbeit):
1. Punkte mit niedriger Existenz-Wahrscheinlichkeit (`exist_row`) vor dem Anpassen weglassen. Die KI
   sagt selbst, wo sie unsicher ist.
2. RANSAC nur als Rückfall, wenn die normale Anpassung schlecht passt, und mit Plausibilitätsprüfung
   (Kurve muss im Bild bleiben, Abstand zu den Nachbarlinien muss stimmen).
3. Gegen das Springen von Bild zu Bild: zeitliche Glättung (z. B. Kalman-Filter).

⚠️
- Für die XAI-Experimente immer die **rohe** Ausgabe der KI messen, nicht die geglättete. Sonst
  misst man die Glättung statt der KI.
- Die Sprünge enthalten auch echte Bewegung des Fahrzeugs. Aufeinanderfolgende Bilder sind sich aber
  sehr ähnlich, deshalb sind große Sprünge fast immer Fehler.
- RANSAC-Einstellungen (Schwelle 4 px, 100 Versuche) nicht optimiert. Andere Werte können besser sein.

❓ *„Warum nicht einfach immer RANSAC?“* RANSAC geht davon aus, dass die Mehrheit der Punkte richtig
ist. Bei Blendung liegen oft viele Punkte falsch, dann kann RANSAC die falsche Gruppe als „richtig“
wählen. Bei sauberen Linien wirft es dagegen gute Punkte weg und wird dadurch unruhiger.

---

## Offene Punkte

- [x] Ergebnis Batch E/F in Abschnitt 5 eingetragen (2026-11-13)
- [x] Sanity Check für HiResCAM gemacht und bestanden (2026-11-16, Abschnitt 3)
- [ ] `docs/FOLIEN.md` ist veraltet (kennt E/F und Blendung nicht, sagt „Grad-CAM bleibt“). Löschen oder an diese Datei angleichen
- [ ] Optional: Blendung kontrolliert testen (echte Lichtstreifen aus Bild 380 auf saubere Bilder legen)
