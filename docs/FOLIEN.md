# Folien: Gliederung und Sprechtext

Ca. 10 Folien, ca. 10–12 Minuten. Alle Bilder liegen in `docs/bilder/`.
Die Sprechtexte sind Stichpunkte – sag sie in deinen eigenen Worten, nicht ablesen.

---

## Folie 1 – Titel

**Nutzt die Spurerkennung benachbarte Markierungen?**
Verdeckungsexperimente und CAM-Erklärungen an UFLDv2 (Smart Rollerz)

---

## Folie 2 – Die Frage

**Auf der Folie:** ein Kamerabild mit drei erkannten Linien (z. B. Bild 2 aus `schritte_2026_inpaint.png`).

**Sprechtext:**
- Unser Auto erkennt drei Linien: links, Mitte, rechts.
- Im echten Betrieb ist oft ein Stück einer Linie weg: abgefahren, Schatten, Blendung.
- Frage: Wenn ein Stück fehlt – schaut das Netz dann auf die **Nachbarlinie**, um es trotzdem zu finden?
- Warum wichtig: Wenn ja, kann *ein* Fehler (Nachbar fehlt) zwei Linien kaputt machen.

---

## Folie 3 – Das Modell

**Auf der Folie:** das Kettenbild aus `lernen/LERNPFAD.md` („Das große Bild“).

**Sprechtext:**
- UFLDv2 mit kleinem ResNet (9 Schichten), Eingabe: Draufsicht in Graustufen, 512×384.
- Es malt keine Linien, sondern legt **55 waagerechte Zeilen** übers Bild und sagt pro Zeile und Linie:
  *wo* (welches von 100 Kästchen) und *ob* die Linie da ist.
- Ergebnis: ein Punkt pro Zeile und Linie.

**Das musst du erklären können:** Was sind Zeilenanker? Was ist `loc_row`, was `exist_row`?
Warum ist der Kopf vollverbunden (wichtig für Folie 7)?

---

## Folie 4 – Methode: Verdecken

**Auf der Folie:** `schritte_2026_inpaint.png`

**Sprechtext:**
- Wir nehmen das mittlere Stück einer Linie (Anker 19–34) …
- … legen einen Streifen darüber (Maske) und retuschieren die Linie weg (inpaint).
- Das neue Bild geht wieder ins Modell. Wir messen: Wie weit verschiebt sich die Linie im verdeckten Stück?
- Gemessen in **% der Bildbreite**, damit beide Modelle vergleichbar sind.

---

## Folie 5 – Methode: die vier Bedingungen

**Auf der Folie:** Tabelle

| | verdeckt | Frage |
|---|---|---|
| A | Stück von T | Wie gut schätzt es die Lücke? |
| B | Stück von T + ganzer Nachbar N | Wird es schlechter ohne Nachbar? |
| C | nur N | Stört N's Fehlen sogar die sichtbare Linie? |
| D | Stück von T + Streifen auf leerer Straße | Kontrolle: schadet *jedes* Übermalen? |

**Sprechtext:**
- **B > A und B > D** → das Netz nutzt den Nachbarn (und nicht nur „mehr Bild kaputt“).
- **C groß** → das Netz hängt *immer* am Nachbarn → gelernte Spurbreite.
- T ← N heißt: Ziellinie T beobachten, Nachbar N wegnehmen. 4 Paare, weil links und rechts keine Nachbarn sind.

---

## Folie 6 – Ergebnis: altes gegen neues Modell (Hauptfolie!)

**Auf der Folie:** `vergleich_modelle_inpaint.png` (beide Modelle, gleiche y-Achse)

**Sprechtext:**
- Je ca. 40–60 Testbilder pro Paar.
- **Altes Modell:** B deutlich über A und D – bis 2,9 % der Bildbreite. Auch C ist groß.
  → Es nutzt den Nachbarn, vor allem die Außenlinien hängen an der Mitte. Wahrscheinlich gelernte Spurbreite.
- **2026er-Modell:** alle Balken klein, B höchstens 0,55 %. Ungefähr **5-mal weniger** Abhängigkeit.
  B > A ist zwar oft noch statistisch signifikant, aber praktisch winzig (bei 512 px ca. 1–2 px).
- Fazit: Das neue Modell nutzt vor allem den sichtbaren Rest der **eigenen** Linie.

**Vorsicht:** `batch_bev_2026_inpaint.png` allein hat eine y-Achse bis nur 0,55 % – da sehen die
Unterschiede groß aus. Für den Vergleich immer die gemeinsame Achse zeigen.

---

## Folie 7 – Erklärung: Grad-CAM gegen HiResCAM

**Auf der Folie:** untere Reihe von `bedingungen_2026_frame0.png`

**Sprechtext:**
- Rot = diese Bildstelle war wichtig für die Position der verdeckten Linie.
- **Grad-CAM:** Flecken übers halbe Bild verteilt, schwer zu deuten.
- **HiResCAM:** rot genau auf der Mittellinie und an den sichtbaren Enden des verdeckten Stücks.
- **Warum:** Grad-CAM mittelt die Gradienten eines Kanals über das ganze Bild. UFLDv2 hat einen
  vollverbundenen Kopf – da ist der Gradient an jeder Stelle anders, das Mitteln verwischt den Ort.
  HiResCAM rechnet Zelle für Zelle.
- Über alle Bilder: Der HiResCAM-Anteil auf dem Nachbarn hängt stärker mit dem gemessenen Effekt
  zusammen (Spearman 0,28–0,53) als bei Grad-CAM (−0,06–0,25).
  Bei der linken Linie zeigt HiResCAM ca. 3-mal mehr auf die Mitte, als es ihrer Fläche entspricht –
  passt dazu, dass die Außenlinien am meisten an der Mitte hängen.

---

## Folie 8 – Eine Schwäche: fehlende Linien werden erfunden

**Auf der Folie:** obere Reihe von `bedingungen_2026_frame0.png`, Bilder B und C

**Sprechtext:**
- Rechte Linie komplett wegretuschiert – das Modell setzt trotzdem rote Punkte, wild verstreut.
- Über alle Bilder sagt es bei einer weggenommenen Nachbarlinie im Schnitt mit **98 %**, dass sie da ist.
- Es sagt also nicht „keine Linie“, sondern rät. Für ein Auto ist „ich weiß es nicht“ sicherer als falsch raten.

---

## Folie 9 – Grenzen

- **Keine Labels:** Wir messen, wie stark sich das Modell *ändert*, nicht, ob es *richtig* liegt.
- **Masken folgen den Vorhersagen:** Wo das Modell eine Linie nur vermutet (z. B. Kreuzung, Bild 600),
  übermalen wir Straße statt Markierung.
- Nur **eine Testfahrt**, ca. 60 Bilder pro Paar.
- Die Füllung kann das Ergebnis beeinflussen – mit `median` statt `inpaint` sind die Zahlen etwas anders.
- CAM-Karten sind grob (12×16 Zellen).

---

## Folie 10 – Fazit

1. Das alte Modell nutzt Nachbarlinien messbar – eher als feste Spurbreite denn als gezieltes Lückenfüllen.
2. Das 2026er-Modell hängt ca. 5-mal weniger am Nachbarn und nutzt die eigene Linie.
3. Beide erfinden fehlende Linien, statt sie als fehlend zu melden.
4. Für UFLDv2 ist HiResCAM das passendere Erklärverfahren als Grad-CAM.

**Ausblick (ein Satz):** Training mit gezielt verdeckten Linien (Patch in `training/` liegt bereit),
damit das Modell lernt, „keine Linie“ zu sagen.

---

## Fragen, die der Prof wahrscheinlich stellt

- Warum braucht ihr Bedingung D?
- Warum % der Bildbreite und nicht Pixel?
- Was ist der Unterschied zwischen Grad-CAM und HiResCAM – in einem Satz?
- Was genau erklärt die CAM-Karte? (Antwort: die Punktzahl der gewählten Kästchen in den verdeckten Ankern)
- Signifikant, aber nur 1–2 px – ist das wichtig? (Ehrlich: statistisch ja, praktisch kaum)
- Woher wisst ihr, dass die Vorhersage auf dem Originalbild richtig ist? (Gar nicht – siehe Grenzen)
- Zeig mir im Code, wo die Maske gebaut wird. (`lernen/schritt5_verdecken.py` bzw. `src/xai_lane/occlusion.py`)
