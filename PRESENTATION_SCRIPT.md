# Präsentationsskript — BBC News Klassifikation (15 Min)

## Aufbau: McKinsey Storytelling-Framework (SCR)
- **Situation** → Komplikation → **Resolution** (Lösung)
- Roter Faden: Von der realen Problemstellung → durch technische Entscheidungen → zur funktionierenden App
- Fokus auf das WARUM (nicht nur WAS)

---

## Folie 1: Titel (30 Sek)

> Hallo zusammen, heute präsentiere ich mein Projekt: **BBC News Klassifikation mit Machine Learning**.
>
> Die zentrale Frage, die mich angetrieben hat, war:
> Kann ein Computer automatisch erkennen, ob ein Nachrichtenartikel über Sport, Politik, Business, Technologie oder Entertainment handelt — und zwar zuverlässig genug, dass man sich darauf verlassen kann?

---

## Folie 2: Situation — Das Problem (1 Min)

> Um das einzuordnen: Große Nachrichtenredaktionen wie die BBC veröffentlichen täglich Hunderte Artikel. Jeder einzelne muss einer Kategorie zugeordnet werden — für die Website, für den Newsletter, für die App.
>
> Bisher macht das ein Mensch. Aber das ist zeitaufwändig, es ist teuer, und vor allem: Es passieren Fehler. Ein Artikel über einen Wirtschaftsskandal bei einem Fußballverein — ist das Sport oder Business?
>
> Die Motivation hinter meinem Projekt war also: **Diesen Prozess zu automatisieren** — und dabei zu lernen, wie eine vollständige Machine-Learning-Pipeline funktioniert: von Rohdaten über die Datenbank, das Modell bis hin zur Benutzeroberfläche.

---

## Folie 3: Komplikation — Warum ist das schwierig? (1 Min)

> Das klingt erstmal simpel. Aber für einen Computer ist das alles andere als einfach.
>
> **Computer verstehen keine Wörter.** Sie können nicht lesen. Sie arbeiten nur mit Zahlen.
>
> Das heißt: Wir müssen drei grundlegende Probleme lösen:
>
> 1. **Datenhaltung:** 2.225 Textdateien in 5 Ordnern — das ist unstrukturiert. Wir brauchen eine sinnvolle Datenbank.
> 2. **Feature Extraction:** Wir müssen einen Weg finden, Text in Zahlen umzuwandeln — so, dass die Bedeutung erhalten bleibt.
> 3. **Modellwahl:** Welcher Algorithmus passt am besten zu unserem Problem?
>
> Jede dieser Entscheidungen habe ich bewusst getroffen, und ich möchte euch jetzt erklären, **warum**.

---

## Folie 4: Mein Ansatz — Die Pipeline im Überblick (1 Min)

> Mein Ansatz ist eine vierstufige Pipeline. Ich zeige euch kurz den Überblick, dann gehen wir Schritt für Schritt durch:
>
> **Schritt 1:** Datenimport in MongoDB — strukturierte Speicherung mit Metadaten.
>
> **Schritt 2:** Feature Extraction mit TF-IDF — Text wird zu Zahlenvektoren.
>
> **Schritt 3:** Modelltraining — ich vergleiche Naive Bayes und SVM.
>
> **Schritt 4:** Streamlit Web-App — damit ein Nutzer das Modell tatsächlich verwenden kann.
>
> Der rote Faden ist dabei: **Rohdaten → Struktur → Verstehen → Anwenden.** Jeder Schritt baut auf dem vorherigen auf.

---

## Folie 5: Schritt 1 — Datenimport & Medienwahl MongoDB (1,5 Min)

> Der Datensatz kommt von Kaggle — 2.225 BBC-Artikel aus dem Jahr 2004/2005, verteilt auf 5 Kategorien:
>
> - Sport: 511 Artikel
> - Business: 510
> - Politics: 417
> - Tech: 401
> - Entertainment: 386
>
> Jeder Artikel ist eine einfache `.txt`-Datei in einem Ordner.
>
> **Warum habe ich MongoDB gewählt — und nicht einfach ein CSV?**
>
> Drei Gründe:
> 1. **Flexibilität:** MongoDB speichert Dokumente als JSON-ähnliche Objekte. Ich kann jederzeit neue Felder hinzufügen — zum Beispiel Veröffentlichungsdatum oder Autor — ohne die ganze Struktur zu ändern.
> 2. **Praxisnähe:** In der realen Welt würden neue Artikel ständig dazukommen. Mit MongoDB kann ich einfach neue Dokumente einfügen, ohne eine Datei komplett neu zu schreiben.
> 3. **Aufgabenstellung:** Die Aufgabe verlangte explizit eine MongoDB-Integration.
>
> Jedes Dokument in der Datenbank hat sinnvolle Metadaten-Felder: Dateiname, Kategorie, Titel — also die erste Zeile des Artikels — der volle Text und die Wortanzahl.

---

## Folie 6: Schritt 2 — Feature Extraction mit TF-IDF (3 Min) ⭐

> Jetzt kommen wir zum Kernstück: **Wie wandeln wir Text in Zahlen um?**
>
> Ich habe mich für **TF-IDF** entschieden — und ich erkläre euch gleich, warum gerade diese Methode.
>
> TF-IDF steht für *Term Frequency – Inverse Document Frequency*.
>
> ---
>
> **Schritt 1: Term Frequency (TF)**
>
> Das ist intuitiv: Wie oft kommt ein Wort in einem bestimmten Artikel vor?
>
> Wenn „goal" in einem Sportartikel 5 Mal vorkommt, hat es eine hohe Term Frequency.
>
> ---
>
> **Schritt 2: Inverse Document Frequency (IDF)**
>
> Jetzt wird es spannend. Manche Wörter wie „the", „is", „and" kommen in *jedem* Artikel vor. Die helfen uns nicht bei der Klassifikation.
>
> IDF bestraft genau solche Wörter. Die Logik:
>
> - Wort kommt in **allen** 2.225 Dokumenten vor → IDF ≈ 0 → **unwichtig**
> - Wort kommt nur in **wenigen** Dokumenten vor → IDF hoch → **sehr informativ!**
>
> Konkret:
> - „the" → überall → IDF fast 0 → nutzlos
> - „goal" → fast nur in Sport → hoher IDF → **starkes Signal!**
> - „profit" → fast nur in Business → hoher IDF → **starkes Signal!**
>
> ---
>
> **TF × IDF = TF-IDF-Wert**
>
> Wir multiplizieren beide. Das Ergebnis sagt: **Wie wichtig ist dieses Wort für genau diesen Artikel — im Vergleich zu allen anderen?**
>
> ---
>
> **Warum genau 10.000 Features?**
>
> Der TfidfVectorizer aus scikit-learn baut ein Vokabular aus allen Artikeln auf. Ich habe `max_features=10000` gewählt — das sind die 10.000 aussagekräftigsten Wörter und **Wortpaare** (sogenannte Bigrams).
>
> Warum Bigrams? Weil „Champions League" als Wortpaar viel informativer ist als die einzelnen Wörter „Champions" und „League" getrennt. „Stock market" sagt mehr als „stock" allein.
>
> Am Ende wird jeder Artikel zu einem **Vektor mit 10.000 Zahlen** — eine Art Fingerabdruck:
>
> | Artikel | goal | profit | election | film | ... |
> |---------|------|--------|----------|------|-----|
> | Sport-001 | 0.45 | 0.00 | 0.00 | 0.01 | ... |
> | Business-001 | 0.00 | 0.52 | 0.03 | 0.00 | ... |
>
> So „sieht" der Computer die Texte — als Muster von Zahlen.
>
> ---
>
> **Warum TF-IDF und nicht etwas anderes?**
>
> Es gibt natürlich Alternativen — zum Beispiel Word Embeddings wie Word2Vec, oder moderne Transformer-Modelle wie BERT.
>
> Aber für meinen Anwendungsfall — 2.225 kurze Nachrichtenartikel mit 5 klar abgegrenzten Kategorien — ist TF-IDF die richtige Wahl:
> - **Effizient** — das Training dauert Sekunden, nicht Stunden
> - **Interpretierbar** — ich kann sehen, welche Wörter wichtig sind
> - **Bewährt** — TF-IDF ist seit Jahrzehnten der Standard in der Textklassifikation
>
> Deep Learning wäre hier Overkill — wie mit einer Kanone auf Spatzen schießen.

---

## Folie 7: Schritt 3 — Modellwahl: Naive Bayes vs. SVM (3 Min) ⭐

> Jetzt haben wir Zahlen. Aber wie lernt der Computer daraus, Kategorien zu erkennen?
>
> Ich habe **zwei Modelle** trainiert und verglichen. Warum zwei? Weil ich nachvollziehbar zeigen wollte, welches besser passt — nicht einfach blind eines nehmen.
>
> Die Daten habe ich aufgeteilt: 80% zum Trainieren (1.780 Artikel), 20% zum Testen (445 Artikel). Das Modell sieht die Testdaten **nie** während des Trainings — so messen wir, wie gut es wirklich generalisiert.
>
> ---
>
> **Modell 1: Multinomial Naive Bayes**
>
> Die Grundidee ist Wahrscheinlichkeit. Naive Bayes lernt:
>
> „Wenn in einem Artikel die Wörter ‚goal', ‚match', ‚player' hohe TF-IDF-Werte haben — mit welcher Wahrscheinlichkeit ist das Sport?"
>
> Er berechnet für jede der 5 Kategorien eine Wahrscheinlichkeit und wählt die höchste.
>
> Warum heißt es „Naive"? Weil der Algorithmus annimmt, dass alle Wörter **unabhängig** voneinander sind. Das stimmt natürlich nicht — „Champions" und „League" gehören zusammen. Aber — und das ist das Überraschende — in der Praxis funktioniert diese naive Annahme bei TF-IDF-Vektoren erstaunlich gut. Der Grund: TF-IDF normalisiert die Werte bereits so stark, dass die Korrelationen zwischen Wörtern weniger ins Gewicht fallen.
>
> **Ergebnis: 99,10% Accuracy** — nur 4 Fehler bei 445 Testartikeln.
>
> ---
>
> **Modell 2: LinearSVC (Support Vector Machine)**
>
> SVM denkt komplett anders. Stellt euch vor: Jeder Artikel ist ein Punkt in einem 10.000-dimensionalen Raum.
>
> SVM versucht, eine **Trennebene** (Hyperebene) zu finden, die die Kategorien voneinander trennt — und zwar so, dass der **Abstand** zu den nächsten Punkten **maximal** ist. Das nennt man „Maximum Margin".
>
> Warum LinearSVC? Weil bei 10.000 Features ein linearer Kernel ausreicht — die Daten sind in so einem hochdimensionalen Raum oft bereits linear trennbar. Ein komplexerer Kernel (z.B. RBF) würde nur die Trainingszeit erhöhen, ohne bessere Ergebnisse zu liefern.
>
> **Ergebnis: 98,88% Accuracy** — nur 5 Fehler bei 445 Testartikeln.
>
> ---
>
> **Meine Entscheidung:**
>
> Beide Modelle sind hervorragend. Aber ich habe mich für **Naive Bayes** als finales Modell entschieden. Warum?
>
> 1. **Minimal bessere Accuracy:** 99,10% vs. 98,88%
> 2. **Schneller:** Naive Bayes trainiert in Millisekunden
> 3. **Wahrscheinlichkeiten:** Naive Bayes liefert echte Konfidenzwerte — ich kann dem Nutzer zeigen: „70% Sport, 20% Business, 10% Rest". SVM gibt das nicht direkt aus.
>
> Gerade der dritte Punkt war mir für die Streamlit-App wichtig — die Nutzerführung profitiert davon, dass man die Sicherheit der Vorhersage sehen kann.

---

## Folie 8: Confusion Matrix — Evaluation (1 Min)

> Zur Evaluation: Hier sehen wir die Confusion Matrices beider Modelle.
>
> Kurze Erklärung: Auf der Y-Achse steht die **tatsächliche** Kategorie, auf der X-Achse die **vorhergesagte**. Die Diagonale zeigt korrekte Vorhersagen.
>
> Bei Naive Bayes: Die Diagonale ist fast perfekt — 99,1% korrekt. Es gab nur 4 Verwechslungen bei 445 Tests.
>
> Was ich besonders interessant fand: Die wenigen Fehler passieren dort, wo es auch für Menschen schwierig wäre — zum Beispiel ein Artikel über Technologie-Investitionen, der zwischen Tech und Business liegt.
>
> Das zeigt: Das Modell macht nicht zufällige Fehler, sondern scheitert an denselben Grenzfällen wie ein Mensch.

---

## Folie 9: Streamlit App — Interaktionskonzept (1,5 Min)

> Der letzte Schritt war mir besonders wichtig: **Eine Benutzeroberfläche bauen**, damit man das Modell tatsächlich ausprobieren kann.
>
> Ich habe Streamlit gewählt — ein Python-Framework für einfache Web-Apps. Warum?
>
> 1. **Minimaler Aufwand:** Weniger als 80 Zeilen Code für eine funktionierende App
> 2. **Python-nativ:** Ich bleibe in der gleichen Sprache wie mein ML-Code
> 3. **Sofortiges Feedback:** Kein Deployment auf einem Server nötig — `streamlit run app.py` und es läuft
>
> **Zum Interaktionskonzept:**
>
> Die Nutzerführung ist bewusst simpel gehalten:
>
> - **Eingabe:** Ein einzelnes Textfeld — der Nutzer fügt einen Nachrichtenartikel ein
> - **Aktion:** Ein Button „Klassifizieren"
> - **Feedback:** Sofort erscheint die vorhergesagte Kategorie — mit einem farbigen Emoji — UND ein Balkendiagramm mit den Konfidenzwerten aller 5 Kategorien
>
> Warum das Balkendiagramm? Weil ein Nutzer nicht nur wissen will **was**, sondern auch **wie sicher** sich das Modell ist. Wenn Sport bei 95% steht, kann man sich darauf verlassen. Wenn es 45% Sport und 40% Business zeigt, weiß man: Dieser Artikel ist ein Grenzfall.
>
> Das ist bewusstes **Feedback-Design** — der Nutzer versteht die Stärken und Grenzen des Modells.
>
> *(Jetzt zeige ich kurz eine Demo)*

---

## Folie 10: LIVE-DEMO (1 Min)

> *(Streamlit-App öffnen)*
>
> Ich gebe hier mal einen kurzen Text ein:
>
> *„The team scored three goals in the second half to win the championship."*
>
> *(Klick auf Klassifizieren)*
>
> Wie erwartet: **Sport** — mit hoher Konfidenz.
>
> Versuchen wir noch eins:
>
> *„The company reported record profits and plans to expand into new markets."*
>
> **Business.** Auch hier ist das Modell sehr sicher.
>
> Man sieht: Das funktioniert in Echtzeit und ist sofort verständlich.

---

## Folie 11: Kritische Reflexion (1,5 Min)

> Zum Schluss möchte ich mein Ergebnis auch kritisch hinterfragen — denn 99% Accuracy klingt fast zu gut.
>
> **Was hat gut funktioniert?**
>
> - Die Kombination TF-IDF + Naive Bayes ist ideal für dieses Problem: schnell, effizient, interpretierbar
> - Die Pipeline ist vollständig reproduzierbar — vom Rohdaten-Import bis zur Web-App
> - Die Streamlit-App macht das Ergebnis greifbar und benutzbar
>
> **Was sind die Grenzen?**
>
> - **Der Datensatz ist relativ klein** — 2.225 Artikel. Bei 100.000 Artikeln müsste man prüfen, ob die Modelle noch so gut performen.
> - **Nur 5 Kategorien** — in der Realität gibt es dutzende Themenbereiche, die sich stärker überlappen.
> - **Nur englische Texte** — das Modell funktioniert nicht für deutsche oder andere Sprachen, weil das TF-IDF-Vokabular sprachspezifisch ist.
> - **Die Daten sind von 2004/2005** — Begriffe und Themen haben sich verändert. „Brexit" oder „COVID" kennt das Modell nicht.
>
> **Was würde ich beim nächsten Mal anders machen?**
>
> - **Cross-Validation** statt eines einzigen Train/Test-Splits — das gibt robustere Ergebnisse
> - **Hyperparameter-Tuning** — zum Beispiel systematisch verschiedene `max_features`-Werte testen
> - **Einen moderneren Ansatz vergleichen** — zum Beispiel einen vortrainierten BERT-Classifier, um zu sehen, ob die Mehraufwand an Komplexität wirklich bessere Ergebnisse bringt
> - **Mehr Nutzertests** für die Streamlit-App — wie versteht ein nicht-technischer Nutzer die Konfidenzwerte?

---

## Folie 12: Tech Stack & Zusammenfassung (30 Sek)

> Kurze Zusammenfassung der verwendeten Technologien:
>
> - **Daten:** MongoDB + pymongo
> - **Machine Learning:** scikit-learn (TF-IDF, Naive Bayes, SVM)
> - **Visualisierung:** matplotlib + seaborn
> - **Web-App:** Streamlit
> - **Serialisierung:** joblib für die .pkl-Modelldateien
>
> Die Kernaussage meines Projekts: **Man braucht kein Deep Learning, um Textklassifikation zuverlässig zu lösen.** Die richtigen klassischen Methoden, sinnvoll kombiniert, erreichen 99% — schnell, interpretierbar und einsetzbar.
>
> Vielen Dank für eure Aufmerksamkeit! Ich freue mich auf Fragen.

---

## Mögliche Fragen & Antworten

**F: Warum nicht Deep Learning / BERT / ChatGPT?**
> Gute Frage. Für diesen Datensatz — 2.225 Artikel, 5 klar getrennte Kategorien — wäre Deep Learning Overkill. Naive Bayes erreicht bereits 99% in Millisekunden. BERT müsste man fine-tunen, bräuchte eine GPU, und der Mehraufwand würde vermutlich weniger als 1% Verbesserung bringen. Aber: Bei einem größeren, komplexeren Datensatz würde ich definitiv einen Transformer-Ansatz in Betracht ziehen.

**F: Warum MongoDB und nicht SQLite oder PostgreSQL?**
> MongoDB ist dokumentenorientiert — das passt perfekt zu Nachrichtenartikeln, die im Grunde Dokumente sind. Außerdem ist das Schema flexibel: Ich kann jederzeit neue Felder hinzufügen, ohne Migration. Für ein produktives System mit vielen gleichzeitigen Zugriffen wäre PostgreSQL aber möglicherweise die bessere Wahl.

**F: Was sind Bigrams?**
> Wortpaare. „stock market" als Bigram ist informativer als die einzelnen Wörter „stock" und „market". Mein TF-IDF-Vokabular enthält sowohl Einzelwörter (Unigrams) als auch Wortpaare (Bigrams) — das gibt dem Modell mehr Kontext.

**F: Funktioniert das auch mit deutschen Texten?**
> Nicht mit diesem Modell — das Vokabular ist komplett englisch. Man müsste neue deutsche Trainingsdaten sammeln, ein neues TF-IDF-Vokabular aufbauen und das Modell neu trainieren. Das Prinzip bleibt aber dasselbe.

**F: Was passiert bei einem Artikel über ein völlig neues Thema?**
> Das Modell wählt trotzdem eine der 5 Kategorien — die mit der höchsten Wahrscheinlichkeit. In der Praxis sollte man einen Schwellenwert einbauen: Wenn keine Kategorie über z.B. 40% Konfidenz liegt, gibt man „Nicht zuordenbar" aus. Das wäre eine sinnvolle Erweiterung.

**F: Wie lang hat das Training gedauert?**
> Naive Bayes: unter 1 Sekunde. SVM: wenige Sekunden. Das ist einer der großen Vorteile klassischer Methoden. Ein BERT-Modell bräuchte für denselben Datensatz mehrere Minuten bis Stunden, je nach Hardware.

**F: Gibt es Overfitting bei 99% Accuracy?**
> Berechtigte Frage. Der Datensatz hat sehr klare Kategorien — Sportartikel verwenden ganz andere Wörter als Politikartikel. Deshalb ist 99% hier realistisch und kein Zeichen von Overfitting. Trotzdem wäre Cross-Validation eine gute Absicherung — das würde ich beim nächsten Mal ergänzen.
