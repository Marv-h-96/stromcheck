# ⚡ StromCheck Deutschland

Echtzeit-Dashboard für den deutschen Strommix – mit KI-Kommentar, Wetterprognose, Geräteplaner und Solar-Analyse.

**Live-App:** [wann-waschmaschine-anstellen.streamlit.app](https://wann-waschmaschine-anstellen.streamlit.app/)

---

## Was die App zeigt

### Hero-Bereich
- **Aktueller Erneuerbaren-Anteil** mit Farbampel (grün / gelb / rot)
- **Bestes Zeitfenster heute & morgen** – wann lohnt es sich Waschmaschine, Spülmaschine oder E-Auto zu starten? Zeigt „Jetzt", wenn der aktuelle Wert besser ist als jedes prognostizierte Fenster
- **CO₂-Intensität** – Gesamtemissionen heute vs. Klimaziel 2030 (inkl. Überschreitungszeitpunkt)
- **KI-Tageskommentar** – personalisierter Text mit Alltagsvergleichen (z. B. wie viele Bierflaschen das Brauen einer Waschmaschine entspricht) und leichtem Humor

### Solaranlage *(optional, Sidebar-Toggle)*
- Anlage konfigurieren: kWp, Ausrichtung, Neigungswinkel
- **PV-Ertragsprognose** – erwartete kWh für heute und morgen
- **Kombinierter Chart** – eigene Erzeugung vs. Grünstrom-Anteil im Netz
- **Eigenverbrauchsampel** – stundengenaue Empfehlung: Eigenverbrauch, Einspeisung oder Warten

### Geräteplaner
- Gerät auswählen (Waschmaschine, Trockner, E-Auto, Wärmepumpe …)
- Deadline setzen → App findet das grünste Zeitfenster im 48h-Forecast
- Zeigt CO₂-Einsparung gegenüber sofortigem Start
- KI erklärt in 1–2 Sätzen warum dieses Fenster empfohlen wird

### Verlauf & Prognose
- **Erneuerbarer-Anteil** – konfigurierbarer Zeitraum (Heute bis 4 Wochen)
- **48h-Heatmap** – stündliche Prognose für heute und morgen
- **Wetterprognose** – Wind und Globalstrahlung am gewählten Standort (Open-Meteo)
- **Forecast-Kurve** – Erneuerbare-Prognose mit Unsicherheitsband (lineares Wettermodell, R² angezeigt)

### Details & Hintergrund *(aufklappbar)*
- Gestapelter Strommix-Chart (erneuerbar vs. fossil)
- Grünstrom-Heatmap nach Wochentag & Uhrzeit
- CO₂-Intensität im Verlauf
- **CO₂-Tageszähler** – kumulierte Emissionen heute mit Fülldiagramm und Klimaziel-Linie
- Rohdaten-Tabellen (SMARD + Open-Meteo)

---

## Architektur

```
stromcheck/
├── app.py           # Streamlit-Dashboard (Hauptdatei)
├── fetcher.py       # SMARD-API-Abruf + Disk-Cache
├── forecast.py      # Wettermodell (Open-Meteo + lineares Regressionsmodell)
├── fetch_data.py    # Standalone-Script für GitHub Actions
├── cache/
│   └── strommix_28d.json   # Vorberechneter 28-Tage-Cache
└── .github/workflows/
    └── update_cache.yml    # Wird per cron-job.org stündlich getriggert
```

### Datenfluss
1. **cron-job.org** triggert stündlich die GitHub Action
2. GitHub Action führt `fetch_data.py` aus → aktualisiert `cache/strommix_28d.json`
3. GitHub committed die Datei → Streamlit Cloud deployed automatisch
4. App liest beim Start aus der JSON-Datei statt direkt aus der API → **Ladezeit < 1 Sekunde**

---

## Datenquellen

| Quelle | Inhalt | Kosten |
|---|---|---|
| [SMARD.de](https://www.smard.de) – Bundesnetzagentur | Stündlicher Strommix (MW je Quelle, Gesamtlast) | kostenlos, keine Auth |
| [Open-Meteo](https://open-meteo.com) | Wetter-Forecast + Historik (Wind, Strahlung) | kostenlos, keine Auth |
| [Open-Meteo Geocoding](https://open-meteo.com/en/docs/geocoding-api) | Ortssuche (Deutschland) | kostenlos, keine Auth |
| [Google Gemini](https://ai.google.dev) | KI-Tageskommentar & Geräteplaner-Erklärung | kostenlos (Free Tier) |

**CO₂-Faktoren:** IPCC AR6 Medianwerte (Lebenszyklus) – Braunkohle 820, Erdgas 490, Solar 45, Wind ~11 g CO₂/kWh

---

## Lokale Installation

```bash
git clone https://github.com/Marv-h-96/stromcheck.git
cd stromcheck
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Gemini API-Key in `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "dein-api-key"
```

App starten:
```bash
streamlit run app.py
```

Initialen Cache befüllen (einmalig):
```bash
python fetch_data.py
```

---

## Tech Stack

- **[Streamlit](https://streamlit.io)** – Web-App-Framework
- **[Plotly](https://plotly.com)** – interaktive Charts
- **[Pandas](https://pandas.pydata.org)** – Datenverarbeitung
- **[NumPy](https://numpy.org)** – lineares Regressionsmodell (lstsq)
- **[google-genai](https://ai.google.dev)** – Gemini API Client
- **GitHub Actions** – CI/CD + Cache-Automatisierung
