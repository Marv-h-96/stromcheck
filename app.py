import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import requests
from fetcher import get_strommix, load_from_cache
from forecast import wetter_prognose

st.set_page_config(page_title="StromCheck Deutschland", page_icon="⚡", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _geocode(query: str):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 5, "language": "de", "format": "json"},
            timeout=5,
        )
        r.raise_for_status()
        out = []
        for res in r.json().get("results", []):
            if res.get("country_code") != "DE":
                continue
            label = res["name"]
            if res.get("admin1"):
                label += f", {res['admin1']}"
            out.append({"label": label, "lat": res["latitude"], "lon": res["longitude"]})
        return out
    except Exception:
        return []


CO2_FAKTOREN = {
    "Braunkohle": 820, "Steinkohle": 740, "Erdgas": 490, "Kernenergie": 12,
    "Wind Onshore": 11, "Wind Offshore": 12, "Solar": 45, "Biomasse": 230,
    "Wasserkraft": 24, "Pumpspeicher": 30, "Sonstige Konventionelle": 600,
    "Sonstige Erneuerbare": 50,
}

def berechne_co2(row):
    gesamt_mw = sum(row.get(q, 0) or 0 for q in CO2_FAKTOREN)
    if gesamt_mw <= 0:
        return np.nan
    return round(sum((row.get(q, 0) or 0) * f for q, f in CO2_FAKTOREN.items()) / gesamt_mw, 1)


AUSRICHTUNG_FAKTOR = {
    "Süd": 1.00, "Süd-West": 0.97, "Süd-Ost": 0.97,
    "West": 0.82, "Ost": 0.82, "Nord": 0.55,
}

def berechne_pv(df_fc, kwp, ausrichtung, neigung):
    """Schätzt stündlichen PV-Ertrag (kW) aus Strahlungsdaten."""
    if df_fc is None or "strahlung" not in df_fc.columns:
        return None
    orient_f  = AUSRICHTUNG_FAKTOR.get(ausrichtung, 1.0)
    neigung_f = max(0.70, 1.0 - abs(neigung - 32) * 0.006)
    df = df_fc[["zeit", "prognose", "strahlung"]].copy()
    df["pv_kw"] = (df["strahlung"] / 1000 * kwp * 0.80 * orient_f * neigung_f).clip(lower=0)
    return df


def pv_empfehlung(pv_kw, renewable_pct, mit_speicher):
    if pv_kw >= 0.1:
        if renewable_pct < 50:
            return "⚡ Eigenverbrauch maximieren", "#27ae60"
        return "↗ Einspeisung — Netz ist grün", "#1abc9c"
    if renewable_pct > 65 and mit_speicher:
        return "🔋 Grüner Strom — Speicher laden", "#3498db"
    if renewable_pct < 40:
        return "🚫 Fossiles Netz — Verbrauch reduzieren", "#c0392b"
    return "~ Gemischter Mix", "#d4ac0d"


def _pattern_fallback(src, now):
    df_h = src[["Erneuerbare_Anteil_%"]].copy()
    df_h.index = df_h.index.tz_localize(None) if df_h.index.tz else df_h.index
    df_h["wd"] = df_h.index.dayofweek
    df_h["h"]  = df_h.index.hour
    mu = df_h.groupby(["wd", "h"])["Erneuerbare_Anteil_%"].agg(["mean", "std"]).reset_index()
    mu["std"] = mu["std"].fillna(8)
    zeiten = [now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i) for i in range(1, 49)]
    rows = []
    for t in zeiten:
        m = mu[(mu["wd"] == t.weekday()) & (mu["h"] == t.hour)]
        if not m.empty:
            v, s = m["mean"].values[0], m["std"].values[0]
            rows.append({"zeit": t, "prognose": round(v, 1),
                         "untere": round(max(0, v - s), 1), "obere": round(min(130, v + s), 1)})
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def _ki_tageskommentar(
    aktuell: float, co2: float, top_str: str, fenster_h: str, fenster_m: str,
    standort: str, wind_kmh: float | None, strahlung_wm2: float | None,
    solar_kwp: float | None, solar_ausrichtung: str | None,
) -> str | None:
    try:
        from google import genai
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        wetter_ctx = (
            f"Aktuelles Wetter in {standort}: Wind {wind_kmh:.0f} km/h, "
            f"Globalstrahlung {strahlung_wm2:.0f} W/m². "
        ) if wind_kmh is not None and strahlung_wm2 is not None else (
            f"Standort des Nutzers: {standort}. "
        )

        solar_ctx = (
            f"Der Nutzer betreibt in {standort} eine Solaranlage mit {solar_kwp} kWp "
            f"({solar_ausrichtung} ausgerichtet). Beziehe die aktuelle Strahlungssituation "
            f"konkret auf seinen Ertrag ein. "
        ) if solar_kwp is not None and solar_ausrichtung is not None else ""

        klimaziel_status = (
            f"im Zielbereich – Glückwunsch!" if co2 <= 100
            else f"{co2/100:.1f}× über dem Klimaziel 2030 (Ø 100 g/kWh Jahresdurchschnitt)"
        )
        # Konkrete Gerätevergleiche (greifbarer als "pro kWh")
        co2_waschmaschine = round(co2 * 1.0)    # ~1 kWh
        co2_spuelmaschine = round(co2 * 1.2)    # ~1.2 kWh
        co2_eauto_100km   = round(co2 * 20 / 1000, 2)  # ~20 kWh / 100 km → kg
        co2_bier_wm       = round(co2_waschmaschine / 500, 2)  # Bierflaschen (500g CO₂/0,5l)
        prompt = (
            f"Du bist ein freundlicher, leicht humorvoller Energie-Assistent für eine deutsche Strommix-App. "
            f"Der Nutzer ist in {standort}. "
            f"Aktuelle Lage: {aktuell:.0f}% Erneuerbare, {co2:.0f} g CO₂/kWh – {klimaziel_status}. "
            f"Konkrete Gerätevergleiche für den aktuellen Strommix (verwende NUR diese, KEIN 'pro kWh'): "
            f"Eine Waschmaschine (1 kWh) erzeugt gerade {co2_waschmaschine}g CO₂ ≈ {co2_bier_wm:.1f} Bierflaschen. "
            f"Eine Spülmaschine ({co2_spuelmaschine}g). "
            f"100 km E-Auto ({co2_eauto_100km:.2f} kg CO₂). "
            f"Stärkste Quellen: {top_str}. "
            f"{wetter_ctx}"
            f"{solar_ctx}"
            f"Bestes Zeitfenster – heute: {fenster_h}, morgen: {fenster_m}. "
            f"Schreibe 2–3 Sätze auf Deutsch direkt an den Nutzer in {standort}. "
            f"Beziehe das lokale Wetter konkret ein. "
            f"Verwende einen der Gerätevergleiche mit Alltagshumor – Bier, E-Auto oder Spülmaschine. "
            f"Das Klimaziel kurz erwähnen, gerne mit Augenzwinkern wenn es verfehlt wird. "
            f"Alltagsnaher Tipp am Ende. Kein Markdown, kein Fettdruck."
        )
        response = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
        return response.text.strip()
    except KeyError:
        return None  # Kein API-Key konfiguriert
    except Exception as e:
        st.toast(f"KI-Kommentar: {e}", icon="⚠️")
        return None


def bestes_fenster(df_tag):
    if df_tag is None or len(df_tag) < 2:
        return None
    rollen = df_tag["prognose"].rolling(2).mean()
    idx = rollen.idxmax()
    return df_tag.loc[idx, "zeit"] - timedelta(hours=1), df_tag.loc[idx, "zeit"], round(rollen[idx], 1)


def co2_schaetzen(ren_pct: float) -> float:
    """Schätzt CO2-Intensität (g/kWh) aus Erneuerbaren-Anteil (lineares Modell)."""
    return max(30.0, round(650 - 6.0 * ren_pct, 0))


def bestes_geraete_fenster(df_fc, dauer_h, deadline_ts, now):
    """Findet das grünste N-Stunden-Fenster vor der Deadline."""
    n = max(1, round(dauer_h))
    df = df_fc[
        (df_fc["zeit"] > now) &
        (df_fc["zeit"] + pd.Timedelta(hours=dauer_h) <= deadline_ts)
    ].reset_index(drop=True)
    if len(df) < n:
        return None
    best_avg, best_i = -1, 0
    for i in range(len(df) - n + 1):
        avg = df["prognose"].iloc[i:i + n].mean()
        if avg > best_avg:
            best_avg, best_i = avg, i
    start = df["zeit"].iloc[best_i]
    return start, start + pd.Timedelta(hours=dauer_h), round(best_avg, 1)


@st.cache_data(ttl=3600, show_spinner=False)
def _ki_geraet(geraet: str, start_str: str, ende_str: str,
               avg_ren: float, aktuell_ren: float, standort: str) -> str | None:
    try:
        from google import genai
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt = (
            f"Ein Nutzer in {standort} möchte {geraet} betreiben. "
            f"Das grünste Zeitfenster im 48h-Forecast ist {start_str}–{ende_str} Uhr "
            f"mit Ø {avg_ren:.0f}% Erneuerbaren (aktuell: {aktuell_ren:.0f}%). "
            f"Erkläre in 1–2 Sätzen auf Deutsch warum dieses Fenster empfohlen wird "
            f"und ob es sich lohnt so lange zu warten. "
            f"Kein Markdown, locker und direkt formuliert."
        )
        return client.models.generate_content(
            model="gemini-3.1-flash-lite", contents=prompt
        ).text.strip()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "ort_lat" not in st.session_state:
    st.session_state.ort_lat  = 51.2
    st.session_state.ort_lon  = 10.4
    st.session_state.ort_name = "Deutschland (Mitte)"

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Einstellungen")

    # Zeitraum
    zeitraum_optionen = {"Heute": 1, "3 Tage": 3, "1 Woche": 7, "2 Wochen": 14, "4 Wochen": 28}
    zeitraum_label = st.selectbox("Zeitraum (Verlauf)", list(zeitraum_optionen.keys()), index=2)
    tage = zeitraum_optionen[zeitraum_label]

    st.markdown("---")

    # ── Standort ──────────────────────────────────────────────────────────────
    st.subheader("📍 Standort")
    ort_eingabe = st.text_input("Ort suchen", placeholder="z.B. München, Freiburg, Kiel …")

    if st.button("🔍 Suchen", use_container_width=True) and ort_eingabe.strip():
        treffer = _geocode(ort_eingabe.strip())
        st.session_state.geo_treffer = treffer
        if not treffer:
            st.warning("Kein Ort in Deutschland gefunden.")

    if st.session_state.get("geo_treffer"):
        treffer  = st.session_state.geo_treffer
        namen    = [t["label"] for t in treffer]
        auswahl  = st.selectbox("Ergebnis wählen", namen)
        if st.button("✓ Übernehmen", use_container_width=True):
            e = next(t for t in treffer if t["label"] == auswahl)
            st.session_state.ort_lat   = e["lat"]
            st.session_state.ort_lon   = e["lon"]
            st.session_state.ort_name  = e["label"]
            st.session_state.geo_treffer = []
            st.rerun()

    st.caption(f"📌 {st.session_state.ort_name}")
    st.caption(f"{st.session_state.ort_lat:.2f}°N · {st.session_state.ort_lon:.2f}°E")

    stadtname = st.session_state.ort_name
    stadt_lat = st.session_state.ort_lat
    stadt_lon = st.session_state.ort_lon

    st.markdown("---")

    # ── Solaranlage ───────────────────────────────────────────────────────────
    st.subheader("☀️ Meine Solaranlage")
    solar_aktiv = st.toggle("Solaranlage vorhanden", value=False)

    if solar_aktiv:
        solar_kwp         = st.slider("Anlagenleistung (kWp)", 1.0, 30.0, 8.0, 0.5)
        solar_ausrichtung = st.selectbox("Ausrichtung", ["Süd", "Süd-West", "Süd-Ost", "West", "Ost", "Nord"])
        solar_neigung     = st.selectbox("Neigungswinkel", [15, 20, 25, 30, 35, 40], index=3)

# ═══════════════════════════════════════════════════════════════════════════════
# DATEN LADEN
# ═══════════════════════════════════════════════════════════════════════════════
df = load_from_cache(tage)
if df is None:
    with st.spinner("Lade Daten von SMARD (kein Cache vorhanden) …"):
        df = get_strommix(tage=tage)
    st.toast("Daten direkt von SMARD geladen – Cache wird beim nächsten GitHub-Actions-Lauf aktualisiert.", icon="ℹ️")

df["CO2_g_kWh"] = df.apply(berechne_co2, axis=1)

aktuell      = df["Erneuerbare_Anteil_%"].dropna().iloc[-1]
aktuell_co2  = df["CO2_g_kWh"].dropna().iloc[-1]
letzter      = df.dropna(subset=["Erneuerbare_Anteil_%"]).iloc[-1]
letzter_zeit = df.dropna(subset=["Erneuerbare_Anteil_%"]).index[-1]
erster_zeit  = df.dropna(subset=["Erneuerbare_Anteil_%"]).index[0]

if aktuell >= 70:
    farbe, status, hero_bg = "🟢", "Sehr grün",  "#1a4a1a"
elif aktuell >= 40:
    farbe, status, hero_bg = "🟡", "Mittel",      "#4a3d00"
else:
    farbe, status, hero_bg = "🔴", "Viel Fossil", "#4a1a1a"

berlin = pytz.timezone("Europe/Berlin")
now    = datetime.now(berlin)

# ── CO2-Tageswerte (für Hero-Kachel + Tageszähler) ────────────────────────────
_df_tag = df[df.index.date == now.date()].dropna(subset=["CO2_g_kWh", "Gesamt Last"]).copy()
_co2_heute_kt   = 0.0
_budget_heute_kt = 0.0
_co2_breach_zeit = None
if len(_df_tag) >= 1:
    _df_tag["_co2_kt"]    = _df_tag["Gesamt Last"] * _df_tag["CO2_g_kWh"] / 1_000_000
    _df_tag["_kumuliert"] = _df_tag["_co2_kt"].cumsum()
    _co2_heute_kt    = _df_tag["_kumuliert"].iloc[-1]
    _budget_heute_kt = _df_tag["Gesamt Last"].sum() * 100 / 1_000_000
    _ueber           = _df_tag[_df_tag["_kumuliert"] > _budget_heute_kt]
    if not _ueber.empty:
        _co2_breach_zeit = _ueber.index[0]

# Donut-Daten
farb_map = {
    "Wind Onshore": "#27ae60", "Wind Offshore": "#2ecc71", "Solar": "#f1c40f",
    "Biomasse": "#8e44ad", "Wasserkraft": "#1abc9c", "Sonstige Erneuerbare": "#16a085",
    "Erdgas": "#e74c3c", "Steinkohle": "#c0392b", "Braunkohle": "#922b21",
    "Kernenergie": "#f39c12", "Pumpspeicher": "#3498db", "Sonstige Konventionelle": "#95a5a6",
}
donut_labels, donut_werte, donut_farben = [], [], []
for q in CO2_FAKTOREN:
    if q in df.columns and letzter[q] > 0:
        donut_labels.append(q)
        donut_werte.append(letzter[q])
        donut_farben.append(farb_map.get(q, "#bdc3c7"))

# ── Forecast ──────────────────────────────────────────────────────────────────

MODELL_TAGE = 28

with st.spinner("Berechne Wetterprognose …"):
    _cached_train = load_from_cache(MODELL_TAGE)
    df_train = _cached_train if _cached_train is not None else df
    df_fc_raw, r2, fehler = wetter_prognose(df_train, MODELL_TAGE, lat=stadt_lat, lon=stadt_lon)

if df_fc_raw is not None:
    df_fc = df_fc_raw.reset_index()
    modell_info = f"Open-Meteo ({stadtname}) · R²={r2} · trainiert auf {MODELL_TAGE} Tagen"
else:
    st.toast(f"Wetterdaten nicht verfügbar – nutze historisches Muster ({fehler})", icon="⚠️")
    df_fc = _pattern_fallback(df_train, now)
    modell_info = f"Historisches Muster ({MODELL_TAGE} Tage)"

df_fc["datum"] = df_fc["zeit"].dt.date.astype(str)
hat_wetterdaten = "strahlung" in df_fc.columns

heute_str  = now.date().isoformat()
morgen_str = (now + timedelta(days=1)).date().isoformat()
fenster_heute  = bestes_fenster(df_fc[df_fc["datum"] == heute_str])
fenster_morgen = bestes_fenster(df_fc[df_fc["datum"] == morgen_str])

# Wenn der aktuelle Anteil >= bestem prognostizierten Fenster: jetzt ist optimal
jetzt_ist_optimal = fenster_heute is not None and aktuell >= fenster_heute[2]

heute  = now.date()
morgen = (now + timedelta(days=1)).date()

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1 – HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.title("⚡ StromCheck Deutschland")
st.caption(f"Stand: {letzter_zeit.strftime('%d.%m.%Y %H:%M')} Uhr · Quelle: SMARD / Bundesnetzagentur")

h1, h2, h3, h4 = st.columns([1, 1.3, 1.3, 1])

with h1:
    st.markdown(f"""
    <div style="padding:1.5rem;border-radius:.75rem;background:{hero_bg};text-align:center;height:140px;display:flex;flex-direction:column;justify-content:center">
        <div style="font-size:2.5rem">{farbe}</div>
        <div style="font-size:2rem;font-weight:bold">{aktuell:.0f}%</div>
        <div style="font-size:.85rem;color:#ccc">{status} · {aktuell_co2:.0f} g CO₂/kWh</div>
    </div>""", unsafe_allow_html=True)

with h4:
    if _budget_heute_kt > 0:
        _pct = _co2_heute_kt / _budget_heute_kt * 100
        if _co2_breach_zeit is not None:
            _h4_bg   = "#4a1a1a"
            _faktor  = _co2_heute_kt / _budget_heute_kt
            _h4_zeile1 = f"🔴 {_faktor:.1f}× über dem Ziel"
            _h4_zeile2 = f"Limit seit {_co2_breach_zeit.strftime('%H:%M')} Uhr überschritten"
        else:
            _h4_bg   = "#1a3a2a"
            _h4_zeile1 = f"✅ {_pct:.0f}% des Tagesbudgets"
            _h4_zeile2 = "Klimaziel heute noch eingehalten"
        st.markdown(f"""
        <div style="padding:1.5rem;border-radius:.75rem;background:{_h4_bg};height:140px;
                    display:flex;flex-direction:column;justify-content:center">
            <div style="font-size:.78rem;color:#bbb">🌍 CO₂ heute</div>
            <div style="font-size:1.5rem;font-weight:bold;margin:.15rem 0">
                {_co2_heute_kt:.0f} kt
            </div>
            <div style="font-size:.78rem;color:#ccc">
                Tagesziel 2030: {_budget_heute_kt:.0f} kt
            </div>
            <div style="font-size:.8rem;font-weight:bold;margin-top:.2rem">{_h4_zeile1}</div>
            <div style="font-size:.7rem;color:#aaa">{_h4_zeile2}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:1.5rem;border-radius:.75rem;background:#2a2a2a;height:140px;
                    display:flex;align-items:center;justify-content:center;color:#777">
            Keine Tagesdaten
        </div>""", unsafe_allow_html=True)

for col, fenster, label, zeige_jetzt in [
    (h2, fenster_heute, "Heute", jetzt_ist_optimal),
    (h3, fenster_morgen, "Morgen", False),
]:
    with col:
        if fenster and zeige_jetzt:
            # Jetzt ist besser als jedes prognostizierte Fenster
            wert = aktuell
            if wert >= 70:   emo, bg = "🟢", "#1a4a1a"
            elif wert >= 40: emo, bg = "🟡", "#4a3d00"
            else:            emo, bg = "🔴", "#4a1a1a"
            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:.75rem;background:{bg};height:140px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.8rem;color:#bbb">Heute – bestes Zeitfenster</div>
                <div style="font-size:1.9rem;font-weight:bold;margin:.2rem 0">Jetzt – {now.strftime('%H:%M')} Uhr</div>
                <div style="font-size:1rem">{emo} {wert:.0f}% Erneuerbare</div>
                <div style="font-size:.75rem;color:#999;margin-top:.3rem">Waschmaschine · Spülmaschine · E-Auto</div>
            </div>""", unsafe_allow_html=True)
        elif fenster:
            start, ende, wert = fenster
            if wert >= 70:   emo, bg = "🟢", "#1a4a1a"
            elif wert >= 40: emo, bg = "🟡", "#4a3d00"
            else:            emo, bg = "🔴", "#4a1a1a"
            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:.75rem;background:{bg};height:140px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.8rem;color:#bbb">{label} – bestes Zeitfenster</div>
                <div style="font-size:1.9rem;font-weight:bold;margin:.2rem 0">{start.strftime('%H:%M')} – {ende.strftime('%H:%M')} Uhr</div>
                <div style="font-size:1rem">{emo} Ø {wert:.0f}% Erneuerbare</div>
                <div style="font-size:.75rem;color:#999;margin-top:.3rem">Waschmaschine · Spülmaschine · E-Auto</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:.75rem;background:#2a2a2a;height:140px;display:flex;align-items:center;justify-content:center;color:#777">
                Keine Prognose für {label}
            </div>""", unsafe_allow_html=True)

st.markdown("")

# ── KI-Tageskommentar ─────────────────────────────────────────────────────────
_top_str = ", ".join(
    f"{l} ({w:.0f} MW)"
    for l, w in sorted(zip(donut_labels, donut_werte), key=lambda x: x[1], reverse=True)[:3]
)
_fh = (f"{fenster_heute[0].strftime('%H:%M')}–{fenster_heute[1].strftime('%H:%M')} Uhr "
       f"({fenster_heute[2]:.0f}% erneuerbar)") if fenster_heute else "keine Daten"
_fm = (f"{fenster_morgen[0].strftime('%H:%M')}–{fenster_morgen[1].strftime('%H:%M')} Uhr "
       f"({fenster_morgen[2]:.0f}% erneuerbar)") if fenster_morgen else "keine Daten"

_wind    = float(df_fc["wind_100m"].iloc[0]) if hat_wetterdaten else None
_strahl  = float(df_fc["strahlung"].iloc[0]) if hat_wetterdaten else None
_sol_kwp = float(solar_kwp) if solar_aktiv else None
_sol_aus = solar_ausrichtung if solar_aktiv else None

_kommentar = _ki_tageskommentar(
    aktuell, aktuell_co2, _top_str, _fh, _fm,
    stadtname, _wind, _strahl, _sol_kwp, _sol_aus,
)
if _kommentar:
    st.markdown(f"""
    <div style="padding:.85rem 1.25rem;border-radius:.6rem;background:#0e1a2e;
                border-left:3px solid #3498db;margin:.25rem 0 1rem 0;
                font-size:.95rem;line-height:1.6;color:#d0d8e8">
        🤖 {_kommentar}
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1.4 – SOLARANLAGE (nur wenn aktiv)
# ═══════════════════════════════════════════════════════════════════════════════
if solar_aktiv:
    st.markdown("---")
    st.subheader("☀️ Solaranlage – Ertrag & Empfehlungen")

    df_pv = berechne_pv(df_fc, solar_kwp, solar_ausrichtung, solar_neigung)

    if df_pv is None:
        st.warning(
            "Für die PV-Ertragsprognose werden Strahlungsdaten von Open-Meteo benötigt. "
            "Stelle sicher, dass der Standort korrekt ist und die Wetterprognose verfügbar ist."
        )
    else:
        df_pv["datum"] = df_pv["zeit"].dt.date.astype(str)
        heute_kwh   = df_pv[df_pv["datum"] == heute_str]["pv_kw"].sum()
        morgen_kwh  = df_pv[df_pv["datum"] == morgen_str]["pv_kw"].sum()
        naechste_kw = df_pv["pv_kw"].iloc[0] if len(df_pv) > 0 else 0.0
        jetzt_ren   = df_fc["prognose"].iloc[0] if len(df_fc) > 0 else 50.0

        empf_text, empf_farbe = pv_empfehlung(naechste_kw, jetzt_ren, False)
        st.markdown(f"""
        <div style="padding:1rem 1.5rem;border-radius:.75rem;background:{empf_farbe}20;
                    border-left:4px solid {empf_farbe};margin-bottom:.75rem">
            <span style="font-size:1.15rem;font-weight:bold;color:{empf_farbe}">{empf_text}</span>
            <span style="font-size:.8rem;color:#aaa;margin-left:1rem">
                nächste Stunde · {naechste_kw:.1f} kW Solar · {jetzt_ren:.0f}% Erneuerbare im Netz
            </span>
        </div>""", unsafe_allow_html=True)

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("☀️ Heute erwartet", f"{heute_kwh:.1f} kWh",
            help=f"{solar_kwp} kWp · {solar_ausrichtung} · {solar_neigung}° · η=80%")
        sm2.metric("📅 Morgen erwartet", f"{morgen_kwh:.1f} kWh")
        sm3.metric("⚡ Nächste Stunde", f"{naechste_kw:.1f} kW")
        sm4.metric("🌐 Netz gerade", f"{jetzt_ren:.0f}% erneuerbar")

        st.markdown("**Ertragsprognose & Netzstrom (48h)**")
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(
            x=df_pv["zeit"], y=df_pv["pv_kw"],
            name=f"PV-Ertrag ({solar_kwp} kWp)",
            fill="tozeroy", fillcolor="rgba(241,196,15,0.30)",
            line=dict(color="#f1c40f", width=2),
            hovertemplate="%{x|%a %H:%M}: %{y:.2f} kW<extra></extra>",
            yaxis="y1",
        ))
        fig_pv.add_trace(go.Scatter(
            x=df_fc["zeit"], y=df_fc["prognose"],
            name="Erneuerbare % (Netz)",
            line=dict(color="#2ecc71", width=2, dash="dot"),
            hovertemplate="%{x|%a %H:%M}: %{y:.0f}%<extra></extra>",
            yaxis="y2",
        ))
        fig_pv.add_vline(x=now.isoformat(), line_dash="solid", line_color="white", line_width=1)
        fig_pv.update_layout(
            yaxis=dict(title="PV-Leistung (kW)", side="left", rangemode="nonnegative"),
            yaxis2=dict(title="Erneuerbare % (Netz)", side="right", overlaying="y",
                        range=[0, 120], showgrid=False),
            xaxis_title="", legend=dict(orientation="h", y=-0.18),
            margin=dict(t=10, b=10), height=300,
        )
        st.plotly_chart(fig_pv, use_container_width=True)
        st.caption(
            f"Strahlungsdaten: Open-Meteo · Standort: {stadtname} · "
            f"Ausrichtung {solar_ausrichtung} · Neigung {solar_neigung}° · Systemwirkungsgrad 80%"
        )

        st.markdown("**Eigenverbrauchsampel – Stundengenaue Empfehlung**")
        pv_heute_h  = {row["zeit"].hour: row["pv_kw"] for _, row in df_pv[df_pv["datum"] == heute_str].iterrows()}
        pv_morgen_h = {row["zeit"].hour: row["pv_kw"] for _, row in df_pv[df_pv["datum"] == morgen_str].iterrows()}
        COLOR_SCORE = {
            "⚡ Eigenverbrauch maximieren": 0.85, "↗ Einspeisung — Netz ist grün": 0.65,
            "🔋 Grüner Strom — Speicher laden": 0.40, "~ Gemischter Mix": 0.20,
            "🚫 Fossiles Netz — Verbrauch reduzieren": 0.02,
        }
        TEXT_SHORT = {
            "⚡ Eigenverbrauch maximieren": "Eigen-\nverbrauch", "↗ Einspeisung — Netz ist grün": "Einspeisung",
            "🔋 Grüner Strom — Speicher laden": "Laden", "~ Gemischter Mix": "Gemischt",
            "🚫 Fossiles Netz — Verbrauch reduzieren": "Fossil",
        }
        z_ampel, text_ampel = [], []
        for hm_dict, pv_dict in [(hm_heute, pv_heute_h), (hm_morgen, pv_morgen_h)]:
            row_z, row_t = [], []
            for h in range(24):
                ren = hm_dict.get(h) or 50.0
                pv  = pv_dict.get(h, 0.0)
                et, _ = pv_empfehlung(pv, ren, False)
                row_z.append(COLOR_SCORE.get(et, 0.2))
                row_t.append(TEXT_SHORT.get(et, ""))
            z_ampel.append(row_z)
            text_ampel.append(row_t)
        fig_ampel = go.Figure(go.Heatmap(
            z=z_ampel, x=[f"{h}:00" for h in range(24)], y=y_labels,
            text=text_ampel, texttemplate="%{text}",
            colorscale=[
                [0.00, "#7b241c"], [0.15, "#c0392b"], [0.15, "#d4ac0d"], [0.35, "#d4ac0d"],
                [0.35, "#3498db"], [0.55, "#3498db"], [0.55, "#1abc9c"], [0.75, "#1abc9c"],
                [0.75, "#27ae60"], [1.00, "#27ae60"],
            ],
            zmin=0, zmax=1, showscale=False,
            hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
        ))
        fig_ampel.add_vline(x=f"{now.hour}:00", line_dash="dash", line_color="white", line_width=2)
        fig_ampel.update_layout(height=180, margin=dict(t=10, b=10, l=0, r=0),
                                xaxis=dict(side="top", tickfont=dict(size=10)),
                                yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_ampel, use_container_width=True)
        legende_html = " &nbsp;·&nbsp; ".join([
            '<span style="color:#27ae60">■</span> Eigenverbrauch',
            '<span style="color:#1abc9c">■</span> Einspeisung ok',
            '<span style="color:#d4ac0d">■</span> Gemischt',
            '<span style="color:#c0392b">■</span> Fossil',
        ])
        st.markdown(f'<div style="font-size:.8rem;color:#aaa">{legende_html}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1.5 – GERÄTEPLANER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🏠 Geräteplaner – Wann ist der grünste Moment?")
st.caption("Wähle ein Gerät und eine Deadline – die App findet das grünste Zeitfenster im 48h-Forecast.")

GERAETE = {
    "Waschmaschine":     {"dauer": 1.5, "icon": "🫧",  "kwh": 1.0},
    "Spülmaschine":      {"dauer": 1.5, "icon": "🍽️", "kwh": 1.2},
    "Trockner":          {"dauer": 2.0, "icon": "🌀",  "kwh": 2.5},
    "E-Auto – 20 kWh":  {"dauer": 2.5, "icon": "🚗",  "kwh": 20.0},
    "E-Auto – 50 kWh":  {"dauer": 6.0, "icon": "🚗",  "kwh": 50.0},
    "Wärmepumpe (2 h)": {"dauer": 2.0, "icon": "🌡️", "kwh": 3.0},
    "Geschirrspüler":    {"dauer": 1.5, "icon": "🍽️", "kwh": 1.0},
}

_heute_22  = now.replace(hour=22, minute=0, second=0, microsecond=0)
_morgen_06 = (now + timedelta(days=1)).replace(hour=6,  minute=0, second=0, microsecond=0)
_morgen_12 = (now + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
_morgen_22 = (now + timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
_in_48h    = now + timedelta(hours=48)

DEADLINES = {}
if _heute_22 > now + timedelta(hours=2):
    DEADLINES["Heute Abend (22:00)"] = _heute_22
DEADLINES["Heute Nacht / morgen früh (06:00)"] = _morgen_06
DEADLINES["Morgen Mittag (12:00)"]              = _morgen_12
DEADLINES["Morgen Abend (22:00)"]               = _morgen_22
DEADLINES["In 48 Stunden"]                      = _in_48h

gp1, gp2, gp3 = st.columns([1.5, 1.5, 1])
geraet_name    = gp1.selectbox("Gerät auswählen", list(GERAETE.keys()))
deadline_label = gp2.selectbox("Fertig bis spätestens", list(DEADLINES.keys()))
suchen = gp3.button("🔍 Berechnen", use_container_width=True, help="Findet das grünste Zeitfenster vor der Deadline")

if suchen:
    g = GERAETE[geraet_name]
    ergebnis = bestes_geraete_fenster(df_fc, g["dauer"], DEADLINES[deadline_label], now)
    st.session_state.geraet_plan = {
        "geraet": geraet_name, "deadline": deadline_label,
        "ergebnis": ergebnis, "aktuell": aktuell,
    }

if "geraet_plan" in st.session_state:
    plan    = st.session_state.geraet_plan
    ergebnis = plan["ergebnis"]

    if ergebnis is None:
        st.warning("Kein passendes Fenster vor dieser Deadline gefunden. Wähle eine spätere Deadline.")
    else:
        start, ende, avg_ren = ergebnis
        g             = GERAETE[plan["geraet"]]
        jetzt_ren     = plan["aktuell"]
        verbesserung  = avg_ren - jetzt_ren
        delta_sign    = "+" if verbesserung >= 0 else ""
        farbe_delta   = "#4caf7d" if verbesserung >= 0 else "#e74c3c"

        if avg_ren >= 70:   emo, bg = "🟢", "#1a4a1a"
        elif avg_ren >= 40: emo, bg = "🟡", "#4a3d00"
        else:               emo, bg = "🔴", "#4a1a1a"

        st.markdown(f"""
        <div style="padding:1.25rem 1.5rem;border-radius:.75rem;background:{bg};margin:.5rem 0">
            <div style="font-size:.78rem;color:#aaa;margin-bottom:.6rem">
                {g['icon']} <b style="color:#ddd">{plan['geraet']}</b>
                &nbsp;·&nbsp; Laufzeit ca. {g['dauer']:.1f} h
                &nbsp;·&nbsp; Verbrauch ca. {g['kwh']:.0f} kWh
                &nbsp;·&nbsp; Deadline: {plan['deadline']}
            </div>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1.5rem;flex-wrap:wrap">
                <div>
                    <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem">
                        Empfohlene Startzeit
                    </div>
                    <div style="font-size:1.9rem;font-weight:bold;line-height:1.1">
                        {start.strftime('%H:%M')} Uhr
                    </div>
                    <div style="font-size:.9rem;color:#ccc;margin-top:.2rem">
                        {start.strftime('%A, %d.%m.')} &nbsp;·&nbsp; fertig um {ende.strftime('%H:%M')} Uhr
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:.7rem;color:#aaa;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem">
                        Erneuerbare Energie
                    </div>
                    <div style="font-size:1.9rem;font-weight:bold;line-height:1.1">
                        {emo} {avg_ren:.0f}%
                    </div>
                    <div style="font-size:.85rem;color:#ccc;margin-top:.2rem">
                        Aktuell: {jetzt_ren:.0f}%
                        &nbsp;→&nbsp;
                        <span style="color:{farbe_delta};font-weight:bold">{delta_sign}{verbesserung:.0f}% grüner durch Warten</span>
                    </div>
                    <div style="font-size:.8rem;color:#aaa;margin-top:.3rem">
                        CO₂ jetzt: <b style="color:#e0e0e0">{co2_schaetzen(jetzt_ren) * g['kwh'] / 1000:.2f} kg</b>
                        &nbsp;→&nbsp; Empfohlen: <b style="color:#4caf7d">{co2_schaetzen(avg_ren) * g['kwh'] / 1000:.2f} kg</b>
                        &nbsp;·&nbsp; <span style="color:#4caf7d">−{(co2_schaetzen(jetzt_ren) - co2_schaetzen(avg_ren)) * g['kwh'] / 1000:.2f} kg CO₂ gespart 🌿</span>
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        ki_text = _ki_geraet(
            plan["geraet"], start.strftime("%H:%M"), ende.strftime("%H:%M"),
            avg_ren, jetzt_ren, stadtname,
        )
        if ki_text:
            st.markdown(f"""
            <div style="padding:.75rem 1.25rem;border-radius:.6rem;background:#0e1a2e;
                        border-left:3px solid #3498db;margin:.4rem 0 .5rem 0;
                        font-size:.9rem;line-height:1.5;color:#d0d8e8">
                🤖 {ki_text}
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1.7 – AKTUELLER STROMMIX (DONUT)
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🍩 Aktueller Strommix")
fig_d = go.Figure(go.Pie(
    labels=donut_labels, values=donut_werte, hole=0.45, marker_colors=donut_farben,
    hovertemplate="%{label}: %{value:.0f} MW (%{percent})<extra></extra>",
    textinfo="percent", textposition="inside",
))
fig_d.update_layout(
    showlegend=True,
    legend=dict(orientation="v", font=dict(size=11), x=1, y=0.5),
    margin=dict(t=10, b=10, l=0, r=0),
    height=320,
)
st.plotly_chart(fig_d, use_container_width=True)
st.caption(
    f"Snapshot: {letzter_zeit.strftime('%d.%m.%Y %H:%M')} Uhr · MW Einspeisung · "
    f"Zeitraum: {erster_zeit.strftime('%d.%m.%Y')} – {letzter_zeit.strftime('%d.%m.%Y')} ({zeitraum_label})"
)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 2 – VERLAUF + 48h KALENDER
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader(f"📈 Erneuerbarer-Anteil – {zeitraum_label}")
df_reset = df.reset_index()
y_max = max(110, df_reset["Erneuerbare_Anteil_%"].max() * 1.05)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_reset["zeit"], y=[40]*len(df_reset),
    fill="tozeroy", fillcolor="rgba(231,76,60,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig1.add_trace(go.Scatter(x=df_reset["zeit"], y=[70]*len(df_reset),
    fill="tonexty", fillcolor="rgba(241,196,15,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig1.add_trace(go.Scatter(x=df_reset["zeit"], y=[100]*len(df_reset),
    fill="tonexty", fillcolor="rgba(46,204,113,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig1.add_trace(go.Scatter(x=df_reset["zeit"], y=[y_max]*len(df_reset),
    fill="tonexty", fillcolor="rgba(52,152,219,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig1.add_hline(y=100, line_dash="dot", line_color="#3498db",
               annotation_text="100% – Netto-Export", annotation_position="top left")
fig1.add_trace(go.Scatter(x=df_reset["zeit"], y=df_reset["Erneuerbare_Anteil_%"],
    name="Erneuerbare %", line=dict(color="#2ecc71", width=2),
    hovertemplate="%{y:.1f}%<extra></extra>"))
fig1.update_layout(yaxis=dict(title="Anteil (%)", range=[0, y_max]),
                   xaxis_title="", margin=dict(t=10, b=10), height=280)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Erneuerbare / Gesamt-Last · Grün ≥70% · Gelb 40–70% · Rot <40% · Blau >100% = Netto-Export")

st.subheader("📅 Nächste 48 Stunden – Prognose")

hm_heute  = {h: None for h in range(24)}
hm_morgen = {h: None for h in range(24)}

for _, row in df_fc.iterrows():
    t = row["zeit"]
    if isinstance(t, str):
        continue
    if t.date() == heute:
        hm_heute[t.hour] = row["prognose"]
    elif t.date() == morgen:
        hm_morgen[t.hour] = row["prognose"]

z = [[hm_heute[h] for h in range(24)], [hm_morgen[h] for h in range(24)]]
y_labels = [
    heute.strftime("%a, %d.%m.") + "  (heute)",
    morgen.strftime("%a, %d.%m.") + "  (morgen)",
]
text_kal = [[f"{v:.0f}%" if v is not None else "" for v in row_data] for row_data in z]

fig_kal = go.Figure(go.Heatmap(
    z=z, x=[f"{h}:00" for h in range(24)], y=y_labels,
    text=text_kal, texttemplate="%{text}",
    colorscale=[[0, "#7b241c"], [0.4, "#d4ac0d"], [0.7, "#1e8449"], [1.0, "#1a6fa8"]],
    zmin=0, zmax=100,
    colorbar=dict(title="% Erneuerbar", ticksuffix="%", len=0.8),
    hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
))
fig_kal.add_vline(x=f"{now.hour}:00", line_dash="dash", line_color="white", line_width=2)
fig_kal.add_annotation(x=f"{now.hour}:00", y=1.05, yref="paper",
                       text="jetzt", showarrow=False, font=dict(color="white", size=11))
fig_kal.update_layout(
    height=200, margin=dict(t=30, b=10, l=0, r=0),
    xaxis=dict(side="top", tickfont=dict(size=10)),
    yaxis=dict(tickfont=dict(size=11)),
)
st.plotly_chart(fig_kal, use_container_width=True)
st.caption(f"Prognose · {modell_info}")

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 3 – WETTERPROGNOSE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader(f"🌤 Wetterprognose – {stadtname}")

if hat_wetterdaten:
    ww1, ww2 = st.columns(2)
    with ww1:
        fig_wind = px.area(df_fc, x="zeit", y="wind_100m",
            labels={"wind_100m": "km/h", "zeit": ""},
            color_discrete_sequence=["#3498db"],
            title="Wind in 100 m Höhe")
        fig_wind.update_layout(margin=dict(t=40, b=10), height=240)
        st.plotly_chart(fig_wind, use_container_width=True)

    with ww2:
        fig_str = px.area(df_fc, x="zeit", y="strahlung",
            labels={"strahlung": "W/m²", "zeit": ""},
            color_discrete_sequence=["#f1c40f"],
            title="Globalstrahlung")
        fig_str.update_layout(margin=dict(t=40, b=10), height=240)
        st.plotly_chart(fig_str, use_container_width=True)

    if r2 is not None:
        wm1, wm2, wm3 = st.columns(3)
        wm1.metric("Modell R²", f"{r2:.2f}",
            help="1.0 = perfekte Vorhersage · 0.0 = nicht besser als Durchschnitt")
        wm2.metric(f"Wind jetzt", f"{df_fc['wind_100m'].iloc[0]:.1f} km/h",
            help="Windgeschwindigkeit 100 m Höhe · Open-Meteo")
        wm3.metric(f"Strahlung jetzt", f"{df_fc['strahlung'].iloc[0]:.0f} W/m²",
            help="Globalstrahlung · Open-Meteo")
else:
    st.info("Wetterdaten von Open-Meteo sind derzeit nicht verfügbar. Die Prognose basiert auf historischen Mustern.")

# Forecast-Kurve mit Unsicherheitsband
st.markdown("**Erneuerbare-Prognose (48h) mit Unsicherheitsband**")
y_fc_max = max(110, df_fc["obere"].max() * 1.05)
fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=pd.concat([df_fc["zeit"], df_fc["zeit"][::-1]]),
    y=pd.concat([df_fc["obere"], df_fc["untere"][::-1]]),
    fill="toself", fillcolor="rgba(52,152,219,0.15)",
    line=dict(width=0), name="Unsicherheitsbereich", hoverinfo="skip"))
fig_fc.add_trace(go.Scatter(
    x=df_fc["zeit"], y=df_fc["prognose"], name="Prognose",
    line=dict(color="#3498db", width=2, dash="dot"),
    hovertemplate="%{x|%a %H:%M}: %{y:.1f}%<extra></extra>"))
fig_fc.add_vline(x=now.isoformat(), line_dash="solid", line_color="white")
fig_fc.add_annotation(x=now.isoformat(), y=1, yref="paper",
                      text="Jetzt", showarrow=False, yanchor="bottom")
fig_fc.add_hline(y=70, line_dash="dot", line_color="#27ae60", annotation_text="70%")
fig_fc.add_hline(y=40, line_dash="dot", line_color="#f39c12", annotation_text="40%")
fig_fc.update_layout(
    yaxis=dict(title="Erneuerbare (%)", range=[0, y_fc_max]),
    xaxis_title="", legend=dict(orientation="h", y=-0.15),
    margin=dict(t=10, b=10), height=280,
)
st.plotly_chart(fig_fc, use_container_width=True)
st.caption(modell_info)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 5 – DETAILS
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📊 Details & Hintergrund", expanded=False):

    st.markdown("### Strommix & Muster")
    sc1, sc2 = st.columns([2, 1])

    with sc1:
        st.subheader("🔋 Strommix im Verlauf")
        erneuerbare = ["Wind Onshore", "Wind Offshore", "Solar", "Biomasse", "Wasserkraft"]
        fossil      = ["Erdgas", "Steinkohle", "Braunkohle"]
        fg = ["#27ae60", "#2ecc71", "#f1c40f", "#8e44ad", "#1abc9c"]
        fr = ["#e74c3c", "#c0392b", "#922b21"]
        fig2 = go.Figure()
        for i, c in enumerate(erneuerbare):
            if c in df.columns:
                fig2.add_trace(go.Scatter(x=df.index, y=df[c], name=c,
                    stackgroup="one", fillcolor=fg[i], line=dict(color=fg[i])))
        for i, c in enumerate(fossil):
            if c in df.columns:
                fig2.add_trace(go.Scatter(x=df.index, y=df[c], name=c,
                    stackgroup="one", fillcolor=fr[i], line=dict(color=fr[i])))
        fig2.update_layout(yaxis_title="Erzeugung (MW)", xaxis_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Tatsächliche Einspeisung je Quelle in MW · gestapelt · Quelle: SMARD, stündlich")

    with sc2:
        st.subheader("🗓️ Grünstrom-Heatmap")
        df_hm = df[["Erneuerbare_Anteil_%"]].copy()
        df_hm.index = df_hm.index.tz_localize(None) if df_hm.index.tz else df_hm.index
        df_hm["stunde"] = df_hm.index.hour
        df_hm["wt"] = df_hm.index.day_name().map({
            "Monday": "Mo", "Tuesday": "Di", "Wednesday": "Mi", "Thursday": "Do",
            "Friday": "Fr", "Saturday": "Sa", "Sunday": "So"})
        pivot_hm = (df_hm.groupby(["wt", "stunde"])["Erneuerbare_Anteil_%"].mean()
                    .unstack("stunde").reindex(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]))
        fig_hm = go.Figure(go.Heatmap(
            z=pivot_hm.values, x=[f"{h}:00" for h in pivot_hm.columns],
            y=pivot_hm.index.tolist(),
            colorscale=[[0, "#922b21"], [0.4, "#f1c40f"], [1, "#27ae60"]],
            zmin=0, zmax=100, colorbar=dict(title="% Erneuerbar", ticksuffix="%"),
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
        fig_hm.update_layout(xaxis_title="Uhrzeit", yaxis_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(f"Durchschnitt pro Wochentag & Stunde · letzte {tage} Tage")

    st.markdown("---")
    st.subheader("🌿 CO2-Intensität im Verlauf")
    fig_co2 = px.area(df.reset_index(), x="zeit", y="CO2_g_kWh",
        labels={"CO2_g_kWh": "g CO2/kWh", "zeit": ""}, color_discrete_sequence=["#e67e22"])
    fig_co2.add_hline(y=100, line_dash="dot", line_color="#27ae60", annotation_text="Ziel 2030: ~100g")
    fig_co2.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_co2, use_container_width=True)
    st.caption("Gewichteter Durchschnitt Lebenszyklusemissionen nach IPCC AR6 · "
               "Braunkohle 820 · Erdgas 490 · Solar 45 · Wind ~11 g CO2/kWh")

    st.markdown("---")
    st.subheader("🌍 CO₂-Tageszähler Deutschland")
    st.caption("Wie viel CO₂ hat der deutsche Stromsektor heute ausgestoßen – und wann wurde das Klimabudget überschritten?")

    if len(_df_tag) >= 2:
        _df_plot  = _df_tag.reset_index()
        _zeiten   = _df_plot["zeit"]
        _kumuliert = _df_plot["_kumuliert"].values
        _pct = _co2_heute_kt / _budget_heute_kt * 100 if _budget_heute_kt > 0 else 0
        _ueber_text = (
            f"Tagesbudget um {_co2_breach_zeit.strftime('%H:%M')} Uhr überschritten"
            if _co2_breach_zeit else "Tagesbudget heute noch eingehalten ✅"
        )
        cz1, cz2, cz3 = st.columns(3)
        cz1.metric("Tatsächliche Emissionen heute", f"{_co2_heute_kt:.0f} kt CO₂",
                   help="Kumulierte CO₂-Emissionen seit Mitternacht (Kilotonnen)")
        cz2.metric("Erlaubtes Tagesbudget (Ziel 2030)", f"{_budget_heute_kt:.0f} kt CO₂",
                   help="Heutige Stromlast × 100 g CO₂/kWh (Klimaziel 2030)")
        cz3.metric("Budget ausgeschöpft", f"{_pct:.0f}%",
                   delta=f"+{_pct - 100:.0f}% über Ziel" if _pct > 100 else None,
                   delta_color="inverse")
        st.caption(f"📍 {_ueber_text}")
        _gruen = np.minimum(_kumuliert, _budget_heute_kt)
        fig_co2_tag = go.Figure()
        fig_co2_tag.add_trace(go.Scatter(
            x=_zeiten, y=_gruen, name="Innerhalb Tagesbudget",
            fill="tozeroy", fillcolor="rgba(39,174,96,0.35)",
            line=dict(color="#27ae60", width=1.5),
            hovertemplate="%{x|%H:%M}: %{y:.1f} kt CO₂<extra></extra>",
        ))
        fig_co2_tag.add_trace(go.Scatter(
            x=_zeiten, y=_kumuliert, name="Über Tagesbudget",
            fill="tonexty", fillcolor="rgba(192,57,43,0.35)",
            line=dict(color="#c0392b", width=1.5),
            hovertemplate="%{x|%H:%M}: %{y:.1f} kt CO₂<extra></extra>",
        ))
        fig_co2_tag.add_hline(y=_budget_heute_kt, line_dash="dash", line_color="#27ae60",
                              annotation_text=f"Tagesbudget 2030 · {_budget_heute_kt:.0f} kt",
                              annotation_position="top left")
        if _co2_breach_zeit is not None:
            fig_co2_tag.add_vline(x=_co2_breach_zeit.strftime("%Y-%m-%dT%H:%M:%S"),
                                  line_dash="dot", line_color="#e74c3c", line_width=1.5)
            fig_co2_tag.add_annotation(x=_co2_breach_zeit.strftime("%Y-%m-%dT%H:%M:%S"),
                                       y=1, yref="paper",
                                       text=f"Limit seit {_co2_breach_zeit.strftime('%H:%M')} Uhr",
                                       showarrow=False, yanchor="bottom",
                                       font=dict(color="#e74c3c", size=11))
        fig_co2_tag.update_layout(yaxis_title="Kumulierte CO₂-Emissionen heute (kt)",
                                  xaxis_title="", legend=dict(orientation="h", y=-0.18),
                                  margin=dict(t=10, b=10), height=280)
        st.plotly_chart(fig_co2_tag, use_container_width=True)
        st.caption("Tagesbudget = heutige Stromlast × 100 g CO₂/kWh (Klimaziel 2030) · Quelle: SMARD + IPCC AR6")
    else:
        st.info("Für heute noch keine ausreichenden Daten vorhanden.")

    st.markdown("---")
    st.subheader("📋 Rohdaten")
    tab1, tab2 = st.tabs(["Stromerzeugung (SMARD)", "Wetterprognose (Open-Meteo)"])

    with tab1:
        spalten = [s for s in [
            "Wind Onshore", "Wind Offshore", "Solar", "Biomasse", "Wasserkraft",
            "Sonstige Erneuerbare", "Erdgas", "Steinkohle", "Braunkohle",
            "Kernenergie", "Pumpspeicher", "Sonstige Konventionelle",
            "Gesamt Last", "Erneuerbare", "Erneuerbare_Anteil_%", "CO2_g_kWh",
        ] if s in df.columns]
        df_tab = df[spalten].copy()
        df_tab.index = df_tab.index.strftime("%d.%m.%Y %H:%M")
        df_tab.index.name = "Zeitpunkt"
        st.dataframe(df_tab.sort_index(ascending=False).style.format({
            **{s: "{:.0f} MW" for s in spalten if s not in ("Erneuerbare_Anteil_%", "CO2_g_kWh")},
            "Erneuerbare_Anteil_%": "{:.1f} %", "CO2_g_kWh": "{:.0f} g/kWh",
        }), use_container_width=True, height=400)

    with tab2:
        if hat_wetterdaten:
            df_fc_tab = df_fc[["zeit", "prognose", "untere", "obere", "strahlung", "wind_100m"]].copy()
            df_fc_tab["zeit"] = df_fc_tab["zeit"].dt.strftime("%d.%m.%Y %H:%M")
            df_fc_tab = df_fc_tab.rename(columns={
                "zeit": "Zeitpunkt", "prognose": "Prognose %", "untere": "Untergrenze %",
                "obere": "Obergrenze %", "strahlung": "Strahlung W/m²", "wind_100m": "Wind km/h",
            }).set_index("Zeitpunkt")
            st.caption(f"Standort: {stadtname} · {stadt_lat:.2f}°N, {stadt_lon:.2f}°E · Quelle: Open-Meteo")
            st.dataframe(df_fc_tab.style.format("{:.1f}"), use_container_width=True, height=400)
        else:
            st.info("Wetterdaten nicht verfügbar – Prognose basiert auf historischen Mustern.")

st.caption("Datenquelle: SMARD.de – Bundesnetzagentur | CO2-Faktoren: IPCC AR6 Medianwerte")
