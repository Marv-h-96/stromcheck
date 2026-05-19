import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from fetcher import get_strommix
from forecast import wetter_prognose

st.set_page_config(page_title="StromCheck Deutschland", page_icon="⚡", layout="wide")

STAEDTE = {
    "Deutschland (Mitte)": (51.2, 10.4),
    "Berlin":              (52.5, 13.4),
    "Hamburg":             (53.6, 10.0),
    "München":             (48.1, 11.6),
    "Köln":                (50.9,  6.9),
    "Frankfurt":           (50.1,  8.7),
    "Stuttgart":           (48.8,  9.2),
    "Düsseldorf":          (51.2,  6.8),
    "Leipzig":             (51.3, 12.4),
    "Dortmund":            (51.5,  7.5),
    "Bremen":              (53.1,  8.8),
    "Dresden":             (51.1, 13.7),
    "Hannover":            (52.4,  9.7),
    "Nürnberg":            (49.5, 11.1),
    "Kiel":                (54.3, 10.1),
    "Freiburg":            (48.0,  7.8),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Einstellungen")
    zeitraum_optionen = {"Heute": 1, "3 Tage": 3, "1 Woche": 7, "2 Wochen": 14, "4 Wochen": 28}
    zeitraum_label = st.selectbox("Zeitraum (Verlauf)", list(zeitraum_optionen.keys()), index=2)
    tage = zeitraum_optionen[zeitraum_label]
    st.markdown("---")
    stadtname = st.selectbox("Standort (Wetterprognose)", list(STAEDTE.keys()))
    stadt_lat, stadt_lon = STAEDTE[stadtname]
    st.caption(f"{stadt_lat}°N, {stadt_lon}°E")

# ── Daten laden ───────────────────────────────────────────────────────────────
with st.spinner("Lade Daten..."):
    df = get_strommix(tage=tage)

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

df["CO2_g_kWh"] = df.apply(berechne_co2, axis=1)

aktuell    = df["Erneuerbare_Anteil_%"].dropna().iloc[-1]
aktuell_co2 = df["CO2_g_kWh"].dropna().iloc[-1]

if aktuell >= 70:
    farbe, status, hero_bg = "🟢", "Sehr grün",   "#1a4a1a"
elif aktuell >= 40:
    farbe, status, hero_bg = "🟡", "Mittel",       "#4a3d00"
else:
    farbe, status, hero_bg = "🔴", "Viel Fossil",  "#4a1a1a"

# ── Forecast (früh berechnen – wird in Hero + Timeline gebraucht) ─────────────
berlin = pytz.timezone("Europe/Berlin")
now    = datetime.now(berlin)
df_fc  = None
r2     = None
modell_info = ""

def _pattern_fallback(src):
    df_h = src[["Erneuerbare_Anteil_%"]].copy()
    df_h.index = df_h.index.tz_localize(None) if df_h.index.tz else df_h.index
    df_h["wd"] = df_h.index.dayofweek
    df_h["h"]  = df_h.index.hour
    mu = df_h.groupby(["wd","h"])["Erneuerbare_Anteil_%"].agg(["mean","std"]).reset_index()
    mu["std"] = mu["std"].fillna(8)
    zeiten = [now.replace(minute=0,second=0,microsecond=0) + timedelta(hours=i) for i in range(1,49)]
    rows = []
    for t in zeiten:
        m = mu[(mu["wd"]==t.weekday())&(mu["h"]==t.hour)]
        if not m.empty:
            v, s = m["mean"].values[0], m["std"].values[0]
            rows.append({"zeit":t,"prognose":round(v,1),
                         "untere":round(max(0,v-s),1),"obere":round(min(130,v+s),1)})
    return pd.DataFrame(rows)

MODELL_TAGE = 28  # Trainingszeitraum immer fix – unabhängig vom Anzeige-Zeitraum

with st.spinner("Berechne Wetterprognose..."):
    df_train = get_strommix(tage=MODELL_TAGE) if tage < MODELL_TAGE else df
    df_fc_raw, r2, fehler = wetter_prognose(df_train, MODELL_TAGE, lat=stadt_lat, lon=stadt_lon)

if df_fc_raw is not None:
    df_fc = df_fc_raw.reset_index()
    modell_info = f"Open-Meteo ({stadtname}) · R²={r2} · trainiert auf {MODELL_TAGE} Tagen"
else:
    st.toast(f"Wetterdaten nicht verfügbar – nutze historisches Muster ({fehler})", icon="⚠️")
    df_fc = _pattern_fallback(df_train)
    modell_info = f"Historisches Muster ({MODELL_TAGE} Tage)"

df_fc["datum"] = df_fc["zeit"].dt.date.astype(str)

def bestes_fenster(df_tag):
    if df_tag is None or len(df_tag) < 2:
        return None
    rollen = df_tag["prognose"].rolling(2).mean()
    idx = rollen.idxmax()
    return df_tag.loc[idx,"zeit"] - timedelta(hours=1), df_tag.loc[idx,"zeit"], round(rollen[idx],1)

heute_str  = now.date().isoformat()
morgen_str = (now + timedelta(days=1)).date().isoformat()
fenster_heute  = bestes_fenster(df_fc[df_fc["datum"] == heute_str])
fenster_morgen = bestes_fenster(df_fc[df_fc["datum"] == morgen_str])

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 1 – HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.title("⚡ StromCheck Deutschland")
st.caption(f"Stand: {df.dropna(subset=['Erneuerbare_Anteil_%']).index[-1].strftime('%d.%m.%Y %H:%M')} Uhr · Quelle: SMARD / Bundesnetzagentur")

h1, h2, h3 = st.columns([1, 1.5, 1.5])

with h1:
    st.markdown(f"""
    <div style="padding:1.5rem;border-radius:.75rem;background:{hero_bg};text-align:center;height:140px;display:flex;flex-direction:column;justify-content:center">
        <div style="font-size:2.5rem">{farbe}</div>
        <div style="font-size:2rem;font-weight:bold">{aktuell:.0f}%</div>
        <div style="font-size:.85rem;color:#ccc">{status} · {aktuell_co2:.0f} g CO₂/kWh</div>
    </div>""", unsafe_allow_html=True)

for col, fenster, label in [(h2, fenster_heute, "Heute"), (h3, fenster_morgen, "Morgen")]:
    with col:
        if fenster:
            start, ende, wert = fenster
            if wert >= 70:   emo, bg = "🟢", "#1a4a1a"
            elif wert >= 40: emo, bg = "🟡", "#4a3d00"
            else:            emo, bg = "🔴", "#4a1a1a"
            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:.75rem;background:{bg};height:140px;display:flex;flex-direction:column;justify-content:center">
                <div style="font-size:.8rem;color:#bbb">{label} – bestes Zeitfenster</div>
                <div style="font-size:1.9rem;font-weight:bold;margin:.2rem 0">{start.strftime('%H:%M')} – {ende.strftime('%H:%M')} Uhr</div>
                <div style="font-size:1rem">{emo} Ø {wert}% Erneuerbare</div>
                <div style="font-size:.75rem;color:#999;margin-top:.3rem">Waschmaschine · Spülmaschine · E-Auto</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding:1.5rem;border-radius:.75rem;background:#2a2a2a;height:140px;display:flex;align-items:center;justify-content:center;color:#777">
                Keine Prognose für {label}
            </div>""", unsafe_allow_html=True)

st.markdown("")

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE 2 – 48h KALENDER
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Nächste 48 Stunden")

heute  = now.date()
morgen = (now + timedelta(days=1)).date()

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

z = [
    [hm_heute[h]  for h in range(24)],
    [hm_morgen[h] for h in range(24)],
]
y_labels = [
    heute.strftime("%a, %d.%m.") + "  (heute)",
    morgen.strftime("%a, %d.%m.") + "  (morgen)",
]

# Text pro Zelle
text = []
for row_data in z:
    text.append([f"{v:.0f}%" if v is not None else "" for v in row_data])

fig_kal = go.Figure(go.Heatmap(
    z=z, x=[f"{h}:00" for h in range(24)], y=y_labels,
    text=text, texttemplate="%{text}",
    colorscale=[[0,"#7b241c"],[0.4,"#d4ac0d"],[0.7,"#1e8449"],[1.0,"#1a6fa8"]],
    zmin=0, zmax=100,
    colorbar=dict(title="% Erneuerbar", ticksuffix="%", len=0.8),
    hovertemplate="%{y}<br>%{x}: %{z:.1f}%<extra></extra>",
))

# Aktuelle Stunde markieren
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
# ZONE 3 – DETAILS
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("📊 Details & Hintergrund", expanded=False):

    # Erneuerbanen-Anteil Verlauf + Donut
    st.markdown("### Verlauf")
    dc1, dc2 = st.columns([2, 1])

    with dc1:
        st.subheader(f"📈 Erneuerbanen-Anteil – {zeitraum_label}")
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
                           xaxis_title="", margin=dict(t=10,b=10))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Erneuerbare / Gesamt-Last · Grün >=70% · Gelb 40-70% · Rot <40% · Blau >100% = Netto-Export")

    with dc2:
        st.subheader("🍩 Aktueller Mix")
        letzter = df.dropna(subset=["Erneuerbare_Anteil_%"]).iloc[-1]
        farb_map = {
            "Wind Onshore":"#27ae60","Wind Offshore":"#2ecc71","Solar":"#f1c40f",
            "Biomasse":"#8e44ad","Wasserkraft":"#1abc9c","Sonstige Erneuerbare":"#16a085",
            "Erdgas":"#e74c3c","Steinkohle":"#c0392b","Braunkohle":"#922b21",
            "Kernenergie":"#f39c12","Pumpspeicher":"#3498db","Sonstige Konventionelle":"#95a5a6",
        }
        dl, dw, df_ = [], [], []
        for q in CO2_FAKTOREN:
            if q in df.columns and letzter[q] > 0:
                dl.append(q); dw.append(letzter[q]); df_.append(farb_map.get(q,"#bdc3c7"))
        fig_d = go.Figure(go.Pie(labels=dl, values=dw, hole=0.45, marker_colors=df_,
            hovertemplate="%{label}: %{value:.0f} MW (%{percent})<extra></extra>",
            textinfo="percent", textposition="inside"))
        fig_d.update_layout(showlegend=True, legend=dict(orientation="v", font=dict(size=10)),
                            margin=dict(t=10,b=10,l=0,r=0))
        st.plotly_chart(fig_d, use_container_width=True)
        letzter_zeit = df.dropna(subset=["Erneuerbare_Anteil_%"]).index[-1]
        erster_zeit = df.dropna(subset=["Erneuerbare_Anteil_%"]).index[0]
        st.caption(
            f"Snapshot: {letzter_zeit.strftime('%d.%m.%Y %H:%M')} Uhr · MW Einspeisung · "
            f"Zeitraum: {erster_zeit.strftime('%d.%m.%Y')} – {letzter_zeit.strftime('%d.%m.%Y')} ({zeitraum_label})"
        )

    st.markdown("---")

    # Strommix + Heatmap
    st.markdown("### Strommix & Muster")
    sc1, sc2 = st.columns([2, 1])

    with sc1:
        st.subheader("🔋 Strommix im Verlauf")
        erneuerbare = ["Wind Onshore","Wind Offshore","Solar","Biomasse","Wasserkraft"]
        fossil      = ["Erdgas","Steinkohle","Braunkohle"]
        fg = ["#27ae60","#2ecc71","#f1c40f","#8e44ad","#1abc9c"]
        fr = ["#e74c3c","#c0392b","#922b21"]
        fig2 = go.Figure()
        for i, c in enumerate(erneuerbare):
            if c in df.columns:
                fig2.add_trace(go.Scatter(x=df.index, y=df[c], name=c,
                    stackgroup="one", fillcolor=fg[i], line=dict(color=fg[i])))
        for i, c in enumerate(fossil):
            if c in df.columns:
                fig2.add_trace(go.Scatter(x=df.index, y=df[c], name=c,
                    stackgroup="one", fillcolor=fr[i], line=dict(color=fr[i])))
        fig2.update_layout(yaxis_title="Erzeugung (MW)", xaxis_title="", margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Tatsächliche Einspeisung je Quelle in MW · gestapelt · Quelle: SMARD, stündlich")

    with sc2:
        st.subheader("🗓️ Grünstrom-Heatmap")
        df_hm = df[["Erneuerbare_Anteil_%"]].copy()
        df_hm.index = df_hm.index.tz_localize(None) if df_hm.index.tz else df_hm.index
        df_hm["stunde"] = df_hm.index.hour
        df_hm["wt"] = df_hm.index.day_name().map({
            "Monday":"Mo","Tuesday":"Di","Wednesday":"Mi","Thursday":"Do",
            "Friday":"Fr","Saturday":"Sa","Sunday":"So"})
        pivot_hm = (df_hm.groupby(["wt","stunde"])["Erneuerbare_Anteil_%"].mean()
                    .unstack("stunde").reindex(["Mo","Di","Mi","Do","Fr","Sa","So"]))
        fig_hm = go.Figure(go.Heatmap(
            z=pivot_hm.values, x=[f"{h}:00" for h in pivot_hm.columns],
            y=pivot_hm.index.tolist(),
            colorscale=[[0,"#922b21"],[0.4,"#f1c40f"],[1,"#27ae60"]],
            zmin=0, zmax=100, colorbar=dict(title="% Erneuerbar", ticksuffix="%"),
            hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
        fig_hm.update_layout(xaxis_title="Uhrzeit", yaxis_title="", margin=dict(t=10,b=10))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption(f"Durchschnitt pro Wochentag & Stunde · letzte {tage} Tage")

    st.markdown("---")

    # CO2
    st.subheader("🌿 CO2-Intensität im Verlauf")
    fig_co2 = px.area(df.reset_index(), x="zeit", y="CO2_g_kWh",
        labels={"CO2_g_kWh":"g CO2/kWh","zeit":""}, color_discrete_sequence=["#e67e22"])
    fig_co2.add_hline(y=100, line_dash="dot", line_color="#27ae60", annotation_text="Ziel 2030: ~100g")
    fig_co2.update_layout(margin=dict(t=10,b=10))
    st.plotly_chart(fig_co2, use_container_width=True)
    st.caption("Gewichteter Durchschnitt Lebenszyklusemissionen nach IPCC AR6 · "
               "Braunkohle 820 · Erdgas 490 · Solar 45 · Wind ~11 g CO2/kWh")

    st.markdown("---")

    # Forecast-Details (nur wenn Wetter-Modell aktiv)
    st.subheader("🔮 Prognose – technische Details")
    if r2 is not None:
        fd1, fd2, fd3 = st.columns(3)
        fd1.metric("Modell R²", f"{r2:.2f}",
            help="1.0 = perfekte Vorhersage · 0.0 = nicht besser als Durchschnitt")
        if "strahlung" in df_fc.columns:
            fd2.metric(f"Wind jetzt ({stadtname})", f"{df_fc['wind_100m'].iloc[0]:.1f} km/h",
                help="Windgeschwindigkeit 100 m Höhe · Open-Meteo")
            fd3.metric(f"Strahlung jetzt ({stadtname})", f"{df_fc['strahlung'].iloc[0]:.0f} W/m²",
                help="Globalstrahlung · Open-Meteo")

    if df_fc is not None and "strahlung" in df_fc.columns:
        fw1, fw2 = st.columns(2)
        with fw1:
            fig_w = px.area(df_fc, x="zeit", y="wind_100m",
                labels={"wind_100m":"km/h","zeit":""}, color_discrete_sequence=["#3498db"],
                title=f"Wind 100 m – {stadtname}")
            fig_w.update_layout(margin=dict(t=30,b=10))
            st.plotly_chart(fig_w, use_container_width=True)
        with fw2:
            fig_s = px.area(df_fc, x="zeit", y="strahlung",
                labels={"strahlung":"W/m²","zeit":""}, color_discrete_sequence=["#f1c40f"],
                title=f"Globalstrahlung – {stadtname}")
            fig_s.update_layout(margin=dict(t=30,b=10))
            st.plotly_chart(fig_s, use_container_width=True)

    # Forecast-Kurve mit Unsicherheitsband
    y_fc_max = max(110, df_fc["obere"].max() * 1.05)
    fig_fc_det = go.Figure()
    fig_fc_det.add_trace(go.Scatter(
        x=pd.concat([df_fc["zeit"], df_fc["zeit"][::-1]]),
        y=pd.concat([df_fc["obere"], df_fc["untere"][::-1]]),
        fill="toself", fillcolor="rgba(52,152,219,0.15)",
        line=dict(width=0), name="Unsicherheitsbereich", hoverinfo="skip"))
    fig_fc_det.add_trace(go.Scatter(
        x=df_fc["zeit"], y=df_fc["prognose"], name="Prognose",
        line=dict(color="#3498db", width=2, dash="dot"),
        hovertemplate="%{x|%a %H:%M}: %{y:.1f}%<extra></extra>"))
    fig_fc_det.add_vline(x=now.isoformat(), line_dash="solid", line_color="white")
    fig_fc_det.add_annotation(x=now.isoformat(), y=1, yref="paper",
                              text="Jetzt", showarrow=False, yanchor="bottom")
    fig_fc_det.add_hline(y=70, line_dash="dot", line_color="#27ae60", annotation_text="70%")
    fig_fc_det.add_hline(y=40, line_dash="dot", line_color="#f39c12", annotation_text="40%")
    fig_fc_det.update_layout(yaxis=dict(title="Erneuerbare (%)", range=[0, y_fc_max]),
                             xaxis_title="", legend=dict(orientation="h", y=-0.15),
                             margin=dict(t=10,b=10))
    st.plotly_chart(fig_fc_det, use_container_width=True)
    st.caption(modell_info)

    st.markdown("---")

    # Rohdaten
    st.subheader("📋 Rohdaten")
    tab1, tab2 = st.tabs(["Stromerzeugung (SMARD)", "Wetterprognose (Open-Meteo)"])

    with tab1:
        spalten = [s for s in [
            "Wind Onshore","Wind Offshore","Solar","Biomasse","Wasserkraft",
            "Sonstige Erneuerbare","Erdgas","Steinkohle","Braunkohle",
            "Kernenergie","Pumpspeicher","Sonstige Konventionelle",
            "Gesamt Last","Erneuerbare","Erneuerbare_Anteil_%","CO2_g_kWh"
        ] if s in df.columns]
        df_tab = df[spalten].copy()
        df_tab.index = df_tab.index.strftime("%d.%m.%Y %H:%M")
        df_tab.index.name = "Zeitpunkt"
        st.dataframe(df_tab.sort_index(ascending=False).style.format({
            **{s:"{:.0f} MW" for s in spalten if s not in ("Erneuerbare_Anteil_%","CO2_g_kWh")},
            "Erneuerbare_Anteil_%":"{:.1f} %", "CO2_g_kWh":"{:.0f} g/kWh",
        }), use_container_width=True, height=400)

    with tab2:
        if "strahlung" in df_fc.columns:
            df_fc_tab = df_fc[["zeit","prognose","untere","obere","strahlung","wind_100m"]].copy()
            df_fc_tab["zeit"] = df_fc_tab["zeit"].dt.strftime("%d.%m.%Y %H:%M")
            df_fc_tab = df_fc_tab.rename(columns={
                "zeit":"Zeitpunkt","prognose":"Prognose %","untere":"Untergrenze %",
                "obere":"Obergrenze %","strahlung":"Strahlung W/m²","wind_100m":"Wind km/h",
            }).set_index("Zeitpunkt")
            st.caption(f"Standort: {stadtname} · {stadt_lat}°N, {stadt_lon}°E · Quelle: Open-Meteo")
            st.dataframe(df_fc_tab.style.format("{:.1f}"), use_container_width=True, height=400)
        else:
            st.info("Nur verfügbar wenn Wetter-Modell aktiv (mind. 1 Woche Zeitraum).")

st.caption("Datenquelle: SMARD.de – Bundesnetzagentur | CO2-Faktoren: IPCC AR6 Medianwerte")
