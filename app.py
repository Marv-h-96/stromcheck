import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from fetcher import get_strommix

st.set_page_config(
    page_title="StromCheck Deutschland",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ StromCheck Deutschland")
st.caption("Live-Daten vom deutschen Stromnetz – powered by SMARD / Bundesnetzagentur")

# Zeitraum-Auswahl
with st.sidebar:
    st.header("Einstellungen")
    zeitraum_optionen = {
        "Heute":      1,
        "3 Tage":     3,
        "1 Woche":    7,
        "2 Wochen":  14,
        "4 Wochen":  28,
    }
    zeitraum_label = st.selectbox("Zeitraum", list(zeitraum_optionen.keys()), index=2)
    tage = zeitraum_optionen[zeitraum_label]

# Daten laden
with st.spinner("Lade aktuelle Strommix-Daten..."):
    df = get_strommix(tage=tage)

# CO2-Intensität berechnen (g CO2/kWh je Quelle, Median-Schätzwerte)
CO2_FAKTOREN = {
    "Braunkohle": 820,
    "Steinkohle": 740,
    "Erdgas": 490,
    "Kernenergie": 12,
    "Wind Onshore": 11,
    "Wind Offshore": 12,
    "Solar": 45,
    "Biomasse": 230,
    "Wasserkraft": 24,
    "Pumpspeicher": 30,
    "Sonstige Konventionelle": 600,
    "Sonstige Erneuerbare": 50,
}

def berechne_co2(row):
    gesamt_mw = sum(row.get(q, 0) or 0 for q in CO2_FAKTOREN)
    if gesamt_mw <= 0:
        return np.nan
    co2 = sum((row.get(q, 0) or 0) * f for q, f in CO2_FAKTOREN.items())
    return round(co2 / gesamt_mw, 1)

df["CO2_g_kWh"] = df.apply(berechne_co2, axis=1)

# Aktueller Erneuerbaren-Anteil (letzter vollständiger Datenpunkt)
aktuell = df["Erneuerbare_Anteil_%"].dropna().iloc[-1]
aktuell_co2 = df["CO2_g_kWh"].dropna().iloc[-1]

# Ampel-Farbe
if aktuell >= 70:
    farbe = "🟢"
    status = "Sehr grün"
elif aktuell >= 40:
    farbe = "🟡"
    status = "Mittel"
else:
    farbe = "🔴"
    status = "Viel Fossil"

# KPI-Zeile
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Erneuerbaren-Anteil", f"{aktuell}%", status)
col2.metric("CO2-Intensität", f"{aktuell_co2:.0f} g/kWh")
col3.metric("Wind Onshore", f"{df['Wind Onshore'].dropna().iloc[-1]:.0f} MW")
col4.metric("Solar", f"{df['Solar'].dropna().iloc[-1]:.0f} MW")
col5.metric("Erdgas", f"{df['Erdgas'].dropna().iloc[-1]:.0f} MW")

st.markdown(f"### Stromampel: {farbe} {status}")
st.markdown("---")

# ── Zeile 1: Erneuerbaren-Anteil + Donut ──────────────────────────────────────
col_links, col_rechts = st.columns([2, 1])

with col_links:
    # Ampel-Verlauf: Fläche eingefärbt nach Grün/Gelb/Rot
    st.subheader(f"📈 Erneuerbaren-Anteil – {zeitraum_label}")
    df_reset = df.reset_index()

    fig1 = go.Figure()
    # Rote Zone (0–40%)
    fig1.add_trace(go.Scatter(
        x=df_reset["zeit"], y=[40] * len(df_reset),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Gelbe Zone (40–70%)
    fig1.add_trace(go.Scatter(
        x=df_reset["zeit"], y=[70] * len(df_reset),
        fill="tonexty", fillcolor="rgba(241,196,15,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Grüne Zone (70–100%)
    fig1.add_trace(go.Scatter(
        x=df_reset["zeit"], y=[100] * len(df_reset),
        fill="tonexty", fillcolor="rgba(46,204,113,0.15)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Linie
    fig1.add_trace(go.Scatter(
        x=df_reset["zeit"], y=df_reset["Erneuerbare_Anteil_%"],
        name="Erneuerbare %", line=dict(color="#2ecc71", width=2),
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))
    fig1.update_layout(
        yaxis=dict(title="Anteil (%)", range=[0, 105]),
        xaxis_title="",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_rechts:
    # Donut: aktueller Strommix
    st.subheader("🍩 Aktueller Mix")
    letzter = df.dropna(subset=["Erneuerbare_Anteil_%"]).iloc[-1]
    alle_quellen = list(CO2_FAKTOREN.keys())
    donut_labels, donut_werte, donut_farben = [], [], []
    farb_map = {
        "Wind Onshore": "#27ae60", "Wind Offshore": "#2ecc71",
        "Solar": "#f1c40f", "Biomasse": "#8e44ad", "Wasserkraft": "#1abc9c",
        "Sonstige Erneuerbare": "#16a085",
        "Erdgas": "#e74c3c", "Steinkohle": "#c0392b", "Braunkohle": "#922b21",
        "Kernenergie": "#f39c12", "Pumpspeicher": "#3498db",
        "Sonstige Konventionelle": "#95a5a6",
    }
    for q in alle_quellen:
        if q in df.columns and letzter[q] > 0:
            donut_labels.append(q)
            donut_werte.append(letzter[q])
            donut_farben.append(farb_map.get(q, "#bdc3c7"))

    fig_donut = go.Figure(go.Pie(
        labels=donut_labels, values=donut_werte,
        hole=0.45, marker_colors=donut_farben,
        hovertemplate="%{label}: %{value:.0f} MW (%{percent})<extra></extra>",
        textinfo="percent", textposition="inside"
    ))
    fig_donut.update_layout(
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=11)),
        margin=dict(t=10, b=10, l=0, r=0)
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# ── Zeile 2: Strommix-Verlauf + Heatmap ───────────────────────────────────────
col_l2, col_r2 = st.columns([2, 1])

with col_l2:
    st.subheader("🔋 Strommix im Verlauf")
    erneuerbare = ["Wind Onshore", "Wind Offshore", "Solar", "Biomasse", "Wasserkraft"]
    fossil = ["Erdgas", "Steinkohle", "Braunkohle"]
    farben_gruen = ["#27ae60", "#2ecc71", "#f1c40f", "#8e44ad", "#1abc9c"]
    farben_rot = ["#e74c3c", "#c0392b", "#922b21"]

    fig2 = go.Figure()
    for i, col in enumerate(erneuerbare):
        if col in df.columns:
            fig2.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                stackgroup="one", fillcolor=farben_gruen[i],
                line=dict(color=farben_gruen[i])
            ))
    for i, col in enumerate(fossil):
        if col in df.columns:
            fig2.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                stackgroup="one", fillcolor=farben_rot[i],
                line=dict(color=farben_rot[i])
            ))
    fig2.update_layout(yaxis_title="Erzeugung (MW)", xaxis_title="", margin=dict(t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with col_r2:
    st.subheader("🗓️ Grünstrom-Heatmap")
    df_hm = df[["Erneuerbare_Anteil_%"]].copy()
    df_hm.index = df_hm.index.tz_localize(None) if df_hm.index.tz is not None else df_hm.index
    df_hm["stunde"] = df_hm.index.hour
    df_hm["wochentag"] = df_hm.index.day_name()

    wochentage_de = {
        "Monday": "Mo", "Tuesday": "Di", "Wednesday": "Mi",
        "Thursday": "Do", "Friday": "Fr", "Saturday": "Sa", "Sunday": "So"
    }
    reihenfolge = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
    df_hm["wochentag"] = df_hm["wochentag"].map(wochentage_de)

    pivot = (
        df_hm.groupby(["wochentag", "stunde"])["Erneuerbare_Anteil_%"]
        .mean()
        .unstack(level="stunde")
        .reindex(reihenfolge)
    )

    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#922b21"], [0.4, "#f1c40f"], [1, "#27ae60"]],
        zmin=0, zmax=100,
        colorbar=dict(title="% Erneuerbar", ticksuffix="%"),
        hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"
    ))
    fig_hm.update_layout(
        xaxis_title="Uhrzeit",
        yaxis_title="",
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# ── CO2-Intensität im Zeitverlauf ─────────────────────────────────────────────
st.subheader("🌿 CO2-Intensität im Verlauf")
fig_co2 = px.area(
    df.reset_index(),
    x="zeit", y="CO2_g_kWh",
    labels={"CO2_g_kWh": "g CO₂/kWh", "zeit": ""},
    color_discrete_sequence=["#e67e22"]
)
fig_co2.add_hline(y=100, line_dash="dot", line_color="#27ae60", annotation_text="Ziel 2030: ~100g")
fig_co2.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(fig_co2, use_container_width=True)

st.markdown("---")

# ── Rohdaten-Tabelle ───────────────────────────────────────────────────────────
with st.expander("📋 Rohdaten anzeigen"):
    spalten_anzeige = [s for s in [
        "Wind Onshore", "Wind Offshore", "Solar", "Biomasse", "Wasserkraft",
        "Sonstige Erneuerbare", "Erdgas", "Steinkohle", "Braunkohle",
        "Kernenergie", "Pumpspeicher", "Sonstige Konventionelle",
        "Gesamt Last", "Erneuerbare", "Erneuerbare_Anteil_%", "CO2_g_kWh"
    ] if s in df.columns]

    df_tabelle = df[spalten_anzeige].copy()
    df_tabelle.index = df_tabelle.index.strftime("%d.%m.%Y %H:%M")
    df_tabelle.index.name = "Zeitpunkt"

    st.dataframe(
        df_tabelle.sort_index(ascending=False).style.format({
            **{s: "{:.0f} MW" for s in spalten_anzeige if s not in ("Erneuerbare_Anteil_%", "CO2_g_kWh")},
            "Erneuerbare_Anteil_%": "{:.1f} %",
            "CO2_g_kWh": "{:.0f} g/kWh",
        }),
        use_container_width=True,
        height=400,
    )

st.caption("Datenquelle: SMARD.de – Bundesnetzagentur | Aktualisierung: stündlich | CO2-Faktoren: Medianwerte IPCC AR6")
