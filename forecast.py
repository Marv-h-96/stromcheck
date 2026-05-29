import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import pytz


@st.cache_data(ttl=3600, show_spinner=False)
def get_wetter(tage_vergangenheit=7, tage_prognose=3, lat=51.2, lon=10.4):
    """Historische + Forecast-Wetterdaten von Open-Meteo in einem Aufruf."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,wind_speed_100m",
        "timezone": "Europe/Berlin",
        "past_days": min(tage_vergangenheit, 92),
        "forecast_days": tage_prognose,
    }
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=10)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(
        {"strahlung": h["shortwave_radiation"], "wind_100m": h["wind_speed_100m"]},
        index=pd.to_datetime(h["time"]),
    )
    df.index = df.index.tz_localize("Europe/Berlin", ambiguous="infer", nonexistent="shift_forward")
    df.index.name = "zeit"
    return df.dropna()


def _features(df):
    return np.column_stack([
        df["strahlung"].values,
        df["wind_100m"].values,
        np.sin(2 * np.pi * df.index.hour / 24),
        np.cos(2 * np.pi * df.index.hour / 24),
        np.sin(2 * np.pi * df.index.dayofweek / 7),
        np.cos(2 * np.pi * df.index.dayofweek / 7),
        np.ones(len(df)),
    ])


def wetter_prognose(df_smard, tage, lat=51.2, lon=10.4):
    """
    Trainiert lineares Modell (Wetter -> Erneuerbaren-Anteil) und gibt
    Prognose fuer die naechsten 48 h zurueck.

    Returns:
        df_fc   – DataFrame mit Index=zeit, Spalten: prognose, untere, obere,
                  strahlung, wind_100m  (None bei Fehler)
        r2      – Modellguete R^2 auf Trainingsdaten (None bei Fehler)
        fehler  – Fehlermeldung als str oder None
    """
    try:
        df_wetter = get_wetter(tage_vergangenheit=tage, tage_prognose=3, lat=lat, lon=lon)
    except Exception as e:
        return None, None, f"Open-Meteo nicht erreichbar: {e}"

    df_train = df_smard[["Erneuerbare_Anteil_%"]].join(df_wetter, how="inner").dropna()
    if len(df_train) < 24:
        return None, None, "Zu wenig Ueberschneidung zwischen SMARD- und Wetterdaten"

    X = _features(df_train)
    y = df_train["Erneuerbare_Anteil_%"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ coeffs
    residuen = y - y_hat
    std = float(np.std(residuen))
    ss_res = float(np.sum(residuen ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = round(1 - ss_res / ss_tot, 2) if ss_tot > 0 else 0.0

    berlin = pytz.timezone("Europe/Berlin")
    now = datetime.now(berlin)
    df_future = df_wetter[df_wetter.index > now].head(48)

    if df_future.empty:
        return None, r2, "Keine Wetter-Forecast-Daten verfuegbar"

    p = np.clip(_features(df_future) @ coeffs, 0, 130)
    df_fc = pd.DataFrame({
        "prognose":  p.round(1),
        "untere":    np.clip(p - std, 0, 130).round(1),
        "obere":     np.clip(p + std, 0, 130).round(1),
        "strahlung": df_future["strahlung"].values,
        "wind_100m": df_future["wind_100m"].values,
    }, index=df_future.index)

    return df_fc, r2, None
