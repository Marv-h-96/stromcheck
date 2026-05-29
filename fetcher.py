import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import pytz

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "strommix_28d.json"


def load_from_cache(tage=7):
    """Liest aus dem gespeicherten 28-Tage-Cache und schneidet auf tage zu.
    Gibt None zurück wenn die Datei fehlt oder nicht lesbar ist."""
    if not CACHE_FILE.exists():
        return None
    try:
        df = pd.read_json(CACHE_FILE, orient="index")
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
        df.index.name = "zeit"
        cutoff = pd.Timestamp.now(tz="Europe/Berlin") - pd.Timedelta(days=tage)
        df = df[df.index >= cutoff]
        if len(df) < 24:
            return None
        return df
    except Exception:
        return None


def _montag_timestamps_fuer_zeitraum(tage):
    berlin = pytz.timezone('Europe/Berlin')
    now = datetime.now(berlin)
    start = now - timedelta(days=tage)

    erster_montag = start - timedelta(days=start.weekday())
    erster_montag = erster_montag.replace(hour=0, minute=0, second=0, microsecond=0)

    timestamps = []
    m = erster_montag
    while m <= now:
        timestamps.append((int(m.timestamp()) * 1000, start))
        m += timedelta(weeks=1)
    return timestamps


def get_zeitreihe(filter_id, name, tage=7):
    """Holt Stundendaten für einen einzelnen Energietyp für die letzten N Tage."""
    alle_dfs = []
    for timestamp, start in _montag_timestamps_fuer_zeitraum(tage):
        url = f"https://www.smard.de/app/chart_data/{filter_id}/DE/{filter_id}_DE_hour_{timestamp}.json"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"  FEHLER bei {name} (ID {filter_id}): HTTP {response.status_code}")
            continue
        data = response.json()
        series = data.get("series", [])
        df = pd.DataFrame(series, columns=["timestamp_ms", "wert_mwh"])
        df["zeit"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.tz_convert("Europe/Berlin")
        df = df.dropna(subset=["wert_mwh"])
        alle_dfs.append((df, start))

    if not alle_dfs:
        return None

    combined = pd.concat([d for d, _ in alle_dfs])
    combined = combined.drop_duplicates(subset=["timestamp_ms"]).sort_values("zeit")
    start_filter = alle_dfs[0][1]
    combined = combined[combined["zeit"] >= start_filter]
    print(f"  OK {name}: {len(combined)} Datenpunkte ({tage} Tage)")
    return combined


def get_strommix(tage=7):
    """Holt alle relevanten Energietypen von der SMARD-API und kombiniert sie."""
    quellen = {
        "Braunkohle":             1223,
        "Kernenergie":            1224,
        "Wind Offshore":          1225,
        "Wasserkraft":            1226,
        "Sonstige Konventionelle":1227,
        "Sonstige Erneuerbare":   1228,
        "Biomasse":               4066,
        "Wind Onshore":           4067,
        "Solar":                  4068,
        "Steinkohle":             4069,
        "Pumpspeicher":           4070,
        "Erdgas":                 4071,
        "Gesamt Last":            410,
    }

    alle = {}
    for name, filter_id in quellen.items():
        df = get_zeitreihe(filter_id, name, tage=tage)
        if df is not None:
            alle[name] = df.set_index("zeit")["wert_mwh"]

    kombiniert = pd.DataFrame(alle)
    erneuerbare_spalten = [s for s in ["Wind Offshore", "Wind Onshore", "Solar", "Biomasse", "Wasserkraft", "Sonstige Erneuerbare"] if s in kombiniert.columns]
    kombiniert["Erneuerbare"] = kombiniert[erneuerbare_spalten].sum(axis=1)
    if "Gesamt Last" in kombiniert.columns:
        kombiniert["Erneuerbare_Anteil_%"] = (kombiniert["Erneuerbare"] / kombiniert["Gesamt Last"] * 100).round(1)
    return kombiniert


if __name__ == "__main__":
    print("🔄 Lade Strommix-Daten von SMARD...\n")
    df = get_strommix(tage=7)
    print(f"\n📊 Datenpunkte geladen: {len(df)}")
    print(f"⚡ Aktueller Erneuerbaren-Anteil: {df['Erneuerbare_Anteil_%'].dropna().iloc[-1]}%")
    print(df[["Wind Onshore", "Solar", "Erdgas", "Erneuerbare_Anteil_%"]].tail())
