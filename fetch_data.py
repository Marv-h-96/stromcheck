"""
Standalone-Script: holt 28-Tage-Strommix von SMARD und speichert ihn als JSON.
Wird per GitHub Actions alle 2 Stunden ausgeführt.
Kein Streamlit-Import – läuft in jeder Python-Umgebung.
"""

from pathlib import Path
from fetcher import get_strommix

CACHE_DIR = Path(__file__).parent / "cache"


def main():
    print("Lade 28-Tage-Strommix von SMARD...")
    df = get_strommix(tage=28)

    CACHE_DIR.mkdir(exist_ok=True)

    df_save = df.copy()
    df_save.index = df_save.index.astype(str)
    path = CACHE_DIR / "strommix_28d.json"
    path.write_text(df_save.to_json(orient="index"), encoding="utf-8")

    aktuell = df["Erneuerbare_Anteil_%"].dropna().iloc[-1]
    print(f"OK: {len(df)} Datenpunkte gespeichert -> {path}")
    print(f"Aktueller Erneuerbaren-Anteil: {aktuell:.1f}%")


if __name__ == "__main__":
    main()
