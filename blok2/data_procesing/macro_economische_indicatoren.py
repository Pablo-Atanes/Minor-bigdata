"""
Macro Economische Data Fetcher
==============================
Haalt de volgende indicatoren op en schrijft ze weg als JSON:

Via FRED API:
  - Nederlandse inflatie (CPI)       → NLCPIALLMINMEI.json
  - Nederlandse werkloosheid         → LRUN74TTNLQ156S.json
  - Nederlands BBP                   → NLNGDPRPCPCPPPT.json
  - ECB beleidsrente                 → ECBDFR.json
  - Eurozone inflatie (HICP)         → CP0000EZ19M086NEST.json

Via CBS Open Data (optioneel):
  - Nederlandse inflatie             → CBS_70936ned.json
  - Nederlandse werkloosheid         → CBS_85224NED.json

JSON formaat per bestand:
  {
    "ticker": "NLCPIALLMINMEI",
    "label": "NL Inflatie (CPI, %)",
    "source": "FRED",
    "last_updated": "2024-04-14",
    "data": [
      { "date": "2024-01-01", "value": 2.9 },
      ...
    ]
  }

Bestandsnamen volgen de conventie:
  safe_ticker_name = ticker_symbol.replace('^', '').replace('=', '_')
  output_file = os.path.join(output_dir, f"{safe_ticker_name}.json")

Installatie (eenmalig):
  pip install fredapi cbsodata pandas

FRED API key aanvragen (gratis):
  https://fred.stlouisfed.org/docs/api/api_key.html
"""

import os
import json
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATIE
# ─────────────────────────────────────────────

FRED_API_KEY = "604abac43bf3e914ca41274770801929"     # Aanvragen op: fred.stlouisfed.org

START_DATE   = "2000-01-01"            # Hoe ver terug je wilt
END_DATE     = datetime.today().strftime("%Y-%m-%d")  # Tot vandaag

OUTPUT_DIR   = "./macro_data"          # Map waar de JSON-bestanden komen


# ─────────────────────────────────────────────
# FRED INDICATOREN — ticker_symbol → label
# ─────────────────────────────────────────────

FRED_INDICATOREN = {
    "ECBDFR":             "ECB Beleidsrente (%)",
    "CP0000EZ19M086NEST": "Eurozone Inflatie HICP (%)",
}


# ─────────────────────────────────────────────
# CBS INDICATOREN — tabel_id → label
# ─────────────────────────────────────────────

CBS_INDICATOREN = {
    "70936ned": "NL CPI (CBS)",
    "85224NED": "NL Werkloosheid (CBS)",
    "83131NED": "NL Inflatie (CBS)"
}


# ─────────────────────────────────────────────
# BESTANDSNAAM HELPER
# ─────────────────────────────────────────────

def maak_bestandsnaam(ticker_symbol: str, output_dir: str) -> str:
    """
    Zet een ticker symbool om naar een veilige bestandsnaam.
    Zelfde conventie als Yahoo Finance tickers:
      ^ wordt verwijderd
      = wordt _
    """
    safe_ticker_name = ticker_symbol.replace('^', '').replace('=', '_')
    output_file = os.path.join(output_dir, f"{safe_ticker_name}.json")
    return output_file


# ─────────────────────────────────────────────
# JSON SCHRIJVEN
# ─────────────────────────────────────────────

def schrijf_naar_json(
    ticker_symbol: str,
    label: str,
    source: str,
    series: pd.Series,
    output_dir: str
) -> str:
    """
    Schrijft een Pandas Series weg als JSON bestand.

    Output structuur:
    {
        "ticker": "NLCPIALLMINMEI",
        "label": "NL Inflatie (CPI, %)",
        "source": "FRED",
        "last_updated": "2024-04-14",
        "data": [
            { "date": "2024-01-01", "value": 2.9 },
            ...
        ]
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    # Bouw data lijst op — NaN waarden overslaan
    data_lijst = [
        {
            "date":  datum.strftime("%Y-%m-%d"),
            "value": round(float(waarde), 6)
        }
        for datum, waarde in series.items()
        if pd.notna(waarde)
    ]

    # Sorteer op datum oplopend (oudste eerst)
    data_lijst.sort(key=lambda x: x["date"])

    payload = {
        "ticker":       ticker_symbol,
        "label":        label,
        "source":       source,
        "last_updated": datetime.today().strftime("%Y-%m-%d"),
        "data":         data_lijst
    }

    output_file = maak_bestandsnaam(ticker_symbol, output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"   💾 Weggeschreven: {output_file} ({len(data_lijst)} datapunten)")
    return output_file


# ─────────────────────────────────────────────
# DATA OPHALEN VIA FRED
# ─────────────────────────────────────────────

def haal_fred_data_op(output_dir: str):
    """
    Haalt alle FRED macro-indicatoren op en schrijft
    elk als apart JSON bestand weg naar output_dir.
    """
    try:
        from fredapi import Fred
    except ImportError:
        print("❌ fredapi niet geïnstalleerd. Voer uit: pip install fredapi")
        return

    fred = Fred(api_key=FRED_API_KEY)

    print(f"\n📡 FRED — {len(FRED_INDICATOREN)} indicatoren ophalen...")

    for ticker_symbol, label in FRED_INDICATOREN.items():
        try:
            print(f"\n📥 {label} ({ticker_symbol})")

            series = fred.get_series(
                ticker_symbol,
                observation_start=START_DATE,
                observation_end=END_DATE
            )

            print(f"   ✅ {len(series)} datapunten "
                  f"({series.index.min().date()} → {series.index.max().date()})")

            schrijf_naar_json(
                ticker_symbol=ticker_symbol,
                label=label,
                source="FRED",
                series=series,
                output_dir=output_dir
            )

        except Exception as e:
            print(f"   ❌ Fout bij ophalen {ticker_symbol}: {e}")


# ─────────────────────────────────────────────
# DATA OPHALEN VIA CBS (optioneel)
# ─────────────────────────────────────────────

def haal_cbs_data_op(output_dir: str):
    """
    Haalt Nederlandse macro-data op via CBS Open Data.
    Geen API key nodig — directe publieke bron.
    Schrijft elk als apart JSON bestand weg naar output_dir.
    """
    try:
        import cbsodata
    except ImportError:
        print("❌ cbsodata niet geïnstalleerd. Voer uit: pip install cbsodata")
        return

    print(f"\n📡 CBS — {len(CBS_INDICATOREN)} indicatoren ophalen...")

    for tabel_id, label in CBS_INDICATOREN.items():
        try:
            print(f"\n📥 {label} ({tabel_id})")

            raw = cbsodata.get_data(tabel_id)
            df  = pd.DataFrame(raw)

            print(f"   ✅ {len(df)} rijen opgehaald")
            print(f"   Kolommen: {list(df.columns)}")

            os.makedirs(output_dir, exist_ok=True)

            # CBS ticker volgt dezelfde naamconventie
            cbs_ticker  = f"CBS_{tabel_id}"
            safe_name   = cbs_ticker.replace('^', '').replace('=', '_')
            output_file = os.path.join(output_dir, f"{safe_name}.json")

            payload = {
                "ticker":       cbs_ticker,
                "label":        label,
                "source":       "CBS",
                "last_updated": datetime.today().strftime("%Y-%m-%d"),
                "data":         df.to_dict(orient="records")
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

            print(f"   💾 Weggeschreven: {output_file}")

        except Exception as e:
            print(f"   ❌ Fout bij ophalen {tabel_id}: {e}")


# ─────────────────────────────────────────────
# HOOFDPROGRAMMA
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Macro Economische Data Fetcher")
    print("=" * 55)
    print(f"  Periode:    {START_DATE} → {END_DATE}")
    print(f"  Output map: {OUTPUT_DIR}")
    print("=" * 55)

    # Controleer API key
    if FRED_API_KEY == "JOUW_API_KEY_HIER":
        print("\n⚠️  Vul eerst je FRED API key in bovenaan het script.")
        print("   Gratis aanvragen op: https://fred.stlouisfed.org/docs/api/api_key.html\n")
        exit(1)

    # FRED data ophalen en wegschrijven als JSON
    haal_fred_data_op(OUTPUT_DIR)
    haal_cbs_data_op(OUTPUT_DIR)
    # CBS data ophalen en wegschrijven als JSON (optioneel)
    # Verwijder het commentaar hieronder om CBS data ook op te halen:
    # haal_cbs_data_op(OUTPUT_DIR)

    print("\n" + "=" * 55)
    print("  ✅ Klaar! Bestanden staan in:", OUTPUT_DIR)
    print("=" * 55)

    # Overzicht van gegenereerde bestanden
    if os.path.exists(OUTPUT_DIR):
        bestanden = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
        print(f"\n  📁 {len(bestanden)} JSON bestand(en) aangemaakt:")
        for b in sorted(bestanden):
            pad     = os.path.join(OUTPUT_DIR, b)
            grootte = os.path.getsize(pad)
            print(f"     • {b} ({grootte:,} bytes)")