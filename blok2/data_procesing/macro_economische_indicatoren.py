"""
Macro Economische Data Fetcher
==============================
Haalt de volgende indicatoren op en schrijft ze weg als JSON.
De data wordt direct gecleaned (NaNs naar null, ffill, Kalender-correcties naar dagelijks, en diffs berekend).

Via FRED API:
  - Nederlandse inflatie (CPI)       → NLCPIALLMINMEI.json
  - Nederlandse werkloosheid         → LRUN74TTNLQ156S.json
  - Nederlands BBP                   → NLNGDPRPCPCPPPT.json
  - ECB beleidsrente                 → ECBDFR.json
  - Eurozone inflatie (HICP)         → CP0000EZ19M086NEST.json

Via CBS Open Data:
  - Nederlandse inflatie             → CBS_70936ned.json
  - Nederlandse werkloosheid         → CBS_85224NED.json
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATIE
# ─────────────────────────────────────────────

FRED_API_KEY = "604abac43bf3e914ca41274770801929"     

START_DATE   = "2010-01-01"  # Zorg dat alles minimaal vanaf 2010 begint voor ffill        
END_DATE     = datetime.today().strftime("%Y-%m-%d")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "ticker_data", "macro_data")

# ─────────────────────────────────────────────
# FRED INDICATOREN — ticker_symbol → label
# ─────────────────────────────────────────────

FRED_INDICATOREN = {
    "ECBDFR":             "ECB Beleidsrente (%)",
    "CP0000EZ19M086NEST": "Eurozone Inflatie HICP (%)",
}

# ─────────────────────────────────────────────
# CBS INDICATOREN — tabel_id → (label, value_col)
# ─────────────────────────────────────────────

CBS_INDICATOREN = {
    "70936ned": {"label": "NL CPI (CBS)", "value_col": "JaarmutatieCPI_1"},
    "85224NED": {"label": "NL Werkloosheid (CBS)", "value_col": "Werkloosheidspercentage_25"},
    "83131NED": {"label": "NL Inflatie (CBS)", "value_col": "CPI_1"}
}

def parse_cbs_period(period_str):
    if pd.isna(period_str):
        return pd.NaT
    
    period_str = str(period_str).strip()
    
    # 2013 1e kwartaal
    if "kwartaal" in period_str:
        year = period_str[:4]
        kw = period_str.split(" ")[1][:1] # 1, 2, 3, 4
        month = (int(kw) - 1) * 3 + 1
        return pd.to_datetime(f"{year}-{month:02d}-01")
        
    # 2024 januari 
    dutch_months = {
        "januari": "01", "februari": "02", "maart": "03", "april": "04",
        "mei": "05", "juni": "06", "juli": "07", "augustus": "08",
        "september": "09", "oktober": "10", "november": "11", "december": "12"
    }
    
    parts = period_str.split(" ")
    if len(parts) == 2:
        year = parts[0]
        month = dutch_months.get(parts[1].lower())
        if month:
            return pd.to_datetime(f"{year}-{month}-01")
            
    # Alleen een jaar (bijv. "2024") negeren of als januari pakken.
    # Data is vaak maandelijks en jaarlijks gelijktijdig, we filteren jaarlijkse eruit.
    return pd.NaT

def clean_and_reindex(df, value_col):
    # Converteer alles naar de 1e van de maand en pak de eerste beschikbare waarde
    df = df.resample('MS').first()
    
    # Resample kan maanden toevoegen waar geen data voor is (als er gaten zijn), dus drop NaNs
    df.dropna(subset=[value_col], inplace=True)
    
    df.index.name = "date"
    df.sort_index(inplace=True)
    
    # Stationarity
    df['value_diff'] = df[value_col].diff().fillna(0)
    
    # Afronden op 4 decimalen en NaN naar None
    df = df.round(4)
    df.replace({np.nan: None}, inplace=True)
    
    # Terug naar een list of dicts met "date", "value", "value_diff"
    result = []
    for date, row in df.iterrows():
        result.append({
            "date": date.strftime("%Y-%m-%d"),
            "value": row[value_col],
            "value_diff": row["value_diff"]
        })
    return result


def haal_fred_data_op(output_dir: str):
    from fredapi import Fred
    fred = Fred(api_key=FRED_API_KEY)

    print(f"\\n📡 FRED — {len(FRED_INDICATOREN)} indicatoren ophalen en cleanen...")

    for ticker_symbol, label in FRED_INDICATOREN.items():
        try:
            print(f"\\n📥 {label} ({ticker_symbol})")
            series = fred.get_series(ticker_symbol, observation_start=START_DATE, observation_end=END_DATE)
            df = series.to_frame(name="value")
            df.index.name = "date"
            df.index = pd.to_datetime(df.index)
            
            # Kalender en NaNs
            data_lijst = clean_and_reindex(df, "value")

            payload = {
                "ticker": ticker_symbol,
                "label": label,
                "source": "FRED",
                "last_updated": datetime.today().strftime("%Y-%m-%d"),
                "data": data_lijst
            }

            output_file = os.path.join(output_dir, f"{ticker_symbol}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ Succesvol weggeschreven (gecleaned): {output_file}")

        except Exception as e:
            print(f"   ❌ Fout bij ophalen {ticker_symbol}: {e}")


def haal_cbs_data_op(output_dir: str):
    import cbsodata
    print(f"\\n📡 CBS — {len(CBS_INDICATOREN)} indicatoren ophalen en cleanen...")

    for tabel_id, meta in CBS_INDICATOREN.items():
        try:
            label = meta['label']
            val_col = meta['value_col']
            print(f"\\n📥 {label} ({tabel_id}) | Value column: {val_col}")

            raw = cbsodata.get_data(tabel_id)
            df = pd.DataFrame(raw)
            
            if 'Perioden' not in df.columns or val_col not in df.columns:
                print(f"   ❌ Data mist de vereiste kolommen.")
                continue
                
            df['date'] = df['Perioden'].apply(parse_cbs_period)
            
            # Verwijder non-dates (bijv hele jaren) en NaN values in target
            df = df.dropna(subset=['date'])
            # Als er duplicates zijn op datum (soms heeft CBS dat), pak de eerste
            df = df.drop_duplicates(subset=['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            # Selecteer  alleen de value column
            df = df[[val_col]]
            
            data_lijst = clean_and_reindex(df, val_col)

            cbs_ticker  = f"CBS_{tabel_id}"
            output_file = os.path.join(output_dir, f"{cbs_ticker}.json")

            payload = {
                "ticker": cbs_ticker,
                "label": label,
                "source": "CBS",
                "last_updated": datetime.today().strftime("%Y-%m-%d"),
                "data": data_lijst
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            print(f"   ✅ Succesvol weggeschreven (gecleaned): {output_file}")

        except Exception as e:
            print(f"   ❌ Fout bij ophalen {tabel_id}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    haal_fred_data_op(OUTPUT_DIR)
    haal_cbs_data_op(OUTPUT_DIR)
    print("\\nKlaar met het Macro DataFrame generatieproces.")