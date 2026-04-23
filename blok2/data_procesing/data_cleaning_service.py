import os
import json
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURATIE & CATEGORISERING
# ==============================================================================

# Paden naar de bronfolders 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TICKER_DATA_DIR = os.path.join(SCRIPT_DIR, 'ticker_data')
MACRO_DATA_DIR = os.path.join(SCRIPT_DIR, 'ticker_data', 'macro_data')

# Output pad voor de 3 schone databestanden
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'cleaned_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tickers categoriseren volgens opdracht.md en tickers.md
FINANCIAL_TICKERS = [
    "MC.PA", "BRK-B", "MSFT", "JPM",     # Aandelen
    "TNX", "IRX", "TYX",                 # Obligaties (zonder ^ voor veilige bestandsnamen)
    "BTC-USD", "ETH-USD", "SOL-USD"      # Crypto
]

COMMODITY_TICKERS = [
    "GC_F", "SI_F", "CL_F", "BZ_F", "NG_F", 
    "HG_F", "ZC_F", "ZW_F", "KC_F", "ZS_F"
]

MACRO_INDICATORS = [
    "ECBDFR", "CP0000EZ19M086NEST",      # FRED
    "CBS_70936ned", "CBS_85224NED", "CBS_83131NED" # CBS
]

# ==============================================================================
# SCHONINGSMECHANISMEN
# ==============================================================================

def handle_outliers(df, column, sigma=3):
    """
    Identificeert en beperkt extreme waarden (outliers) met de Z-score methode.
    Waarden die meer dan 'sigma' standaarddeviaties afwijken worden gecapped.
    """
    if column not in df.columns:
        return df
        
    mean = df[column].mean()
    std = df[column].std()
    
    if std == 0: # Voorkom deling door nul
        return df
        
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    
    # Gebruik Winsorizing (cappen) in plaats van verwijderen om continuïteit te behouden
    outliers_found = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if outliers_found > 0:
        print(f"    -> {outliers_found} uitschieters gevonden in {column}. Gecorrigeerd naar {sigma} sigma.")
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        
    return df

def clean_series(df, date_col='Date', value_col='Close', is_macro=False):
    """
    Core cleaning pipeline voor een enkele tijdreeks:
    1. Datatypes afdwingen
    2. Dubbele data verwijderen
    3. Missing data opvullen (Kalender-reindex + ffill/bfill), tenzij is_macro=True
    4. Outliers aanpakken
    """
    # 1. Datatypes & Index
    df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df.set_index(date_col, inplace=True)
    df.index = df.index.normalize()
    
    # Zorg dat de waarde kolom een getal is
    if value_col in df.columns:
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # 2. Dubbele records verwijderen (Punt 4 opdracht.md)
    # Verwijder zowel dubbele data op dezelfde datum als exact identieke rijen
    df = df[~df.index.duplicated(keep='last')]
    df = df.drop_duplicates()
    
    # 3. Missing data oplossing (Punt 3 opdracht.md)
    if not is_macro:
        # Reindex naar een volledige dagelijkse kalender 
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_range)
        df.index.name = 'Date'
        
        # Vul ontbrekende gegevens in (weekenden/feestdagen)
        df.ffill(inplace=True)
        df.bfill(inplace=True) # Fallback voor start van serie
    else:
        df.index.name = 'Date'
        df.sort_index(inplace=True)
    
    # 4. Extreme waarden (Punt 5 opdracht.md)
    if value_col in df.columns:
        df = handle_outliers(df, value_col)
        
    # Stationarity berekeningen (handig voor analyse later)
    if value_col in df.columns:
        df['Returns_pct'] = df[value_col].pct_change().fillna(0)
        df['Diff_abs'] = df[value_col].diff().fillna(0)
        
    return df.reset_index()

# ==============================================================================
# AGGREGATIE & UITVOERING
# ==============================================================================

def process_category(tickers, source_dir, category_name, is_macro=False):
    """
    Verwerkt een groep tickers en voegt deze samen in één databestand.
    """
    print(f"\n--- Verwerken van categorie: {category_name} ---")
    all_series = []
    
    for ticker in tickers:
        file_path = os.path.join(source_dir, f"{ticker}.json")
        if not os.path.exists(file_path):
            print(f"  [WAARSCHUWING] Bestand niet gevonden: {file_path}")
            continue
            
        print(f"  -> Laden van {ticker}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = json.load(f)
            
        # Macro data heeft een andere JSON structuur (met 'data' key)
        if isinstance(raw_content, dict) and 'data' in raw_content:
            df = pd.DataFrame(raw_content['data'])
            clean_df = clean_series(df, date_col='date', value_col='value', is_macro=is_macro)
            value_col_name = 'value'
        else:
            df = pd.DataFrame(raw_content)
            clean_df = clean_series(df, date_col='Date', value_col='Close', is_macro=is_macro)
            value_col_name = 'Close'
            
        # Selecteer alleen Date en de relevante waarde kolom
        temp_df = clean_df[['Date', value_col_name]].copy()
        
        # Hernoem de waarde kolom naar de specifieke ticker naam
        temp_df.rename(columns={value_col_name: ticker}, inplace=True)
        
        # Zorg dat Date een string is in 'YYYY-MM-DD' formaat, en set als index
        temp_df['Date'] = temp_df['Date'].dt.strftime('%Y-%m-%d')
        temp_df.set_index('Date', inplace=True)
        
        all_series.append(temp_df)
        
    if all_series:
        # Voeg alle kolommen samen op basis van de Date index
        combined_df = pd.concat(all_series, axis=1)
        combined_df.index.name = 'Date'
        combined_df.reset_index(inplace=True)
        
        if not is_macro:
            # Vul eventuele overgebleven NaNs na de samenvoeging
            combined_df.ffill(inplace=True)
            combined_df.bfill(inplace=True)
        
        # Vervang NaNs met None voor geldige JSON
        combined_df.replace({np.nan: None}, inplace=True)
        
        output_data = combined_df.to_dict(orient="records")
    else:
        output_data = []
        
    # Opslaan van het geaggregeerde bestand (Punt 2 opdracht.md)
    output_file = os.path.join(OUTPUT_DIR, f"portfolio_{category_name.lower().replace(' ', '_')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Categorie '{category_name}' opgeslagen in: {output_file}")

def main():
    print("==================================================")
    print("      DIENST VOOR SCHONE DATA & AGGREGATIE        ")
    print("==================================================")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Financiële producten (10)
    process_category(FINANCIAL_TICKERS, TICKER_DATA_DIR, "Financial Products", is_macro=False)
    
    # 2. Grondstoffen (10)
    process_category(COMMODITY_TICKERS, TICKER_DATA_DIR, "Commodities", is_macro=False)
    
    # 3. Macro-economische indicatoren (5)
    process_category(MACRO_INDICATORS, MACRO_DATA_DIR, "Macro indicators", is_macro=True)
    
    print("\n==================================================")
    print("          SCHONING EN OPSLAG VOLTOOID             ")
    print("==================================================")
    print("Controleer de map 'cleaned_output' voor de resultaten.")

if __name__ == "__main__":
    main()
