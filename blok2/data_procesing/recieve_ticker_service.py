import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Hardcoded list of tickers to track 
TICKERS = [
    # Aandelen
    "MC.PA", "BRK-B", "MSFT", "JPM",
    # Obligaties (Treasury Yields)
    "^TNX", "^IRX", "^TYX",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Grondstoffen
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "KC=F", "ZS=F"
]

START_DATE = '2019-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

def clean_and_reindex(df):
    """
    Apply standard cleaning pipeline:
    1. Reindex to everyday calendar frequency 'D'
    2. Forward-Fill missing values (weekends or holidays)
    3. Calculate Return / Pct Change and diff
    4. Replace NaNs with None for pure JSON output
    """
    # 1. Behandel kalender correctie (naar dagelijks Mon-Sun)
    daily_idx = pd.date_range(start=max(pd.to_datetime(START_DATE).tz_localize('UTC'), df.index.min()), 
                              end=min(pd.to_datetime(END_DATE).tz_localize('UTC'), df.index.max()), 
                              freq='D')
    
    df = df.reindex(daily_idx)
    df.index.name = "Date"
    
    # 2. Ffill and Bfill
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # 3. Differencing en Returns
    if 'Close' in df.columns:
        df['Close_diff'] = df['Close'].diff().fillna(0)
        df['Returns_pct'] = df['Close'].pct_change().fillna(0)
        
    # Rond waarden af
    df = df.round(6)

    # 4. NaNs naar null
    df.replace({np.nan: None}, inplace=True)
    
    # Zet index terug en normaliseer datum voor de output
    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    return df.to_dict(orient="records")

def fetch_and_save_data(tickers, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker_symbol in tickers:
        print(f"Fetching data for {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Fetch maximum available historical data
            hist = ticker.history(period="10y")
            
            if hist.empty:
                print(f"--> No data found for {ticker_symbol}")
                continue
            
            # Formateer index om een timezone-aware datetime index te zijn
            if 'Date' in hist.columns:
                 hist.set_index('Date', inplace=True)
            hist.index = pd.to_datetime(hist.index, utc=True).normalize()
            
            # Verwijder gedupliceerde dates if any
            hist = hist[~hist.index.duplicated(keep='last')]
                
            clean_data_list = clean_and_reindex(hist)
            
            # Create a safe file name (e.g. ^TNX -> TNX, GC=F -> GC_F)
            safe_ticker_name = ticker_symbol.replace('^', '').replace('=', '_')
            output_file = os.path.join(output_dir, f"{safe_ticker_name}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_data_list, f, indent=4)
                
            print(f"--> Successfully saved and cleaned {ticker_symbol} data to {output_file}")
            
        except Exception as e:
            print(f"--> Error fetching data for {ticker_symbol}: {e}")

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output directory for json files
    output_dir = os.path.join(current_dir, 'ticker_data')
    
    print(f"Starting data collection and native cleaning for {len(TICKERS)} hardcoded tickers.")
    fetch_and_save_data(TICKERS, output_dir)
    print("\\nAll data processing complete.")

if __name__ == "__main__":
    main()
