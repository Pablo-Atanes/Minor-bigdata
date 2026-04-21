import os
import json
import yfinance as yf

# Hardcoded list of tickers to track 
TICKERS = [
    # Aandelen
    "MC.PA", "BRK-B", "MSFT", "JPM"
    # Obligaties (Treasury Yields)
    "^TNX", "^IRX", "^TYX",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Grondstoffen
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "KC=F", "ZS=F"
]

def fetch_and_save_data(tickers, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for ticker_symbol in tickers:
        print(f"Fetching data for {ticker_symbol}...")
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Fetch maximum available historical data
            hist = ticker.history(period="5y")
            
            if hist.empty:
                print(f"--> No data found for {ticker_symbol}")
                continue
            
            # Reset index to have Date as a column, and convert dates to string for JSON serialization
            hist.reset_index(inplace=True)
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].astype(str)
            else:
                hist.index = hist.index.astype(str)
            
            # Convert the dataframe to a dictionary we will save as JSON
            data_dict = hist.to_dict(orient="records")
            
            # Create a safe file name (e.g. ^TNX -> TNX, GC=F -> GC_F)
            safe_ticker_name = ticker_symbol.replace('^', '').replace('=', '_')
            output_file = os.path.join(output_dir, f"{safe_ticker_name}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4)
                
            print(f"--> Successfully saved {ticker_symbol} data to {output_file}")
            
        except Exception as e:
            print(f"--> Error fetching data for {ticker_symbol}: {e}")

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output directory for json files
    output_dir = os.path.join(current_dir, 'ticker_data')
    
    print(f"Starting data collection for {len(TICKERS)} hardcoded tickers.")
    fetch_and_save_data(TICKERS, output_dir)
    print("\nAll data processing complete.")

if __name__ == "__main__":
    main()
