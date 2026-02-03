import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def extract_to_json(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    """Read all Excel/CSV files from the data folder and save each as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(filepath)
        elif filename.endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            continue

        json_name = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, json_name)

        df.to_json(output_path, orient="records", indent=2)
        print(f"Saved {json_name} ({len(df)} rows)")


if __name__ == "__main__":
    extract_to_json()
