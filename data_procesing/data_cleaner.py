import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "questions")

# Columns relevant to the research questions
TARGET = "Diabetes_binary"
LIFESTYLE_FEATURES = ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]
CONTINUOUS_FEATURES = ["BMI"]
BINARY_COLUMNS = LIFESTYLE_FEATURES + [
    "HighBP", "HighChol", "CholCheck", "Stroke",
    "HeartDiseaseorAttack", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex",
]
ORDINAL_COLUMNS = ["GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]


def load_data(filename="diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Load a CSV from the data folder."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # Convert Diabetes_012 (3-class) to Diabetes_binary (0/1) if needed
    if "Diabetes_012" in df.columns and TARGET not in df.columns:
        df[TARGET] = (df["Diabetes_012"] > 0).astype(int)
        df = df.drop(columns=["Diabetes_012"])

    return df


def clean_data(df):
    """Clean the diabetes dataset for lifestyle-focused research questions.

    Steps:
      1. Drop duplicate rows
      2. Drop rows with missing values
      3. Cast binary and ordinal columns to int
      4. Validate value ranges
      5. Filter BMI outliers
    """
    rows_before = len(df)

    # 1. Drop exact duplicate rows
    df = df.drop_duplicates()

    # 2. Drop rows with any missing value
    df = df.dropna()

    # 3. Cast types — binary/ordinal columns should be integers
    for col in BINARY_COLUMNS + [TARGET]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in ORDINAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 4. Validate value ranges
    # Binary columns must be 0 or 1
    for col in BINARY_COLUMNS + [TARGET]:
        if col in df.columns:
            df = df[df[col].isin([0, 1])]

    # Ordinal range checks
    range_checks = {
        "GenHlth": (1, 5),
        "Age": (1, 13),
        "Education": (1, 6),
        "Income": (1, 8),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30),
    }
    for col, (low, high) in range_checks.items():
        if col in df.columns:
            df = df[(df[col] >= low) & (df[col] <= high)]

    # 5. Filter BMI outliers (keep values within a plausible range)
    if "BMI" in df.columns:
        df = df[(df["BMI"] >= 12) & (df["BMI"] <= 98)]

    rows_after = len(df)
    print(f"Cleaning: {rows_before} -> {rows_after} rows ({rows_before - rows_after} removed)")

    return df.reset_index(drop=True)


def get_data_for_questions(df):
    """Return a dict with a tailored DataFrame for each research question.

    Q1: Diabetes prevalentie vs fysieke activiteit
    Q2: Correlatie lifestyle factors met diabetes
    Q3: K-means lifestyle clusters
    Q4: Apriori association rules
    Q5: Gecombineerde lifestyle factors en diabetes risico
    """
    questions = {}

    # Q1 — PhysActivity vs Diabetes prevalentie
    questions["q1_physical_activity"] = df[[TARGET, "PhysActivity"]].copy()

    # Q2 — Alle lifestyle factors correlatie met diabetes
    questions["q2_lifestyle_correlation"] = df[
        [TARGET] + LIFESTYLE_FEATURES + CONTINUOUS_FEATURES
    ].copy()

    # Q3 — K-means clustering op lifestyle + BMI (no target needed for clustering)
    questions["q3_kmeans_clusters"] = df[
        LIFESTYLE_FEATURES + CONTINUOUS_FEATURES
    ].copy()

    # Q4 — Apriori association rules (all binary columns including target)
    questions["q4_apriori"] = df[
        [TARGET] + LIFESTYLE_FEATURES
    ].copy()

    # Q5 — Combined lifestyle factors effect on diabetes risk
    questions["q5_combined_risk"] = df[
        [TARGET] + LIFESTYLE_FEATURES + CONTINUOUS_FEATURES
        + ["HighBP", "HighChol", "GenHlth", "Age"]
    ].copy()

    return questions


def save_questions_to_json(questions, output_dir=OUTPUT_DIR):
    """Save each question's DataFrame as a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in questions.items():
        filepath = os.path.join(output_dir, f"{name}.json")
        data.to_json(filepath, orient="records", indent=2)
        print(f"Saved {name}.json ({len(data)} rows, {len(data.columns)} columns)")


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    questions = get_data_for_questions(df)
    save_questions_to_json(questions)
