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
    """Clean the diabetes dataset for research questions.

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

    Q1: Fruit consumptie & diabetes (Chi-square, logistische regressie)
    Q2: Lifestyle patronen & diabetes (Apriori associatieregels)
    Q3: Socio-economische status & diabetes (Multipele logistische regressie)
    Q4: Cumulatief effect lifestyle factors (Polynomial logistische regressie)
    Q5: Cholesterol + bloeddruk & diabetes (Interactie-effect)
    """
    questions = {}

    # Q1 — Fruit consumptie vs diabetes
    questions["q1_fruit_diabetes"] = df[[TARGET, "Fruits"]].copy()

    # Q2 — Lifestyle patronen (apriori associatieregels)
    questions["q2_lifestyle_patterns"] = df[
        [TARGET] + LIFESTYLE_FEATURES
    ].copy()

    # Q3 — Socio-economische status & diabetes
    questions["q3_socioeconomic"] = df[
        [TARGET, "Income", "Education"]
    ].copy()

    # Q4 — Cumulatief effect lifestyle factors + afgeleide score
    q4_cols = [TARGET, "Smoker", "PhysActivity", "Fruits", "HvyAlcoholConsump"]
    q4 = df[q4_cols].copy()
    q4["Unhealthy_Lifestyle_Score"] = (
        q4["Smoker"]
        + (1 - q4["PhysActivity"])
        + (1 - q4["Fruits"])
        + q4["HvyAlcoholConsump"]
    )
    questions["q4_cumulative_lifestyle"] = q4

    # Q5 — Cholesterol + bloeddruk & diabetes (met interactieterm)
    q5 = df[[TARGET, "HighBP", "HighChol"]].copy()
    q5["HighBP_x_HighChol"] = q5["HighBP"] * q5["HighChol"]
    questions["q5_cholesterol_bloodpressure"] = q5

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
