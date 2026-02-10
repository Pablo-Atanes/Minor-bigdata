import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "questions")

TARGET = "Diabetes_binary"

# -- Validation rules per column (single source of truth) --

COLUMN_RULES = {
    # Binary columns: must be 0 or 1
    "Diabetes_binary":      {"type": "binary"},
    "Smoker":               {"type": "binary"},
    "PhysActivity":         {"type": "binary"},
    "Fruits":               {"type": "binary"},
    "Veggies":              {"type": "binary"},
    "HvyAlcoholConsump":    {"type": "binary"},
    "HighBP":               {"type": "binary"},
    "HighChol":             {"type": "binary"},
    "CholCheck":            {"type": "binary"},
    "Stroke":               {"type": "binary"},
    "HeartDiseaseorAttack": {"type": "binary"},
    "AnyHealthcare":        {"type": "binary"},
    "NoDocbcCost":          {"type": "binary"},
    "DiffWalk":             {"type": "binary"},
    "Sex":                  {"type": "binary"},
    # Ordinal columns: int within a range
    "GenHlth":    {"type": "ordinal", "range": (1, 5)},
    "Age":        {"type": "ordinal", "range": (1, 13)},
    "Education":  {"type": "ordinal", "range": (1, 6)},
    "Income":     {"type": "ordinal", "range": (1, 8)},
    "MentHlth":   {"type": "ordinal", "range": (0, 30)},
    "PhysHlth":   {"type": "ordinal", "range": (0, 30)},
    # Continuous columns
    "BMI":        {"type": "continuous", "range": (12, 98)},
}


def load_data(filename="diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Load a CSV from the data folder."""
    filepath = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(filepath)

    # Convert Diabetes_012 (3-class) to Diabetes_binary (0/1) if needed
    if "Diabetes_012" in df.columns and TARGET not in df.columns:
        df[TARGET] = (df["Diabetes_012"] > 0).astype(int)
        df = df.drop(columns=["Diabetes_012"])

    return df


# -- Per-question derivation helpers --

def _derive_q4(df):
    """Add Unhealthy_Lifestyle_Score to Q4 data."""
    df["Unhealthy_Lifestyle_Score"] = (
        df["Smoker"]
        + (1 - df["PhysActivity"])
        + (1 - df["Fruits"])
        + df["HvyAlcoholConsump"]
    )
    return df


def _derive_q5(df):
    """Add HighBP_x_HighChol interaction term to Q5 data."""
    df["HighBP_x_HighChol"] = df["HighBP"] * df["HighChol"]
    return df


# -- Declarative question definitions --

QUESTION_DEFINITIONS = [
    {
        "name": "q1_fruit_diabetes",
        "columns": [TARGET, "Fruits"],
        "post_process": None,
    },
    {
        "name": "q2_lifestyle_patterns",
        "columns": [TARGET, "Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
        "post_process": None,
    },
    {
        "name": "q3_socioeconomic",
        "columns": [TARGET, "Income", "Education"],
        "post_process": None,
    },
    {
        "name": "q4_cumulative_lifestyle",
        "columns": [TARGET, "Smoker", "PhysActivity", "Fruits", "HvyAlcoholConsump"],
        "post_process": _derive_q4,
    },
    {
        "name": "q5_cholesterol_bloodpressure",
        "columns": [TARGET, "HighBP", "HighChol"],
        "post_process": _derive_q5,
    },
]


# -- Core cleaning logic --

def _clean_subset(df, columns):
    """Clean a subset of the DataFrame, validating only the given columns.

    Steps:
      1. Select only the requested columns
      2. Drop duplicate rows (within this subset)
      3. Drop rows with any NaN (within this subset)
      4. Cast types and validate ranges per column using COLUMN_RULES
    """
    subset = df[columns].copy()
    rows_before = len(subset)

    subset = subset.dropna()

    for col in columns:
        rule = COLUMN_RULES.get(col)
        if rule is None:
            continue

        col_type = rule["type"]

        if col_type == "binary":
            subset[col] = subset[col].astype(int)
            subset = subset[subset[col].isin([0, 1])]

        elif col_type == "ordinal":
            subset[col] = subset[col].astype(int)
            low, high = rule["range"]
            subset = subset[(subset[col] >= low) & (subset[col] <= high)]

        elif col_type == "continuous":
            low, high = rule["range"]
            subset = subset[(subset[col] >= low) & (subset[col] <= high)]

    rows_after = len(subset)
    print(f"  {rows_before} -> {rows_after} rijen ({rows_before - rows_after} verwijderd)")

    return subset.reset_index(drop=True)


# -- Public API --

def load_and_clean_all(filename="diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Load data and return a cleaned DataFrame per research question.

    Each question's data is cleaned independently: only the columns
    relevant to that question are validated, so rows are never dropped
    due to irrelevant columns.

    Returns:
        dict[str, pd.DataFrame]: {question_name: cleaned_dataframe}
    """
    raw = load_data(filename)
    rows_before = len(raw)
    raw = raw.drop_duplicates()
    print(f"Geladen: {rows_before} rijen, {len(raw.columns)} kolommen")
    print(f"Na deduplicatie: {len(raw)} rijen ({rows_before - len(raw)} duplicaten verwijderd)\n")

    questions = {}
    for qdef in QUESTION_DEFINITIONS:
        name = qdef["name"]
        print(f"{name}:")

        cleaned = _clean_subset(raw, qdef["columns"])

        if qdef["post_process"] is not None:
            cleaned = qdef["post_process"](cleaned)

        questions[name] = cleaned

    print()
    return questions


def save_questions_to_json(questions, output_dir=OUTPUT_DIR):
    """Save each question's DataFrame as a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in questions.items():
        filepath = os.path.join(output_dir, f"{name}.json")
        data.to_json(filepath, orient="records", indent=2)
        print(f"Saved {name}.json ({len(data)} rijen, {len(data.columns)} kolommen)")


if __name__ == "__main__":
    questions = load_and_clean_all()
    save_questions_to_json(questions)
