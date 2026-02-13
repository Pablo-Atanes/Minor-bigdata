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
        # keep target as small unsigned integer for downstream numeric ops
        df[TARGET] = (df["Diabetes_012"] > 0).astype("uint8")
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

def _clean_subset(df, columns, report=False):
    """Clean a subset of the DataFrame, validating only the given columns.

    When report=True the function also returns a DataFrame describing which
    original rows were removed and for which reason.

    Steps:
      1. Select only the requested columns
      2. Drop rows with any NaN (within this subset)
      3. Validate values per column according to COLUMN_RULES
      4. Cast types and return cleaned subset
    """
    subset = df[columns].copy()
    # keep original index for reporting
    subset["_orig_index"] = subset.index
    rows_before = len(subset)

    removal_logs = []

    # 1) drop rows with any NaN (within this subset)
    na_mask = subset.isna().any(axis=1)
    if na_mask.any():
        for _, row in subset[na_mask].iterrows():
            removal_logs.append({
                "orig_index": int(row["_orig_index"]),
                "step": "dropna",
                "column": None,
                "value": None,
            })
        subset = subset[~na_mask]

    # 2) per-column validation & casting (record removals)
    for col in columns:
        rule = COLUMN_RULES.get(col)
        if rule is None:
            continue

        col_type = rule["type"]

        if col_type == "binary":
            # record invalid binary values before casting
            invalid_mask = ~subset[col].isin([0, 1])
            if invalid_mask.any():
                for _, row in subset[invalid_mask].iterrows():
                    removal_logs.append({
                        "orig_index": int(row["_orig_index"]),
                        "step": "invalid_binary",
                        "column": col,
                        "value": row[col],
                    })
            subset = subset[~invalid_mask]
            subset[col] = subset[col].astype("uint8")

        elif col_type == "ordinal":
            low, high = rule["range"]
            out_mask = ~subset[col].between(low, high)
            if out_mask.any():
                for _, row in subset[out_mask].iterrows():
                    removal_logs.append({
                        "orig_index": int(row["_orig_index"]),
                        "step": "out_of_range",
                        "column": col,
                        "value": row[col],
                    })
            subset = subset[~out_mask]
            subset[col] = subset[col].astype("uint8")

        elif col_type == "continuous":
            low, high = rule["range"]
            out_mask = ~subset[col].between(low, high)
            if out_mask.any():
                for _, row in subset[out_mask].iterrows():
                    removal_logs.append({
                        "orig_index": int(row["_orig_index"]),
                        "step": "out_of_range",
                        "column": col,
                        "value": row[col],
                    })
            subset = subset[~out_mask]
            subset[col] = subset[col].astype("float32")

    # 3) drop duplicate rows inside the subset (if any)
    if subset.duplicated().any():
        dup_mask = subset.duplicated(keep="first")
        for _, row in subset[dup_mask].iterrows():
            removal_logs.append({
                "orig_index": int(row["_orig_index"]),
                "step": "duplicate_in_subset",
                "column": None,
                "value": None,
            })
        subset = subset.drop_duplicates(keep="first")

    rows_after = len(subset)
    print(f"  {rows_before} -> {rows_after} rijen ({rows_before - rows_after} verwijderd)")

    # prepare report DataFrame if requested
    report_df = None
    if report:
        report_df = pd.DataFrame(removal_logs)
        if not report_df.empty:
            report_df = report_df.sort_values(["step", "column"]).reset_index(drop=True)

    # cleanup helper column and return
    subset = subset.drop(columns=["_orig_index"]) if "_orig_index" in subset.columns else subset
    if report:
        return subset.reset_index(drop=True), report_df
    return subset.reset_index(drop=True)


# -- Public API --

def load_and_clean_all(filename="diabetes_binary_health_indicators_BRFSS2015.csv", report=False):
    """Load data and return a cleaned DataFrame per research question.

    Each question's data is cleaned independently: only the columns
    relevant to that question are validated, so rows are never dropped
    due to irrelevant columns.

    If report=True the function returns a tuple (questions, reports)
    where `reports` is a dict mapping question_name -> DataFrame listing
    removed rows and reasons.

    Returns:
        dict[str, pd.DataFrame] or (dict, dict): cleaned data per question
    """
    raw = load_data(filename)
    rows_before = len(raw)
    raw = raw.drop_duplicates()
    print(f"Geladen: {rows_before} rijen, {len(raw.columns)} kolommen")
    print(f"Na deduplicatie: {len(raw)} rijen ({rows_before - len(raw)} duplicaten verwijderd)\n")

    questions = {}
    reports = {}

    for qdef in QUESTION_DEFINITIONS:
        name = qdef["name"]
        print(f"{name}:")

        if report:
            cleaned, report_df = _clean_subset(raw, qdef["columns"], report=True)
            reports[name] = report_df
        else:
            cleaned = _clean_subset(raw, qdef["columns"], report=False)

        if qdef["post_process"] is not None:
            cleaned = qdef["post_process"](cleaned)

        questions[name] = cleaned

    print()
    if report:
        return questions, reports
    return questions


def save_questions_to_json(questions, output_dir=OUTPUT_DIR):
    """Save each question's DataFrame as a separate JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    for name, data in questions.items():
        filepath = os.path.join(output_dir, f"{name}.json")
        data.to_json(filepath, orient="records", indent=2)
        print(f"Saved {name}.json ({len(data)} rijen, {len(data.columns)} kolommen)")


def save_cleaning_reports(reports, output_dir=OUTPUT_DIR):
    """Save per-question cleaning reports (DataFrames) as CSV + JSON for review."""
    os.makedirs(output_dir, exist_ok=True)
    for name, rpt in reports.items():
        if rpt is None or rpt.empty:
            print(f"No removals for {name}")
            continue
        csv_path = os.path.join(output_dir, f"{name}_removals.csv")
        json_path = os.path.join(output_dir, f"{name}_removals.json")
        rpt.to_csv(csv_path, index=False)
        rpt.to_json(json_path, orient="records", indent=2)
        print(f"Saved cleaning report for {name}: {len(rpt)} removed rows")


def compare_cleaning_strategies(filename="diabetes_binary_health_indicators_BRFSS2015.csv"):
    """Compare per-question cleaning (current) vs global-clean-then-partition.

    Prints row counts per question for both strategies and the memory usage
    (approx) of the raw vs cleaned DataFrames.
    """
    raw = load_data(filename)
    raw = raw.drop_duplicates()
    print(f"Raw rows after dedup: {len(raw)}")

    # A) current approach (per-question cleaning)
    print("\nA) Per-question cleaning (current):")
    questions = load_and_clean_all(filename)
    for name, df in questions.items():
        print(f"  {name}: {len(df)} rijen")

    # B) global clean then partition
    print("\nB) Global clean -> partition:")
    # determine all relevant cols across questions
    all_cols = sorted({c for q in QUESTION_DEFINITIONS for c in q["columns"]})
    df_global = raw[all_cols].copy()
    rows_before = len(df_global)
    df_global = df_global.dropna()

    for col in all_cols:
        rule = COLUMN_RULES.get(col)
        if rule is None:
            continue
        if rule["type"] == "binary":
            df_global = df_global[df_global[col].isin([0, 1])]
        elif rule["type"] == "ordinal":
            low, high = rule["range"]
            df_global = df_global[df_global[col].between(low, high)]
        elif rule["type"] == "continuous":
            low, high = rule["range"]
            df_global = df_global[df_global[col].between(low, high)]

    print(f"  global: {rows_before} -> {len(df_global)} rijen (removed {rows_before - len(df_global)})")
    for qdef in QUESTION_DEFINITIONS:
        subset = df_global[qdef["columns"]].dropna()
        print(f"  {qdef['name']}: {len(subset)} rijen")

    try:
        print("\nMemory usage (approx):")
        print(f"  raw memory (bytes): {raw.memory_usage(deep=True).sum()}")
        print(f"  global-clean memory (bytes): {df_global.memory_usage(deep=True).sum()}")
    except Exception:
        pass


if __name__ == "__main__":
    questions = load_and_clean_all()
    save_questions_to_json(questions)
