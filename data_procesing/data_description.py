import pandas as pd


def describe_question(name, df):
    """Print beschrijvende statistieken voor een onderzoeksvraag-dataset.

    Toont: aantal, gemiddelde, mediaan, modus, standaardafwijking,
    min, max, scheefheid en kurtosis per kolom.
    """
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Aantal rijen: {len(df)}")
    print(f"Kolommen: {df.columns.tolist()}\n")

    rows = []
    for col in df.columns:
        series = df[col]
        mode_val = series.mode()
        rows.append({
            "Kolom": col,
            "Aantal": series.count(),
            "Gemiddelde": round(series.mean(), 4),
            "Mediaan": round(series.median(), 4),
            "Modus": mode_val.iloc[0] if len(mode_val) > 0 else None,
            "Std": round(series.std(), 4),
            "Min": series.min(),
            "Max": series.max(),
            "Scheefheid": round(series.skew(), 4),
            "Kurtosis": round(series.kurtosis(), 4),
        })

    stats_df = pd.DataFrame(rows).set_index("Kolom")
    print(stats_df.to_string())
    return stats_df


def describe_all(questions):
    """Print beschrijvende statistieken voor alle onderzoeksvragen."""
    all_stats = {}
    for name, df in questions.items():
        all_stats[name] = describe_question(name, df)
    return all_stats
