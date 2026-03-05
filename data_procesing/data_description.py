import pandas as pd
import matplotlib.pyplot as plt


# Kolommen die binair zijn (waarden 0 en 1)
_BINARY_COLS = {
    "Diabetes_binary", "Smoker", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "HighBP", "HighChol", "HighBP_x_HighChol",
}


def _is_binary(series):
    """Geeft True als een kolom alleen de waarden 0 en 1 bevat."""
    return set(series.dropna().unique()).issubset({0, 1})


def plot_histograms(df, title=""):
    """Histogram per kolom in df.

    - Binaire kolommen (0/1): staafdiagram met aantal per waarde
    - Ordinale/continue kolommen: histogram met 10 bins
    """
    cols = df.columns.tolist()
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, col in zip(axes, cols):
        series = df[col]
        if _is_binary(series):
            counts = series.value_counts().sort_index()
            bars = ax.bar(
                [str(int(v)) for v in counts.index],
                counts.values,
                color=["#4a90d9", "#d94a4a"],
                edgecolor="black",
            )
            ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
            ax.set_title(f"{col}\n(binair: 0/1)", fontsize=9)
        else:
            ax.hist(series.dropna(), bins=10, color="#4a90d9", edgecolor="black")
            ax.set_title(f"{col}\n(ordinaal/continu)", fontsize=9)
        ax.set_xlabel("Waarde")
        ax.set_ylabel("Aantal")

    plt.tight_layout()
    plt.show()


def plot_boxplots(df, title=""):
    """Boxplot per kolom in df.

    Voor binaire kolommen wordt een notitie getoond omdat de boxplot
    weinig informatie geeft (IQR = 0 of 1).
    """
    cols = df.columns.tolist()
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    if n == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, col in zip(axes, cols):
        series = df[col]
        ax.boxplot(series.dropna(), patch_artist=True,
                   boxprops=dict(facecolor="#4a90d9", color="black"),
                   medianprops=dict(color="#d94a4a", linewidth=2))
        ax.set_title(f"{col}", fontsize=9)
        ax.set_ylabel("Waarde")
        if _is_binary(series):
            ax.text(0.5, 0.02, "Binaire variabele\n(boxplot beperkt informatief)",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=7, color="gray", style="italic")

    plt.tight_layout()
    plt.show()


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
