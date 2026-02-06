import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(df, title="Correlatiematrix"):
    """Toon een correlatiematrix als heatmap."""
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_proportional_bar(df, group_col, title=None):
    """Proportioneel (100%) gestapeld staafdiagram: diabetes verhouding per groep."""
    ct = pd.crosstab(df[group_col], df["Diabetes_binary"], normalize="index") * 100
    ct.columns = ["Geen diabetes", "Diabetes"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=["#4a90d9", "#d94a4a"], edgecolor="black")
    ax.set_title(title or f"Diabetes verhouding per {group_col}")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel(group_col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Status")
    # Percentages op de balken
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_pie(df, col, title=None, labels_map=None):
    """Cirkeldiagram: verdeling van een kolom."""
    counts = df[col].value_counts().sort_index()
    if labels_map:
        labels = [labels_map.get(v, str(v)) for v in counts.index]
    else:
        labels = [f"{col}={v}" for v in counts.index]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90,
           colors=["#4a90d9", "#d94a4a", "#5cb85c", "#f0ad4e", "#9b59b6"])
    ax.set_title(title or f"Verdeling van {col}")
    plt.tight_layout()
    plt.show()


def plot_line(df, x_col, title=None):
    """Lijndiagram: diabetes percentage per waarde van x_col."""
    rates = df.groupby(x_col)["Diabetes_binary"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind="line", marker="o", ax=ax, color="steelblue", linewidth=2)
    ax.set_title(title or f"Diabetes percentage per {x_col}")
    ax.set_ylabel("Diabetes (%)")
    ax.set_xlabel(x_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_horizontal_comparison(df, columns, title=None):
    """Horizontaal staafdiagram: vergelijk diabetes percentage over meerdere factoren."""
    rates = {}
    for col in columns:
        rates[col] = df[df[col] == 1]["Diabetes_binary"].mean() * 100
    rates_series = pd.Series(rates).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#d94a4a" if v > df["Diabetes_binary"].mean() * 100 else "#4a90d9"
              for v in rates_series.values]
    bars = ax.barh(rates_series.index, rates_series.values, color=colors, edgecolor="black")
    ax.axvline(df["Diabetes_binary"].mean() * 100, color="gray", linestyle="--",
               label=f"Gemiddeld: {df['Diabetes_binary'].mean()*100:.1f}%")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("Diabetes (%)")
    ax.set_title(title or "Diabetes percentage per factor")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_scatter_colored(df, x_col, y_col, title=None):
    """Scatter plot (x/y-diagram) gekleurd op diabetes status."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, name in [(0, "#4a90d9", "Geen diabetes"), (1, "#d94a4a", "Diabetes")]:
        subset = df[df["Diabetes_binary"] == label]
        ax.scatter(subset[x_col], subset[y_col], c=color, label=name,
                   alpha=0.3, s=10, edgecolors="none")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or f"{x_col} vs {y_col}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap_2x2(df, row_col, col_col, title=None):
    """2x2 heatmap: diabetes percentage per combinatie van twee binaire variabelen."""
    pivot = df.groupby([row_col, col_col])["Diabetes_binary"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Diabetes (%)"})
    ax.set_title(title or f"Diabetes % per {row_col} x {col_col}")
    ax.set_ylabel(row_col)
    ax.set_xlabel(col_col)
    plt.tight_layout()
    plt.show()


def plot_area(df, x_col, title=None):
    """Vlakdiagram: diabetes percentage per waarde van x_col."""
    rates = df.groupby(x_col)["Diabetes_binary"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(rates.index, rates.values, alpha=0.4, color="steelblue")
    ax.plot(rates.index, rates.values, marker="o", color="steelblue", linewidth=2)
    ax.set_title(title or f"Diabetes percentage per {x_col}")
    ax.set_ylabel("Diabetes (%)")
    ax.set_xlabel(x_col)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -- Functies per onderzoeksvraag --

def plot_q1(q1):
    """Grafieken voor Q1: Fruit Consumptie & Diabetes.

    - Proportioneel staafdiagram: vergelijkt diabetes verhouding tussen fruiteters en niet-fruiteters
    - Cirkeldiagram: toont hoeveel mensen wel/niet dagelijks fruit eten
    """
    plot_proportional_bar(q1, "Fruits",
                          "Q1: Diabetes verhouding per fruitconsumptie")
    plot_pie(q1, "Fruits", "Q1: Verdeling fruitconsumptie",
             labels_map={0: "< 1x per dag", 1: ">= 1x per dag"})


def plot_q2(q2):
    """Grafieken voor Q2: Lifestyle Patronen & Diabetes.

    - Horizontaal staafdiagram: vergelijkt diabetes percentage over alle lifestyle factoren
      (beter dan 5 losse staafdiagrammen, makkelijker te vergelijken)
    - Correlatiematrix: toont samenhang tussen alle factoren
    """
    plot_horizontal_comparison(
        q2, ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
        "Q2: Diabetes percentage per lifestyle factor (waarde=1)")
    plot_correlation_matrix(q2, "Q2: Correlatiematrix lifestyle patronen")


def plot_q3(q3):
    """Grafieken voor Q3: Socio-economische Status & Diabetes.

    - Lijndiagrammen: tonen de trend van diabetes percentage over inkomen/opleiding
      (beter dan staafdiagrammen voor ordinale data — de volgorde en trend zijn belangrijk)
    - Scatter plot: toont de relatie tussen inkomen en opleiding, gekleurd op diabetes
    - Correlatiematrix: toont de samenhang tussen de variabelen
    """
    plot_line(q3, "Income", "Q3: Diabetes percentage per inkomenscategorie (1=laag, 8=hoog)")
    plot_line(q3, "Education", "Q3: Diabetes percentage per opleidingsniveau (1=laag, 6=hoog)")
    plot_scatter_colored(q3, "Income", "Education",
                         "Q3: Inkomen vs Opleiding (kleur = diabetes status)")
    plot_correlation_matrix(q3, "Q3: Correlatiematrix socio-economisch")


def plot_q4(q4):
    """Grafieken voor Q4: Cumulatief Effect Lifestyle Factors.

    - Vlakdiagram: toont hoe diabetes risico stijgt met de score
      (beter dan staafdiagram — benadrukt de cumulatieve opbouw)
    - Cirkeldiagram: toont de verdeling van de score in de populatie
    - Bubbeldiagram: combineert score, diabetes percentage en groepsgrootte
    """
    plot_area(q4, "Unhealthy_Lifestyle_Score",
              "Q4: Diabetes percentage per Unhealthy Lifestyle Score")

    plot_pie(q4, "Unhealthy_Lifestyle_Score",
             "Q4: Verdeling Unhealthy Lifestyle Score",
             labels_map={0: "0 (gezond)", 1: "1", 2: "2", 3: "3", 4: "4 (ongezond)"})

    # Bubbeldiagram
    summary = q4.groupby("Unhealthy_Lifestyle_Score").agg(
        diabetes_pct=("Diabetes_binary", "mean"),
        aantal=("Diabetes_binary", "count"),
    ).reset_index()
    summary["diabetes_pct"] *= 100
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        summary["Unhealthy_Lifestyle_Score"], summary["diabetes_pct"],
        s=summary["aantal"] / 30, alpha=0.6, edgecolors="black", color="steelblue")
    for _, row in summary.iterrows():
        ax.annotate(f"n={int(row['aantal']):,}",
                    (row["Unhealthy_Lifestyle_Score"], row["diabetes_pct"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8)
    ax.set_xlabel("Unhealthy Lifestyle Score")
    ax.set_ylabel("Diabetes (%)")
    ax.set_title("Q4: Bubbeldiagram — Score vs Diabetes (grootte = aantal personen)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_q5(q5):
    """Grafieken voor Q5: Cholesterol + Bloeddruk & Diabetes.

    - 2x2 Heatmap: toont diabetes percentage per combinatie van HighBP en HighChol
      (beter dan staafdiagrammen — de 2x2 structuur maakt het interactie-effect direct zichtbaar)
    - Proportioneel staafdiagram: per combinatie de diabetes verhouding
    - Correlatiematrix: toont de samenhang inclusief interactieterm
    """
    plot_heatmap_2x2(q5, "HighBP", "HighChol",
                     "Q5: Diabetes % per HighBP x HighChol combinatie")

    q5_copy = q5.copy()
    label_map = {
        (0, 0): "Geen BP\nGeen Chol",
        (0, 1): "Geen BP\nHoog Chol",
        (1, 0): "Hoog BP\nGeen Chol",
        (1, 1): "Hoog BP\nHoog Chol",
    }
    q5_copy["Groep"] = list(zip(q5_copy["HighBP"], q5_copy["HighChol"]))
    q5_copy["Groep"] = q5_copy["Groep"].map(label_map)
    plot_proportional_bar(q5_copy, "Groep",
                          "Q5: Diabetes verhouding per BP/Chol combinatie")
    plot_correlation_matrix(q5, "Q5: Correlatiematrix bloeddruk & cholesterol")


def plot_all(questions):
    """Alle grafieken voor alle onderzoeksvragen."""
    plot_q1(questions["q1_fruit_diabetes"])
    plot_q2(questions["q2_lifestyle_patterns"])
    plot_q3(questions["q3_socioeconomic"])
    plot_q4(questions["q4_cumulative_lifestyle"])
    plot_q5(questions["q5_cholesterol_bloodpressure"])
