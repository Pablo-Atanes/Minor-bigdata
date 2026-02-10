import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


TARGET = "Diabetes_binary"


def plot_elbow(df, features, k_range=range(2, 9)):
    """Elbow-methode: plot inertia (within-cluster sum of squares) per k.

    Het 'knikpunt' in de grafiek suggereert het optimale aantal clusters.
    """
    X = df[features].values
    inertias = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertias.append(model.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, marker="o", linewidth=2, color="steelblue")
    ax.set_xlabel("Aantal clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow-methode: optimaal aantal clusters")
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return inertias


def plot_silhouette_scores(df, features, k_range=range(2, 9)):
    """Silhouette scores per k.

    Hogere score = betere clustering (max 1.0).
    """
    X = df[features].values
    scores = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(10000, len(X)),
                                 random_state=42)
        scores.append(score)
        print(f"  k={k}: silhouette score = {score:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(k_range), scores, color="steelblue", edgecolor="black")
    ax.set_xlabel("Aantal clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette score per aantal clusters")
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    best_k = list(k_range)[np.argmax(scores)]
    print(f"\n  Hoogste silhouette score bij k={best_k} ({max(scores):.4f})")
    return scores, best_k


def run_clustering(df, features, n_clusters):
    """Fit K-Means en voeg 'Cluster' kolom toe aan het DataFrame.

    Returns:
        (DataFrame met Cluster kolom, gefitte KMeans model)
    """
    X = df[features].values
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = model.fit_predict(X)

    print(f"  K-Means clustering met k={n_clusters}")
    print(f"  Inertia: {model.inertia_:.1f}")
    print(f"\n  Clustergroottes:")
    for c in range(n_clusters):
        n = (df["Cluster"] == c).sum()
        pct = n / len(df) * 100
        print(f"    Cluster {c}: {n:,} respondenten ({pct:.1f}%)")
    print()

    return df, model


def plot_cluster_profiles(df, features):
    """Heatmap: gemiddelde waarde per feature per cluster.

    Toont het 'profiel' van elk cluster — welke lifestyle kenmerken
    dominant zijn in elke groep.
    """
    profiles = df.groupby("Cluster")[features].mean()

    fig, ax = plt.subplots(figsize=(8, max(3, len(profiles) * 0.8 + 1)))
    sns.heatmap(profiles, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                vmin=0, vmax=1, linewidths=0.5,
                cbar_kws={"label": "Gemiddelde (0-1)"})
    ax.set_title("Cluster profielen: gemiddelde waarde per lifestyle factor")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.show()

    return profiles


def plot_diabetes_per_cluster(df):
    """Staafdiagram: diabetes percentage per cluster."""
    rates = df.groupby("Cluster")[TARGET].mean() * 100
    sizes = df.groupby("Cluster")[TARGET].count()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d94a4a" if r > df[TARGET].mean() * 100 else "#4a90d9"
              for r in rates.values]
    bars = ax.bar(rates.index, rates.values, color=colors, edgecolor="black")
    ax.axhline(df[TARGET].mean() * 100, color="gray", linestyle="--",
               label=f"Gemiddeld: {df[TARGET].mean()*100:.1f}%")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)

    # Clustergrootte als annotatie
    for i, (rate, size) in enumerate(zip(rates.values, sizes.values)):
        ax.annotate(f"n={size:,}", (i, rate), textcoords="offset points",
                    xytext=(0, -15), ha="center", fontsize=8, color="white",
                    fontweight="bold")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Diabetes (%)")
    ax.set_title("Diabetes percentage per cluster")
    ax.set_xticks(rates.index)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    return rates


def cluster_q2(q2, n_clusters=4):
    """Volledige clusteranalyse pipeline voor Q2 lifestyle data.

    Stappen:
    1. Elbow-methode en silhouette scores
    2. K-Means clustering met gekozen k
    3. Cluster profielen (heatmap)
    4. Diabetes percentage per cluster
    5. Nederlandse interpretatie

    Parameters:
        q2: DataFrame met Q2 lifestyle data
        n_clusters: aantal clusters (standaard 4)

    Returns:
        (DataFrame met Cluster kolom, cluster profielen, diabetes rates)
    """
    features = ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"]

    print("="*70)
    print("  CLUSTERANALYSE: LIFESTYLE PROFIELEN (Q2)")
    print("="*70)

    # Stap 1: Optimaal k bepalen
    print("\n--- Stap 1: Elbow-methode ---")
    plot_elbow(q2, features)

    print("\n--- Stap 2: Silhouette scores ---")
    scores, best_k = plot_silhouette_scores(q2, features)

    # Stap 2: Clustering uitvoeren
    print(f"\n--- Stap 3: K-Means clustering (k={n_clusters}) ---")
    q2_clustered, model = run_clustering(q2, features, n_clusters)

    # Stap 3: Profielen
    print("\n--- Stap 4: Cluster profielen ---")
    profiles = plot_cluster_profiles(q2_clustered, features)

    print("\nCluster profielen (gemiddelde per feature):")
    print(profiles.to_string())
    print()

    # Stap 4: Diabetes per cluster
    print("\n--- Stap 5: Diabetes percentage per cluster ---")
    rates = plot_diabetes_per_cluster(q2_clustered)

    print("\nDiabetes percentage per cluster:")
    for cluster, rate in rates.items():
        size = (q2_clustered["Cluster"] == cluster).sum()
        print(f"  Cluster {cluster}: {rate:.1f}% diabetes (n={size:,})")

    print()
    return q2_clustered, profiles, rates
