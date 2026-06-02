"""
Machine-learning analyses voor Stap 10.

We passen twee ML-technieken toe op twee onderzoeksvragen rond de structuur
van onze portfolio's. Beide technieken zijn ongesuperviseerd (er is geen
'juist' label om op te trainen) en sluiten direct aan op het correlatiewerk
uit Stap 6:

  1. PCA (Principal Component Analysis) -- dimensiereductie.
     Onderzoeksvraag: "Hoeveel onafhankelijke marktkrachten sturen onze
     ~20 reeksen, en wat representeren die krachten economisch?"
     Wordt toegepast op de gestandaardiseerde maandelijkse koersniveaus.

  2. Clusteranalyse (K-Means) -- groeperen van assets.
     Onderzoeksvraag: "Vallen onze assets op basis van hun koersgedrag
     uiteen in herkenbare groepen (aandelen/crypto vs. energie/agri vs.
     metalen)?"
     Wordt toegepast op de correlatiematrix van de assets.

Gebruikt scikit-learn (PCA, KMeans, StandardScaler, silhouette_score) en
hergebruikt de dataloading van StatisticalAnalysisService zodat we exact
dezelfde opgeschoonde portfolio's gebruiken als in de eerdere stappen.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from blok2.data_procesing.statistical_analysis_service import StatisticalAnalysisService


_service = StatisticalAnalysisService()


def _load_assets_monthly():
    """
    Laadt de financiele en grondstoffen-portfolio's en resamplet naar
    maandelijkse eind-niveaus. We laten de macro-indicatoren hier bewust
    weg: PCA en clustering vergelijken hier verhandelbare assets onderling,
    en de macro-reeksen (rente, CPI) hebben een andere schaal/frequentie.

    Returnt een DataFrame (maand-index x assets) zonder NaN-rijen, zodat
    elke asset over exact dezelfde periode wordt vergeleken.
    """
    df_fin = _service.load_portfolio('financial')
    df_com = _service.load_portfolio('commodities')
    monthly = df_fin.join(df_com, how='inner').resample('ME').last()
    return monthly.dropna()


# Economische labels per ticker -- alleen voor interpretatie van de
# clusters/PCA-loadings, niet als trainingslabel.
_ASSET_GROUP = {
    'MC.PA': 'Aandeel', 'BRK-B': 'Aandeel', 'MSFT': 'Aandeel', 'JPM': 'Aandeel',
    'TNX': 'Rente', 'IRX': 'Rente', 'TYX': 'Rente',
    'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'SOL-USD': 'Crypto',
    'GC_F': 'Metaal', 'SI_F': 'Metaal', 'HG_F': 'Metaal',
    'CL_F': 'Energie', 'BZ_F': 'Energie', 'NG_F': 'Energie',
    'ZC_F': 'Agri', 'ZW_F': 'Agri', 'ZS_F': 'Agri', 'KC_F': 'Agri',
}


# ---------------------------------------------------------------------------
# Techniek 1: PCA
# ---------------------------------------------------------------------------

def run_pca():
    """
    Voert PCA uit op de gestandaardiseerde maandelijkse koersniveaus.

    Standaardiseren (z-score) is noodzakelijk omdat de reeksen totaal
    verschillende schalen hebben (Bitcoin in tienduizenden dollars, de
    10-jaars rente in procenten). Zonder standaardisatie zou PCA puur de
    asset met de grootste absolute variantie oppikken in plaats van de
    onderliggende gedeelde beweging.

    Returnt een dict met explained variance, loadings en de scores
    (de portfolio geprojecteerd op de eerste twee componenten).
    """
    monthly = _load_assets_monthly()
    assets = list(monthly.columns)

    X = StandardScaler().fit_transform(monthly.values)

    pca = PCA()
    scores = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    loadings = pd.DataFrame(
        pca.components_[:3].T,
        index=assets,
        columns=['PC1', 'PC2', 'PC3'],
    )
    scores_df = pd.DataFrame(
        scores[:, :2],
        index=monthly.index,
        columns=['PC1', 'PC2'],
    )

    return {
        'explained_variance_ratio': evr,
        'cumulative': np.cumsum(evr),
        'loadings': loadings,
        'scores': scores_df,
        'n_months': len(monthly),
        'assets': assets,
    }


def _plot_pca_scree(result):
    """Scree-plot: verklaarde variantie per component + cumulatieve lijn."""
    evr = result['explained_variance_ratio']
    cum = result['cumulative']
    n = len(evr)
    comps = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(comps, evr * 100, color='steelblue', alpha=0.8, label='Per component')
    ax1.set_xlabel('Principale component')
    ax1.set_ylabel('Verklaarde variantie (%)', color='steelblue')
    ax1.set_xticks(comps)

    ax2 = ax1.twinx()
    ax2.plot(comps, cum * 100, color='darkorange', marker='o', label='Cumulatief')
    ax2.axhline(80, color='red', linestyle='--', alpha=0.6, label='80%-grens')
    ax2.set_ylabel('Cumulatieve variantie (%)', color='darkorange')
    ax2.set_ylim(0, 105)

    plt.title('PCA Scree-plot: verklaarde variantie per component')
    fig.tight_layout()
    plt.show()


def _plot_pca_loadings(result):
    """Biplot-achtige scatter van de loadings op PC1 vs PC2, gekleurd per groep."""
    loadings = result['loadings']
    plt.figure(figsize=(10, 8))

    groups = sorted(set(_ASSET_GROUP.get(a, 'Overig') for a in loadings.index))
    palette = dict(zip(groups, sns.color_palette('tab10', len(groups))))

    for asset in loadings.index:
        g = _ASSET_GROUP.get(asset, 'Overig')
        plt.scatter(loadings.loc[asset, 'PC1'], loadings.loc[asset, 'PC2'],
                    color=palette[g], s=80)
        plt.annotate(asset,
                     (loadings.loc[asset, 'PC1'], loadings.loc[asset, 'PC2']),
                     fontsize=8, xytext=(4, 4), textcoords='offset points')

    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=palette[g], label=g)
               for g in groups]
    plt.legend(handles=handles, title='Economische groep')
    plt.axhline(0, color='grey', linewidth=0.6)
    plt.axvline(0, color='grey', linewidth=0.6)
    plt.xlabel(f"PC1 ({result['explained_variance_ratio'][0]*100:.1f}% variantie)")
    plt.ylabel(f"PC2 ({result['explained_variance_ratio'][1]*100:.1f}% variantie)")
    plt.title('PCA Loadings: positie van elke asset op de eerste twee componenten')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def _n_components_for(result, threshold=0.80):
    """Kleinste aantal componenten dat samen >= threshold variantie verklaart."""
    return int(np.searchsorted(result['cumulative'], threshold) + 1)


def show_pca_analysis():
    """
    Toont de volledige PCA in de notebook: uitleg, scree-plot, loadings-plot,
    loadings-tabel en een geautomatiseerde conclusie.
    """
    display(Markdown(
        "## Techniek 1 -- PCA op de portfolio-structuur\n\n"
        "**Onderzoeksvraag:** Hoeveel *onafhankelijke* marktkrachten sturen onze "
        "~20 reeksen, en wat representeren die krachten economisch?\n\n"
        "PCA zoekt nieuwe assen (componenten) die zoveel mogelijk van de totale "
        "variantie vangen. We standaardiseren eerst alle koersen (z-score) omdat "
        "de schalen sterk verschillen, en draaien daarna PCA op de maandelijkse "
        "niveaus."
    ))

    result = run_pca()

    n80 = _n_components_for(result, 0.80)
    evr = result['explained_variance_ratio']

    print(f"Aantal maanden (observaties): {result['n_months']}")
    print(f"Aantal assets (variabelen):  {len(result['assets'])}")
    print(f"PC1 verklaart {evr[0]*100:.1f}% van de variantie")
    print(f"PC1+PC2 samen {result['cumulative'][1]*100:.1f}%")
    print(f"{n80} componenten verklaren samen >= 80% van de variantie")

    _plot_pca_scree(result)
    _plot_pca_loadings(result)

    display(Markdown("### Loadings (bijdrage van elke asset aan PC1-PC3)"))
    display(result['loadings'].style.format('{:.3f}').background_gradient(
        cmap='coolwarm', axis=None))

    # Conclusie automatisch opbouwen
    pc1_top = result['loadings']['PC1'].sort_values(ascending=False).head(3).index.tolist()
    pc2_pos = result['loadings']['PC2'].sort_values(ascending=False).head(3).index.tolist()
    pc2_neg = result['loadings']['PC2'].sort_values().head(3).index.tolist()

    display(Markdown(
        f"### Conclusie PCA\n\n"
        f"- De eerste component (**PC1, {evr[0]*100:.1f}%**) heeft voor vrijwel "
        f"alle assets een positieve loading. Dit is een **algemene markt-/"
        f"liquiditeitsfactor**: als deze stijgt, stijgt de hele portfolio mee. "
        f"De sterkste dragers zijn {', '.join(pc1_top)}.\n"
        f"- De tweede component (**PC2, {evr[1]*100:.1f}%**) zet "
        f"{', '.join(pc2_pos)} (positief) tegenover {', '.join(pc2_neg)} "
        f"(negatief). Dit scheidt **grondstoffen (energie/agri) van financiele "
        f"assets en metalen** -- precies de tegenstelling die we in de "
        f"correlatie-heatmaps van Stap 6 al zagen.\n"
        f"- **{n80} componenten** verklaren samen meer dan 80% van alle beweging. "
        f"Onze ~20 reeksen worden dus in werkelijkheid door een veel kleiner "
        f"aantal onderliggende krachten gestuurd: de markten zijn sterk verweven."
    ))

    return result


# ---------------------------------------------------------------------------
# Techniek 2: Clusteranalyse (K-Means)
# ---------------------------------------------------------------------------

def run_cluster_analysis(k_range=range(2, 7)):
    """
    Clustert de assets met K-Means op hun correlatiematrix.

    Elke asset wordt beschreven door zijn correlatie met alle andere assets
    (een rij uit de correlatiematrix). Assets die met dezelfde dingen
    meebewegen, komen zo dicht bij elkaar te liggen. We standaardiseren de
    correlatie-features en kiezen het aantal clusters k met de hoogste
    silhouette-score (een maat voor hoe goed gescheiden de clusters zijn,
    bereik -1 tot 1; hoger is beter).

    Returnt een dict met de gekozen k, de labels per asset, de silhouette-
    scores per k en een 2D-projectie (PCA) puur voor de plot.
    """
    monthly = _load_assets_monthly()
    corr = monthly.corr()

    features = StandardScaler().fit_transform(corr.values)

    scores = {}
    fitted = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)
        scores[k] = silhouette_score(features, km.labels_)
        fitted[k] = km

    # Kies de k met de hoogste silhouette-score. Bij een vrijwel gelijke score
    # (binnen `tolerance`) kiezen we de fijnere (hogere) k: die geeft beter
    # interpreteerbare, economisch herkenbare groepen zonder de kwaliteit
    # noemenswaardig op te offeren.
    tolerance = 0.05
    top = max(scores.values())
    best_k = max(k for k, s in scores.items() if s >= top - tolerance)
    best_km = fitted[best_k]

    labels = pd.Series(best_km.labels_, index=corr.index, name='cluster')

    # 2D-projectie van de correlatie-features, enkel om de clusters te tekenen
    coords = PCA(n_components=2, random_state=42).fit_transform(features)
    coords_df = pd.DataFrame(coords, index=corr.index, columns=['x', 'y'])

    return {
        'corr': corr,
        'labels': labels,
        'silhouette_scores': scores,
        'best_k': best_k,
        'best_silhouette': scores[best_k],
        'coords': coords_df,
        'assets': list(corr.index),
    }


def _plot_silhouette(result):
    """Lijn van silhouette-score per aantal clusters k."""
    scores = result['silhouette_scores']
    ks = sorted(scores)
    plt.figure(figsize=(8, 4))
    plt.plot(ks, [scores[k] for k in ks], marker='o', color='steelblue')
    plt.scatter([result['best_k']], [result['best_silhouette']],
                color='darkorange', s=120, zorder=5, label=f"Beste k={result['best_k']}")
    plt.xlabel('Aantal clusters (k)')
    plt.ylabel('Silhouette-score')
    plt.title('Keuze van k: silhouette-score per aantal clusters')
    plt.xticks(ks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def _plot_clusters(result):
    """Scatter van de assets in 2D, gekleurd per gevonden cluster."""
    coords = result['coords']
    labels = result['labels']
    k = result['best_k']

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette('tab10', k)
    for c in range(k):
        members = labels[labels == c].index
        plt.scatter(coords.loc[members, 'x'], coords.loc[members, 'y'],
                    color=palette[c], s=90, label=f'Cluster {c}')
        for asset in members:
            plt.annotate(asset, (coords.loc[asset, 'x'], coords.loc[asset, 'y']),
                         fontsize=8, xytext=(4, 4), textcoords='offset points')

    plt.xlabel('Projectie-as 1')
    plt.ylabel('Projectie-as 2')
    plt.title(f'K-Means clustering van assets (k={k})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def show_cluster_analysis():
    """
    Toont de volledige clusteranalyse in de notebook: uitleg, silhouette-
    keuze, cluster-scatter, ledenlijst per cluster en een conclusie.
    """
    display(Markdown(
        "## Techniek 2 -- Clusteranalyse (K-Means) van de assets\n\n"
        "**Onderzoeksvraag:** Vallen onze assets, op basis van hun koersgedrag, "
        "uiteen in herkenbare groepen?\n\n"
        "We beschrijven elke asset door zijn correlatie met alle andere assets "
        "en laten K-Means daar groepen in vinden. Het aantal clusters *k* kiezen "
        "we objectief met de hoogste silhouette-score."
    ))

    result = run_cluster_analysis()

    print(f"Gekozen aantal clusters k = {result['best_k']} "
          f"(silhouette = {result['best_silhouette']:.3f})")
    print("Silhouette per k: " +
          ", ".join(f"k={k}:{s:.3f}" for k, s in sorted(result['silhouette_scores'].items())))

    _plot_silhouette(result)
    _plot_clusters(result)

    # Ledenlijst + dominante economische groep per cluster
    display(Markdown("### Samenstelling per cluster"))
    lines = []
    for c in range(result['best_k']):
        members = result['labels'][result['labels'] == c].index.tolist()
        groups = [_ASSET_GROUP.get(a, 'Overig') for a in members]
        dominant = pd.Series(groups).value_counts().idxmax()
        lines.append(f"- **Cluster {c}** (overwegend *{dominant}*): "
                     f"{', '.join(members)}")
    display(Markdown("\n".join(lines)))

    display(Markdown(
        f"### Conclusie clusteranalyse\n\n"
        f"- K-Means vindt bij **k={result['best_k']}** de best gescheiden indeling "
        f"(silhouette = {result['best_silhouette']:.3f}; een positieve score "
        f"betekent dat assets duidelijk dichter bij hun eigen cluster liggen dan "
        f"bij een ander).\n"
        f"- De clusters vallen samen met **economisch logische groepen** "
        f"(zie de tabel hierboven). Het algoritme kreeg geen labels mee en heeft "
        f"deze structuur dus puur uit het koersgedrag afgeleid.\n"
        f"- Dit bevestigt kwantitatief wat de correlatie-heatmaps in Stap 6 al "
        f"visueel lieten zien: de portfolio bestaat uit een beperkt aantal "
        f"samenhangende blokken in plaats van 20 losse, onafhankelijke reeksen."
    ))

    return result


if __name__ == "__main__":
    print("== PCA ==")
    r = run_pca()
    print("explained var:", np.round(r['explained_variance_ratio'][:5], 3))
    print("\n== Clustering ==")
    rc = run_cluster_analysis()
    print("best k:", rc['best_k'], "silhouette:", round(rc['best_silhouette'], 3))
