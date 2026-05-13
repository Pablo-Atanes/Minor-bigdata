"""
Stap 8 — Seizoensanalyse, Timeshift en Resampling.

Drie publieke functies voor de notebook:
- show_seasonal_analysis(portfolio, ticker, label, is_macro=False)
- show_timeshift_analysis(x_portfolio, x_col, y_portfolio, y_col, lag_range)
- show_volatility_resampling(portfolio, ticker)

Gebruikt StatisticalAnalysisService voor het laden van de drie portfolio-bestanden
(financial / commodities / macro) zodat we de cleaning-pipeline niet dupliceren.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from blok2.data_procesing.statistical_analysis_service import StatisticalAnalysisService


_service = StatisticalAnalysisService()

NL_MONTHS = [
    'Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun',
    'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec',
]


def _load_monthly_series(portfolio, ticker, is_macro=False):
    """
    Laadt één serie en resamplet naar maandelijkse eind-waarde.
    Voor macro is de bron al maandelijks (forward-filled naar daily); we nemen
    de laatste waarde per maand zodat de seizoenscomponent niet uitgesmeerd is.
    """
    df = _service.load_portfolio(portfolio)
    if ticker not in df.columns:
        raise ValueError(
            f"Ticker '{ticker}' niet gevonden in portfolio '{portfolio}'. "
            f"Beschikbaar: {list(df.columns)}"
        )
    series = df[ticker].dropna()
    monthly = series.resample('ME').last().dropna()
    return monthly


# ============================================================================
# 1. SEIZOENSANALYSE
# ============================================================================

def show_seasonal_analysis(portfolio, ticker, label=None, is_macro=False,
                           heatmap_recent_years=10):
    """
    Maandelijkse seizoensanalyse.

    - Voor financieel/grondstof: maandelijks procentueel rendement (pct_change).
    - Voor macro-indicator: maandelijkse mutatie (absoluut verschil, diff),
      omdat het al een index/percentage is en pct_change op een rente weinig
      betekenis heeft.

    Toont:
      1. Barplot van het gemiddelde per kalendermaand (±1 std)
      2. Heatmap jaar × maand (laatste N jaar) voor jaar-op-jaar consistentie
      3. Markdown-toelichting met sterkste/zwakste maand en interpretatie
    """
    label = label or ticker
    monthly = _load_monthly_series(portfolio, ticker, is_macro=is_macro)

    if is_macro:
        change = monthly.diff().dropna()
        unit = 'absolute mutatie'
        fmt = '{:+.4f}'
    else:
        change = monthly.pct_change().dropna() * 100
        unit = '% rendement'
        fmt = '{:+.2f}%'

    df = change.to_frame('change')
    df['month'] = df.index.month
    df['year'] = df.index.year

    monthly_stats = df.groupby('month')['change'].agg(['mean', 'std', 'count'])

    display(Markdown(f"### Seizoensanalyse: {label} ({unit})"))
    display(Markdown(
        f"Periode: {monthly.index.min().strftime('%Y-%m')} t/m "
        f"{monthly.index.max().strftime('%Y-%m')} — "
        f"{len(monthly)} maanden, {len(df)} mutaties."
    ))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Plot 1: barplot met errorbar
    ax1 = axes[0]
    colors = ['#2ca02c' if v >= 0 else '#d62728' for v in monthly_stats['mean']]
    ax1.bar(range(1, 13), monthly_stats['mean'],
            yerr=monthly_stats['std'], color=colors, alpha=0.75,
            capsize=4, edgecolor='black', linewidth=0.5)
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(NL_MONTHS)
    ax1.set_xlabel('Maand')
    ax1.set_ylabel(f'Gemiddelde {unit}')
    ax1.set_title(f'Gemiddelde per kalendermaand — {label}')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Plot 2: heatmap jaar x maand
    ax2 = axes[1]
    pivot = df.pivot_table(index='year', columns='month', values='change', aggfunc='mean')
    pivot = pivot.tail(heatmap_recent_years)
    pivot.columns = NL_MONTHS[:len(pivot.columns)] if len(pivot.columns) <= 12 else pivot.columns
    sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=True,
                fmt='.1f', cbar_kws={'label': unit}, ax=ax2,
                linewidths=0.3, linecolor='white')
    ax2.set_title(f'Heatmap jaar × maand (laatste {heatmap_recent_years} jaar)')
    ax2.set_xlabel('Maand')
    ax2.set_ylabel('Jaar')

    plt.suptitle(f'Seizoensanalyse {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Tabel met statistieken
    display(Markdown("**Gemiddelde per maand:**"))
    stats_display = monthly_stats.copy()
    stats_display.index = [NL_MONTHS[i - 1] for i in stats_display.index]
    stats_display.columns = ['Gemiddelde', 'Std', 'N']
    display(stats_display.style.format({'Gemiddelde': fmt, 'Std': '{:.3f}', 'N': '{:.0f}'}))

    # Toelichting
    best_m = monthly_stats['mean'].idxmax()
    worst_m = monthly_stats['mean'].idxmin()
    best_v = monthly_stats['mean'].max()
    worst_v = monthly_stats['mean'].min()
    spread = best_v - worst_v

    md = f"""**Toelichting seizoensanalyse {label}:**

- **Sterkste maand:** {NL_MONTHS[best_m - 1]} (gemiddeld {fmt.format(best_v)})
- **Zwakste maand:** {NL_MONTHS[worst_m - 1]} (gemiddeld {fmt.format(worst_v)})
- **Spreiding tussen beste en slechtste maand:** {abs(spread):.3f} ({unit})
- **Aantal jaren in steekproef:** {df['year'].nunique()}

De barplot toont het gemiddelde per kalendermaand met ±1 standaarddeviatie als errorbar.
Wanneer een errorbar de nullijn kruist, is het seizoenseffect statistisch zwak en niet
betrouwbaar voor één enkele maand. De heatmap laat zien of het patroon **consistent** is
over de jaren: kleurkolommen die overwegend groen of overwegend rood zijn, duiden op een
stabiel seizoenspatroon; kolommen met gemengde kleuren betekenen dat het seizoenseffect
sterk varieert per jaar."""
    display(Markdown(md))

    return monthly_stats


# ============================================================================
# 2. TIMESHIFT-ANALYSE (leading indicator)
# ============================================================================

def show_timeshift_analysis(x_portfolio, x_col, y_portfolio, y_col,
                            lag_range=range(-6, 13), x_as_return=True,
                            y_as_return=True):
    """
    Onderzoekt cross-correlatie tussen X en Y over verschillende time-lags.

    Positieve lag = X.shift(+lag) wordt vergeleken met Y, d.w.z. X loopt VOOR
    op Y (X is de leading indicator). Negatieve lag = X loopt achter op Y.

    Default: beide series worden eerst stationair gemaakt (pct_change). Voor
    macro-indicatoren kun je y_as_return=False zetten om de absolute mutatie
    te gebruiken (`diff`).
    """
    x_monthly = _load_monthly_series(x_portfolio, x_col)
    y_monthly = _load_monthly_series(y_portfolio, y_col, is_macro=(y_portfolio == 'macro'))

    x = x_monthly.pct_change().dropna() * 100 if x_as_return else x_monthly.diff().dropna()
    y = y_monthly.pct_change().dropna() * 100 if y_as_return else y_monthly.diff().dropna()

    lags = list(lag_range)
    correlations = []
    n_obs = []
    for lag in lags:
        joined = pd.concat([x.shift(lag), y], axis=1, join='inner').dropna()
        joined.columns = ['x', 'y']
        if len(joined) < 5:
            correlations.append(np.nan)
            n_obs.append(len(joined))
            continue
        correlations.append(joined['x'].corr(joined['y']))
        n_obs.append(len(joined))

    result = pd.DataFrame({
        'lag': lags,
        'correlation': correlations,
        'n_obs': n_obs,
    }).set_index('lag')

    # Beste lag op basis van absolute correlatie
    valid = result.dropna()
    best_lag = int(valid['correlation'].abs().idxmax())
    best_r = valid.loc[best_lag, 'correlation']

    display(Markdown(f"### Timeshift-analyse: {x_col} → {y_col}"))
    display(Markdown(
        f"Lag-bereik: {min(lags)} t/m {max(lags)} maanden. "
        f"X = {x_col} ({'rendement' if x_as_return else 'mutatie'}), "
        f"Y = {y_col} ({'rendement' if y_as_return else 'mutatie'})."
    ))

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['#1f77b4' if lag != best_lag else '#ff7f0e' for lag in lags]
    ax.bar(lags, result['correlation'], color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.6, label='lag = 0 (gelijktijdig)')
    ax.axvline(best_lag, color='red', linestyle='--', alpha=0.7,
               label=f'piek bij lag = {best_lag} (r = {best_r:.3f})')
    ax.set_xlabel('Lag in maanden (positief = X loopt voor op Y)')
    ax.set_ylabel('Pearson-correlatie')
    ax.set_title(f'Cross-correlatie {x_col} (geshift) vs {y_col}')
    ax.set_xticks(lags)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()

    display(Markdown("**Correlatie per lag:**"))
    display(result.style.format({'correlation': '{:+.4f}', 'n_obs': '{:.0f}'}))

    # Conclusie
    if best_lag > 0:
        leading_msg = (
            f"**{x_col} loopt {best_lag} maand(en) voor op {y_col}** — "
            f"oftewel: een verandering in {x_col} vandaag is geassocieerd "
            f"met een verandering in {y_col} over {best_lag} maand(en)."
        )
    elif best_lag < 0:
        leading_msg = (
            f"**{y_col} loopt {abs(best_lag)} maand(en) voor op {x_col}** — "
            f"in dit lag-bereik is {x_col} dus géén leading indicator voor {y_col}; "
            f"het omgekeerde lijkt eerder waar."
        )
    else:
        leading_msg = (
            f"De sterkste correlatie zit bij lag = 0 (gelijktijdig). "
            f"Er is geen duidelijke leading-relatie in deze richting."
        )

    strength = abs(best_r)
    if strength >= 0.5:
        strength_msg = f"De gevonden correlatie (|r| = {strength:.3f}) is **sterk**."
    elif strength >= 0.3:
        strength_msg = f"De gevonden correlatie (|r| = {strength:.3f}) is **matig**."
    elif strength >= 0.1:
        strength_msg = f"De gevonden correlatie (|r| = {strength:.3f}) is **zwak**."
    else:
        strength_msg = f"De gevonden correlatie (|r| = {strength:.3f}) is **verwaarloosbaar**."

    md = f"""**Toelichting timeshift-analyse:**

- **Piek-lag:** {best_lag} maand(en)
- **Piek-correlatie:** r = {best_r:+.4f}
- **Aantal observaties bij piek:** {int(result.loc[best_lag, 'n_obs'])}

{leading_msg}

{strength_msg}

**Methode-uitleg:** voor elke lag in het bereik schuiven we de X-reeks `lag`
posities op en berekenen we de Pearson-correlatie met Y over alle datums waar
beide reeksen waarden hebben. Door beide reeksen eerst stationair te maken
(pct_change/diff) voorkomen we spurious correlatie door trends. De lag met
de hoogste **absolute** correlatie is het beste kandidaat-lead-lag-paar; positief
betekent dat X **vooruit** loopt op Y."""
    display(Markdown(md))

    return result


# ============================================================================
# 3. RESAMPLING — KOERSVOLATILITEIT PER TIJDSNIVEAU
# ============================================================================

# Pandas resample-codes met bijbehorend aantal periodes per jaar
RESAMPLE_LEVELS = [
    ('D',  'Dag',      252),   # 252 handelsdagen per jaar
    ('W',  'Week',     52),
    ('ME', 'Maand',    12),
    ('QE', 'Kwartaal', 4),
]


def show_volatility_resampling(portfolio, ticker, levels=None):
    """
    Toont volatiliteit van koers-returns op verschillende tijdsniveaus.

    Volatiliteit = standaardafwijking van de pct_change returns op dat niveau.
    Geannualiseerde volatiliteit = std × √(periodes per jaar). Onder de
    aanname van independent identically distributed returns (random walk)
    zou de geannualiseerde vol over alle niveaus ongeveer gelijk moeten zijn.
    Afwijkingen wijzen op autocorrelatie / volatility clustering / mean reversion.
    """
    levels = levels or RESAMPLE_LEVELS

    df = _service.load_portfolio(portfolio)
    if ticker not in df.columns:
        raise ValueError(
            f"Ticker '{ticker}' niet gevonden in portfolio '{portfolio}'. "
            f"Beschikbaar: {list(df.columns)}"
        )

    price = df[ticker].dropna()
    display(Markdown(f"### Resampling-volatiliteit: {ticker}"))
    display(Markdown(
        f"Periode: {price.index.min().strftime('%Y-%m-%d')} t/m "
        f"{price.index.max().strftime('%Y-%m-%d')} — {len(price)} dagen ruwe data."
    ))

    rows = []
    returns_by_level = {}
    for code, naam, periodes_per_jaar in levels:
        resampled = price.resample(code).last().dropna()
        returns = resampled.pct_change().dropna() * 100  # in %
        std = returns.std()
        annualized = std * np.sqrt(periodes_per_jaar)
        rows.append({
            'Niveau': naam,
            'N obs': len(returns),
            'Std (%)': std,
            'Annualized vol (%)': annualized,
            'Min return (%)': returns.min(),
            'Max return (%)': returns.max(),
        })
        returns_by_level[naam] = returns

    table = pd.DataFrame(rows).set_index('Niveau')
    display(Markdown("**Volatiliteitsstatistieken per niveau:**"))
    display(table.style.format({
        'N obs': '{:.0f}',
        'Std (%)': '{:.3f}',
        'Annualized vol (%)': '{:.2f}',
        'Min return (%)': '{:+.2f}',
        'Max return (%)': '{:+.2f}',
    }))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: geannualiseerde vol per niveau
    ax1 = axes[0]
    ax1.bar(table.index, table['Annualized vol (%)'],
            color=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(table)],
            alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Geannualiseerde volatiliteit (%)')
    ax1.set_title(f'Annualized volatility per tijdsniveau — {ticker}')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    for i, v in enumerate(table['Annualized vol (%)']):
        ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Plot 2: boxplot van returns
    ax2 = axes[1]
    data_for_box = [returns_by_level[naam].values for naam in table.index]
    bp = ax2.boxplot(data_for_box, labels=table.index, patch_artist=True,
                     showfliers=True)
    for patch, color in zip(bp['boxes'],
                            ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'][:len(table)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_ylabel('Return (%) per periode')
    ax2.set_title(f'Verdeling van returns per niveau — {ticker}')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f'Resampling-volatiliteit {ticker}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Toelichting
    max_ann = table['Annualized vol (%)'].idxmax()
    min_ann = table['Annualized vol (%)'].idxmin()
    max_ann_v = table['Annualized vol (%)'].max()
    min_ann_v = table['Annualized vol (%)'].min()
    spread_pct = (max_ann_v - min_ann_v) / min_ann_v * 100 if min_ann_v > 0 else float('nan')

    md = f"""**Toelichting resampling-volatiliteit {ticker}:**

- **Hoogste geannualiseerde vol:** {max_ann} ({max_ann_v:.2f}%)
- **Laagste geannualiseerde vol:** {min_ann} ({min_ann_v:.2f}%)
- **Relatief verschil:** {spread_pct:.1f}% tussen hoogste en laagste niveau

**Methode:** voor elk tijdsniveau (dag / week / maand / kwartaal) resamplen we
de koers naar de laatste waarde van de periode, berekenen we de procentuele
returns, en bepalen we de standaarddeviatie. Vervolgens annualiseren we met
de wortel-tijd-regel (std × √N, waarbij N het aantal periodes per jaar is:
252 voor dag, 52 voor week, 12 voor maand, 4 voor kwartaal).

**Interpretatie:** onder de random-walk hypothese zou de geannualiseerde
volatiliteit op alle niveaus ongeveer gelijk moeten zijn. Wanneer de
geannualiseerde vol op langere niveaus **lager** is dan op kortere, wijst dit
op **mean reversion**: korte-termijn-bewegingen worden deels weer ongedaan
gemaakt. Wanneer de geannualiseerde vol op langere niveaus **hoger** is, wijst
dit op **trending** of volatility clustering (lange perioden van consistente
beweging in één richting). De boxplot rechts laat tegelijk zien dat de
absolute spreiding van returns toeneemt naarmate het tijdsniveau langer is —
dat is normaal, want langere periodes accumuleren meer beweging."""
    display(Markdown(md))

    return table
