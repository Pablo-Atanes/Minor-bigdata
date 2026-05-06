"""
Meervoudige regressieanalyses voor de 5 onderzoeksvragen (Stap 7).

Gebruikt scikit-learn LinearRegression voor het schatten van de coefficienten
en scipy.stats voor het berekenen van p-waarden (sklearn levert deze niet
standaard). Data wordt geresamplet naar maandelijkse frequentie omdat de
macro-economische indicatoren (CBS/ECB) maandelijks worden gepubliceerd.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from blok2.data_procesing.statistical_analysis_service import StatisticalAnalysisService


_service = StatisticalAnalysisService()


def _load_combined_monthly():
    """
    Laadt alle drie portfolio's, resamplet naar maandelijkse eind-waarden,
    en mergt op datum. Returnt een DataFrame met DatetimeIndex (month-end).

    Macro-indicatoren (ECB-rente, werkloosheid) worden niet elke maand
    gepubliceerd — sommige zijn kwartaal of veranderen pas bij een
    beleidsbesluit. We forward-fillen de macro-reeksen na de resample
    zodat het laatst bekende niveau blijft gelden tot een update volgt
    (correct gedrag voor 'stock' indicatoren).
    """
    df_fin = _service.load_portfolio('financial')
    df_com = _service.load_portfolio('commodities')
    df_mac = _service.load_portfolio('macro')

    df_fin_m = df_fin.resample('ME').last()
    df_com_m = df_com.resample('ME').last()
    df_mac_m = df_mac.resample('ME').last().ffill()

    combined = df_fin_m.join(df_com_m, how='outer').join(df_mac_m, how='outer')
    return combined


def _build_design_matrix(df_monthly, y_col, x_specs, y_as_return=False):
    """
    Bouwt de X- en y-matrix volgens de specificatie.

    x_specs is een lijst van tuples:
      (kolomnaam, transform, label)
    waarbij transform een van:
      'level'        — ruwe waarde
      'return'       — pct_change (procentueel rendement)
      'diff'         — eerste verschil
      ('lag', n)     — value verschoven met n maanden
      ('return_lag', n) — pct_change verschoven met n maanden

    Geeft (X_df, y_series) terug zonder NaN-rijen.
    """
    work = pd.DataFrame(index=df_monthly.index)

    # Y bouwen
    y_raw = df_monthly[y_col]
    if y_as_return:
        y = y_raw.pct_change() * 100  # procentuele return
        y_label = f"{y_col} (% return)"
    else:
        y = y_raw
        y_label = y_col

    # X bouwen
    x_labels = []
    for spec in x_specs:
        col, transform, label = spec
        base = df_monthly[col]
        if transform == 'level':
            series = base
        elif transform == 'return':
            series = base.pct_change() * 100
        elif transform == 'diff':
            series = base.diff()
        elif isinstance(transform, tuple) and transform[0] == 'lag':
            series = base.shift(transform[1])
        elif isinstance(transform, tuple) and transform[0] == 'return_lag':
            series = (base.pct_change() * 100).shift(transform[1])
        else:
            raise ValueError(f"Onbekende transform: {transform}")
        work[label] = series
        x_labels.append(label)

    work['__y__'] = y
    work = work.dropna()

    X = work[x_labels].copy()
    y_clean = work['__y__'].copy()
    return X, y_clean, y_label


def _compute_pvalues(X, y, model):
    """
    Bereken handmatig std-errors, t-stats en p-waarden voor een
    sklearn LinearRegression fit. Formule:
      var(beta) = sigma^2 * (X'X)^-1, waarbij sigma^2 = SSR / (n - k - 1).
    """
    n = len(y)
    k = X.shape[1]  # aantal predictors (exclusief intercept)

    # Voeg intercept-kolom toe voor matrix-berekening
    X_design = np.hstack([np.ones((n, 1)), X.values])

    y_pred = model.predict(X)
    residuals = y.values - y_pred
    ssr = np.sum(residuals ** 2)
    df_resid = n - k - 1
    sigma_squared = ssr / df_resid if df_resid > 0 else np.nan

    try:
        cov_matrix = sigma_squared * np.linalg.inv(X_design.T @ X_design)
        std_errors_all = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_errors_all = np.full(k + 1, np.nan)

    coefs_all = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs_all / std_errors_all
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df_resid))

    return coefs_all, std_errors_all, t_stats, p_values


def run_multiple_regression(y_col, x_specs, y_as_return=False, test_size=0.2):
    """
    Voert een meervoudige regressieanalyse uit en retourneert een dict met
    alle resultaten (coefs, p-values, R2, RMSE, voorspellingen).
    """
    df = _load_combined_monthly()
    X, y, y_label = _build_design_matrix(df, y_col, x_specs, y_as_return=y_as_return)

    if len(y) < (len(x_specs) + 5):
        raise ValueError(
            f"Te weinig observaties ({len(y)}) voor {len(x_specs)} predictors."
        )

    # Train/test split met shuffle=False (tijdreeks: eerste 80% train, laatste 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Statistieken op de train-set (waar het model op gefit is)
    coefs_all, std_errors_all, t_stats, p_values = _compute_pvalues(X_train, y_train, model)

    r2_train = r2_score(y_train, y_train_pred)
    n_train = len(y_train)
    k = X_train.shape[1]
    adj_r2 = 1 - (1 - r2_train) * (n_train - 1) / (n_train - k - 1)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Coefficienten-tabel bouwen
    rows = ['Intercept'] + list(X.columns)
    coef_df = pd.DataFrame({
        'Coefficient': coefs_all,
        'Std Error': std_errors_all,
        't-stat': t_stats,
        'p-value': p_values,
    }, index=rows)

    return {
        'coef_df': coef_df,
        'r2': r2_train,
        'adj_r2': adj_r2,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'n_train': n_train,
        'n_test': len(y_test),
        'y_label': y_label,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'x_labels': list(X.columns),
    }


def _plot_actual_vs_predicted(result, title):
    """Scatter van werkelijke vs voorspelde Y, met train- en test-punten."""
    plt.figure(figsize=(9, 6))

    plt.scatter(result['y_train'], result['y_train_pred'],
                alpha=0.6, label=f"Train (n={result['n_train']})", color='steelblue')
    plt.scatter(result['y_test'], result['y_test_pred'],
                alpha=0.8, label=f"Test (n={result['n_test']})", color='darkorange', marker='^')

    all_y = np.concatenate([result['y_train'].values, result['y_test'].values])
    all_pred = np.concatenate([result['y_train_pred'], result['y_test_pred']])
    lo = min(all_y.min(), all_pred.min())
    hi = max(all_y.max(), all_pred.max())
    plt.plot([lo, hi], [lo, hi], 'r--', alpha=0.7, label='Perfecte voorspelling')

    plt.xlabel(f"Werkelijke {result['y_label']}")
    plt.ylabel(f"Voorspelde {result['y_label']}")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def _generate_conclusion(result, hypothesis_h1, expected_signs):
    """
    Genereert een Markdown-conclusie. expected_signs is een dict
    {label: '+' | '-' | None} die zegt welk teken de hypothese voorspelt.
    """
    coef_df = result['coef_df']

    significant_lines = []
    for label in result['x_labels']:
        coef = coef_df.loc[label, 'Coefficient']
        p = coef_df.loc[label, 'p-value']
        sig = "**significant**" if p < 0.05 else "niet significant"
        direction = "positief" if coef > 0 else "negatief"

        expected = expected_signs.get(label)
        if expected is None:
            verdict = ""
        elif (expected == '+' and coef > 0) or (expected == '-' and coef < 0):
            verdict = " — richting komt overeen met H1"
        else:
            verdict = " — richting is **tegengesteld** aan H1"

        significant_lines.append(
            f"- **{label}**: coefficient = {coef:.4f} ({direction}), p = {p:.4f} ({sig}){verdict}"
        )

    sig_predictors = [
        label for label in result['x_labels']
        if coef_df.loc[label, 'p-value'] < 0.05
    ]

    if sig_predictors:
        h0_oordeel = (
            f"Minimaal één predictor ({', '.join(sig_predictors)}) is significant bij p<0.05. "
            "We **verwerpen H0** voor deze predictor(en) en vinden ondersteuning voor H1."
        )
    else:
        h0_oordeel = (
            "Geen enkele predictor is significant bij p<0.05. "
            "We kunnen **H0 niet verwerpen** — de data biedt onvoldoende bewijs voor H1."
        )

    md = f"""**Hypothese (H1):** {hypothesis_h1}

**Modelfit:**
- R² (train) = {result['r2']:.4f}
- Adjusted R² = {result['adj_r2']:.4f}
- RMSE train = {result['rmse_train']:.4f}
- RMSE test = {result['rmse_test']:.4f}
- N train = {result['n_train']}, N test = {result['n_test']}

**Per predictor:**
{chr(10).join(significant_lines)}

**Conclusie:** {h0_oordeel}
"""
    return md


def show_regression_analysis(rq_number, title, hypothesis_h1, y_col, x_specs,
                             y_as_return=False, expected_signs=None):
    """
    Toont een complete regressieanalyse in de notebook: titel, hypothese,
    coefficientensoorten, modelfit-statistieken, scatter werkelijk-vs-voorspeld,
    en een geautomatiseerde conclusie.
    """
    if expected_signs is None:
        expected_signs = {}

    display(Markdown(f"## Onderzoeksvraag {rq_number}: {title}"))
    display(Markdown(f"**H0:** Er is geen relatie tussen de gekozen predictors en {y_col}.  \n"
                     f"**H1:** {hypothesis_h1}"))

    result = run_multiple_regression(y_col, x_specs, y_as_return=y_as_return)

    display(Markdown("### Coefficientenschattingen"))
    styled = result['coef_df'].style.format({
        'Coefficient': '{:.6f}',
        'Std Error': '{:.6f}',
        't-stat': '{:.3f}',
        'p-value': '{:.4f}',
    }).map(
        lambda v: 'background-color: #d4edda' if isinstance(v, float) and v < 0.05 else '',
        subset=['p-value']
    )
    display(styled)

    print(f"R² (train): {result['r2']:.4f}    Adjusted R²: {result['adj_r2']:.4f}")
    print(f"RMSE train: {result['rmse_train']:.4f}    RMSE test: {result['rmse_test']:.4f}")
    print(f"N train: {result['n_train']}    N test: {result['n_test']}")

    _plot_actual_vs_predicted(result, f"RQ{rq_number}: {title} — Werkelijk vs Voorspeld")

    display(Markdown(_generate_conclusion(result, hypothesis_h1, expected_signs)))

    return result
