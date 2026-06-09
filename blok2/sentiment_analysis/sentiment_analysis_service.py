import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x


DEFAULT_ARTICLES_PATH = 'blok2/sentiment_analysis/articles_lvmh.json'
DEFAULT_PRICES_PATH = 'blok2/data_procesing/ticker_data/MC.PA.json'

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def _resolve_path(path, fallback_dir):
    """Probeer eerst path, val anders terug op fallback_dir/basename (Colab-safe)."""
    if os.path.exists(path):
        return path
    alt = os.path.join(fallback_dir, os.path.basename(path))
    if os.path.exists(alt):
        return alt
    return path


def load_articles(path=DEFAULT_ARTICLES_PATH):
    """Laadt het LVMH-nieuwsartikelen JSON-bestand en retourneert een DataFrame
    gesorteerd op datum."""
    resolved = _resolve_path(path, 'blok2/sentiment_analysis')
    with open(resolved, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def score_articles(df):
    """Past VADER toe op de 'text'-kolom en voegt compound, pos, neu, neg en label toe."""
    analyzer = SentimentIntensityAnalyzer()
    scores = df['text'].apply(analyzer.polarity_scores).apply(pd.Series)
    scored = pd.concat([df, scores], axis=1)

    def to_label(c):
        if c >= POS_THRESHOLD:
            return 'positief'
        if c <= NEG_THRESHOLD:
            return 'negatief'
        return 'neutraal'

    scored['label'] = scored['compound'].apply(to_label)
    return scored


def compute_net_sentiment(df):
    """Net Sentiment Index = gemiddelde van de VADER compound scores."""
    return float(df['compound'].mean())


def show_sentiment_summary(df):
    """Toon Markdown-overzicht met dataset-statistieken en Net Sentiment Index."""
    n = len(df)
    net = compute_net_sentiment(df)
    counts = df['label'].value_counts().reindex(['positief', 'neutraal', 'negatief']).fillna(0).astype(int)
    pct = (counts / n * 100).round(1)
    period_start = df['date'].min().strftime('%Y-%m-%d')
    period_end = df['date'].max().strftime('%Y-%m-%d')
    sources = df['source'].value_counts()

    lines = [
        f"### Sentimentanalyse LVMH",
        f"- **Aantal artikelen:** {n}",
        f"- **Periode:** {period_start} t/m {period_end}",
        f"- **Bronnen:** " + ", ".join(f"{s} ({c})" for s, c in sources.items()),
        "",
        "| Categorie | Aantal | Percentage |",
        "| :--- | ---: | ---: |",
        f"| Positief | {counts['positief']} | {pct['positief']}% |",
        f"| Neutraal | {counts['neutraal']} | {pct['neutraal']}% |",
        f"| Negatief | {counts['negatief']} | {pct['negatief']}% |",
        "",
        f"**Net Sentiment Index:** `{net:.3f}` (schaal -1 tot +1)",
    ]
    display(Markdown("\n".join(lines)))
    return {'n': n, 'net_sentiment': net, 'counts': counts.to_dict(), 'percentages': pct.to_dict()}


def show_sentiment_distribution(df):
    """Staafdiagram van de label-verdeling."""
    counts = df['label'].value_counts().reindex(['positief', 'neutraal', 'negatief']).fillna(0).astype(int)
    colors = ['#2ca02c', '#7f7f7f', '#d62728']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title('Verdeling van het nieuwssentiment over LVMH-artikelen')
    ax.set_ylabel('Aantal artikelen')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.1, str(int(v)), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def show_sentiment_over_time(df, prices_path=DEFAULT_PRICES_PATH, window=7):
    """Lijnplot met rollend gemiddelde van het sentiment en de LVMH-koers
    op een tweede y-as."""
    df = df.sort_values('date').copy()
    df.set_index('date', inplace=True)
    rolling = df['compound'].rolling(window=window, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(rolling.index, rolling.values, color='#1f77b4', label=f'Sentiment ({window}-daags rollend)')
    ax1.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax1.set_ylabel('VADER compound score', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xlabel('Datum')

    prices_resolved = _resolve_path(prices_path, 'blok2/data_procesing/ticker_data')
    if os.path.exists(prices_resolved):
        with open(prices_resolved, 'r', encoding='utf-8') as f:
            prices = pd.DataFrame(json.load(f))
        prices['Date'] = pd.to_datetime(prices['Date'])
        prices = prices.sort_values('Date')
        mask = (prices['Date'] >= df.index.min()) & (prices['Date'] <= df.index.max())
        prices_window = prices.loc[mask]

        ax2 = ax1.twinx()
        ax2.plot(prices_window['Date'], prices_window['Close'], color='#d62728', alpha=0.6, label='LVMH koers (EUR)')
        ax2.set_ylabel('LVMH koers (EUR)', color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')

    plt.title('Sentiment LVMH-nieuws vs LVMH koers')
    fig.tight_layout()
    plt.show()


def export_scores(df, path='blok2/sentiment_analysis/sentiment_scores.csv'):
    """Schrijf de per-artikel scores naar CSV."""
    cols = ['id', 'date', 'source', 'title', 'compound', 'pos', 'neu', 'neg', 'label']
    out = df[cols].copy()
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    return path
