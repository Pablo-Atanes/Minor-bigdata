import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy.stats import skew, kurtosis

# Voor Colab ondersteuning (indien IPython beschikbaar is)
try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

class StatisticalAnalysisService:
    def __init__(self, data_dir='cleaned_output'):
        # In Colab moeten we mogelijk het volledige pad opgeven als we niet in de juiste map zitten
        self.data_dir = data_dir
        self.portfolios = {
            'financial': 'portfolio_financial_products.json',
            'commodities': 'portfolio_commodities.json',
            'macro': 'portfolio_macro_indicators.json'
        }
        
    def load_portfolio(self, name):
        """Laadt een portfolio JSON bestand."""
        # Zoek naar de juiste key (flexibel voor afkortingen)
        key = next((k for k in self.portfolios if name.lower() in k), None)
        if not key:
            raise ValueError(f"Portfolio '{name}' niet gevonden. Kies uit: {list(self.portfolios.keys())}")
            
        file_path = os.path.join(self.data_dir, self.portfolios[key])
        if not os.path.exists(file_path):
             # Probeer een niveau hoger als we in een submap zitten (handig voor Colab)
             file_path = os.path.join('blok2/data_procesing', self.data_dir, self.portfolios[key])
             
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df

    def get_stats(self, df):
        """Calculates all basic statistics."""
        stats_dict = {}
        for col in df.columns:
            series = df[col].dropna()
            mode_val = series.mode()
            mode_val = mode_val.iloc[0] if not mode_val.empty else np.nan
            
            stats_dict[col] = {
                'Mean': series.mean(),
                'Median': series.median(),
                'Mode': mode_val,
                'Standard Deviation': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Skewness': skew(series),
                'Kurtosis': kurtosis(series)
            }
        return pd.DataFrame(stats_dict).T

    def get_combined_df(self):
        """Voegt alle drie de portfolio's samen op basis van datum."""
        df_fin = self.load_portfolio('financial')
        df_com = self.load_portfolio('commodities')
        df_mac = self.load_portfolio('macro')
        
        # Merge op index (Date)
        combined = df_fin.join(df_com, how='outer').join(df_mac, how='outer')
        return combined

# --- Standalone functions for Colab/Jupyter ---

service = StatisticalAnalysisService()

def show_correlation_heatmap(portfolio_name='combined'):
    """Generates a correlation matrix/heatmap."""
    if portfolio_name == 'combined':
        df = service.get_combined_df()
    else:
        df = service.load_portfolio(portfolio_name)
    
    corr = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Matrix: {portfolio_name}")
    plt.show()
    
    display(Markdown(f"### Correlation Table: {portfolio_name}"))
    display(corr.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

def show_scatter_plot(x_data_name, x_col, y_data_name, y_col, title=None):
    """Generates a scatter plot (x,y) and a data table for a research question."""
    # Load and join data
    if x_data_name == y_data_name:
        df = service.load_portfolio(x_data_name)[[x_col, y_col]]
    else:
        df_x = service.load_portfolio(x_data_name)[[x_col]]
        df_y = service.load_portfolio(y_data_name)[[y_col]]
        df = df_x.join(df_y, how='inner')
    
    # 1. Toon de Tabel (Eerste 5 rijen + correlatie)
    display(Markdown(f"### Data Tabel: {x_col} vs {y_col}"))
    corr_val = df.corr().iloc[0, 1]
    
    # Maak een kleine samenvattingstabel
    summary_df = df.describe().T
    summary_df['Correlation'] = [corr_val, corr_val]
    display(summary_df.style.format("{:.2f}"))
    
    # 2. Toon de Grafiek
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    if not title:
        title = f"Scatter Plot: {x_col} vs {y_col} (r = {corr_val:.2f})"
    else:
        title = f"{title} (r = {corr_val:.2f})"
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def show_central_tendency(portfolio_name):
    """Shows Mean, Median, and Mode."""
    df = service.load_portfolio(portfolio_name)
    stats = service.get_stats(df)
    cols = ['Mean', 'Median', 'Mode']
    display(Markdown(f"### Central Tendency: {portfolio_name}"))
    display(stats[cols].style.format("{:.2f}"))

def show_dispersion(portfolio_name):
    """Shows Standard Deviation, Min, and Max."""
    df = service.load_portfolio(portfolio_name)
    stats = service.get_stats(df)
    cols = ['Standard Deviation', 'Min', 'Max']
    display(Markdown(f"### Dispersion and Risk: {portfolio_name}"))
    display(stats[cols].style.format("{:.2f}"))

def show_shape_statistics(portfolio_name):
    """Shows Skewness and Kurtosis."""
    df = service.load_portfolio(portfolio_name)
    stats = service.get_stats(df)
    cols = ['Skewness', 'Kurtosis']
    display(Markdown(f"### Shape of Distribution: {portfolio_name}"))
    display(stats[cols].style.format("{:.2f}"))

def show_distribution_plots(portfolio_name, ticker=None):
    """Generates Histogram and Boxplot."""
    df = service.load_portfolio(portfolio_name)
    tickers = [ticker] if ticker else df.columns
    
    for t in tickers:
        if t not in df.columns: continue
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[t], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title(f'Histogram - {t}')
        
        sns.boxplot(x=df[t], ax=axes[1], color='lightgreen')
        axes[1].set_title(f'Boxplot - {t}')
        
        plt.suptitle(f"Distribution Analysis: {t}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test call
    print("Test: Combined Correlation Heatmap")
    show_correlation_heatmap('combined')
