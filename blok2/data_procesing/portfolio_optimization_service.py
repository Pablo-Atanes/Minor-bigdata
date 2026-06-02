import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Try to import IPython display tools for pretty printing in Colab
try:
    from IPython.display import display, Markdown
except ImportError:
    display = print
    Markdown = lambda x: x

from blok2.data_procesing.statistical_analysis_service import StatisticalAnalysisService

class PortfolioOptimizationService:
    def __init__(self, data_dir='cleaned_output', num_portfolios=20000, random_seed=42):
        self.num_portfolios = num_portfolios
        self.random_seed = random_seed
        
        # Check if we are running in the subdirectory or root
        if not os.path.exists(data_dir) and os.path.exists(os.path.join('blok2/data_procesing', data_dir)):
            self.stats_service = StatisticalAnalysisService(data_dir=os.path.join('blok2/data_procesing', data_dir))
        else:
            self.stats_service = StatisticalAnalysisService(data_dir=data_dir)
            
        # Define the personas and their selected assets (6 assets each)
        self.personas = {
            'student': {
                'name': 'Jonge Dynamische Student (High Growth)',
                'assets': ['BTC-USD', 'SOL-USD', 'MSFT', 'MC.PA', 'CL_F', 'NG_F'],
                'risk_profile': 'High Risk / High Growth',
                'description': 'Dit portfolio is ontworpen voor een jonge student met een lange horizon en hoge risicobereidheid. Het bevat cryptocurrencies (BTC, SOL), technologie (MSFT), luxe goederen (LVMH) en zeer volatiele energie-grondstoffen (ruwe olie, aardgas) om maximaal te profiteren van groeipotentieel.'
            },
            'docent': {
                'name': 'Oude Statische Docent (Capital Preservation)',
                'assets': ['BRK-B', 'JPM', 'GC_F', 'TYX', 'ZC_F', 'ZS_F'],
                'risk_profile': 'Low Risk / High Stability',
                'description': 'Dit portfolio is ontworpen voor een oudere docent die dicht bij zijn pensioen zit. De focus ligt op stabiliteit en koopkrachtbehoud. Het bevat stabiele waarde-aandelen (Berkshire Hathaway, JPMorgan), goud als veilige haven, obligatierentes (TYX) en defensieve agrarische grondstoffen (maïs, soja).'
            }
        }
        
        # Cache for simulated portfolios to avoid recalculating
        self._simulation_cache = {}

    def _get_returns_data(self, assets):
        """Laadt de data en berekent de maandelijkse returns, mean returns, covariance en risk-free rate."""
        combined_df = self.stats_service.get_combined_df()
        
        # Resample naar maandelijks (einde van de maand)
        monthly_prices = combined_df.resample('ME').last()
        
        # Bereken procentuele veranderingen (returns)
        monthly_returns = monthly_prices[assets].pct_change().dropna()
        
        # Jaarlijkse statistieken
        mean_returns = monthly_returns.mean() * 12
        cov_matrix = monthly_returns.cov() * 12
        
        # Bepaal gemiddelde risicovrije rentevoet uit de IRX (2-jaars Yield)
        rf = (monthly_prices['IRX'].mean() / 100)
        
        return monthly_returns, mean_returns, cov_matrix, rf

    def run_monte_carlo_simulation(self, persona_key):
        """
        Voert een Monte Carlo-simulatie uit door willekeurige portfolios te genereren.
        Slaat de resultaten op in een cache om dubbele berekeningen te voorkomen.
        """
        if persona_key in self._simulation_cache:
            return self._simulation_cache[persona_key]
            
        assets = self.personas[persona_key]['assets']
        _, mean_returns, cov_matrix, rf = self._get_returns_data(assets)
        num_assets = len(assets)
        
        # Arrays om resultaten in op te slaan
        results = np.zeros((3, self.num_portfolios))
        weights_record = []
        
        # Stel de random seed in voor reproduceerbaarheid
        np.random.seed(self.random_seed)
        
        for i in range(self.num_portfolios):
            # 1. Genereer willekeurige gewichten
            weights = np.random.random(num_assets)
            # 2. Normaliseer de gewichten zodat ze optellen tot exact 1.0 (100%)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            # 3. Bereken het verwachte portfoliorendement
            p_return = np.dot(weights, mean_returns)
            
            # 4. Bereken het portfoliorisico (volatiliteit/standaarddeviatie)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # 5. Bereken de Sharpe-ratio
            p_sharpe = (p_return - rf) / p_vol if p_vol > 0 else 0
            
            # Sla de resultaten op
            results[0, i] = p_return
            results[1, i] = p_vol
            results[2, i] = p_sharpe
            
        sim_data = {
            'results': results,
            'weights': weights_record,
            'mean_returns': mean_returns,
            'cov_matrix': cov_matrix,
            'rf': rf,
            'assets': assets
        }
        
        self._simulation_cache[persona_key] = sim_data
        return sim_data

    def optimize_portfolio_mc(self, persona_key):
        """
        Zoekt de optimale portfolios (Min Volatility en Max Sharpe) in de simulatiedata.
        """
        sim_data = self.run_monte_carlo_simulation(persona_key)
        results = sim_data['results']
        weights = sim_data['weights']
        assets = sim_data['assets']
        
        # Indexen vinden voor de beste opties
        max_sharpe_idx = np.argmax(results[2])
        min_vol_idx = np.argmin(results[1])
        
        # Max Sharpe Portfolio gegevens ophalen
        max_sharpe_weights = pd.Series(weights[max_sharpe_idx], index=assets)
        max_sharpe_stats = {
            'weights': max_sharpe_weights,
            'return': results[0, max_sharpe_idx],
            'volatility': results[1, max_sharpe_idx],
            'sharpe': results[2, max_sharpe_idx]
        }
        
        # Minimum Volatiliteit Portfolio gegevens ophalen
        min_vol_weights = pd.Series(weights[min_vol_idx], index=assets)
        min_vol_stats = {
            'weights': min_vol_weights,
            'return': results[0, min_vol_idx],
            'volatility': results[1, min_vol_idx],
            'sharpe': results[2, min_vol_idx]
        }
        
        return min_vol_stats, max_sharpe_stats

    def show_portfolio_optimization(self, persona_key):
        """Toont de resultaten van de Monte Carlo-optimalisatie in nette tabellen."""
        if persona_key not in self.personas:
            raise ValueError(f"Persona '{persona_key}' onbekend.")
            
        persona = self.personas[persona_key]
        
        # Haal de gesimuleerde portfolios op
        opt_min_risk, opt_max_sharpe = self.optimize_portfolio_mc(persona_key)
        
        # Bouw dataframe voor gewichten
        df_weights = pd.DataFrame({
            'Minimum Risico Gewichten': opt_min_risk['weights'],
            'Max Sharpe Gewichten': opt_max_sharpe['weights']
        })
        
        # Formatteer gewichten als percentages
        df_weights = df_weights * 100
        
        # Bouw dataframe voor de statistieken
        stats_data = {
            'Statistiek': ['Verwacht Jaarlijks Rendement', 'Jaarlijkse Volatiliteit (Risico)', 'Sharpe-ratio'],
            'Minimum Risico Portfolio': [
                f"{opt_min_risk['return']*100:.2f}%",
                f"{opt_min_risk['volatility']*100:.2f}%",
                f"{opt_min_risk['sharpe']:.3f}"
            ],
            'Max Sharpe Portfolio': [
                f"{opt_max_sharpe['return']*100:.2f}%",
                f"{opt_max_sharpe['volatility']*100:.2f}%",
                f"{opt_max_sharpe['sharpe']:.3f}"
            ]
        }
        df_stats = pd.DataFrame(stats_data).set_index('Statistiek')
        
        # Display met Markdown
        display(Markdown(f"### Monte Carlo Simulatieresultaten voor **{persona['name']}**"))
        display(Markdown(f"**Risicoprofiel:** {persona['risk_profile']}  \n*{persona['description']}*"))
        display(Markdown(f"*Simulatiegrootte: {self.num_portfolios:,} willekeurige portfolios*"))
        display(Markdown("#### 📈 Portfoliostatistieken"))
        display(df_stats)
        display(Markdown("#### ⚖️ Gevonden Gewichtsverdeling per Asset"))
        display(df_weights.style.format("{:.2f}%").background_gradient(cmap='Greens', axis=None))
        
        # Toon ook de individuele assets ter referentie
        sim_data = self.run_monte_carlo_simulation(persona_key)
        mean_returns = sim_data['mean_returns']
        cov_matrix = sim_data['cov_matrix']
        
        df_individual = pd.DataFrame({
            'Verwacht Rendement (Jaarlijks)': mean_returns * 100,
            'Volatiliteit (Jaarlijks)': np.sqrt(np.diag(cov_matrix)) * 100
        })
        display(Markdown("#### 🔍 Ter Vergelijking: Individuele Assets"))
        display(df_individual.style.format("{:.2f}%"))

    def plot_efficient_frontier(self, persona_key):
        """Visualiseert de Monte Carlo puntenwolk gekleurd op Sharpe-ratio."""
        if persona_key not in self.personas:
            raise ValueError(f"Persona '{persona_key}' onbekend.")
            
        persona = self.personas[persona_key]
        sim_data = self.run_monte_carlo_simulation(persona_key)
        results = sim_data['results']
        rf = sim_data['rf']
        assets = sim_data['assets']
        
        # Haal de twee optimale portfolios op
        opt_min_risk, opt_max_sharpe = self.optimize_portfolio_mc(persona_key)
        
        # Individuele assets statistieken
        mean_returns = sim_data['mean_returns']
        cov_matrix = sim_data['cov_matrix']
        ind_returns = mean_returns.values * 100
        ind_vols = np.sqrt(np.diag(cov_matrix)) * 100
        
        plt.figure(figsize=(12, 8))
        
        # Scatter van alle gesimuleerde portfolios, gekleurd op Sharpe-ratio
        sc = plt.scatter(results[1]*100, results[0]*100, c=results[2], cmap='viridis', marker='o', s=10, alpha=0.3)
        cb = plt.colorbar(sc)
        cb.set_label('Sharpe-ratio', fontsize=11, labelpad=10)
        
        # Plot de individuele assets
        print(type(ind_vols))
        print(type(ind_returns))
        print(ind_vols)
        print(ind_returns)
        plt.scatter(ind_vols, ind_returns, color='red', marker='o', s=100, zorder=3, edgecolors='black', label='Individuele Assets')

        for i, asset in enumerate(assets):
            plt.annotate(asset, (ind_vols[i], ind_returns[i]), fontsize=10, fontweight='bold', 
                         xytext=(6, 6), textcoords='offset points', bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.6, ec="gray"))
            
        # Plot Minimum Risico Portfolio
        plt.scatter(opt_min_risk['volatility'] * 100, opt_min_risk['return'] * 100, color='darkorange', marker='*', s=300, zorder=5, edgecolors='black', label='Simulated Minimum Risico Portfolio')
        
        # Plot Max Sharpe Portfolio
        plt.scatter(opt_max_sharpe['volatility'] * 100, opt_max_sharpe['return'] * 100, color='magenta', marker='D', s=150, zorder=5, edgecolors='black', label='Simulated Max Sharpe Portfolio')
        
        # Opmaken van de grafiek
        plt.title(f"Monte Carlo Portfolio Simulatie & Efficient Frontier: {persona['name']}", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Volatiliteit / Risico (Standaarddeviatie per jaar in %)", fontsize=11, labelpad=10)
        plt.ylabel("Verwacht Jaarlijks Rendement (%)", fontsize=11, labelpad=10)
        plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
        plt.axvline(rf * 100, color='blue', linewidth=1, linestyle='--', label=f'Risicovrije Rente ({rf*100:.2f}%)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', fontsize=10, frameon=True)
        plt.tight_layout()
        plt.show()

# Standalone helper om de code te testen
if __name__ == "__main__":
    service = PortfolioOptimizationService(num_portfolios=20000)
    print("Testing Student Monte Carlo Simulation...")
    service.show_portfolio_optimization('student')
    print("Testing Docent Monte Carlo Simulation...")
    service.show_portfolio_optimization('docent')
