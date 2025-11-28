"""
Black-Litterman Portfolio Optimization
Features:
- Full Automation (Data -> LLM -> Optimization -> Backtest)
- Raw Sentiment Visibility in CSV
- Visualization
"""
import pandas as pd
import numpy as np
import os
import shutil
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- MODULE IMPORTS ---
from data_handler import (download_hsi_constituents, download_monthly_prices, 
                          calculate_monthly_returns, get_current_market_caps, 
                          estimate_historical_caps, prepare_lookback_data)
from llm_auto_query import generate_views_for_period
from black_litterman import (calculate_equilibrium_returns_pi, estimate_risk_aversion_delta, 
                             compute_posterior_returns_mu, optimize_weights, optimize_mvo_portfolio,
                             compute_portfolio_returns)
from backtest import calculate_performance_metrics
from optimizer_tuning import run_bayesian_optimization
from visualization import (plot_cumulative_wealth, plot_weights_stacked_bar, 
                           plot_view_distribution, create_comparison_table)

# --- USER CONFIGURATION ---
CONFIG = {
    "API_KEY": "YOUR API KEY",  # <--- Perplexity API Key
    "MODEL": "sonar-pro",
    "N_QUERIES": 5,
    "VAL_START": "2024-04-01",
    "TEST_END": "2025-07-01",
    "LOOKBACK": 3,
    "FREQ": "3MS"
}

def main():
    print("=" * 60)
    print("      AI PORTFOLIO OPTIMIZER (FINAL VERSION)")
    print("=" * 60)

    # 1. CLEANUP
    if os.path.exists('data'): shutil.rmtree('data')
    if os.path.exists('results'): shutil.rmtree('results')
    os.makedirs('data', exist_ok=True)
    os.makedirs('results/views', exist_ok=True)

    # 2. DATA COLLECTION
    print("\n[STEP 1] Data Collection...")
    tickers = download_hsi_constituents()
    prices = download_monthly_prices(tickers, "2023-01-01", CONFIG["TEST_END"])
    prices.to_csv('data/monthly_prices.csv')
    
    returns = calculate_monthly_returns(prices)
    returns.to_csv('data/monthly_returns.csv')
    
    current_caps = get_current_market_caps(tickers)
    latest_prices = prices.iloc[-1]
    print("  ✓ Data saved.")

    # 3. SCHEDULE
    val_dates = pd.date_range(CONFIG["VAL_START"], "2024-10-01", freq=CONFIG["FREQ"])
    test_dates = pd.date_range("2024-10-01", CONFIG["TEST_END"], freq=CONFIG["FREQ"])

    # 4. MINING & OPTIMIZATION
    print("\n[STEP 2] Mining Views & Running Optimization...")
    llm_views_cache = {}
    
    for val_date in tqdm(val_dates, desc="Mining Validation Views"):
        try:
            # 1. Get Preview Data (using default params)
            Q_preview, _, meta = generate_views_for_period(
                prices, returns, tickers, val_date,
                lookback_months=CONFIG["LOOKBACK"], api_key=CONFIG["API_KEY"],
                n_queries=CONFIG["N_QUERIES"], model=CONFIG["MODEL"],
                alpha=0.5, beta=0.0015 
            )
            
            # 2. Cache Raw Data for Optimizer
            if 'raw_sentiments' in meta:
                llm_views_cache[val_date] = meta
            
            # 3. EXTRACT RAW SCORES FOR CSV 
            raw_s_list = []
            raw_c_list = []
            
            for t in tickers:
                s_vals = meta.get('raw_sentiments', {}).get(t, [])
                c_vals = meta.get('raw_confidences', {}).get(t, [])
                
                avg_s = np.mean(s_vals) if s_vals else 0.5
                avg_c = np.mean(c_vals) if c_vals else 0.5
                
                raw_s_list.append(avg_s)
                raw_c_list.append(avg_c)

            # 4. Save Detailed CSV
            pd.DataFrame({
                'Ticker': tickers,
                'Sentiment_S': raw_s_list,       
                'Confidence_C': raw_c_list,     
                'Q_Preview': Q_preview          
            }).to_csv(f'results/views/view_{val_date.strftime("%Y%m%d")}.csv')
            
        except Exception: pass

    # Run Optimization
    best_alpha, best_beta, best_tau = run_bayesian_optimization(
        prices, returns, tickers, val_dates, llm_views_cache, CONFIG["LOOKBACK"]
    )
    print(f"  > Optimized: Alpha={best_alpha:.3f}, Beta={best_beta:.4f}, Tau={best_tau:.3f}")

    # 5. BACKTEST
    print("\n[STEP 3] Running Backtest...")
    
    weights_store = {'LLM-BL': {}, 'Market': {}, 'MVO': {}, 'EW': {}}
    returns_store = {'LLM-BL': [], 'Market': [], 'MVO': [], 'EW': []}
    
    pbar = tqdm(test_dates, desc="Simulating")
    for i, rebal_date in enumerate(pbar):
        try:
            # A. Data Prep
            lookback = prepare_lookback_data(prices, rebal_date, CONFIG["LOOKBACK"])
            lb_ret = lookback.pct_change().dropna()
            if lb_ret.empty: continue
            
            cov = lb_ret.cov().values
            mu_hist = lb_ret.mean().values
            
            p_rebal = prices.loc[prices.index < rebal_date].iloc[-1]
            h_caps = estimate_historical_caps(tickers, current_caps, p_rebal, latest_prices)
            
            # B. Generate LLM View (Using Optimized Params)
            Q, Omega, meta = generate_views_for_period(
                prices, returns, tickers, rebal_date,
                lookback_months=CONFIG["LOOKBACK"], api_key=CONFIG["API_KEY"],
                n_queries=CONFIG["N_QUERIES"], model=CONFIG["MODEL"],
                alpha=best_alpha, beta=best_beta
            )
            
            # C. Calculate Equilibrium
            delta = estimate_risk_aversion_delta(returns.mean(axis=1).values)
            pi = calculate_equilibrium_returns_pi(pd.Series(h_caps), cov, delta)
            
            # --- STRATEGIES ---
            
            # 1. LLM-BL (Pass Delta & Optimized Tau)
            mu_bl = compute_posterior_returns_mu(pi, np.eye(len(tickers)), Q, Omega, cov, tau=best_tau)
            w_bl = optimize_weights(mu_bl, cov, delta=delta)
            weights_store['LLM-BL'][rebal_date] = w_bl

            # 2. Market (Cap Weighted)
            w_mkt = h_caps / np.sum(h_caps)
            weights_store['Market'][rebal_date] = w_mkt

            # 3. MVO
            w_mvo = optimize_mvo_portfolio(mu_hist, cov, delta=delta)
            weights_store['MVO'][rebal_date] = w_mvo
            
            # 4. EW
            w_ew = np.ones(len(tickers)) / len(tickers)
            weights_store['EW'][rebal_date] = w_ew
            
            # --- RETURNS ---
            next_idx = i + 1
            end_idx = test_dates[next_idx] if next_idx < len(test_dates) else None
            holding = returns.loc[(returns.index >= rebal_date) & (returns.index < end_idx)] if end_idx else returns.loc[returns.index >= rebal_date]
            
            if not holding.empty:
                returns_store['LLM-BL'].extend(compute_portfolio_returns(w_bl, holding))
                returns_store['Market'].extend(compute_portfolio_returns(w_mkt, holding))
                returns_store['MVO'].extend(compute_portfolio_returns(w_mvo, holding))
                returns_store['EW'].extend(compute_portfolio_returns(w_ew, holding))
            
            # Save Backtest View (with Raw S/C for consistency)
            # We extract raw S/C again for the backtest log
            raw_s_list_test = []
            for t in tickers:
                s_vals = meta.get('raw_sentiments', {}).get(t, [])
                raw_s_list_test.append(np.mean(s_vals) if s_vals else 0.5)

            pd.DataFrame({
                'Ticker': tickers, 
                'Sentiment_S': raw_s_list_test,
                'Q_Optimized': Q
            }).to_csv(f'results/views/view_{rebal_date.strftime("%Y%m%d")}.csv')

        except Exception as e:
            pbar.write(f"Error {rebal_date}: {e}")

    # 6. VISUALIZATION
    print("\n[STEP 4] Generating Report...")
    metrics = {k: calculate_performance_metrics(v) for k, v in returns_store.items()}
    
    plot_dates = returns.index[-len(returns_store['LLM-BL']):]
    
    # Plots
    plot_cumulative_wealth(returns_store, plot_dates)
    plot_weights_stacked_bar(weights_store['LLM-BL'], tickers, strategy_name="LLM-BL Weights")
    plot_view_distribution(view_folder='results/views')
    create_comparison_table(metrics)
    
    # Save Data
    pd.DataFrame(weights_store['LLM-BL']).T.to_csv('results/weights_bl.csv')
    
    print("-" * 40)
    print(f"Strategy | Sharpe | CAGR")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k:<8} | {v['Sharpe']:.2f}   | {v['CAGR']*100:.1f}%")
    print("-" * 40)

if __name__ == "__main__":
    main()