"""
Optimizer Tuning Module
"""
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from data_handler import prepare_lookback_data, get_current_market_caps, estimate_historical_caps
from black_litterman import estimate_risk_aversion_delta, calculate_equilibrium_returns_pi, compute_posterior_returns_mu, optimize_weights, compute_portfolio_returns
from llm_auto_query import calculate_predicted_return

def run_bayesian_optimization(prices, returns, tickers, val_dates, llm_cache, lookback_months=3):
    print(f"\n[BO] Starting Optimization...")
    
    val_data = {}
    cur_caps = get_current_market_caps(tickers); last_price = prices.iloc[-1]
    
    for date in val_dates:
        lb_p = prepare_lookback_data(prices, date, lookback_months)
        lb_r = lb_p.pct_change().dropna()
        if lb_r.empty: continue
        
        cov = lb_r.cov().values
        p_rebal = prices.loc[prices.index < date].iloc[-1]
        h_caps = estimate_historical_caps(tickers, cur_caps, p_rebal, last_price)
        delta = estimate_risk_aversion_delta(returns.mean(axis=1).values)
        pi = calculate_equilibrium_returns_pi(pd.Series(h_caps), cov, delta)
        fut_r = returns.loc[(returns.index>=date) & (returns.index < date+pd.DateOffset(months=3))]
        
        val_data[date] = {'cov':cov, 'pi':pi, 'future':fut_r, 'mu_hist':lb_r.mean().values/21.0, 'delta':delta}

    def objective(alpha, beta, tau):
        sharpes = []
        for date, data in val_data.items():
            if date not in llm_cache: continue
            meta = llm_cache[date]
            raw_s, raw_c = meta.get('raw_sentiments',{}), meta.get('raw_confidences',{})
            
            Q, omg = np.zeros(len(tickers)), np.zeros(len(tickers))
            for i, t in enumerate(tickers):
                if t in raw_s and t in raw_c:
                    s_vals, c_vals = raw_s[t], raw_c[t]
                    preds = []
                    for k in range(len(s_vals)):
                        c = c_vals[k] if k < len(c_vals) else 0.5
                        preds.append(calculate_predicted_return(data['mu_hist'][i], s_vals[k], c, alpha, beta))
                    Q[i] = np.mean(preds)*21.0
                    omg[i] = (np.var(preds) if len(preds)>1 else 1e-5)*441.0
                else: omg[i] = 1e-3
            
            try:
                mu = compute_posterior_returns_mu(data['pi'], np.eye(len(tickers)), Q, np.diag(omg), data['cov'], tau)
                w = optimize_weights(mu, data['cov'], delta=data['delta'])
                pr = compute_portfolio_returns(w, data['future'])
                sharpes.append(np.mean(pr)/np.std(pr) if np.std(pr)>1e-6 else -1.0)
            except: pass
        return np.mean(sharpes) if sharpes else -99

    optimizer = BayesianOptimization(f=objective, pbounds={'alpha':(0.0,1.0), 'beta':(0.0001,0.01), 'tau':(0.01,0.2)}, verbose=2, random_state=42)
    optimizer.maximize(init_points=5, n_iter=15)
    best = optimizer.max['params']
    return best['alpha'], best['beta'], best['tau']