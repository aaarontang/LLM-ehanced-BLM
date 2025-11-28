"""
Backtesting Module
"""
import numpy as np
import pandas as pd

def calculate_transaction_costs(old_weights, new_weights, cost_rate=0.001):
    """
    Calculate transaction costs from weight changes
    """
    return np.sum(np.abs(new_weights - old_weights)) * cost_rate

def compute_portfolio_returns(weights, returns_df):
    """
    Calculate portfolio returns given weights and return matrix
    """
    if len(weights) != len(returns_df.columns):
        # Handle edge case where weights don't match columns (e.g., insufficient data)
        return np.zeros(len(returns_df))
    
    # Portfolio return = weights^T * returns for each period
    port_returns = returns_df.values @ weights
    
    return port_returns

def calculate_performance_metrics(portfolio_returns):
    """
    Compute comprehensive performance metrics
    """
    portfolio_returns = np.array(portfolio_returns)
    
    # Remove NaN values
    portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
    
    # Handle empty returns case
    if len(portfolio_returns) == 0:
        return {
            "CAGR": 0, "Sharpe": 0, "Volatility": 0,
            "MaxDrawdown": 0, "VaR_5%": 0, "CVaR_5%": 0, "TotalReturn": 0
        }
    
    # 1. Total Return & CAGR
    total_return = np.prod(1 + portfolio_returns) - 1
    n_periods = len(portfolio_returns)
    # Annualize assuming monthly data (12 periods)
    cagr = (1 + total_return) ** (12 / n_periods) - 1 if n_periods > 0 else 0
    
    # 2. Sharpe & Volatility
    mean_ret = np.mean(portfolio_returns)
    volatility = np.std(portfolio_returns)
    sharpe = (mean_ret / (volatility + 1e-8)) * np.sqrt(12)
    
    # 3. Drawdown
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    mdd = np.min(drawdown)
    
    # 4. VaR & CVaR
    var_5 = np.percentile(portfolio_returns, 5)
    
    # Calculate Expected Shortfall (CVaR)
    below_var = portfolio_returns[portfolio_returns <= var_5]
    cvar_5 = np.mean(below_var) if len(below_var) > 0 else var_5
    
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Volatility": volatility * np.sqrt(12),
        "MaxDrawdown": mdd,
        "VaR_5%": var_5,
        "CVaR_5%": cvar_5,  # <--- This key caused the error
        "TotalReturn": total_return
    }