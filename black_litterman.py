"""
Black-Litterman Module
"""
import numpy as np
from scipy.optimize import minimize

def calculate_equilibrium_returns_pi(market_caps, cov_matrix, delta):
    weights = market_caps / market_caps.sum()
    pi = delta * (cov_matrix @ weights)
    return pi

def estimate_risk_aversion_delta(market_returns):
    mean_return = np.mean(market_returns)
    var_return = np.var(market_returns)
    if var_return < 1e-8: return 2.5
    delta = mean_return / var_return
    return np.clip(delta, 1.0, 10.0) # Limit range for stability

def compute_posterior_returns_mu(pi, P, Q, omega, cov_matrix, tau, reg=1e-5):
    tau_sigma = tau * cov_matrix
    tau_sigma_reg = tau_sigma + reg * np.eye(len(tau_sigma))
    omega_reg = omega + reg * np.eye(len(omega))
    try:
        tau_sigma_inv = np.linalg.inv(tau_sigma_reg)
        omega_inv = np.linalg.inv(omega_reg)
        post_prec = tau_sigma_inv + P.T @ omega_inv @ P
        post_inv = np.linalg.inv(post_prec + reg * np.eye(len(post_prec)))
        post_mean = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
        return post_inv @ post_mean
    except np.linalg.LinAlgError:
        return pi

def optimize_weights(expected_returns, cov_matrix, delta=2.5, allow_short=False):
    """
    Maximize Utility: U = w'mu - (delta/2) * w'Cov'w
    """
    n_assets = len(expected_returns)
    init_guess = np.ones(n_assets) / n_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Force Diversification: Max 30% per stock
    bounds = tuple((0.0, 0.30) for _ in range(n_assets)) if not allow_short else None
    
    def neg_utility(w):
        ret = np.dot(w, expected_returns)
        var = np.dot(w.T, np.dot(cov_matrix, w))
        util = ret - (delta / 2.0) * var
        return -util # Minimize negative utility

    try:
        res = minimize(neg_utility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
        return res.x if res.success else init_guess
    except:
        return init_guess

def optimize_mvo_portfolio(mean_returns, cov_matrix, delta=2.5):
    # MVO uses the same Utility Logic for fair comparison
    return optimize_weights(mean_returns, cov_matrix, delta=delta)

def compute_portfolio_returns(weights, returns_df):
    if len(weights) != len(returns_df.columns): return np.zeros(len(returns_df))
    return returns_df.values @ weights