import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from heston_mc import HestonMonteCarloPricer


def black_scholes_delta(S, K, T, r, sigma):
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Avoid division by zero for T=0
    T = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)

    # For expired options: delta = 1 if S>K else 0
    expired_mask = T < 1e-6
    delta = np.where(expired_mask, np.where(S > K, 1.0, 0.0), delta)
    return delta


class DeltaTrainer:
    def __init__(self, model = None):
        if model is None:
            model = MLPRegressor(hidden_layer_sizes=(64,64),
                                 activation='tanh',
                                 max_iter=1000,
                                 random_state=42)
            
        self.model = model

    def generate_training_data(self, n_samples = 10000, seed = 42):
        np.random.seed(seed)
        S = np.random.uniform(50, 150, n_samples)
        K = np.random.uniform(50, 150, n_samples)
        T = np.random.uniform(0.05, 2.0, n_samples)
        r = np.random.uniform(0.0, 0.05, n_samples)
        sigma = np.random.uniform(0.1, 0.5, n_samples)
        X = np.column_stack([S, K, T, r, sigma])
        y = black_scholes_delta(S, K, T, r, sigma)
        return X, y
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return rmse, y_pred

    def predict(self, X):
        return self.model.predict(X)
    
class HedgingSimulator:
    def __init__(self, pricer: HestonMonteCarloPricer, r=0.01):
        self.pricer = pricer
        self.r = r

    def simulate(self, delta_func, S0=100, K=100, T=1.0, sigma=0.2, dt=1/252):
        S_paths, v_paths = self.pricer.simulate_paths(T=T, N=1, dt=dt)
        S = S_paths[0]
        n_steps = len(S)
        cash = 0.0
        delta_prev = delta_func(S[0], K, T)
        cash -= delta_prev * S[0]

        for t in range(1, n_steps):
            tau = max(T - t * dt, 1e-6)
            delta_now = delta_func(S[t], K, tau)
            d_delta = delta_now - delta_prev
            cash -= d_delta * S[t]
            cash *= np.exp(self.r * dt)
            delta_prev = delta_now

        payoff = np.maximum(S[-1] - K, 0)
        pnl = cash + delta_prev * S[-1] - payoff
        return pnl

def heston_finite_diff_delta(pricer: HestonMonteCarloPricer,
                             S, K, T, r, kappa, theta, sigma_v, rho, v0,
                             bump=1e-2):
    """
    Compute Heston model delta via central finite difference.
    Delta ≈ [V(S + eps) - V(S - eps)] / (2 * eps)
    """
    S_plus = S + bump
    S_minus = np.maximum(S - bump, 1e-8)

    price_up = pricer.price(S_plus, K, T, r, kappa, theta, sigma_v, rho, v0)
    price_dn = pricer.price(S_minus, K, T, r, kappa, theta, sigma_v, rho, v0)
    return (price_up - price_dn) / (2 * bump)
