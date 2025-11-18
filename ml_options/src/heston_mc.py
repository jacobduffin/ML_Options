import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import time
from scipy.stats import norm
from scipy.optimize import brentq


# -------------------------------------------------
# 1. Heston Monte Carlo Pricer
# -------------------------------------------------
class HestonMonteCarloPricer:
    def __init__(self, n_paths=5000, n_steps=100, seed=42):
        self.n_paths = n_paths
        self.n_steps = n_steps
        np.random.seed(seed)

    def price(self, S0, K, T, r, kappa, theta, sigma_v, rho, v0):
        if T <= 0:
            return max(S0 - K, 0)
        dt = T / self.n_steps
        S = np.full(self.n_paths, S0)
        v = np.full(self.n_paths, v0)
        sqrt_dt = np.sqrt(dt)

        for _ in range(self.n_steps):
            Z1 = np.random.standard_normal(self.n_paths)
            Z2 = np.random.standard_normal(self.n_paths)
            dW1 = sqrt_dt * Z1
            dW2 = sqrt_dt * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)
            v = np.maximum(v + kappa * (theta - v) * dt +
                           sigma_v * np.sqrt(np.maximum(v, 0)) * dW2, 0)
            S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * Z1)

        payoff = np.maximum(S - K, 0)
        return np.exp(-r * T) * np.mean(payoff)
    
    def simulate_paths(self, S0=100, v0=0.04, T=1.0, r=0.01,
                       kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.7,
                       N=None, dt=None):
        if N is None:
            N = self.n_paths
        if dt is None:
            dt = T / self.n_steps

        M = int(T / dt)
        S = np.zeros((N, M + 1))
        v = np.zeros((N, M + 1))
        S[:, 0] = S0
        v[:, 0] = v0

        for t in range(M):
            Z1 = np.random.standard_normal(N)
            Z2 = np.random.standard_normal(N)
            Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

            v[:, t + 1] = np.maximum(
                v[:, t] + kappa * (theta - v[:, t]) * dt +
                sigma_v * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z2,
                0
            )
            S[:, t + 1] = S[:, t] * np.exp(
                (r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t] * dt) * Z1
            )

        return S, v


# -------------------------------------------------
# 2. Dataset Generator
# -------------------------------------------------
class DataGenerator:
    def __init__(self, pricer):
        self.pricer = pricer

    def generate(self, n_samples=1000):
        np.random.seed(0)
        S = np.random.uniform(80, 200, n_samples)
        K = np.random.uniform(80, 200, n_samples)
        T = np.random.uniform(0.1, 2.0, n_samples)
        r = np.random.uniform(0.0, 0.05, n_samples)
        kappa = np.random.uniform(1.0, 4.0, n_samples)
        theta = np.random.uniform(0.02, 0.15, n_samples)
        sigma_v = np.random.uniform(0.2, 1.0, n_samples)
        rho = np.random.uniform(-0.9, -0.1, n_samples)
        v0 = np.random.uniform(0.02, 0.2, n_samples)

        prices = []
        for i in range(n_samples):
            p = self.pricer.price(S[i], K[i], T[i], r[i],
                                  kappa[i], theta[i], sigma_v[i], rho[i], v0[i])
            prices.append(p)

        df = pd.DataFrame({
            "S": S, "K": K, "T": T, "r": r,
            "kappa": kappa, "theta": theta, "sigma_v": sigma_v,
            "rho": rho, "v0": v0, "price": prices
        })
        return df


# -------------------------------------------------
# 3. Surrogate Model
# -------------------------------------------------
class SurrogateModel:
    def __init__(self):
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(64, 64),
                                 activation="tanh", max_iter=2000,
                                 random_state=1))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return rmse, r2, y_pred

    def predict(self, X):
        return self.model.predict(X)

    def benchmark(self, X_sample, pricer, params):
        """Compare MC vs surrogate timing."""
        t0 = time.time()
        for _ in range(30):
            pricer.price(**params)
        t1 = time.time()

        t2 = time.time()
        for _ in range(30):
            self.model.predict(X_sample)
        t3 = time.time()

        mc_time = (t1 - t0) / 30
        ml_time = (t3 - t2) / 30
        return mc_time, ml_time, mc_time / ml_time



def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(price, S, K, T, r):
    if T <= 0 or price < max(S - K*np.exp(-r*T), 0):
        return np.nan
    f = lambda sigma: black_scholes_call(S,K,T,r,sigma) - price
    try:
        return brentq(f, 1e-4, 5.0)
    except ValueError:
        return np.nan
