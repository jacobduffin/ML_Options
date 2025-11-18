import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S /K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

class BlackScholesData:

    def __init__(self, n_samples = 10000, seed = 42):
        np.random.seed(seed)
        S = np.random.uniform(50,150,n_samples)
        K = np.random.uniform(50,150,n_samples)
        T = np.random.uniform(0.1,2.0,n_samples)
        r = np.random.uniform(0.0,0.05,n_samples)
        sigma = np.random.uniform(0.1,0.5,n_samples)
        self.X = np.column_stack([S,K,T,r,sigma])
        self.y = black_scholes_call(S,K,T,r,sigma)

    def split(self, test_size=0.2):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=42)


class PricingModel:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.loss_curve = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        if hasattr(self.model, "loss_curve_"):
            self.loss_curve_ = self.model.loss_curve_
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        return rmse, r2, y_pred