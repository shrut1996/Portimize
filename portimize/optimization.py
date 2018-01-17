import numpy as np
from scipy.optimize import minimize

class MarkowitzOptimize:

    def __init__(self, portfolio_prices, weights):
        self.portfolio_prices = portfolio_prices
        self.weights = weights

    def check_sum(self, weights):
        '''
        Returns 0 if sum of weights is 1.0
        '''
        return np.sum(weights) - 1

    def neg_sharpe(self, weights):
        return self.get_ret_vol_sr(weights)[2] * (-1)

    def get_ret_vol_sr(self, weights):
        """
        Takes in weights, returns array or return,volatility, sharpe ratio
        """
        log_ret = np.log(self.portfolio_prices / self.portfolio_prices.shift(1))
        weights = np.array(weights)
        ret = np.sum(log_ret.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(np.asarray(log_ret.cov()) * 252, weights)))
        sr = ret / vol
        return np.array([ret, vol, sr])

    def minimizeSharpeRatio(self):
        cons = ({'type': 'eq', 'fun': self.check_sum})
        bounds = np.asarray(((0, 1), (0, 1), (0, 1), (0, 1)))
        default = np.asarray([0.25, 0.25, 0.25, 0.25])
        opt_results = minimize(self.neg_sharpe, default, method='SLSQP', bounds=bounds, constraints=cons)
        return opt_results.x