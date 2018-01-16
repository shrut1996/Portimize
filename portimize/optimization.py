import numpy as np
from scipy.optimize import minimize

def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1

def neg_sharpe(portfolio_prices, weights):
    log_ret = np.log(portfolio_prices / portfolio_prices.shift(1))
    return  get_ret_vol_sr(log_ret, weights)[2] * -1

def get_ret_vol_sr(log_ret, weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(np.asarray(log_ret.cov()) * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

def optimize(portfolio_prices, weights):
    # By convention of minimize function it should be a function that returns zero for conditions
    cons = ({'type': 'eq', 'fun': check_sum})
    bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
    default = np.asarray([0.25,0.25,0.25,0.25])

    opt_results = minimize(neg_sharpe, default, method='SLSQP', bounds=bounds)
    return opt_results.x