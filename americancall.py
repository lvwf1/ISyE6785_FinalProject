import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

class AmericanCall(object):
    """Compute European option value, greeks, and implied volatility.

    Parameters
    ==========
    S0 : int or float
        initial asset value
    K : int or float
        strike
    T : int or float
        time to expiration as a fraction of one year
    M : int
        grid or granularity for time (in number of total points)
    r : int or float
        continuously compounded risk free rate, annualized
    i : int
        number of simulations
    sigma : int or float
        continuously compounded standard deviation of returns
    delta : int or float
    rho: int or float
    kind : str, {'call', 'put'}, default 'call'
        type of option

    Resources
    =========
    http://www.thomasho.com/mainpages/?download=&act=model&file=256
    """

    def __init__(self, S0, K, T, M, r, delta, sigma, i, kind='call'):
        if kind.istitle():
            kind = kind.lower()
        if kind not in ['call', 'put']:
            raise ValueError('Option type must be \'call\' or \'put\'')

        self.S0 = S0
        self.K = K
        self.T = T
        self.M = int(M)
        self.r = r
        self.delta = delta
        self.sigma = sigma
        self.i = int(i)
        self.kind = kind
        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

        self.d1 = ((np.log(self.S0 / self.K)
                + (self.r + 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))
        self.d2 = ((np.log(self.S0 / self.K)
                + (self.r - 0.5 * self.sigma ** 2) * self.T)
                / (self.sigma * np.sqrt(self.T)))

        # Several greeks use negated terms dependent on option type
        # For example, delta of call is N(d1) and delta put is N(d1) - 1
        self.sub = {'call' : [0, 1, -1], 'put' : [-1, -1, 1]}

    def value(self):
        """Compute option value."""
        return (self.sub[self.kind][1] * self.S0
               * scipy.stats.norm.cdf(self.sub[self.kind][1] * self.d1)
               + self.sub[self.kind][2] * self.K * np.exp(-self.r * self.T)
               * scipy.stats.norm.cdf(self.sub[self.kind][1] * self.d2))

    def AmericanPutPrice(self, seed):
        """ Returns Monte Carlo price matrix rows: time columns: price-path simulation """
        np.random.seed(seed)
        path = np.zeros((self.M + 1, self.i), dtype=np.float64)
        path[0] = self.S0
        for t in range(1, self.M + 1):
            rand = np.random.standard_normal(int(self.i / 2))
            rand = np.concatenate((rand, -rand))
            path[t] = (path[t - 1] * np.exp((self.r - self.delta) * self.time_unit + self.sigma * np.sqrt(self.time_unit) * rand))

        """ Returns the inner-value of American Option """
        if self.kind == 'call':
            payoff = np.maximum(path - self.K, np.zeros((self.M + 1, self.i), dtype=np.float64))
        else:
            payoff = np.maximum(self.K - path, np.zeros((self.M + 1, self.i), dtype=np.float64))

        value = np.zeros_like(payoff)
        value[-1] = payoff[-1]
        for t in range(self.M - 1, 0, -1):
            regression = np.polyfit(path[t], value[t + 1] * self.discount, 5)
            continuation_value = np.polyval(regression, path[t])
            value[t] = np.where(payoff[t] > continuation_value, payoff[t], value[t + 1] * self.discount)

        return np.sum(value[1] * self.discount) / float(self.i)

# This is a function simulating the price path for a Geometric Brownian Motion price model
def gen_paths(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, I):
    dt = float(T) / M
    path_1 = np.zeros((M + 1, I), np.float64)
    path_2 = np.zeros((M + 1, I), np.float64)
    path_1[0] = S0_1
    path_2[0] = S0_2
    for t in range(1, M + 1):
        rand_1 = np.random.standard_normal(I)
        rand_2 = np.random.standard_normal(I)
        path_1[t] = path_1[t - 1] * np.exp((r - delta_1) * dt +
                                           sigma_1 * np.sqrt(dt) * rand_1)
        path_2[t] = path_2[t - 1] * np.exp((r - delta_2) * dt +
                                           rho * sigma_2 * np.sqrt(dt) * rand_1 +
                                           np.sqrt(1-rho**2) * sigma_2 * np.sqrt(dt) * rand_2)
    return [path_1,path_2]

def gen_paths_antithetic(S0_1, S0_2, r, delta_1, delta_2, sigma_1, sigma_2, rho, T, M, I):
    dt = float(T) / M
    path_1 = np.zeros((2*M + 1, I), np.float64)
    path_2 = np.zeros((2*M + 1, I), np.float64)
    path_1[0] = S0_1
    path_2[0] = S0_2
    for t in range(1, M + 1):
        rand_1 = np.random.standard_normal(I)
        rand_2 = np.random.standard_normal(I)
        rand_anti_1 = -1.0 * rand_1  # antithetic variates
        rand_anti_2 = -1.0 * rand_2  # antithetic variates
        path_1[2 * t] = path_1[2 * (t - 1)] * np.exp((r - delta_1) * dt +
                                                   sigma_1 * np.sqrt(dt) * rand_1)
        path_1[2 * t - 1] = path_1[max(2 * t - 3, 0)] * np.exp((r - delta_1) * dt +
                                                             sigma_1 * np.sqrt(dt) * rand_anti_1)
        path_2[2*t] = path_2[2*(t - 1)] * np.exp((r - delta_2) * dt +
                                               rho * sigma_2 * np.sqrt(dt) * rand_1 +
                                               np.sqrt(1 - rho ** 2) * sigma_2 * np.sqrt(dt) * rand_2)
        path_2[2*t-1] = path_2[max(2*t - 3, 0)] * np.exp((r - delta_2) * dt +
                                                       rho * sigma_2 * np.sqrt(dt) * rand_anti_1 +
                                                       np.sqrt(1 - rho ** 2) * sigma_2 * np.sqrt(dt) * rand_anti_2)
    return [path_1,path_2]

def hist_comp(dist1, dist2, lgnd, bin_num):
    hist_start = min(min(dist1), min(dist2))
    hist_end = max(max(dist1), max(dist2))
    bin_vec = np.linspace(hist_start, hist_end, bin_num)
    plt.hist([dist1, dist2], color=['r','g'], label=[lgnd[0],lgnd[1]], alpha=0.8, bins=bin_vec)
    plt.legend(loc='upper right')
    plt.show()

def computeAmericanMC(S0, K, T, M, r, delta, sigma, i,seed, type):
    return AmericanCall(S0, K, T, M, r, delta, sigma, i, type).AmericanPutPrice(seed)

def computeAmericanBS(S0, K, T, M, r, delta, sigma, i, type):
    return AmericanCall(S0, K, T, M, r, delta, sigma, i, type)