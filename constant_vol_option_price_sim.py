import math
import numpy as np


def constant_vol_option_price_sim(S0, K, T, r, volatility, M, I):
    # np.random.seed(100)
    dt = T / M
    S = S0 * np.exp(np.cumsum((r - 0.5 * volatility ** 2) * dt
                              + volatility * math.sqrt(dt)
                              * np.random.standard_normal((M + 1, I)), axis=0))
    S[0] = S0
    C0 = math.exp(-r * T) * sum(np.maximum(S[-1] - K, 0)) / I
    return C0

