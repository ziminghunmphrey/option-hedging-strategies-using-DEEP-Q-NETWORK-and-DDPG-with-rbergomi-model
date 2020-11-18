import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def g(x, alpha):
    return x ** alpha


def b(i, alpha):
    return ((i ** (alpha + 1) - (i - 1) ** (alpha + 1)) / (alpha + 1)) ** (1 / alpha)


def cov(alpha, num_time):
    cov = np.zeros((2, 2))
    cov[0, 0] = 1. / num_time
    cov[0, 1] = 1. / ((1. * alpha + 1) * num_time ** (1. * alpha + 1))
    cov[1, 1] = 1. / ((2. * alpha + 1) * num_time ** (2. * alpha + 1))
    cov[1, 0] = cov[0, 1]
    return cov


def asset_price_sim(num_paths, num_time, rho, T, alpha, eta, xi, S0):
    # np.random.seed(100)
    dt = T / num_time
    time_list = np.linspace(0, T, num_time + 1).T
    # generate required Brownian increments
    dw_cov = cov(alpha, num_time)
    dW1 = np.random.multivariate_normal([0, 0], dw_cov, (num_paths, num_time))
    dW2 = np.random.randn(num_paths, num_time) * np.sqrt(dt)
    # correlate the orthogonal increments using rho
    dB = rho * dW1[:, :, 0] + np.sqrt(1 - rho ** 2) * dW2
    # construct the volterra process
    Y1 = np.zeros((num_paths, num_time + 1))
    Y2 = np.zeros((num_paths, num_time + 1))
    for i in range(1, num_time + 1):
        Y1[:, i] = dW1[:, i - 1, 1]
    # construct arrays for convolution G
    G = np.zeros(num_time + 1)
    for i in range(2, num_time + 1):
        G[i] = g(b(i, alpha) / num_time, alpha)
    # the result of convolution GX
    GX = np.zeros((num_paths, len(dW1[0, :, 0]) + len(G) - 1))
    for i in range(num_paths):
        GX[i, :] = np.convolve(G, dW1[i, :, 0])
    Y2 = GX[:, :num_time + 1]
    # construct and return full process
    Y = np.sqrt(2 * alpha + 1) * (Y1 + Y2)
    # construct the variance process
    V = xi * np.exp(eta * Y - 0.5 * eta ** 2 * time_list ** (2 * alpha + 1))
    # construct the asset price process
    increments = np.sqrt(V[:, :-1]) * dB - 0.5 * V[:, :-1] * dt
    integral = np.cumsum(increments, axis=1)
    S = np.zeros_like(V)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(integral)
    # C = np.exp(-r * T) * np.mean(np.maximum((S - K), 0))
    return V.T, S.T


def call_option_price_bs_sim(S, V, r, T, K, num_time, num_paths):
    dt = T / num_time
    C = np.zeros_like(S)
    delta = np.zeros_like(C)
    sigma = np.sqrt(V)
    for j in range(num_paths):
        for i in range(num_time):
            t = i * dt
            d1 = ((np.log(S[i, j] / K)) + (r + 0.5 * sigma[i, j] ** 2) * (T - t)) / (sigma[i, j] * np.sqrt(T - t))
            d2 = d1 - sigma[i, j] * np.sqrt(T - t)
            C[i, j] = S[i, j] * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
            delta[i, j] = norm.cdf(d1)
    return C, delta


def generate_sim(num_paths, num_time, rho, T, alpha, eta, xi, S0, K, r):
    dt = T / num_time
    T_list = np.arange(0, T + dt, dt)
    V, S = asset_price_sim(num_paths, num_time, rho, T, alpha, eta, xi, S0)
    C, delta = call_option_price_bs_sim(S, V, r, T, K, num_time, num_paths)
    return T_list, V, S, C, delta


def call_option_price_sim(S, num_time, T, r):
    C = np.zeros(num_time)
    dt = T / num_time
    for i in range(1, num_time + 1):
        S_T = S[i, :]
        T_temp = T - i * dt
        C[i - 1] = np.exp(-r * T_temp) * np.mean(np.maximum(S_T - K, 0))
    return C


# we use the pathwise method for simulation of delta
def delta_sim(S0, S, r, T, num_time, K):
    dt = T / num_time
    # T_list = np.arange(0, T + dt, dt)
    S = np.mean(S, axis=1)
    A = np.where(S > K, S, 0)
    delta = np.zeros_like(A)
    # delta = np.cumsum(np.exp(-r * T_list) * A / S0)
    for i in range(1, len(delta)):
        T_temp = T - i * dt
        delta[i] = np.exp(-r * T_temp) * np.mean(S[i:]) / S[i]
    return delta


def record_sim(S, C, delta, T, num_time):
    dt = T / num_time
    T_list = np.arange(dt, T + dt, dt)
    S = np.mean(S, axis=1)
    pd_data = pd.DataFrame(np.vstack((T_list[:-1], S[1:-1], C[:-1], delta[1:-1])),
                           index=['time', 'asset_price', 'option_price', 'delta'])
    pd_data.to_csv('./rough_vol/data/sim_data.csv')


def sim_figure(V, S, C):
    plt.figure(1)
    plt.plot(V)
    plt.ylabel('Stochastic variance V')
    plt.xlabel('time step')
    plt.title('Simulation of variance V')
    plt.savefig('./rough_vol/figures/variance.png')
    plt.figure(2)
    plt.plot(S)
    plt.ylabel('Simulated asset paths')
    plt.xlabel('time step')
    plt.title('Simulation of asset paths')
    plt.savefig('./rough_vol/figures/asset_price.png')
    plt.figure(3)
    plt.plot(C)
    plt.ylabel('Simulated option price')
    plt.xlabel('time step')
    plt.title('Simulation of option price')
    plt.savefig('./rough_vol/figures/option_price.png')
    plt.show()


'''
num_paths = 100
num_time = 300
rho = -0.02
T = 1
alpha = -0.43
eta = 1.0
xi = 1.0
S0 = 100
K = 80
r = 0.05
T_list, V, S, C, delta = generate_sim(num_paths, num_time, rho, T, alpha, eta, xi, S0, K, r)
'''
# V, S = asset_price_sim(num_paths, num_time, rho, T, alpha, eta, xi, S0)
# C = call_option_price_sim(S, num_time, T, r)
# delta = delta_sim(S0, S, r, T, num_time, K)
# record_sim(S, C, delta, T, num_time)
# sim_figure(V, S, C)
