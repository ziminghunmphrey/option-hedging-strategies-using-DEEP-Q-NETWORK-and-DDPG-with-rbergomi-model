import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def asset_price_sim(num_paths, num_time, rho, S_0, V_0, T, kappa, theta, sigma, r):
    dt = T / num_time
    S = np.zeros((num_time + 1, num_paths))  # simulation matrix of asset price
    S[0, :] = S_0  # set first line as the initial asset price
    V = np.zeros((num_time + 1, num_paths))  # simulation matrix of variance(square of volatility)
    V[0, :] = V_0
    # np.random.seed(100)
    for i in range(num_paths):  # num_path is the column of the matrices
        for j in range(1, num_time + 1):  # num_time is the row of the matrices
            # two stochastic drivers for variance V and asset price S
            Zv = np.random.randn(1)
            Zs = rho * Zv + math.sqrt(1 - rho ** 2) * np.random.randn(1)
            # we use the Euler scheme to simulate the variance
            V[j, i] = V[j - 1, i] + kappa * (theta - V[j - 1, i]) * dt \
                      + sigma * math.sqrt(V[j - 1, i]) * math.sqrt(dt) * Zv
            if V[j, i] < 0:
                V[j, i] = abs(V[j, i])
            # we then simulate the asset price
            S[j, i] = S[j - 1, i] * np.exp((r - V[j - 1, i] / 2) * dt
                                           + math.sqrt(V[j - 1, i]) * math.sqrt(dt) * Zs)
    return V, S


def call_option_price(S, K_list, num_time, T, r):
    C = np.zeros((len(K_list), num_time))
    dt = T / num_time
    for i in range(1, num_time + 1):  # different expiry date
        S_T = S[i, :]
        T_temp = T - i * dt
        for j in range(len(K_list)):  # different strike price
            K = K_list[j]
            C[j, i - 1] = np.exp(-r * T_temp) * np.mean(np.maximum(S_T - K, 0))
    return C


def sim_figure(V, S, C, K_list, T, num_time):
    dt = T / num_time
    T_list = np.arange(dt, T + dt, dt).tolist()
    plt.figure(1)
    plt.plot(V)
    plt.ylabel('Stochastic variance V')
    plt.xlabel('time step')
    plt.title('Simulation of variance V')
    plt.savefig('./heston/figures/variance.png')
    plt.figure(2)
    plt.plot(S)
    plt.ylabel('Simulated asset paths')
    plt.xlabel('time step')
    plt.title('Simulation of asset paths')
    plt.savefig('./heston/figures/asset_price.png')
    fig = plt.figure(3)
    ax = Axes3D(fig)
    xx, yy = np.meshgrid(T_list, K_list)
    ax.plot_surface(xx, yy, C, cmap='rainbow')
    ax.set_xlabel('Expiry date T')
    ax.set_ylabel('Strike price K')
    ax.set_zlabel('Option price C')
    ax.set_title('Simulation of option price')
    plt.savefig('./heston/figures/option_price.png')
    plt.show()


def record_sim(V, S, C, K_list, T, num_time):
    dt = T / num_time
    T_list = np.arange(dt, T + dt, dt).tolist()
    pd_V = pd.DataFrame(V)
    pd_S = pd.DataFrame(S)
    pd_C = pd.DataFrame(C, index=K_list, columns=T_list)
    pd_V.to_csv('./heston/data/sim_volatility.csv')
    pd_S.to_csv('./heston/data/sim_asset_price.csv')
    pd_C.to_csv('./heston/data/sim_option_price.csv')


num_paths = 10000
num_time = 50
rho = -.02  # correlation of the bivariate normal distribution
S_0 = 100  # initial asset price
V_0 = 0.2 ** 2  # square of volatility
kappa = 2
theta = 0.2 ** 2  # long run variance
sigma = 0.1  # volatility of volatility
r = 0.05  # risk-free interest rate
T = 1  # longest expiry date
K_list = np.arange(25, 120, 1).tolist()  # list of strike price
V, S = asset_price_sim(num_paths, num_time, rho, S_0, V_0, T, kappa, theta, sigma, r)
C = call_option_price(S, K_list, num_time, T, r)
record_sim(V, S, C, K_list, T, num_time)
sim_figure(V, S, C, K_list, T, num_time)
