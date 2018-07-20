import numpy as np

def z_normalize(x):
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    norm = (x - mu) / std
    return norm, mu, std

def roi_window(window, invest_at):
    return (window - invest_at) / invest_at

def roi(invest_at, sell_at):
    return (sell_at - invest_at) / invest_at