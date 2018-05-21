import numpy as np


def compute_sharpe_ratio(ret_p, risk_free_rate, st_dev_p):
    """
    :param ret_p: float, portfolio return
    :param risk_free_rate: float
    :param st_dev_p: float, portfolio volatility
    :return: float, computed Sharpe ratio
    """
    return (ret_p - risk_free_rate) / st_dev_p


def get_return_from_prices (v_prices, lag):
    """
    :param v_prices: ndarray, historical prices
    :param lag: long, used to compute return on a specific period
    :return: ndarray, computed returns
    """
    delta = v_prices.diff(lag)
    delta = delta[lag:]
    v_returns = delta/v_prices.shift(lag)
    return v_returns[lag:]


def get_vol_from_ret(v_ret, lag):
    """

    :param v_ret: ndarray, historical returns
    :param lag: long, used to compute on a specific period
    :return: ndarray, computed volatility
    """
    v_vol = np.zeros([v_ret.shape[0]-lag, v_ret.shape[1]])

    for curt_idx in range(0, len(v_ret)-lag, 1):
        v_temp = v_ret[curt_idx:curt_idx+lag]
        v_vol[curt_idx,:] = np.std(v_temp, 1)
    return v_vol


def get_rsi_from_price(v_prices, lag):
    """
    Computes the Relative Strength Index
    :param v_prices: ndarray, historical prices
    :param lag: long, used to compute on a specific period
    :return: ndarray, computed rsi
    """
    delta = v_prices.diff()
    delta = delta[1:]
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    
    RolUp=dUp.rolling(lag).mean()
    RolDown=dDown.rolling(lag).mean().abs()
    
    RS = RolUp / RolDown
    rsi= 100.0 - (100.0 / (1.0 + RS))
    
    return rsi


def get_proc_from_price(v_prices, lag):
    """
    Computes the Price Rate Of Change
    :param v_prices: ndarray, historical prices
    :param lag: long, used to compute on a specific period
    :return: ndarray, computed proc
    """
    proc = v_prices.pct_change(lag)
    return proc





