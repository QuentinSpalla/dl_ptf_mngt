#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:50:48 2018

@author: SPALLA
"""

def get_average_price(high_price, low_price, close_price):
    # fonction pour recuperer le prix moyen a partir du High, Low et Close
    return ((high_price+low_price+close_price)/3)

def get_return_from_prices (v_prices, lag):
    # fonction pour recuperer les rendements a partir des prix (selon une periode)
    delta = v_prices.diff(lag)
    delta = delta[lag:]
    v_returns = delta/v_prices.shift(lag)
    return v_returns[lag:]

def get_rsi_from_price(v_prices, lag):
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

def get_sto_osc_from_price(v_prices, v_low, v_high, lag): 
    stok = ((v_prices - v_low.rolling(lag).min()) / (v_high.rolling(lag).max() - v_low.rolling(lag_days).min())) * 100
    return stok

def get_williams_from_price(v_prices, v_low, v_high, lag): 
    williams = ((v_high.rolling(lag).max() - v_prices) / (v_high.rolling(lag).max() - v_low.rolling(lag_days).min())) * (-100)
    return williams

def get_macd_from_price(v_prices, short_days, long_days, signal_days=9): 
    ema_short = v_prices.ewm(span=short_days).mean()
    ema_long = v_prices.ewm(span=long_days).mean()
    ema_signal = v_prices.ewm(span=signal_days).mean()
    macd = ema_short[long_days:] - ema_long[long_days:]
    return macd-ema_signal


def get_proc_from_price(v_prices, lag): 
    proc = v_prices.pct_change(lag)
    return proc

def get_obv_from_volume(v_prices, v_volumes, lag): 
    delta = v_prices.diff(lag)
    obv = v_volumes.copy()
    obv[delta==0] = 0
    obv[delta<0] = -v_volumes[delta<0]
    obv = obv.cumsum()
    return obv

def get_atr_from_price(v_prices, v_lows, v_highs, lag, n=20):     
    tr = pd.concat([(v_highs - v_lows).abs(), (v_highs - v_prices.shift(lag)).abs(), (v_lows - v_prices.shift(lag)).abs()],axis=1).T.max()
    atr = tr
    
    for i in range(1,len(atr)):          
        atr[i] = ((n-1)*atr[i-1]+tr[i])/n
        
    return atr
