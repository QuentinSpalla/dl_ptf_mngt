# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:45:49 2018

@author: SPALLA
"""

import pandas as pd

data_sp = pd.read_csv('data_stocks.csv')
data_dj = data_sp[['DATE', 'NYSE.MMM'	,
'NYSE.AXP'	,
'NASDAQ.AAPL'	,
'NYSE.BA'	,
'NYSE.CAT'	,
'NYSE.CVX'	,
'NASDAQ.CSCO'	,
'NYSE.KO'	,
'NYSE.DIS'	,
'NYSE.DD'	,
'NYSE.XOM'	,
'NYSE.GE'	,
'NYSE.GS'	,
'NYSE.HD'	,
'NYSE.IBM'	,
'NASDAQ.INTC'	,
'NYSE.JNJ'	,
'NYSE.JPM'	,
'NYSE.MCD'	,
'NYSE.MRK'	,
'NASDAQ.MSFT'	,
'NYSE.NKE'	,
'NYSE.PFE'	,
'NYSE.PG'	,
'NYSE.TRV'	,
'NYSE.UTX'	,
'NYSE.UNH'	,
'NYSE.VZ'	,
'NYSE.V'	,
'NYSE.WMT'	]]

data_dj.to_csv('data_stocks_dj.csv')
print(data_dj.shape[1])

