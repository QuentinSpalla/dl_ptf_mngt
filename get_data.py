# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:45:49 2018

@author: SPALLA
"""

import pandas as pd
import fin_tools as ft
from tools import rename_df
from constants import RG_RSI_LAG, RG_RET_LAG, RG_PROC_LAG, RISK_FREE_RATE, NBR_MINUTES_STEP


class AllData():
    """
    Data from file with methods adding financial indicators
    """
    def __init__(self, in_file):       
        self.df_prices = pd.read_csv(in_file, sep=';')
        self.data = self.df_prices.copy()
        self.df_target = None
        self.first_idx_ret = 0

    def add_indicators(self):
        """
        Adds financial indicators to the data to have more inputs
        """
        # RETURN
        for i in RG_RET_LAG:
            temp = ft.get_return_from_prices(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='RET_' + str(i))

            if i == NBR_MINUTES_STEP:
                self.first_idx_ret = self.data.shape[1]
            self.data = pd.concat([self.data, temp], axis=1)

        # RSI
        for i in RG_RSI_LAG:
            temp = ft.get_rsi_from_price(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='RSI_' + str(i))
            self.data = pd.concat([self.data, temp], axis=1)

        # PROC
        for i in RG_PROC_LAG:
            temp = ft.get_proc_from_price(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='PROC_' + str(i))
            self.data = pd.concat([self.data, temp], axis=1)

    def create_target(self):
        """
        Computes All Sharpe Ratios for the data.
        Return for each step but vol for NBR_MINUTES_STEP steps.
        """
        temp_ret = ft.get_return_from_prices(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], 1)
        temp_vol = ft.get_vol_from_ret(temp_ret.values, NBR_MINUTES_STEP)
        temp_ret = temp_ret[NBR_MINUTES_STEP:]
        self.df_target = ft.compute_sharpe_ratio(temp_ret.values, RISK_FREE_RATE, temp_vol)

    def clean_data(self):
        """
        Cleans raw data from NaN or too short series
        """
        self.df_prices = None
        max_to_remove = max((max(RG_RSI_LAG), max(RG_PROC_LAG), max(RG_RET_LAG), NBR_MINUTES_STEP))

        if max_to_remove == NBR_MINUTES_STEP:
            self.data = self.data[max_to_remove+1:]
        else:
            self.target = self.target[max_to_remove:]
        # self.data.dropna(axis=0, how='any')
        # self.df_target = self.df_target[len(self.df_target)-len(self.data):]
