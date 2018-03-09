# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:45:49 2018

@author: SPALLA
"""

import pandas as pd
import fin_tools as ft
from tools import rename_df
from constants import RG_RSI_LAG, RG_RET_LAG, RG_PROC_LAG


class AllData():
    """
    Data from file with methods adding financial indicators
    """
    def __init__(self, in_file):       
        self.df_prices = pd.read_csv(in_file, sep=';')
        self.data = self.df_prices.copy()

    def add_indicators(self):
        """
        Adds financial indicators to the data to have more inputs
        """
        # RETURN
        for i in RG_RET_LAG:
            temp = ft.get_return_from_prices(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='RET_', suffix='_'+str(i))
            self.data = pd.concat([self.data, temp], axis=1)

        # RSI
        for i in RG_RSI_LAG:
            temp = ft.get_rsi_from_price(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='RSI_', suffix='_'+str(i))
            self.data = pd.concat([self.data, temp], axis=1)

        # PROC
        for i in RG_PROC_LAG:
            temp = ft.get_proc_from_price(self.df_prices.loc[:, self.df_prices.columns != 'DATE'], i)
            temp = rename_df(temp, prefix='PROC_', suffix='_'+str(i))
            self.data = pd.concat([self.data, temp], axis=1)

    def clean_data(self):
        """
        Cleans raw data from NaN or too short series
        """
        self.df_prices = None
        self.data