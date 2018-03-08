# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 17:45:49 2018

@author: SPALLA
"""

import pandas as pd
import fin_tools as ft


class AllData():
    """
    Data from file with methods adding financial indicators
    """
    def __init__(self, in_file):       
        self.prices = pd.read_csv(in_file)
        
        
    def add_indicators(self):
        """
        Adds financial indicators to the data to more inputs
        """
        rt = ft.get_return_from_prices(self.prices, 1)
        self.data = self.prices.copy()
        #for i in range(6,25,6):
        i = 6
        self.a = ft.get_rsi_from_price(self.prices, i)
        b = pd.Series(a, name='RSI' + str(i))
        self.data = pd.concat([self.data, pd.Series(ft.get_rsi_from_price(self.prices, i), name='RSI' + str(i))], axis = 1)


        pass
        
    def clean_data(self):
        """
        Cleans raw data from NaN or too short series
        """
        pass