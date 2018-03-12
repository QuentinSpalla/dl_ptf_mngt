#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 21:21:52 2018

@author: SPALLA
"""

import constants
from get_data import AllData
from strategy import Strategy

dj = AllData(constants.INPUT_FILE)
dj.add_indicators()
dj.create_target()
dj.clean_data()

my_strategy = Strategy(dj.data.loc[:, dj.data.columns != 'DATE'].values, dj.data['DATE'].values, dj.df_target)
my_strategy.create_lstm()
my_strategy.train()

print('the end')
