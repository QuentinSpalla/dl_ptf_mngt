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
dj.clean_data()

my_strategy = Strategy(dj.loc[:, dj.columns != 'DATE'], dj['DATE'])

print('the end')
