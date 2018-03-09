#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:33:40 2018

@author: SPALLA
"""

# DATA
INPUT_FILE = 'data_stocks_dj.csv'
OUTPUT_FILE = 'output_ptf_values.csv'

# INDICATORS
NBR_MIN_STEP = 30
RG_RSI_LAG = range(6, 25, 6)
RG_PROC_LAG = range(6, 25, 6)
RG_RET_LAG = range(1, 30, 10)

# NEURAL NETWORK
FC_1_NEURONS = 200
FC_1_POS = 1
FC_2_NEURONS = 60
FC_2_POS = 2
