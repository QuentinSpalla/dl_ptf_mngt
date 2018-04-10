#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:33:40 2018

@author: SPALLA
"""

# DATA
INPUT_FILE = 'data_stocks_dj.csv'
OUTPUT_FILE = 'output_ptf_values.csv'

# FINANCIAL INDICATORS
RG_RSI_LAG = range(6, 25, 6)
RG_PROC_LAG = range(6, 25, 6)
RG_RET_LAG = [1, 10, 20, 30]

# PORTFOLIO MANAGEMENT
NBR_MINUTES_STEP = 30
RISK_FREE_RATE = 0
TRANSACTION_FEE_RATE = 0.001
INITIAL_PTF_VALUE = 100
RETURN_TYPE_NAME = 'RET_30'

# NEURAL NETWORK
FC_1_NEURONS = 200
FC_1_POS = 1
FC_2_NEURONS = 60
FC_2_POS = 3
FC_OUTPUT_NEURONS = 30
FC_OUTPUT_POS = 5
RELU_1_POS = 2
RELU_2_POS = 4

# LSTM
SIG_F_POS = 6
SIG_I_POS = 6
SIG_O_POS = 6
TANH_C_POS = 6
TAU_QUANTILE = 0.6

# GRADIENT DESCENT
LEARNING_RATE = 1e-4
BATCH_SIZE = 15
