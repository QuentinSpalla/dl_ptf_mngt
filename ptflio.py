#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:35:42 2018

@author: SPALLA
"""
import numpy as np


class Portfolio:
    def __init__(self, nbr_assets, transac_fee_rate, initial_value, name=''):
        self.weights = np.zeros(nbr_assets)
        self.last_weights = np.zeros(nbr_assets)
        self.curt_time = 0
        self.curt_return = 0
        self.curt_value = 0
        self.curt_transac_value = 0
        self.values = [initial_value]
        self.transac_values = [0]
        self.name = name
        self.transac_fee_rate = transac_fee_rate

    def update_weights(self, new_w):
        self.last_weights = self.weights
        self.weights = new_w

    def update_time(self):
        self.curt_time += 1

    def update_weights_inv_val(self, values):
        temp_w = values / sum(values)
        self.update_weights(temp_w)

    def update_weights_inv_rdt(self, rdt):
        temp_w_r = self.weights * rdt
        self.update_weights(temp_w_r / sum(temp_w_r))

    def update_val_list(self):
        self.values.append(self.curt_value)
        self.transac_values.append(self.curt_transac_value)

    def compute_transaction_fees(self):
        self.curt_transac_value = sum(abs(self.weights - self.last_weights)) * self.curt_value * self.transac_fee_rate

    def compute_return(self, assets_ret):
        self.curt_return = np.dot(self.weights, assets_ret)

    def compute_value(self):
        self.curt_value = self.curt_value * (1.+self.curt_return)
