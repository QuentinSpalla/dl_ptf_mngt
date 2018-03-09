#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:35:42 2018

@author: SPALLA
"""
import numpy as np


class Portfolio:
    def __init__(self, nbr_assets, name=''):
        self.weights = np.zeros(nbr_assets)
        self.curt_time = 0
        self.curt_return = 0
        self.name = name

    def compute_sharpe_ratio(self):
        pass

    def update_weights(self, new_w):
        self.weights = new_w

    def update_time(self):
        self.curt_time += 1

    def update_return(self, new_rt):
        self.curt_return = new_rt
