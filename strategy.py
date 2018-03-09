#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:40:52 2018

@author: SPALLA
"""
import constants
import layer
from neural_network import NNetwork


class Strategy:
    def __init__(self, data, dates):
        self.data = data
        self.dates = dates

    def create_neural_network(self):
        nn = NNetwork()
        curt_lay = layer.FCLayer(self.data.shape[1], constants.FC_1_NEURONS, True)
        nn.add_layer(curt_lay, constants.FC_1_POS)
        curt_lay = layer.FCLayer(constants.FC_1_NEURONS, constants.FC_2_NEURONS, True)
        nn.add_layer(curt_lay, constants.FC_2_POS)
        curt_lay = layer.FCLayer(constants.FC_2_NEURONS, constants.FC_3_NEURONS, True)
        nn.add_layer(curt_lay, constants.FC_3_POS)

