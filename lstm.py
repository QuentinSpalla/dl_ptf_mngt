#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class LSTM:
    def __init__(self, nn_f, nn_i, nn_c_bar, nn_o, tau_quantile):
        self.nn_f = nn_f
        self.nn_i = nn_i
        self.nn_c_bar = nn_c_bar
        self.nn_o = nn_o
        self.tau_quantile = tau_quantile

    def forward(self, h_prev, c_prev, in_data):
        """
        Computes forward step of lstm neural net
        :param h_prev: last predicted output
        :param c_prev: last cell state
        :param in_data: input data with features
        :return: predicted output, new cell state
        """
        z = np.concatenate((h_prev, in_data), axis=0)
        f = self.nn_f.get_output(z)
        i = self.nn_i.get_output(z)
        c_bar = self.nn_c_bar.get_output(z)
        c = f * c_prev + i * c_bar
        o = self.nn_o.get_output(z)
        h = o * np.tanh(c)
        return h, c

    def backward(self, out_data, intermediate_values, targets):
        pass

    def get_loss(self, out_data, target):
        """
        Computes Tau Quantile Loss in order to put more weights on negative returns
        """
        return sum((1-self.tau_quantile) * max((0, out_data-target)) + self.tau_quantile * max((0, target-out_data)))
