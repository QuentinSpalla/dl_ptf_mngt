#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class LSTM:
    def __init__(self, nn_f, nn_i, nn_c_bar, nn_o):
        self.nn_f = nn_f
        self.nn_i = nn_i
        self.nn_c_bar = nn_c_bar
        self.nn_o = nn_o

    def forward(self, h_prev, c_prev, in_data):
        """
        Computes forward step of lstm neural net
        :param h_prev: last predicted output
        :param c_prev: last cell state
        :param in_data: input data with features
        :return: predicted output, new cell state
        """
        z = np.concatenate((h_prev.T, in_data.T), axis=0)
        f = self.nn_f.get_output(z) # sigmoid(np.dot(p.W_f.v, z) + p.b_f.v)
        i = self.nn_i.get_output(z) # sigmoid(np.dot(p.W_i.v, z) + p.b_i.v)
        c_bar = self.nn_c_bar.get_output(z) # tanh(np.dot(p.W_C.v, z) + p.b_C.v)
        c = f * c_prev + i * c_bar
        o = self.nn_o.get_output(z) # sigmoid(np.dot(p.W_o.v, z) + p.b_o.v)
        h = o * np.tanh(c)
        return h, c

    def backward(self):
        pass

    def get_loss(self):
        """
         Computes Tau Quantile Loss in order to put more weights on negative returns
        """
        pass