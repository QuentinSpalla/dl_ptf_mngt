#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lstm_param import Param

class LSTM:
    def __init__(self, nn_f, nn_i, nn_c_bar, nn_o, tau_quantile):
        self.nn_f = nn_f
        self.nn_i = nn_i
        self.nn_c_bar = nn_c_bar
        self.nn_o = nn_o
        self.tau_quantile = tau_quantile
        self.z = Param()
        self.f = Param()
        self.i = Param()
        self.c_bar = Param()
        self.c = Param()
        self.o = Param()
        self.h = Param()

    def forward(self, h_prev, c_prev, in_data):
        """
        Computes forward step of lstm neural net
        :param h_prev: last predicted output
        :param c_prev: last cell state
        :param in_data: input data with features
        :return: predicted output, new cell state
        """
        self.z.v = np.concatenate((h_prev, in_data), axis=0)
        self.f.v = self.nn_f.get_output(self.z.v)
        self.i.v = self.nn_i.get_output(self.z.v)
        self.c_bar.v = self.nn_c_bar.get_output(self.z.v)
        self.c.v = self.f.v * c_prev + self.i.v * self.c_bar.v
        self.o.v = self.nn_o.get_output(self.z.v)
        self.h.v = o.v * np.tanh(self.c.v)
        return self.h.v, self.c.v

    def backward(self, out_data, intermediate_values, targets):
        target, dh_next, dC_next, C_prev,
        z, f, i, C_bar, C, o, h, v, y,
        p = parameters):

        self.o.d = dh * tanh(C)
        do = dsigmoid(o) * do
        p.W_o.d += np.dot(do, z.T)
        p.b_o.d += do

        dC = np.copy(dC_next)
        dC += dh * o * dtanh(tanh(C))
        dC_bar = dC * i
        dC_bar = dtanh(C_bar) * dC_bar
        p.W_C.d += np.dot(dC_bar, z.T)
        p.b_C.d += dC_bar

        di = dC * C_bar
        di = dsigmoid(i) * di
        p.W_i.d += np.dot(di, z.T)
        p.b_i.d += di

        df = dC * C_prev
        df = dsigmoid(f) * df
        p.W_f.d += np.dot(df, z.T)
        p.b_f.d += df

        dz = (np.dot(p.W_f.v.T, df)

    + np.dot(p.W_i.v.T, di)
    + np.dot(p.W_C.v.T, dC_bar)
    + np.dot(p.W_o.v.T, do))
    dh_prev = dz[:H_size, :]
    dC_prev = f * dC


return dh_prev, dC_prev

    def get_loss(self, out_data, target):
        """
        Computes Tau Quantile Loss in order to put more weights on negative returns
        """
        return sum((1-self.tau_quantile) * max((0, out_data-target)) + self.tau_quantile * max((0, target-out_data)))
