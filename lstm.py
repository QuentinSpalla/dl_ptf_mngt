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
        self.inter_val = Param()

    def get_intermediate_values(self):
        temp_dict = {}
        temp_dict['val_nn_f'] = self.nn_f.get_intermediate_values()
        temp_dict['val_nn_i'] = self.nn_i.get_intermediate_values()
        temp_dict['val_nn_c_bar'] = self.nn_c_bar.get_intermediate_values()
        temp_dict['val_nn_o'] = self.nn_o.get_intermediate_values()
        temp_dict['other_val'] = self.inter_val
        return temp_dict

    def forward(self, h_prev, c_prev, in_data):
        """
        Computes forward step of lstm neural net
        :param h_prev: last predicted output
        :param c_prev: last cell state
        :param in_data: input data with features
        :return: predicted output, new cell state
        """
        self.inter_val.z = np.concatenate((h_prev, in_data), axis=0)
        self.inter_val.f = self.nn_f.get_output(self.inter_val.z)
        self.inter_val.i = self.nn_i.get_output(self.inter_val.z)
        self.inter_val.c_bar = self.nn_c_bar.get_output(self.inter_val.z)
        self.inter_val.c = self.inter_val.f * c_prev + self.inter_val.i * self.inter_val.c_bar
        self.inter_val.o = self.nn_o.get_output(self.inter_val.z)
        self.inter_val.h = self.inter_val.o * np.tanh(self.inter_val.c)
        self.inter_val.c_prev = c_prev
        return self.inter_val.h, self.inter_val.c

    def update_inter_val(self, dic_inter_val):
        self.inter_val = dic_inter_val['other_val']
        self.nn_f.update_val(dic_inter_val['val_nn_f'])
        self.nn_i.update_val(dic_inter_val['val_nn_i'])
        self.nn_c_bar.update_val(dic_inter_val['val_nn_c_bar'])
        self.nn_o.update_val(dic_inter_val['val_nn_o'])

    def backprogation(self, d_loss, d_h_next, d_c_next):
        # Transform loss in dh ???

        dh = d_h_next + d_loss
        do = dh * np.tanh(self.inter_val.c)
        # use neural net directly
        d_z_o = self.nn_o.backpropagation(do)
        """
        do = dsigmoid(o) * do
        p.W_o.d += np.dot(do, z.T)
        p.b_o.d += do
        """

        dc = np.copy(d_c_next)
        dc += dh * self.inter_val.o * (1-np.tanh(self.inter_val.c) ** 2)
        dc_bar = dc * self.inter_val.i
        dc_bar = (1-self.inter_val.c_bar)**2 * dc_bar
        # NN direct
        d_z_c_bar = self.nn_c_bar.backpropagation(dc_bar)
        """
        p.W_C.d += np.dot(dC_bar, z.T)
        p.b_C.d += dC_bar
        """

        di = dc * self.inter_val.c_bar
        # nn direct
        d_z_i = self.nn_c_bar.backpropagation(di)
        """
        di = dsigmoid(i) * di
        p.W_i.d += np.dot(di, z.T)
        p.b_i.d += di
        """

        df = dc * self.inter_val.c_prev
        # nn direct
        d_z_f = self.nn_c_bar.backpropagation(df)
        """
        df = dsigmoid(f) * df
        p.W_f.d += np.dot(df, z.T)
        p.b_f.d += df
        """

        dz = (d_z_f
              + d_z_i
              + d_z_c_bar
              + d_z_o)
        """
        dz = (np.dot(p.W_f.v.T, df)
              + np.dot(p.W_i.v.T, di)
              + np.dot(p.W_C.v.T, dC_bar)
              + np.dot(p.W_o.v.T, do))
        """
        dh_prev = dz[:len(d_loss), :]
        dc_prev = self.inter_val.f * dc
        return dh_prev, dc_prev

    def backward(self, out_data, intermediate_values, targets):
        dh_next = np.zeros((targets.shape[1], 1))
        dc_next = np.zeros((targets.shape[1], 1))

        for curt_idx in range(targets.shape[0]):
            self.update_inter_val(intermediate_values[curt_idx])
            curt_loss, curt_d_loss = self.get_loss_and_d(out_data[curt_idx, :], targets[curt_idx, :])
            dh_next, dc_next = self.backprogation(curt_d_loss, dh_next, dc_next)

    def get_loss_and_d(self, out_data, target):
        """
        Computes Tau Quantile Loss in order to put more weights on negative returns
        """
        loss = (1-self.tau_quantile) * np.maximum(0, out_data-target) \
            + self.tau_quantile * np.maximum(0, target-out_data)
        d_loss = np.ones(out_data.shape)*self.tau_quantile - np.maximum(0, target-out_data)/(target-out_data)
        return loss.reshape((loss.shape[0], 1)), d_loss.reshape((loss.shape[0], 1))
