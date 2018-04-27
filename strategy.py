#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:40:52 2018

@author: SPALLA
"""
import constants
import math
import layer
import numpy as np
from neural_network import NNetwork
from ptflio import Portfolio
from lstm import LSTM


class Strategy:
    """
    Contains the neural network and the portfolios
    """
    def __init__(self, data, dates, targets, first_ret_idx):
        self.targets = (targets - np.mean(targets, 0)[None, :]) / np.var(targets, 0)[None, :]**(1/2) # / np.amax(abs(targets), 1)[:, None]
        self.data = (data - np.mean(data, 0)[None, :]) / np.var(data, 0)[None, :]**(1/2)  #data / np.amax(abs(data), 1)[:, None]
        self.dates = dates
        self.bmk = None
        self.ptf = None
        self.lstm = None
        self.first_ret_idx = first_ret_idx

    def create_neural_network(self, act_fun_lay, act_fun_pos):
        """
        Creates 2 hidden fully connected neural network with chosen last activation function
        :param act_fun_lay: activation function
        :param act_fun_pos: activation function position
        :return: neural network created
        """
        nn = NNetwork()
        curt_lay = layer.FCLayer(self.data.shape[1] + self.targets.shape[1], constants.FC_1_NEURONS, constants.EDGE_GRADIENT, True)
        nn.add_layer(curt_lay, constants.FC_1_POS)
        nn.add_layer(layer.ReLULayer(), constants.RELU_1_POS)
        curt_lay = layer.FCLayer(constants.FC_1_NEURONS, constants.FC_2_NEURONS, constants.EDGE_GRADIENT, True)
        nn.add_layer(curt_lay, constants.FC_2_POS)
        nn.add_layer(layer.ReLULayer(), constants.RELU_2_POS)
        curt_lay = layer.FCLayer(constants.FC_2_NEURONS, constants.FC_OUTPUT_NEURONS, constants.EDGE_GRADIENT, True)
        nn.add_layer(curt_lay, constants.FC_OUTPUT_POS)
        nn.add_layer(act_fun_lay, act_fun_pos)
        return nn

    def create_lstm(self):
        """
        Creates the lstm neural network
        """
        nn_f = self.create_neural_network(layer.SigmoidLayer(), constants.SIG_F_POS)
        nn_i = self.create_neural_network(layer.SigmoidLayer(), constants.SIG_I_POS)
        nn_c = self.create_neural_network(layer.TanhLayer(), constants.TANH_C_POS)
        nn_o = self.create_neural_network(layer.SigmoidLayer(), constants.SIG_O_POS)
        self.lstm = LSTM(nn_f, nn_i, nn_c, nn_o, constants.TAU_QUANTILE, constants.LEARNING_RATE)

    def create_portfolio(self):
        self.ptf = Portfolio(constants.FC_OUTPUT_NEURONS, constants.TRANSACTION_FEE_RATE, constants.INITIAL_PTF_VALUE
                             , name='portfolio')
        self.ptf.update_weights(1 / len(self.ptf.weights) * np.ones([len(self.ptf.weights), 1]))

    def create_benchmark(self):
        """
        Creates equal weights benchmark portfolio
        """
        self.bmk = Portfolio(constants.FC_OUTPUT_NEURONS, constants.TRANSACTION_FEE_RATE, constants.INITIAL_PTF_VALUE
                             , name='benchmark')
        self.bmk.update_weights(1/len(self.bmk.weights) * np.ones([len(self.bmk.weights), 1]))

    def train(self, size_train=0.7):
        """
        Updates the Neural Net
        """
        h_prev = np.zeros((constants.FC_OUTPUT_NEURONS, 1))
        c_prev = np.zeros((constants.FC_OUTPUT_NEURONS, 1))
        curt_batch_size = 1
        intermediate_values = []
        out_data = np.zeros((constants.BATCH_SIZE, constants.FC_OUTPUT_NEURONS))

        for curt_index in range(0, int(math.floor(size_train * self.data.shape[0])), 1):
            in_data = self.data[curt_index, :].reshape(self.data.shape[1], 1)
            h_prev, c_prev = self.lstm.forward(h_prev, c_prev, in_data)
            out_data[curt_batch_size-1, :] = h_prev.reshape((constants.FC_OUTPUT_NEURONS, ))
            intermediate_values.append(self.lstm.get_intermediate_values())

            if curt_batch_size == constants.BATCH_SIZE:
                self.lstm.backward(out_data,
                                   intermediate_values,
                                   self.targets[curt_index-constants.BATCH_SIZE+1:curt_index+1])
                self.lstm.update_param()
                if np.sum(np.isnan(self.lstm.nn_c_bar.layers[1].weights)) > 0:
                    print('error')
                if curt_index == 59-15:
                    print('error')
                curt_batch_size = 0
                intermediate_values = []
                out_data = np.zeros((constants.BATCH_SIZE, constants.FC_OUTPUT_NEURONS))
                print(curt_index)

            curt_batch_size += 1

    def test(self, size_test=0.3, is_nn_to_train=False):
        """
        Uses NN to predict Sharpe Ratios.
        Updates the portfolios weights and computes portfolio returns.
        """
        if not is_nn_to_train:
            h_prev = np.zeros((constants.FC_OUTPUT_NEURONS, 1))
            c_prev = np.zeros((constants.FC_OUTPUT_NEURONS, 1))
            curt_index = int(math.floor((1-size_test)*self.data.shape[0]))
            bmk_wgts = 1 / len(self.bmk.weights) * np.ones([len(self.bmk.weights), 1])

            for curt_index in range(curt_index,
                                    self.data.shape[0]-constants.NBR_MINUTES_STEP,
                                    constants.NBR_MINUTES_STEP):
                in_data = self.data[curt_index, :].reshape(self.data.shape[1], 1)
                h_prev, c_prev = self.lstm.forward(h_prev, c_prev, in_data)
                temp_ret = self.data[curt_index + constants.NBR_MINUTES_STEP,
                           self.first_ret_idx:self.first_ret_idx+constants.FC_OUTPUT_NEURONS].reshape(constants.FC_OUTPUT_NEURONS, 1)
                self.ptf.compute_return(temp_ret)
                self.ptf.compute_value()
                self.ptf.update_weights_inv_rdt(temp_ret)
                self.ptf.update_weights_inv_val(h_prev)
                self.ptf.compute_transaction_fees()
                self.ptf.update_val_list()
                self.ptf.update_time()
                self.bmk.compute_return(temp_ret)
                self.bmk.compute_value()
                self.bmk.update_weights_inv_rdt(temp_ret)
                self.bmk.update_weights_inv_val(bmk_wgts)
                self.bmk.compute_transaction_fees()
                self.bmk.update_val_list()
                self.bmk.update_time()
                if self.ptf.curt_time == 147:
                    print('error')
            print('stop')