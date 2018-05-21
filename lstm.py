import numpy as np
from lstm_param import Param
from tools import two_list_add


class LSTM:
    """
    Long Short Term Memory with neural networks.
    Lstm Inputs : financial data as prices, returns, rsi , etc.
    Lstm Outputs : Sharpe ratios
    """
    def __init__(self, nn_f, nn_i, nn_c_bar, nn_o, tau_quantile, initial_learning_rate):
        """
        :param nn_f: NNetwork
        :param nn_i: NNetwork
        :param nn_c_bar: NNetwork
        :param nn_o: NNetwork
        :param tau_quantile: float
        :param initial_learning_rate: float
        """
        self.nn_f = nn_f
        self.nn_i = nn_i
        self.nn_c_bar = nn_c_bar
        self.nn_o = nn_o
        self.tau_quantile = tau_quantile
        self.inter_val = Param()
        self.learning_rate = initial_learning_rate

    def get_intermediate_values(self):
        """
        :return: dictionary, all intermediate values of the lstm
        """
        temp_dict = {}
        temp_dict['val_nn_f'] = self.nn_f.get_intermediate_values()
        temp_dict['val_nn_i'] = self.nn_i.get_intermediate_values()
        temp_dict['val_nn_c_bar'] = self.nn_c_bar.get_intermediate_values()
        temp_dict['val_nn_o'] = self.nn_o.get_intermediate_values()
        temp_dict['other_val'] = self.inter_val
        return temp_dict

    def forward(self, h_prev, c_prev, in_data):
        """
        Inspired from http://blog.varunajayasiri.com/numpy_lstm.html
        Computes forward step of lstm neural net
        :param h_prev: ndarray, last predicted output
        :param c_prev: ndarray, last cell state
        :param in_data: ndarray, input data with features
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
        """
        Updates all intermediate values
        :param dic_inter_val: dictionary, all intermediate values of the lstm
        """
        self.inter_val = dic_inter_val['other_val']
        self.nn_f.update_val(dic_inter_val['val_nn_f'])
        self.nn_i.update_val(dic_inter_val['val_nn_i'])
        self.nn_c_bar.update_val(dic_inter_val['val_nn_c_bar'])
        self.nn_o.update_val(dic_inter_val['val_nn_o'])

    def backprogation(self, d_loss, d_h_next, d_c_next):
        """
        Inspired from http://blog.varunajayasiri.com/numpy_lstm.html
        Computes backward step of lstm neural net
        :param d_loss: ndarray, derived loss
        :param d_h_next: ndarray, next predicted output
        :param d_c_next: ndarray, next cell state
        :return: previous output, previous cell state
        """

        dh = d_h_next + d_loss
        do = dh * np.tanh(self.inter_val.c)
        d_z_o = self.nn_o.backpropagation(do)

        dc = np.copy(d_c_next)
        dc += dh * self.inter_val.o * (1-np.tanh(self.inter_val.c) ** 2)
        dc_bar = dc * self.inter_val.i
        dc_bar = (1-self.inter_val.c_bar)**2 * dc_bar

        d_z_c_bar = self.nn_c_bar.backpropagation(dc_bar)
        di = dc * self.inter_val.c_bar
        d_z_i = self.nn_i.backpropagation(di)
        df = dc * self.inter_val.c_prev
        d_z_f = self.nn_f.backpropagation(df)
        dz = (d_z_f
              + d_z_i
              + d_z_c_bar
              + d_z_o)

        dh_prev = dz[:len(d_loss), :]
        dc_prev = self.inter_val.f * dc
        return dh_prev, dc_prev

    def backward(self, out_data, intermediate_values, targets):
        """
        Computes batch gradient descent
        :param out_data: ndarray, batch neural network outputs
        :param intermediate_values: dictionary, all intermediate values of the lstm
        :param targets: ndarray, batch target values
        """
        dh_next = np.zeros((targets.shape[1], 1))
        dc_next = np.zeros((targets.shape[1], 1))

        for curt_idx in range(targets.shape[0]):
            self.update_inter_val(intermediate_values[curt_idx])
            curt_loss, curt_d_loss = self.get_loss_and_d(out_data[curt_idx, :], targets[curt_idx, :])
            dh_next, dc_next = self.backprogation(curt_d_loss, dh_next, dc_next)

    def get_loss_and_d(self, out_data, target):
        """
        :param out_data: ndarray, neural network output
        :param target: ndarray, target value
        :return: ndarray loss, ndarray derived loss
        """
        loss = (1-self.tau_quantile) * np.maximum(0, out_data-target) \
            + self.tau_quantile * np.maximum(0, target-out_data)
        d_loss = np.ones(out_data.shape)*self.tau_quantile - np.maximum(0, target-out_data)/(target-out_data)
        return loss.reshape((loss.shape[0], 1)), d_loss.reshape((loss.shape[0], 1))

    def get_diff_weights_bias(self):
        """
        :return: dictionary, all derived weights and derived bias of the lstm
        """
        dic_dw_db = {}
        dic_dw_db['nn_f'] = self.nn_f.get_all_diff_weights_bias()
        dic_dw_db['nn_i'] = self.nn_i.get_all_diff_weights_bias()
        dic_dw_db['nn_c'] = self.nn_c_bar.get_all_diff_weights_bias()
        dic_dw_db['nn_o'] = self.nn_o.get_all_diff_weights_bias()
        return dic_dw_db

    def update_param(self):
        """
        Updates all weights and bias of the lstm using accumulated derived weights and derived bias
        """
        temp_dic_diff = self.get_diff_weights_bias()
        # f
        self.nn_f.update_weights_bias(two_list_add(self.nn_f.get_all_weights_bias(),
                                                   temp_dic_diff['nn_f'],
                                                   self.learning_rate))
        # i
        self.nn_i.update_weights_bias(two_list_add(self.nn_i.get_all_weights_bias(),
                                                   temp_dic_diff['nn_i'],
                                                   self.learning_rate))
        # c
        self.nn_c_bar.update_weights_bias(two_list_add(self.nn_c_bar.get_all_weights_bias(),
                                                       temp_dic_diff['nn_c'],
                                                       self.learning_rate))
        # o
        self.nn_o.update_weights_bias(two_list_add(self.nn_o.get_all_weights_bias(),
                                                   temp_dic_diff['nn_o'],
                                                   self.learning_rate))
        temp_dic_diff = None