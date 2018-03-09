# -*- coding: utf-8 -*-

import numpy as np
from tools import conv2, get_height_after_conv, inv_conv2, conv_delta

"""
File inspired from 
https://github.com/muupan/async-rl/blob/master/a3c_ale.py
weights initialization based on muupan's code
"""


class ConvLayer:    
    def __init__(self, input_channel, output_channel, kernel_size, stride
                 , is_weights_init=True):
        self.in_val = 0
        self.is_wb = True
        if is_weights_init:
            d = 1.0 / np.sqrt(input_channel * kernel_size * kernel_size)
            
            self.weights =np.random.uniform(low=-d
                                           , high=d
                                           , size=(input_channel
                                                   , output_channel
                                                   , kernel_size
                                                   , kernel_size))
            self.bias =np.random.uniform(low=-d
                                         , high=d
                                         , size= output_channel)
        else:
            self.weights =np.empty([input_channel, output_channel, kernel_size
                                                   , kernel_size])
            self.bias =np.empty([output_channel])
            
        self.stride = stride
        self.clear_weights_bias()
    
    def get_shape_wb(self):
        """
        Returns weights and bias shapes at current layer
        """
        return self.weights.shape, self.bias.shape
    
    def clear_weights_bias(self):
        self.db = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weights)
        
    def update_val(self, val):
        """
        Updates the input values of the layer
        Useful for backpropagation
        """
        self.in_val = val
        
    def update_weights_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def forward(self, input_data):
        self.in_val = input_data
        in_row, in_col, in_channel  = input_data.shape
        out_channel, kernel_size = self.weights.shape[1], self.weights.shape[2]
        out_row = get_height_after_conv(in_row, kernel_size, self.stride)
        self.top_val = np.zeros((out_row , out_row , out_channel))
                
        for o in range(out_channel):
            for i in range(in_channel):
                self.top_val[:,:,o] += conv2(input_data[:,:,i], 
                            self.weights[i, o], 
                            self.stride)
                if np.min(self.top_val[:,:,o]) < -5:
                    print('ERROR top_val')
                    a = conv2(input_data[:,:,i], 
                            self.weights[i, o], 
                            self.stride)
            self.top_val[:,:,o] += self.bias[o]
        return self.top_val
    
    def backward(self, residuals):
        """
        Backpropagation step for this layer
        Inspired from :
        " http://www.jefkine.com/general/2016/09/05/
        backpropagation-in-convolutional-neural-networks/ "
        """
        in_channel, out_channel, kernel_size, a = self.weights.shape
        dw = np.zeros_like(self.weights)        
        
        for i in range(in_channel):
            for o in range(out_channel):
                dw[i, o] = inv_conv2(self.in_val[:,:,i], 
                                      residuals[:,:,o], 
                                      self.stride)
                if np.min(dw) < -10:
                    print('ERROR dw')
                    a = inv_conv2(self.in_val[:,:,i], 
                                      residuals[:,:,o], 
                                      self.stride)
        self.db += residuals.sum(axis=1).sum(axis=0)
        self.dw += dw        
        if np.min(self.dw) < -10 or np.min(self.db) < -10:
            print('ERROR')
        gradient_x = np.zeros_like(self.in_val)
        
        for i in range(in_channel):
            for o in range(out_channel):
                gradient_x[:,:,i] += conv_delta(residuals[:,:,o] 
                                            , self.weights[i][o]
                                            , self.stride
                                            , self.in_val.shape[0])
        
        return gradient_x
    
    def get_diff_weights_bias(self):
        return self.dw, self.db
    
    def get_weights_bias(self):
        return self.weights, self.bias
        
        
class FCLayer:
    def __init__(self, input_num, output_num, is_weights_init=True):
        self.in_val = 0
        self.is_wb = True
        if is_weights_init:
            d = 1.0 / np.sqrt(input_num)
            self.weights =np.random.uniform(low=-d
                                           , high=d
                                           , size=(input_num, output_num))  
            self.bias =np.random.uniform(low=-d
                                         , high=d
                                         , size=(output_num, 1))
        else:
            self.weights =np.empty([input_num, output_num])      
            self.bias =np.empty([output_num, 1])
        self.clear_weights_bias()
    
    def get_shape_wb(self):
        return self.weights.shape, self.bias.shape
    
    def update_weights_bias(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, input_data):
        self.in_val = input_data
        return np.dot(self.weights.T, input_data) + self.bias
    
    def backward(self, loss):
        self.dw += np.dot(self.in_val, loss.T)
        self.db += np.sum(loss) / len(loss)
        if np.min(self.dw) < -10 or np.min(self.db) < -5:
            print('ERROR db')
        residual_x = np.dot(self.weights, loss)
        return residual_x
    
    def get_diff_weights_bias(self):
        return self.dw, self.db
    
    def clear_weights_bias(self):
        self.db = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weights)
    
    def get_weights_bias(self):
        return self.weights, self.bias

class FlattenLayer:
    def __init__(self):
        self.in_val = 0
        self.is_wb = False
        pass
    
    def get_shape_wb(self):
        pass
    
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, in_data):
        self.r, self.c, self.in_channel  = in_data.shape
        return in_data.reshape(self.in_channel * self.r * self.c, 1)

    def backward(self, residual):
        return residual.reshape(self.r, self.c, self.in_channel)
    
    def get_diff_weights_bias(self):
        pass
    
    def get_weights_bias(self):
        pass


class SoftmaxLayer:
    def __init__(self):
        self.in_val = 0
        self.is_wb = False
        pass
    
    def update_val(self, val):
        self.in_val = val
    
    def forward(self, x):        
        e_x = np.exp(x)
        temp = e_x / sum(e_x)
        self.in_val = temp
        return temp

    def backward(self, residuals):
        return (self.in_val-residuals)

    def get_diff_weights_bias(self):
        pass
    
    def get_weights_bias(self):
        pass
    
    def get_shape_wb(self):
        pass


class ReLULayer:
    def __init__(self):
        self.in_val = 0
        self.is_wb = False
        pass
    
    def update_val(self, val):
        self.in_val = val
    
    def forward(self, in_data):
        self.in_val = in_data
        ret = in_data.copy()
        ret[ret < 0] = 0
        return ret
    
    def backward(self, residual):
        gradient_x = residual.copy()
        gradient_x[self.in_val < 0] = 0
        return gradient_x
    
    def get_diff_weights_bias(self):
        pass
    
    def get_weights_bias(self):
        pass
    
    def get_shape_wb(self):
        pass


class TauQuantileleLayer:
    def __init__(self):
        self.in_val = 0
        self.is_wb = False

    def forward(self, in_data):
        pass