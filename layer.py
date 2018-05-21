import numpy as np
from tools import conv2, get_height_after_conv, inv_conv2, conv_delta

"""
File inspired from https://github.com/muupan/async-rl/blob/master/a3c_ale.p weights initialization based on muupan's 
code
"""
        
        
class FCLayer:
    def __init__(self, input_num, output_num, edge_dw, is_weights_init=True):
        self.in_val = 0
        self.is_wb = True
        self.edge_dw = edge_dw
        if is_weights_init:
            d = 1.0 / np.sqrt(input_num)
            """
            # from https://towardsdatascience.com/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
            self.weights = np.random.randn(input_num, output_num) * np.sqrt(2/input_num)
            self.bias = np.zeros()
            """
            self.weights =np.random.uniform(low=-d
                                           , high=d
                                           , size=(input_num, output_num))  
            self.bias =np.random.uniform(low=-d
                                         , high=d
                                         , size=(output_num, 1))

        else:
            self.weights = np.empty([input_num, output_num])
            self.bias = np.empty([output_num, 1])
        self.clear_weights_bias()
    
    def get_shape_wb(self):
        return self.weights.shape, self.bias.shape
    
    def update_weights_bias(self, weights, bias):
        if np.sum(np.isnan(self.weights)) > 0:
            print('error')
        self.weights = weights
        self.bias = bias
        
    def update_val(self, val):
        self.in_val = val
        
    def forward(self, input_data):
        self.in_val = input_data
        if np.sum(np.isnan(self.in_val))>0:
            print('error nan self.in_val')
        if np.sum(np.isinf(self.in_val))>0:
            print('error inf self.in_val')
        if np.sum(self.in_val>5):
            print('error big value self.in_val')
        return np.dot(self.weights.T, input_data) + self.bias
    
    def backward(self, loss):
        self.dw += np.clip(np.dot(self.in_val, loss.T), -self.edge_dw, self.edge_dw)
        self.db += np.sum(loss) / len(loss)
        if np.min(self.dw) < -10 or np.min(self.dw) > +10:
            print('ERROR dw')
        if np.min(self.db) < -5 or np.max(self.db) > +5:
            print('ERROR db')

        if np.sum(np.isnan(self.dw)) > 0:
            print('error')
        if np.sum(np.isnan(self.db)) > 0:
            print('error')
        residual_x = np.dot(self.weights, loss)
        return residual_x
    
    def get_diff_weights_bias(self):
        return self.dw, self.db
    
    def clear_weights_bias(self):
        self.db = np.zeros_like(self.bias)
        self.dw = np.zeros_like(self.weights)
    
    def get_weights_bias(self):
        return self.weights, self.bias


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
        return self.in_val-residuals

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


class SigmoidLayer:
    """
    Inspired from http://blog.varunajayasiri.com/numpy_lstm.html
    """
    def __init__(self):
        self.in_val = 0
        self.is_wb = False
        pass

    def update_val(self, val):
        self.in_val = val

    def forward(self, in_data):
        self.in_val = in_data
        return 1 / (1 + np.exp(-in_data))

    def backward(self, residual):
        return residual * (1 - residual)

    def get_diff_weights_bias(self):
        pass

    def get_weights_bias(self):
        pass

    def get_shape_wb(self):
        pass


class TanhLayer:
    """
    Inspired from http://blog.varunajayasiri.com/numpy_lstm.html
    """
    def __init__(self):
        self.in_val = 0
        self.is_wb = False
        pass

    def update_val(self, val):
        self.in_val = val

    def forward(self, in_data):
        self.in_val = in_data
        return np.tanh(in_data)

    def backward(self, residual):
        return 1 - residual * residual

    def get_diff_weights_bias(self):
        pass

    def get_weights_bias(self):
        pass

    def get_shape_wb(self):
        pass
