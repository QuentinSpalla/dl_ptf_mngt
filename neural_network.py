# -*- coding: utf-8 -*-

import numpy as np
from tools import get_vect_from_list


class NNetwork():
    """
    Deep Neural Network that takes in 
    Inputs : images of a game (84,84,4)
    Outputs : 
        - probabilities for each actions 
        - value of the current state V(s) = E[Rt\s_t=s]
    
    """
    def __init__(self):        
        # self.pi = []
        # self.value = 0
        self.layers = {}

    def add_layer(self, layer, position):
        self.layers[position] = layer    

    def get_output(self, in_data):
        """
        Computes forward output threw the neural network
        :param in_data: input data of the NN
        """
        out_data = in_data

        for layer_pos in range(1, len(self.layers) + 1, 1):
            layer = self.layers[layer_pos]
            out_data = layer.forward(out_data)
        return out_data
    
    def get_intermediate_values(self):
        """
        Returns values at each neurons 
        of layers which are useful for the backpropagation
        """
        intermediate_values = []
        for layer_pos in range(1, len(self.layers)+1):
            layer = self.layers[layer_pos]               
            intermediate_values.append(layer.in_val)
        return intermediate_values

    def update_val(self, inter_val):
        for layer_pos in range(1, len(self.layers)+1):
            layer = self.layers[layer_pos]
            layer.update_val(inter_val[layer_pos-1])

    def backpropagation(self, d_loss):
        """
        Backpropagation threw the nn's layers
        :param d_loss: derivative loss
        :return: First layer derivative
        """
        out_data = d_loss
        layer_pos = len(self.layers)    # if not activation layer :- 1

        while layer_pos >= 1:
            layer = self.layers[layer_pos]
            out_data = layer.backward(out_data)
            layer_pos -= 1

        return out_data

    def backpropag_pi(self, loss, values):
        """
        Makes the backpropagation on all the convolutional network 
        using the probability final layer
        Returns the weigts difference
        """
        out_data = loss        
        layer_pos = len(self.layers) - 1
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            layer.update_val(values[layer_pos-1])
            out_data = layer.backward(out_data)
            if np.min(out_data)< -5 or np.max(out_data)>5:
                print('ERROR out_data backpropag')
            layer_pos -= 1

    
    def get_all_diff_weights_bias(self):
        """
        Return in a vect (N,1) all the gradients (weights and bias)
        of the NeuralNet
        """
        dw_b = []
        layer_pos = len(self.layers)
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            curt_dw_db = layer.get_diff_weights_bias()
            if not curt_dw_db is None:
                dw_b.append(curt_dw_db[0])
                dw_b.append(curt_dw_db[1])
                layer.clear_weights_bias()
                
            layer_pos -= 1
        
        dw_b.reverse()
        vec_diff_weights_bias = get_vect_from_list(dw_b)
        return vec_diff_weights_bias 
    
    
    def get_all_weights_bias(self):
        """
        Return in a vect (N,1) all the weights and bias of the NeuralNet
        """
        w_b = []
        
        layer_pos = len(self.layers)
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            curt_w_b = layer.get_weights_bias()
            if not curt_w_b is None:
                w_b.append(curt_w_b[0])
                w_b.append(curt_w_b[1])
                layer.clear_weights_bias()
                
            layer_pos -= 1
        
        w_b.reverse()
        vec_weights_bias = get_vect_from_list(w_b)
        return vec_weights_bias 
    
    def get_all_shapes(self):
        """
        Returns the shapes of all layer weights
        """
        w_b_shapes = []
        
        layer_pos = len(self.layers)
        
        while layer_pos >=1:
            layer = self.layers[layer_pos]            
            curt_w_b = layer.get_shape_wb()
            if not curt_w_b is None:
                w_b_shapes.append(curt_w_b[0])
                w_b_shapes.append(curt_w_b[1])
                layer.clear_weights_bias()
                
            layer_pos -= 1
        
        w_b_shapes.reverse()        
        return w_b_shapes 
    
    def backpropag_value(self, loss, values):
        """
        Makes the backpropagation on the last value layer
        Returns the weigts difference of that layer
        """
        out_data = loss        
        layer_pos = len(self.layers)                
        layer = self.layers[layer_pos]            
        layer.update_val(values[layer_pos-1])
        layer.backward(out_data)
         
    def update_weights_bias(self, list_weights_bias):
        """
        Update Weights and Bias
        """
        layer_pos = 1
        i=0
        
        while layer_pos <=len(self.layers):
            layer = self.layers[layer_pos]   
            if layer.is_wb:                
                layer.update_weights_bias(list_weights_bias[i+1]
                , list_weights_bias[i])
                i += 2
            layer_pos += 1   