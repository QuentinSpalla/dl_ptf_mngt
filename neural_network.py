import numpy as np


class NNetwork():
    """
    Neural Network with multiple layers
    """
    def __init__(self):
        self.layers = {}

    def add_layer(self, layer, position):
        """
        Adds a layer to the neural network
        :param layer: Layer, layer to add
        :param position: long
        """
        self.layers[position] = layer    

    def get_output(self, in_data):
        """
        Computes forward output threw the neural network
        :param in_data: ndarray, input data of the NN
        """
        out_data = in_data

        for layer_pos in range(1, len(self.layers) + 1, 1):
            layer = self.layers[layer_pos]
            out_data = layer.forward(out_data)
        return out_data
    
    def get_intermediate_values(self):
        """
        Returns values at each neurons of layers
        Used for backpropagation
        """
        intermediate_values = []
        for layer_pos in range(1, len(self.layers)+1):
            layer = self.layers[layer_pos]               
            intermediate_values.append(layer.in_val)
        return intermediate_values

    def update_val(self, inter_val):
        """
        Updates all intermediate valeus for each layer of the neural network
        :param inter_val: dictionary, all intermediate values of a neural network
        """
        for layer_pos in range(1, len(self.layers)+1):
            layer = self.layers[layer_pos]
            layer.update_val(inter_val[layer_pos-1])

    def backpropagation(self, d_loss):
        """
        Backpropagation threw the neural network's layers
        :param d_loss: ndarray, derived loss
        :return: ndarray, First layer derivative
        """
        out_data = d_loss
        layer_pos = len(self.layers)

        while layer_pos >= 1:
            layer = self.layers[layer_pos]
            out_data = layer.backward(out_data)
            layer_pos -= 1

        return out_data
    
    def get_all_diff_weights_bias(self):
        """
        Retrieves gradients for each layers of the neural network
        :return: ndarray, all gradients (weights and bias) of the neural network
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
        return dw_b

    def get_all_weights_bias(self):
        """
        Retrieves weights/bias for each layers of the neural network
        :return: ndarray, all weights and bias of the neural network
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
        return w_b
         
    def update_weights_bias(self, list_weights_bias):
        """
        Update weights and bias of each layers of the neural network
        :param list_weights_bias: list, all weights and bias of the neural network
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