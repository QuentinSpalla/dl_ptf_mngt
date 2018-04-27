# -*- coding: utf-8 -*-

import numpy as np
import math

"""
Useful functions
"""


def get_vect_from_list(list_values):
    """
    Returns vector (N, 1) of a list containing matrixes 
    """
    vect_values = np.array([]).reshape(0,1)
    for curt_mat in list_values:
        temp_size = 1
        
        for curt_shape in range(len(curt_mat.shape)):
            temp_size *= curt_mat.shape[curt_shape]
        vect_values = np.concatenate((vect_values
                                     , curt_mat.reshape(temp_size, 1))
                                     , axis=0)
    return vect_values


def rename_df(df, prefix='', suffix=''):
    df_colname = df.columns
    new_names = prefix + df_colname + suffix
    df.columns = new_names
    return df


def get_list_from_vect(vect_values, all_shapes):
    """
    Returns list containing matrixes from vector (N, 1)
    """
    list_values = []
    temp_idx = 0
    for curt_mat_shape in all_shapes:
        temp_size = 1
        for i in range(len(curt_mat_shape)):
            temp_size *= curt_mat_shape[i]
        
        list_values.append(vect_values[temp_idx:temp_idx+temp_size].reshape(curt_mat_shape))
        temp_idx += temp_size
        
    return list_values


def conv_delta(out_data, weights, stride, in_data_size):
    """
    Computes gradient deltas in backpropagation for convolution layer
    Equation (20) in : 
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    ret = np.zeros((in_data_size, in_data_size))
    temp_size = int(weights.shape[0]/stride)
    out_data_size = out_data.shape[0] 
    
    for row in range(in_data_size):
        for col in range(in_data_size):
            row_even = row % 2
            col_even = col % 2
            i_ = math.floor(row/stride)
            j_ = math.floor(col/stride)
            for m in range(temp_size):
                for n in range(temp_size):                                         
                    if i_-m >=0 and j_-n >= 0 and i_-m < out_data_size and j_-n < out_data_size:
                        ret[row, col] += out_data[i_-m, j_-n] * weights[row_even + stride * m, col_even + stride * n]
    
    return ret
    
    
def inv_conv2(in_data, out_data, stride):
    """
    Computes gradient weights in backpropagation for convolution layer
    """
    in_row, in_col = in_data.shape
    out_row, out_col = out_data.shape
    
    kernel_size = get_kernel(in_row, out_row, stride)
    ret = np.zeros((kernel_size, kernel_size))
    
    for y in range(0, out_row):
        for x in range(0, out_row):
            sub = in_data[stride*y : stride*y + kernel_size, 
                          stride*x : stride*x + kernel_size]
            ret += np.sum(sub * out_data[y,x])
    return ret


def conv2(X, k, stride):
    """
    Convolution inspired from
    " https://gist.githubusercontent.com/JiaxiangZheng/
    a60cc8fe1bf6e20c1a41abc98131d518/raw/
    3630ae57e2e6c5669868a173b763f00fc6ddfb76/CNN.py "
    """
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    
    ret_row = get_height_after_conv(x_row, k_row, stride)            
    ret = np.zeros((ret_row, ret_row))
    
    for y in range(0, ret_row):
        for x in range(0, ret_row):
            sub = X[stride*y : stride*y + k_row, stride*x : stride*x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret


def get_kernel(init_height, output_size, stride):
    """
    Returns kernel size for weights in convolutional layer
    """
    return int(init_height-(output_size-1)*stride)


def get_height_after_conv(init_height, filter_size, stride):
    """
    Returns height of outputs in convolutional layer
    """
    return int(((init_height-filter_size)/stride+1))


def two_list_add(list1, list2, coeff):
    list_pos = 0
    temp_list = [None] * len(list1)

    while list_pos < len(list1):
        temp_list[list_pos] = list1[list_pos] - coeff*list2[list_pos]
        list_pos += 1

    return temp_list