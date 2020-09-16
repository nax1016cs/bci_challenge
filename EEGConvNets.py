# -*- coding: utf-8 -*-
# author: Chun-Shu Wei
# Mitsubishi Electric Research Laboratories. All rights reserved.
# CONFIDENTIAL - Mitsubishi Electric Research Laboratories.
# Jul. 31, 2017: Chun-Shu Wei

from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout, Permute, Reshape
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1_l2
#from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Flatten
from keras.callbacks import Callback, EarlyStopping
from keras.layers.merge import Concatenate
import keras
import scipy.io
#import matplotlib.pylab as plt
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_first')
import os
import argparse
import platform

n_ch = 56
n_samp = 161
n_class = 2

def square(x):
    return x * x

def safe_log(x):
    return K.log(x + 1e-7)

def ShallowNet(input_shape):

    input_EEG = Input(input_shape)
    # Layer 1
    layer_1 = Conv2D(40, (1, 12), input_shape=(1, n_ch, n_samp), padding='valid', 
			kernel_regularizer=l1_l2(l1=0, l2=0.0001))(input_EEG)
    layer_1 = Conv2D(40, (n_ch, 40), padding='valid')(layer_1)
    layer_1 = BatchNormalization(axis = 1)(layer_1)
    layer_1 = Activation(square)(layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    layer_1 = AveragePooling2D(pool_size=(1, 37), strides=(1, 7))(layer_1) # pooling aft dropout works better
    layer_1 = Activation(safe_log)(layer_1)
    flatten = Flatten()(layer_1)
    dense = Dense(n_class)(flatten)
    softmax = Activation('softmax')(dense)

    return Model(input_EEG, softmax)



def EEGNet(input_shape):

	input_EEG = Input(input_shape)
	# Layer 1
	layer_1 = Conv2D(16, (n_ch,1), padding = 'same',
	                 kernel_regularizer=l1_l2(l1=0.0001, l2=0.0001))(input_EEG)
	layer_1 = BatchNormalization(axis=1)(layer_1)
	layer_1 = ELU()(layer_1)
	layer_1 = Dropout(0.25)(layer_1)
	permute_1 = Permute((2, 1, 3))(layer_1)
	
	# Layer 2
	layer_2 = Conv2D(4, (2,32), padding = 'same', strides=(2, 4))(permute_1)
	layer_2 = BatchNormalization(axis=1)(layer_2)
	layer_2 = ELU()(layer_2)
#	layer_2 = MaxPooling2D(pool_size=(2, 4), strides=None, padding = 'same')(layer_2)
	layer_2 = Dropout(0.25)(layer_2)
	
	# Layer 3
	layer_3 = Conv2D(4, (8,4), padding = 'same', strides=(2, 4))(layer_2)
	layer_3 = BatchNormalization(axis=1)(layer_3)
	layer_3 = ELU()(layer_3)
#	layer_3 = MaxPooling2D(pool_size=(2, 4), strides=None, padding = 'same')(layer_3)
	layer_3 = Dropout(0.25)(layer_3)
	
	flatten = Flatten()(layer_3)
	dense = Dense(n_class)(flatten)
	softmax = Activation('softmax')(dense)
	
	return Model(input_EEG,softmax)

def SCCNet(input_shape):

    input_EEG = Input(input_shape) 
    n_comp = 22
    layer_1 = Conv2D(n_comp, (n_ch, 1), input_shape=(1, n_ch, n_samp), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(input_EEG)
    permute_2 = Permute((2, 1, 3))(layer_1)
    layer_1 = Conv2D(20, (n_comp, 12), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(permute_2)
    layer_1 = BatchNormalization(axis=1, momentum=0.9)(layer_1)
    layer_1 = Activation(square)(layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    layer_1 = AveragePooling2D(pool_size=(1, 62), strides=(1, 12))(layer_1) # pooling aft dropout works better
    layer_1 = Activation(safe_log)(layer_1) 
    flatten = Flatten()(layer_1)
    dense = Dense(n_class)(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(input_EEG, softmax)

def SCCNet_CT(input_shape, C, T):

    input_EEG = Input(input_shape) 
    n_comp = C
    layer_1 = Conv2D(n_comp, (n_ch, T), input_shape=(1, n_ch, n_samp), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(input_EEG)
    permute_2 = Permute((2, 1, 3))(layer_1)
    layer_1 = Conv2D(20, (n_comp, 12), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(permute_2)
    layer_1 = BatchNormalization(axis=1, momentum=0.9)(layer_1)
    layer_1 = Activation(square)(layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    layer_1 = AveragePooling2D(pool_size=(1, 62), strides=(1, 12))(layer_1) # pooling aft dropout works better
    layer_1 = Activation(safe_log)(layer_1) 
    flatten = Flatten()(layer_1)
    dense = Dense(n_class)(flatten)
    softmax = Activation('softmax')(dense)
    
    return Model(inputs=input_EEG, output=softmax)


def SCCNet3TJ(input_shape, n_user):

    input_EEG = Input(input_shape) 
    n_comp = n_ch
    layer_1 = Conv2D(n_comp, (n_ch, 1), input_shape=(1, n_ch, n_samp), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(input_EEG)
    permute_2 = Permute((2, 1, 3))(layer_1)
    layer_1 = Conv2D(20, (n_comp, 12), padding='valid',
                 kernel_regularizer=l1_l2(l1=0, l2=0.0001))(permute_2)
    layer_1 = BatchNormalization(axis=1, momentum=0.9)(layer_1)
    layer_1 = Activation(square)(layer_1)
    layer_1 = Dropout(0.5)(layer_1)
    layer_1 = AveragePooling2D(pool_size=(1, 62), strides=(1, 12))(layer_1) # pooling aft dropout works better
    layer_1 = Activation(safe_log)(layer_1) 
    flatten_a = Flatten()(layer_1)
    dense_u = Dense(n_user)(flatten_a)
    dense_s = Dense(2)(flatten_a)
    softmax_u = Activation('softmax', name='out_u')(dense_u)
    softmax_s = Activation('softmax', name='out_s')(dense_s)
    log_u = Activation(safe_log)(softmax_u)
    log_s = Activation(safe_log)(softmax_s)
    merged_1 = Concatenate()([flatten_a, log_u])
    merged_2 = Concatenate()([merged_1, log_s])
    dense_c = Dense(n_class)(merged_2)
    softmax_c = Activation('softmax', name='out_c')(dense_c)
    
    return Model(inputs=input_EEG, output=[softmax_c, softmax_u, softmax_s])