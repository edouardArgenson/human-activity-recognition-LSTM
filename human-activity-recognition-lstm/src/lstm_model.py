#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:39:22 2019

@author: edouard
"""

#%%

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, LSTM
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TerminateOnNaN, ModelCheckpoint


#%% 

# TODO input shape as argument ?

# Builds and compiles a LSTM neural network, with two LSTM neuron layers,
# an optional intermediary dense layer, and a dense output layer.
# n_lstm_1 : n LSTM neurons in first layer.
# n_lstm_2 : n LSTM neurons in second layer.
# n_dense_3 : n perceptron neurons in third layer, can be 0.
# n_dense_out :  n perceptron neurons in output layer.
# Returns compiled model.
def build_model(n_lstm_1, n_lstm_2, n_dense_3, n_dense_out):
    
    timesteps = 128
    input_dim = 9
    
    input_shape_tuple = (timesteps, input_dim)
    
    model = Sequential() ;
    model.add(LSTM(n_lstm_1, input_shape=input_shape_tuple, activation='tanh', return_sequences=True))
    model.add(LSTM(n_lstm_2, input_shape=input_shape_tuple, activation='tanh', return_sequences=False))
    
    if n_dense_3 > 0 :
        model.add(Dense(n_dense_3, kernel_initializer=initializers.glorot_uniform(seed=None)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
    model.add(Dense(n_dense_out, kernel_initializer=initializers.glorot_uniform(seed=None)))
    model.add(Activation('sigmoid'))
    
    my_optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, 
                               epsilon=None, decay=0.0, amsgrad=False, clipnorm=1.)
 
    model.compile(loss='categorical_crossentropy',
                  optimizer=my_optim,
                  metrics=['accuracy'])
    
    return model


def train_model(m, train_set, validation_set, epchs, btch_size, ckpt_path):
    ton_callback = TerminateOnNaN()
        
    checkpoint = ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', 
                                 save_best_only=False, save_weights_only=False, period=1)
    
    x_train = train_set['x']
    y_train = train_set['y']
    
    x_validation = validation_set['x']
    y_validation = validation_set['y']
    
    histo = m.fit(x_train, y_train,
              epochs=epchs,
              batch_size=btch_size,
              callbacks=[ton_callback, checkpoint],
              validation_data=(x_validation, y_validation)
              )
    
    return histo


def predict(m, x_test):
    test_predictions_ohe = m.predict(x_test)
    test_predictions = np.argmax(test_predictions_ohe, axis=1)
    return test_predictions


def load_best_model(m, hist, ckpt_dir):
  
    validation_loss = hist.history['val_loss']
    validation_acc = hist.history['val_accuracy']
    
    val_loss_as_array = np.asarray(validation_loss)
    
    # Get rid of nans in val_loss_as_array
    for k in range(np.size(val_loss_as_array)):
        if np.isnan(val_loss_as_array[k]):
            val_loss_as_array[k] = 1000
    
    model_idx = np.argmin(val_loss_as_array)
    
    print('model_idx={}'.format(model_idx))
    print('val_loss[model_idx]=' + str(val_loss_as_array[model_idx]))
    print('val_accuracy[model_idx]=' + str(validation_acc[model_idx]))
    
    m.load_weights(ckpt_dir + '/model-' + '{:03d}'.format(model_idx) + '.hdf5')
    
    