#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 19:56:41 2019

@author: edouard
"""

#%%

import os

import numpy as np
import pandas as pd
import math
from tensorflow.keras import utils as kutils


#%%

def load_data_and_build_features(train_percentage, split_random_seed, shuffle_random_seed, data_path):
    
    dataframes_train = read_csv_data(data_path, 'train')
    dataframes_test = read_csv_data(data_path, 'test')

    # Split training set into train and validation sets.
    # We split by volunteer data in order to avoid leakage between train and validation.
    # To perform a ~80/20% split, we draw 4 volunteer ids from the 21 volunteers from the training data.
    (split_train_set, split_validation_set) = split_into_train_and_validation(train_percentage, split_random_seed, dataframes_train)

    # We remap labels from [1;n_classes] to [0;n_classes-1] in order to perform one-hot-encoding at next step.
    remapped_train_set = remap_y_labels(split_train_set)
    remapped_validation_set = remap_y_labels(split_validation_set)
    
    remapped_test_set = remap_y_labels(dataframes_test)
    
    n_classes = count_classes(remapped_train_set['y_df'])
    
    shuffled_train_set = shuffle_dataset(remapped_train_set, shuffle_random_seed)
    
    # TODO doc !! (above fction def)
    train_set = convert_to_nd_arrays(shuffled_train_set, n_classes)
    validation_set = convert_to_nd_arrays(remapped_validation_set, n_classes)
    test_set = convert_to_nd_arrays(remapped_test_set, n_classes, one_hot_encode_y=False)

    return (train_set, validation_set, test_set)


def load_class_labels(data_path):
    filename = data_path + 'activity_labels.txt'
    class_names_df = pd.read_csv(filename, sep='\s+', header=None, names=['Label'])
    return class_names_df['Label'].to_numpy()


# Model checkpoints will be saved in :
# '../data/working_data/model_checkpoints/[timestamped_directory]'
# Timestamped plot of training loss will be saved in :
# '../data/working_data/training_plots/'
def prepare_working_data_directories(timestamp):
    ckpt_dir = '../data/working_data/model_checkpoints/{}'.format(timestamp) 
    os.mkdir(ckpt_dir)
    print('mkdir {}'.format(ckpt_dir))
    
    ckpt_path = '' + ckpt_dir + '/model-{epoch:03d}.hdf5'
    
    training_plt_path = '../data/working_data/training_plots/' + 'training_{}'.format(timestamp)
    confusion_matrix_plt_path = '../data/working_data/confusion_matrices/' + 'confusion_matrix_{}'.format(timestamp)
    
    return {
            'checkpoint_dir' : ckpt_dir, 
            'checkpoint_path' : ckpt_path,
            'training_loss_plt_path' : training_plt_path,
            'confusion_matrix_plt_path' : confusion_matrix_plt_path
            }


# suffix should be 'train' or 'test'. It is the suffix in csv file names.
def read_csv_data(folder_path, suffix):
    
    # Builds paths.
    suffixed_folder_path = folder_path + '/' + suffix + '/' 
    inertial_signals_path = suffixed_folder_path + 'inertial_signals/'
    
    signal_paths = []
    
    signal_paths.append('' + inertial_signals_path + 'total_acc_x_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'total_acc_y_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'total_acc_z_' + suffix + '.txt')
    
    signal_paths.append('' + inertial_signals_path + 'body_acc_x_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'body_acc_y_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'body_acc_z_' + suffix + '.txt')
    
    signal_paths.append('' + inertial_signals_path + 'body_gyro_x_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'body_gyro_y_' + suffix + '.txt')
    signal_paths.append('' + inertial_signals_path + 'body_gyro_z_' + suffix + '.txt')
    
    subject_path = '' + suffixed_folder_path + 'subject_' + suffix + '.txt'
    
    y_path = '' + suffixed_folder_path + 'y_' + suffix + '.txt'
    
    # Reads CSVs.
    signal_dfs = []
    for path in signal_paths:
        signal_dfs.append(pd.read_csv(path, sep='\s+', header=None))
        
    subject_data_df = pd.read_csv(subject_path, sep='\s+', header=None, names=['Subject_id'])
    
    y_data_df = pd.read_csv(y_path, sep='\s+', header=None, names=['Activity'])
    
    return {
            'signal_dfs' : signal_dfs, 
            'subject_dfs' : subject_data_df, 
            'y_df' : y_data_df
            }
    
    
# Splits training data into train and validation set.
def split_into_train_and_validation(train_percentage, random_seed, df):
        
   subject_data_df = df['subject_dfs']
   signals_df = df['signal_dfs']
   y_df = df['y_df']
    
   (train_indexes, validation_indexes) = split_index(train_percentage, random_seed, subject_data_df)
    
   train_signals_dfs = select_by_index(signals_df, train_indexes)
   validation_signals_dfs = select_by_index(signals_df, validation_indexes)
    
   y_train_df = y_df.iloc[train_indexes]
   y_validation_df = y_df.iloc[validation_indexes]
    
   train_set = {'signal_dfs' : train_signals_dfs, 'y_df' : y_train_df}
   validation_set = {'signal_dfs' : validation_signals_dfs, 'y_df' : y_validation_df}
    
   return (train_set, validation_set)


def remap_y_labels(data_set):
    remapped_data_set = {}
    remapped_data_set['signal_dfs'] = data_set['signal_dfs']
    remapped_data_set['y_df'] = data_set['y_df'] - 1
    return remapped_data_set


def count_classes(labels_df):
    return labels_df.iloc[:,0].nunique()
   
    
def shuffle_dataset(arranged_dataset, random_seed):
    arranged_signals_df = arranged_dataset['signal_dfs']
    arranged_y_df = arranged_dataset['y_df']
    
    shuffled_y_df = arranged_y_df.sample(frac=1, replace=False, random_state=random_seed)
    shuffled_index = shuffled_y_df.index
    
    print('shuffled_index.size =', shuffled_index.size)
    
    shuffled_signals_df = []
    for arranged_signal_df in arranged_signals_df:
        shuffled_signals_df.append(arranged_signal_df.loc[shuffled_index,:])
        
    return {'signal_dfs' : shuffled_signals_df, 'y_df' : shuffled_y_df}


# Convert signals to (,,) TODO completer par (batch size, time steps, etc)
# one_hot_encode_y : bool, optional (default=True)
def convert_to_nd_arrays(data_set, n_classes, one_hot_encode_y=True):
    
    signal_dfs = data_set['signal_dfs']
    y_df = data_set['y_df']
    
    signal_nps = []
    for signal_df in signal_dfs:
        signal_nps.append(signal_df.to_numpy())
        
    stacked_signals = np.stack(signal_nps, axis=2)
    print('stacked_signals.shape : ', stacked_signals.shape)
    
    y_np = y_df.to_numpy()
    y = y_np
    
    # One-hot-encodes y if asked for.
    if one_hot_encode_y :
        y = kutils.to_categorical(y_np, num_classes=n_classes)
    
    return {'x' : stacked_signals, 'y' : y}
   

def split_index(train_percentage, seed_number, subject_data_df):
    # Gets volunteer list of ids
    volunteer_ids = subject_data_df['Subject_id'].unique()
    
    np.random.seed(seed_number)
    np.random.shuffle(volunteer_ids)
    
    n_subjects = np.size(volunteer_ids)
    n_subject_train = min(n_subjects-1, math.ceil(n_subjects * train_percentage / 100.))
    
    train_set_ids = volunteer_ids[:n_subject_train]
    validation_set_ids = volunteer_ids[n_subject_train:]
    
    print('train_set_ids : ', train_set_ids)
    print('validation_set_ids : ', validation_set_ids)
    
    train_idxs = subject_data_df[subject_data_df['Subject_id'].isin(train_set_ids)].index
    validation_idxs = subject_data_df[subject_data_df['Subject_id'].isin(validation_set_ids)].index
    
    train_size = train_idxs.size
    validation_size = validation_idxs.size
    
    print('train_size = ', train_size)
    print('validation_size = ', validation_size)
    train_percentage = train_size / (train_size + validation_size)
    validation_percentage = validation_size / (train_size + validation_size)
    print('By training data percentage : train={}%, validation={}%'.format(train_percentage, validation_percentage))
    
    return (train_idxs, validation_idxs)


def select_by_index(signals_df, idxs):
    selected_signals = []
    for signal_df in signals_df:
        selected_signals.append(signal_df.iloc[idxs])
    return selected_signals