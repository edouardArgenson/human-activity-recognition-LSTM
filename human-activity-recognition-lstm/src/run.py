# -*- coding: utf-8 -*-

#%%

from time import time

import tensorflow.keras.backend as K
from sklearn.metrics import accuracy_score

from load_and_prepare_data import load_data_and_build_features, load_class_labels, prepare_working_data_directories
from lstm_model import build_model, train_model, predict, load_best_model
from plot_tools import plot_and_save_loss_training, plot_and_save_confusion_matrix

#%%

def run():
    data_path = '../data/UCI_HAR_dataset/'
    
    # Prepare data parameters.
    train_percentage = 80
    random_seed_for_split = 0   
    random_seed_for_shuffle = 0
    
    # Neural network parameters.
    n_lstm_1 = 40
    n_lstm_2 = 20
    n_dense_3 = 20
    n_dense_out = 6
    
    # Training parameters.
    epochs = 50
    batch_size = 32
    
    # Load and prepare data.
    (train_set, validation_set, test_set) = load_data_and_build_features(
            train_percentage, 
            random_seed_for_split, 
            random_seed_for_shuffle,
            data_path
            )
    
    paths = prepare_working_data_directories(time())
    
    # Build and train model.
    K.clear_session()
    model = build_model(n_lstm_1, n_lstm_2, n_dense_3, n_dense_out)
    histo = train_model(model, train_set, validation_set, epochs, batch_size, paths['checkpoint_path'])
    
    plot_and_save_loss_training(histo, paths['training_loss_plt_path'])
    
    # Find and load model with min validation loss.
    load_best_model(model, histo, paths['checkpoint_dir'])
    
    # Use best model to make test predictions.
    test_preds = predict(model, test_set['x'])
    
    # Compute model accuracy on test set.
    accuracy_score_test = accuracy_score(test_set['y'], test_preds)
    print('Model accuracy score on test set : {}%'.format(100*accuracy_score_test))
    
    classes = load_class_labels(data_path)
    plot_and_save_confusion_matrix(test_set['y'], test_preds, classes, True, paths['confusion_matrix_plt_path'])


if __name__ == "__main__":
    # executes only if run as a script
    run()