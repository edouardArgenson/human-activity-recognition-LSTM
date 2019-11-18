# -*- coding: utf-8 -*-

#%%

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

#%%    

# Plots train + validation loss.
# Saves plot as .png file
def plot_and_save_loss_training(hist, file_path):
    training_loss = hist.history['loss']
    validation_loss = hist.history['val_loss']
    validation_acc = hist.history['val_accuracy']
    
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(training_loss, '.-', label='training_loss')
    ax_loss.plot(validation_loss, '.-', label='validation_loss')
    ax_loss.plot(validation_acc, '.-', label='validation accuracy')
    ax_loss.set_title('Train+Validation loss')
    ax_loss.legend()
    #fig_loss.show()
    
    filename = file_path + '.png'
    
    print('saving {} ...'.format(filename))
    fig_loss.savefig(filename)
    print('saved.')


# Wrapper saving figure from plot_confusion_matrix()
def plot_and_save_confusion_matrix(y_true, y_pred, class_names, normalize, file_path):
    fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize)
    filename = file_path + '.png'
    print('saving {} ...'.format(filename))
    fig.savefig(filename)
    print('saved.')


# Code from :
# www.scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig