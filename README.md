# human-activity-recognition-LSTM
Training a LSTM neural network on the UCI Human Activity Recognition dataset.

UCI Human Activity Recognition dataset :
https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

"Human Activity Recognition database built from the recordings of 30 subjects performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors."

The  subjects performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone on the waist.
The aim is to predict which activity has been performed from the smartphone's accelerometer and gyroscope time series data.

The model is a neural network with two layers of LSTM neurons, and two fully connected layers above it.
It achieves 91.4% accuracy on test set.

The main with the used parameters can be found at './src/run.py'.

A few plots (loss evolution during training and confusion matrices) can be found in './data/working_data/'.
