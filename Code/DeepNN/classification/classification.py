
# coding: utf-8


import time
import pandas as pd
import numpy as np 
import os


from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


# data set version , 2011 to load the 2011 version of the data set, 2014 for loading the 2014 version
# any other value for loading other arbitrary binary classification datasets (for which the class label is the last column of the data set )
DATA_SET_VER = 2014
PERMUTE_DATA_FLAG = True
EXP_TYPE = 'out_of_sample'
# number of rounds that the experiment is repeated
ROUNDS_NO = 20
TRAIN_RATIO = 2/3
PROBABILITY_THRESHOLD = 0.5 #  a simple method to alleviate the class imbalance problem
# sizes of hidden layers
HIDDEN_LAYER_SIZES = (10,10,10)
ACT_FUNCTION = 'relu'


# returns the training and testing features and target variables based on the specified training ratio
def get_data(train_ratio = TRAIN_RATIO):
    current_dir = os.getcwd()
    if DATA_SET_VER == 2011:
         data_set_file_name = current_dir + '/dataset/bank/bank-full-preprocessed.csv'
    elif DATA_SET_VER == 2014:
         data_set_file_name = current_dir + '/dataset/bank-additional/bank-additional-full-preprocessed.csv';
    else:
         data_set_file_name = current_dir + '/dataset/synthetic/synthetic.csv';

    data_set_df = pd.read_csv(data_set_file_name,delimiter = ';')
    if PERMUTE_DATA_FLAG:
        data_set_df = shuffle(data_set_df)
    data_set_rows_no = data_set_df.shape[0]
    if EXP_TYPE == "out_of_sample":
        train_samples_max_index = int(data_set_rows_no * train_ratio)
        train_data_df = data_set_df[0:train_samples_max_index]
        test_data_df = data_set_df[train_samples_max_index:]
    else:
        train_data_df = data_set_df
        test_data_df = data_set_df.copy()
    train_y = train_data_df['y'].values
    train_x = train_data_df.drop(['y'], axis=1).values
    test_y = test_data_df['y'].values
    test_x = test_data_df.drop(['y'], axis=1).values

    return train_x,train_y,test_x,test_y


# makes and trains a NN model and returns  the predicted probabilities back
# based on the values passed for the hidden layer size and the activation function 
def get_nn_predictions(train_x, train_y, test_x, hls_sizes = HIDDEN_LAYER_SIZES, activation_function = ACT_FUNCTION):
    skl_nn_model = MLPClassifier(hidden_layer_sizes = hls_sizes, activation = activation_function , max_iter= 100)
    skl_nn_model.fit(train_x,train_y)
    pred_y = skl_nn_model.predict_proba(test_x)
    return pred_y

# returns the prediction accuracy based the test data and the predicted probabilities 
def get_prediction_accuracy(test_y, predicted_prob):
  predicted_y =  predicted_prob[:,0] < PROBABILITY_THRESHOLD
  prediction_accuracy = sum(predicted_y == test_y) / predicted_y.shape[0]
  return(prediction_accuracy)



sum_accuracy = 0
for r in range(ROUNDS_NO):
  train_x,train_y,test_x,test_y = get_data()
  predicted_prob = get_nn_predictions(train_x, train_y, test_x)
  prediction_accuracy = get_prediction_accuracy(test_y, predicted_prob)
  sum_accuracy += prediction_accuracy;
model_accuracy = sum_accuracy * 100/ ROUNDS_NO;
print("Accuracy : %.2f percent "% model_accuracy) ;

