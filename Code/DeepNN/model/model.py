
# coding: utf-8



import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt


from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


LOGGING = True
DATA_SET_VER = 2011
PERMUTE_DATA_FLAG = True
EXP_TYPE = 'out_of_sample'
# number of times that each experiment is repeated
R = 20
# the default value for fraction of data set to be used for training
TRAIN_RATIO = 2/3


np.set_printoptions(precision=3)




# for logging and debugging purposes
def log(logging_str):
    if LOGGING:
        print(logging_str)
        
# reading the raw or the preprocessed data set 
def read_data_set():
    current_dir = os.getcwd()
    if DATA_SET_VER == 2011:
         data_set_file_name = current_dir + '/dataset/bank/bank-full-preprocessed.csv'
    elif DATA_SET_VER == 2014:
         data_set_file_name = current_dir + '/dataset/bank-additional/bank-additional-full-preprocessed.csv';
    else:
         data_set_file_name = current_dir + '/dataset/synthetic/synthetic.csv';

    data_set_df = pd.read_csv(data_set_file_name,delimiter = ';')
    return data_set_df 




# returns the training and testing features and target variables based on the specifired training ratio
def get_data(train_ratio = TRAIN_RATIO):
    data_set_df = read_data_set()
    if PERMUTE_DATA_FLAG:
        data_set_df = shuffle(data_set_df)
    if EXP_TYPE == "out_of_sample":
        data_set_rows_no = data_set_df.shape[0]
        train_samples_max_index = int(data_set_rows_no * train_ratio)
        train_data_df = data_set_df[0:train_samples_max_index]
        test_data_df = data_set_df[train_samples_max_index:]
        train_y = train_data_df['y'].values
        train_x = train_data_df.drop(['y'], axis=1).values
        test_y = test_data_df['y'].values
        test_x = test_data_df.drop(['y'], axis=1).values
    else:
        train_data_df = data_set_df
        test_data_df = data_set_df.copy()
        train_y = train_data_df['y'].values
        train_x = train_data_df.drop(['y'], axis=1).values
        test_y = test_data_df['y'].values
        test_x = test_data_df.drop(['y'], axis=1).values
    return train_x,train_y,test_x,test_y




# makes a NN model and returns back the predictred probabilities based on the values passed for the hidden layer size and the activation function 
def get_skl_nn_predictions(train_x, train_y, test_x, hls_size=(5, 5), activation_function = 'relu'):
    skl_nn_model = MLPClassifier(hidden_layer_sizes = hls_size, activation = activation_function)
    skl_nn_model.fit(train_x,train_y)
    pred_y = skl_nn_model.predict_proba(test_x)
    return pred_y

# a utility function to make the hidden layers size datastructure based on the sklearn documentation
def get_hl_tuple(nn_depth, nn_width):
    hl_list = []
    for i in range(nn_depth):
        hl_list.append(nn_width)
    hl_tuple = tuple(hl_list)
    return hl_tuple




# makes a LR model and returns back the predictred probabilities
def get_skl_lr_predictions(train_x,train_y,test_x):
    skl_lr_model = LogisticRegression()
    skl_lr_model.fit(train_x,train_y)
    #print(skl_lr_mode.coef_)
    #print(skl_lr_mode.intercept_)
    pred_y = skl_lr_model.predict_proba(test_x)
    return pred_y




# studying the effect of the network capacity 
capacity_list = range(1,7)
capacity_list_len = len(capacity_list)

skl_lr_auc_score = 0 
for i in range(R):
    train_x,train_y,test_x,test_y = get_data()
    pred_y = get_skl_lr_predictions(train_x, train_y, test_x)
    skl_lr_auc_score += roc_auc_score(y_true = test_y, y_score = pred_y[:,1])
skl_lr_auc_score /= R
lr_line  = np.ones(capacity_list_len) * skl_lr_auc_score
plt.figure(figsize=(20, 12))
plt.ylabel('auc score')
plt.xlabel('nn_depth')
plt.plot(capacity_list, lr_line, label='logistic_regression')

auc_scores =[[0 for x in capacity_list] for y in capacity_list] 
for nn_width in capacity_list:
    for nn_depth in capacity_list:
        for i in range(R):
            pred_y = get_skl_nn_predictions(train_x, train_y, test_x, hls_size = get_hl_tuple(nn_depth, nn_width))
            skl_nn_auc_score = roc_auc_score(y_true = test_y, y_score = pred_y[:,1])
            auc_scores[nn_width-1][nn_depth-1] += skl_nn_auc_score
        auc_scores[nn_width-1][nn_depth-1] /= R
    plt.plot(capacity_list,auc_scores[nn_width-1],label = "nn_width = " + str(nn_width))

plt.legend(loc="lower right", fontsize=10);
plt.show()




# studying the effect of the network capacity -x axis is for the network width 
plt.figure(figsize=(20, 12))
plt.ylabel('auc score')
plt.xlabel('nn_width')
plt.plot(capacity_list, lr_line, label='logistic_regression')
for nn_depth in capacity_list:
    plt.plot(capacity_list,[row[nn_depth-1] for row in auc_scores],label = "nn_depth = " + str(nn_depth))

plt.legend(loc="lower right", fontsize=10);
plt.show()




#studying the effect of activation function choice and the training data ratio
plt.figure(figsize=(20, 12))

tr_ratio_list = list(np.arange(0.1,1.0,0.1))
tr_ratio_list_len = len(tr_ratio_list)

lr_auc_scores = []
skl_nn_logistic_auc_scores = []
skl_nn_tanh_auc_scores = []
skl_nn_relu_auc_scores = []

for tr_ratio in tr_ratio_list:
    skl_lr_auc_score = 0
    skl_nn_logistic_auc_score = 0
    skl_nn_tanh_auc_score = 0
    skl_nn_relu_auc_score = 0
    for i in range(R):
        train_x,train_y,test_x,test_y = get_data(train_ratio = tr_ratio)
        pred_y = get_skl_lr_predictions(train_x, train_y, test_x)
        skl_lr_auc_score += roc_auc_score(y_true = test_y, y_score = pred_y[:,1])
        pred_y = get_skl_nn_predictions(train_x, train_y, test_x, hls_size = get_hl_tuple(3,10), activation_function= 'logistic')
        skl_nn_logistic_auc_score += roc_auc_score(y_true = test_y, y_score = pred_y[:,1])
        pred_y = get_skl_nn_predictions(train_x, train_y, test_x, hls_size = get_hl_tuple(3,10), activation_function= 'tanh')
        skl_nn_tanh_auc_score += roc_auc_score(y_true = test_y, y_score = pred_y[:,1])        
        pred_y = get_skl_nn_predictions(train_x, train_y, test_x, hls_size = get_hl_tuple(3,10))
        skl_nn_relu_auc_score += roc_auc_score(y_true = test_y, y_score = pred_y[:,1])
    
    lr_auc_scores.append(skl_lr_auc_score/R)
    skl_nn_logistic_auc_scores.append(skl_nn_logistic_auc_score/R)
    skl_nn_tanh_auc_scores.append(skl_nn_tanh_auc_score/R)
    skl_nn_relu_auc_scores.append(skl_nn_relu_auc_score/R)
    
plt.ylabel('auc score')
plt.xlabel('training_data_ratio')
plt.plot(tr_ratio_list, lr_auc_scores, label = "lr")
plt.plot(tr_ratio_list, skl_nn_logistic_auc_scores, label = "nn logistic 3-10")
plt.plot(tr_ratio_list, skl_nn_tanh_auc_scores, label = "nn tanh 3-10")
plt.plot(tr_ratio_list, skl_nn_relu_auc_scores, label = "nn relu 3-10")
plt.legend(loc="lower right", fontsize=10);
plt.show()

