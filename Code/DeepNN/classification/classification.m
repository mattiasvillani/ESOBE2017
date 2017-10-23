clear;


global DATA_SET_VER PERMUTE_DATA_FLAG EXP_TYPE ROUNDS_NO PROBABILITY_THRESHOLD HIDDEN_LAYER_SIZES TRAIN_RATIO ACT_FUNCTION;

% data set version , 2011 to load the 2011 version of the data set
% 2014 for loading the 2014 version
% any other value for loading other arbitrary binary classification datasets (for which
% the class label is the last column of the data set )
DATA_SET_VER = 2014; 
PERMUTE_DATA_FLAG = true;
EXP_TYPE = 'out_of_sample';
% number of rounds that the experiment is repeated
ROUNDS_NO = 1;
TRAIN_RATIO = 2/3;
PROBABILITY_THRESHOLD = 0.5; % a simple method to alleviate the class imbalance problem
% sizes of hidden layers
HIDDEN_LAYER_SIZES = [10,10,10];
ACT_FUNCTION = 'poslin'; % The relu activation function 



tic()
sum_accuray = 0;
for r = 1 : ROUNDS_NO
    [train_x, train_y, test_x, test_y] = get_data();
    predicted_prob = get_nn_predictions(train_x, train_y, test_x);
    prediction_accuracy = get_prediction_accuracy(test_y, predicted_prob);
    sum_accuray = sum_accuray + prediction_accuracy;
end
model_accuracy = sum_accuray / ROUNDS_NO;
fprintf('Accuracy : %.2f percent', model_accuracy * 100 );




% returns the training and t esting features and target variables based on the specified training ratio
function [train_x, train_y, test_x, test_y] = get_data(train_ratio)
    global TRAIN_RATIO DATA_SET_VER PERMUTE_DATA_FLAG EXP_TYPE 
    if nargin < 1
        train_ratio = TRAIN_RATIO;
    end
    if DATA_SET_VER == 2011
        data_set_file_name = './dataset/bank/bank-full-preprocessed.csv';
    elseif DATA_SET_VER == 2014
        data_set_file_name = './dataset/bank-additional/bank-additional-full-preprocessed.csv';
    else
        data_set_file_name = './dataset/synthetic/synthetic.csv';
    end 
    A = importdata(data_set_file_name, ';');
    data_set_data_frame = A.data;
    data_set_rows_no = size(data_set_data_frame , 1);
    if PERMUTE_DATA_FLAG == true
        rng(toc());
        permutation_pattern = randperm(data_set_rows_no)';
        data_set_data_frame = data_set_data_frame(permutation_pattern , :);
    end
    if strcmp(EXP_TYPE,'out_of_sample')
        train_samples_max_index = floor(train_ratio * data_set_rows_no);
        train_data_frame = data_set_data_frame(1: train_samples_max_index , :);
        test_data_frame = data_set_data_frame(train_samples_max_index + 1 : end , :);
    else
        train_data_frame = data_set_data_frame;
        test_data_frame = data_set_data_frame;
    end
    columns_no = size(train_data_frame , 2);
    train_y = train_data_frame(: , columns_no);
    train_x = train_data_frame(: , 1 : columns_no - 1);
    test_y = test_data_frame(: , columns_no);
    test_x = test_data_frame(: , 1 : columns_no - 1);
end

% makes and trains a NN model and returns  the predicted probabilities back
% based on the values passed for the hidden layer size and the activation function 
function [predicted_prob] = get_nn_predictions(train_x, train_y, test_x, hidden_layer_sizes, activation_function)
    global HIDDEN_LAYER_SIZES ACT_FUNCTION;
    if nargin == 3
        hidden_layer_sizes = HIDDEN_LAYER_SIZES;
        activation_function = ACT_FUNCTION; 
    elseif nargin == 4
        activation_function = ACT_FUNCTION;
    end
    
    % one hot encoding of the target variable
    train_y_one_hot_encoded = zeros(size(train_y, 1), 2 );
    zero_row_indexes = train_y == 0;
    one_row_indexes = train_y == 1;
    train_y_one_hot_encoded(zero_row_indexes , 1 ) = 1;
    train_y_one_hot_encoded(one_row_indexes , 2 ) = 1;

    % the train function requires its parameters in column-based matrixes
    train_x = train_x';
    train_y_one_hot_encoded = train_y_one_hot_encoded';
    
    % defigning the nn
    NN = patternnet(hidden_layer_sizes);
    NN.divideParam.trainRatio = 1.0;
    NN.divideParam.valRatio = 0.0;
    NN.trainParam.epochs = 100;
    for i = 1 : size(hidden_layer_sizes , 2)
        NN.layers{i}.transferFcn = activation_function;
    end
    
    % training the nn
    NN = train(NN,train_x, train_y_one_hot_encoded);
    
    % converting the test data to column based format and passing it
    % through the trained network and getting the generated class probabilites
    test_x = test_x';
    predicted_prob = NN(test_x);
end





% returns the prediction accuracy based the test data and the predicted probabilities 
function prediction_accuracy = get_prediction_accuracy(test_y, predicted_prob)
    global PROBABILITY_THRESHOLD;
    predicted_y = predicted_prob(1, :) < PROBABILITY_THRESHOLD;
    prediction_accuracy = sum(predicted_y == test_y') / size(predicted_y , 2);
end


