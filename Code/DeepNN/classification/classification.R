require(mxnet)

DATA_SET_VER = 2014;
PERMUTE_DATA_FLAG = TRUE;
EXP_TYPE = 'out_of_sample';
# number of rounds that the experiment is repeated
ROUNDS_NO = 20;
TRAIN_RATIO = 2/3;
PROBABILITY_THRESHOLD = 0.5;
# sizes of hidden layers
HIDDEN_LAYER_SIZES = c(10, 10, 10);

# returns the training and testing features and target variables based on the specified training ratio
get_data = function(train_ratio)
{
  if(missing(train_ratio))
    train_ratio = TRAIN_RATIO;
  if(DATA_SET_VER == 2011)
  {
    data_set_file_name = './dataset/bank/bank-full-preprocessed.csv';
  }
  else if(DATA_SET_VER == 2014)
  {
    data_set_file_name = './dataset/bank-additional/bank-additional-full-preprocessed.csv';
  }
  else
  {
    data_set_file_name = './dataset/synthetic/synthetic.csv';
  }
  data_set_data_frame = read.csv(data_set_file_name, header = TRUE, sep = ';');
  data_set_rows_no = dim(data_set_data_frame)[1];
  if(PERMUTE_DATA_FLAG == TRUE)
  {
    set.seed(as.numeric(Sys.time()) / 100000)
    data_set_data_frame = data_set_data_frame[sample(nrow(data_set_data_frame)),];
  }
  if(EXP_TYPE == 'out_of_sample')
  {
    train_samples_max_index = floor(train_ratio * data_set_rows_no);
    train_data_frame = data_set_data_frame[1 : train_samples_max_index , ];
    test_data_frame = data_set_data_frame[(train_samples_max_index+1) : data_set_rows_no , ];
  }
  else
  {
    train_data_frame = data_set_data_frame;
    test_data_frame = data_set_data_frame;
  }
  columns_no = dim(train_data_frame)[2];
  train_y = train_data_frame[ , columns_no];
  train_x = train_data_frame[ , 1 : (columns_no - 1)];
  test_y = test_data_frame[ , columns_no];
  test_x = test_data_frame[ , 1 : (columns_no - 1)];
  
  return(list(trx = train_x , try = train_y , tex = test_x, tey = test_y))
}

# makes and trains a NN model and returns  the predicted probabilities back
# based on the values passed for the hidden layer size and the activation function 
get_nn_predictions = function(train_x, train_y, test_x, hidden_layer_sizes, activation_function)
{
  if(missing(hidden_layer_sizes))
    hidden_layer_sizes = HIDDEN_LAYER_SIZES;
  if(missing(activation_function))
    activation_function = 'relu'
  
  train_x = data.matrix(train_x)
  test_x =  data.matrix(test_x)
  
  # defigning the nn
  input = mx.symbol.Variable("data")
  for(hl_no in 1 : length(hidden_layer_sizes))
  { 
    hl = mx.symbol.FullyConnected(input, name = paste("hl", hl_no, sep = " ") , num_hidden = hidden_layer_sizes[hl_no])
    input = mx.symbol.Activation(hl, name = paste("act", hl_no, sep = " "), act_type = activation_function)
  }
  ol = mx.symbol.FullyConnected(input, name = "output_layer", num_hidden = 2)
  softmax = mx.symbol.SoftmaxOutput(ol, name = "sm")
  # training the nn
  devices = mx.cpu()
  mx.set.seed(as.numeric(Sys.time()) / 100000)
  NN <- mx.model.FeedForward.create(softmax, X=train_x, y=train_y,
                                    ctx = devices, num.round = 100, array.batch.size = 100, 
                                    initializer = mx.init.uniform(0.1), learning.rate=0.01, momentum = 0.9,
                                    eval.metric = mx.metric.accuracy)
  

  
  predicted_prob = predict(NN, test_x)
  return(predicted_prob)
}

# returns the prediction accuracy based the test data and the predicted probabilities 
get_prediction_accuracy = function(test_y, predicted_prob)
{
  predicted_y = as.numeric(predicted_prob[1,] < PROBABILITY_THRESHOLD)
  prediction_accuracy = sum(predicted_y == test_y) / length(predicted_y);
  return(prediction_accuracy)
}

sum_accuracy = 0
for( r in 1:ROUNDS_NO)
{
  data_list = get_data()
  train_x = data_list$trx
  train_y = data_list$try
  test_x = data_list$tex
  test_y = data_list$tey
  predicted_prob = get_nn_predictions(train_x, train_y, test_x);
  prediction_accuracy = get_prediction_accuracy(test_y, predicted_prob);
  sum_accuracy = sum_accuracy + prediction_accuracy;
}
model_accuracy = sum_accuracy / ROUNDS_NO;
sprintf("Accuracy : %.2f percent ", model_accuracy * 100 )
