

# new version of the preprocessing file
import pandas as pd
import numpy as np 
import os
from scipy.stats import zscore 

CATEGORICAL_VALUES_CODING = 'onehot' # onehot and ordinal coding 
DATA_SET_VER = 2011
REPLACING_VALUE_FOR_MISSING_VALUES = -10.0
LOGGING = True
NORMALIZE_DATA_FLAG = True
NORMALIZE_ONEHOT_FEATURES = False



#some utility functions
def log(logging_str):
    if LOGGING:
        print(logging_str)
        
# reading the raw or the preprocessed data set 
def read_data_set(read_preprocessed_data = False):
    current_dir = os.getcwd()
    if DATA_SET_VER == 2014:
        if read_preprocessed_data == False:
            data_set_file_name = current_dir + "/dataset/bank-additional/bank-additional-full.csv"
        else:
            data_set_file_name = current_dir + "/dataset/bank-additional/bank-additional-full-preprocessed.csv"

    else:
        if read_preprocessed_data == False:
            data_set_file_name = current_dir + "/dataset/bank/bank-full.csv"
        else:
            data_set_file_name = current_dir + "/dataset/bank/bank-full-preprocessed.csv"

    data_set_df = pd.read_csv(data_set_file_name,delimiter = ';')
    return data_set_df

def save_data_set(data_set_df):
    current_dir = os.getcwd()
    if DATA_SET_VER == 2014:
        data_set_file_name = current_dir + "/dataset/bank-additional/bank-additional-full-preprocessed.csv"
    else:
        data_set_file_name = current_dir + "/dataset/bank/bank-full-preprocessed.csv"
    data_set_df.to_csv(data_set_file_name , sep = ';' , index = False)
    



def categorical_to_numerical(data_set_df):
    if CATEGORICAL_VALUES_CODING == 'ordinal':
        replacing_format = {
         "job" : { "admin." :  0, "blue-collar" : 1, "technician" : 2, "services" : 3, "management" : 4, "retired" :  5, "entrepreneur" : 6, "self-employed" : 7, "housemaid" : 8, "unemployed" : 9, "student" :   10},
         "marital" : {"single" : 0, "married" : 1, "divorced" : 2},
         "education" :  {"illiterate" : 0, "basic.4y" : 1, "basic.6y" : 2, "basic.9y" : 3,"primary" : 4 , "secondary" : 5, "high.school" : 6, "tertiary" : 7,  "professional.course" : 8, "university.degree" : 10},
         "default" : { "no" : 0, "yes" : 1},
         "housing" : { "no" : 0, "yes" : 1},
         "loan" : { "no" : 0, "yes" : 1},
         "contact" : { "cellular" : 0, "telephone" : 1},
         "month":     {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12},
         "day_of_week": {"mon": 1, "tue": 2, "wed": 3, "thu": 4,"fri": 5},
         "contact" : {"other" : 0, "telephone" : 1 , "cellular" :2},
         "poutcome" : {"other" : 0, "nonexistent" : 1, "failure" : 2, "success" : 3  },
         "y" : { "no" : 0, "yes" : 1}
         }
    else:
        replacing_format = {
         "education" :  {"illiterate" : 0, "basic.4y" : 1, "basic.6y" : 2, "basic.9y" : 3,"primary" : 4 , "secondary" : 5, "high.school" : 6, "tertiary" : 7,  "professional.course" : 8, "university.degree" : 10},
         "month":     {'jan' : 1, 'feb' : 2, 'mar' : 3, 'apr' : 4, 'may' : 5, 'jun' : 6, 'jul' : 7, 'aug' : 8, 'sep' : 9, 'oct' : 10, 'nov' : 11, 'dec' : 12},
         "day_of_week": {"mon": 1, "tue": 2, "wed": 3, "thu": 4,"fri": 5},
         "y" : { "no" : 0, "yes" : 1}
         }
     
    data_set_df.replace(replacing_format, inplace=True)
    if CATEGORICAL_VALUES_CODING == 'onehot':
        data_set_df = pd.get_dummies(data_set_df)
    return data_set_df



# The averaging mode should also be added and supported!
def process_missing_values(data_set_df):
    '''
    Note that:
    for the 2014 version of the dataset, percentage of missing values for each of the columns is as follows : 
    age : 0.0, job : 0.6, marital : 0.0, education : 4.1, default : 0.0, balance : 0.0, housing : 0.0, loan : 0.0, contact : 28.8, day : 0.0, month : 0.0, duration : 0.0, campaign : 0.0, pdays : 0.0, previous : 0.0, poutcome : 81.7
    
    for the 2011 version of the dataset, percentage of missing values for each of the columns is as follows : 
    age : 0.0, job : 0.8, marital : 0.2, education : 4.2, default : 20.9, housing : 2.4, loan : 2.4, contact : 0.0, month : 0.0, day_of_week : 0.0, duration : 0.0, campaign : 0.0, pdays : 0.0, previous : 0.0, poutcome : 0.0, emp.var.rate : 0.0, cons.price.idx : 0.0, cons.conf.idx : 0.0, euribor3m : 0.0, nr.employed : 0.0
    '''
    removing_columns_names = []
    #removing_columns_names = ['poutcome','job','education']
    if CATEGORICAL_VALUES_CODING == 'ordinal':
        replacing_columns_names = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y', 'balance', 'day']
    else:
        replacing_columns_names = ['age', 'duration', 'pdays', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'balance', 'day']
    for column_name in removing_columns_names: 
        data_set_df.drop(column_name, 1, inplace = True)
    for column_name in replacing_columns_names: 
        if column_name in list(data_set_df.columns.values):
            for i in range(len(data_set_df[column_name])):
                if str(data_set_df[column_name][i]) == 'unknown':
                    data_set_df.iloc[i, data_set_df.columns.get_loc(column_name)] = REPLACING_VALUE_FOR_MISSING_VALUES
    return(data_set_df)



data_set_df = read_data_set()
data_set_df = categorical_to_numerical(data_set_df)
data_set_df = process_missing_values(data_set_df)

# Moving the Y column to end
columns_ordering = list(data_set_df.columns)
columns_ordering.remove('y')
columns_ordering.append('y')
data_set_df = data_set_df[columns_ordering]



# to apply the data type modifications
save_data_set(data_set_df)
data_set_df = read_data_set(read_preprocessed_data = True)


#apply normalization and final saving of the modified dataset
if NORMALIZE_DATA_FLAG:
    column_names = list(data_set_df.columns)
    column_names.remove('y')
    if NORMALIZE_ONEHOT_FEATURES == False:
        column_names_temp = []
        for column_name in column_names:
            if not '_' in column_name:
                column_names_temp.append(column_name)
        column_names = column_names_temp
    data_set_df[column_names] = data_set_df[column_names].apply(zscore)
    
save_data_set(data_set_df)



# A helper cell and not necessary to be run 
def percent_of_missing_values(data_set_df,column_name):
    number_of_missing_values = [(str(data_set_df[column_name][i]) == 'unknown') for i in range(len(data_set_df[column_name]))].count(True)
    total_number_of_values = len(data_set_df[column_name])
    return number_of_missing_values * 100.0 / total_number_of_values
    

def missing_values_frequency_info(data_set_df):
    for column_name in data_set_df:
        print(str(percent_of_missing_values(data_set_df,column_name)) + " percent of elements in column " + str(column_name) + " are missing.")
        
# to check that no missing data is remaining
log(data_set_df.dtypes)

#to check the data range of each column and find a suitable substitute for missing values
def report_min_max(data_set_df):
    return data_set_df.describe().loc[['min','max']]

log(report_min_max(data_set_df))
'''
-10 is a good substitute for the missing values.
Just balance(2011) and cons.conf.idx(2014) columns contain -10 in their data rage.
However, these columns do not have missing values. 
'''

# descriptive statistics of the data set
data_set_df.describe()

