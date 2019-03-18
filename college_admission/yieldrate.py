# https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier # CANNOT HANDLE CATEGORICAL VARIABLES
import h2o
from h2o.estimators import H2ORandomForestEstimator


import pandas as pd
import numpy as np
import random 
import csv

# Set random seed
np.random.seed(0)


adm = pd.read_csv('admissions.csv')
print(adm.shape) # (669, 28)


# column names
print(adm.columns)

# Look at some observations
# print(adm.head())
# The first column is the row number so we don't need it
df = adm.drop(['Unnamed: 0'], axis = 1)
Enrolled = adm['Enrolled']
print("Overall distribution of Enrolled yes or no")
Enrolled_table = df.Enrolled.value_counts()
print(Enrolled_table)
print("Yield Rate is", round(Enrolled_table[1]/ len(Enrolled),2 ) * 100 , "%")
##############################
# Create Training and Test Data
df['is_train'] = np.random.uniform(0,1, len(df)) <= .75
print(df.head())

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Features are all except our outcome, Enrolled, and Deposit because Depsit is directly realted to our outcome
features = df.columns.difference(['Deposit', 'Enrolled'])
len(features) # 26 features


############################## H2o #########################
# https://medium.com/tech-vision/random-forest-classification-with-h2o-python-for-beginners-b31f6e4ccf3c
h2o.init()

data = h2o.import_file('admissions.csv')
training_columns = ['ACADEMIC_RATING', 'Application_Type', 'Attempt_Email',
       'Attempt_Facebook', 'Attempt_None', 'Attempt_Phone', 'Attempt_Text',
       'Attempt_Voicemail', 'Brochure_Sent', 'CITIZENSHIP', 'Condition',
       'Contactor_ID', 'Contactor_Role', 'Dept_Choice', 'Early_Admit',
       'FIRST_GEN', 'LEADERSHIP', 'RACE', 'Response_Email',
       'Response_Facebook', 'Response_None', 'Response_Phone', 'Response_Text',
       'SERVICE', 'SEX']
response_column = 'Enrolled'

# Split data into train and testing
train, test = data.split_frame(ratios=[0.8])

# Define model
model = H2ORandomForestEstimator(ntrees=50, max_depth=20, nfolds=10)

# Train model
model.train(x=training_columns, y=response_column, training_frame=train)


# Model performance
performance = model.model_performance(test_data=test)

print(performance)