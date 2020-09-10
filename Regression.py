# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime
import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import sys
    
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
seed = 309
# Freeze the random seed
random.seed(seed)
np.random.seed(seed)
train_test_split_test_size = 0.3

# Training settings
alpha = 0.1  # step size
max_iters = 50  # max iterations

def load_data():
    #loads the file 
    #uncomment to test the program using a command line
    file1 = sys.argv[1]   
    
    file = pd.read_csv(file1)
   
    #file = pd.read_csv('diamonds.csv')   
    #print(df)
    #print(df.columns)
    #Exploratory Data Analysis
    #print(file.info())
    #print(file.shape)
    #print(file.describe())
    #correlations = file.corr(method='pearson')
    #print(correlations)
    #skew = file.skew()
    #print(skew)

    
    return file

def preprocess(file):
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    #encoding of categorical values
    file['cut'] = labelencoder.fit_transform(file['cut'])
    file['clarity'] = labelencoder.fit_transform(file['clarity'])
    file['color'] = labelencoder.fit_transform(file['color'])
    
    #plots the correlations
    corr = file.corr(method = 'pearson') # Correlation Matrix
    sns.heatmap(file.corr(), cmap='coolwarm',annot = True)
    #plt.show()
   
    file.drop(['depth'],axis=1,inplace = True)

    return file

def splitandstandardise(data):
    # Split the data into train and test
    train_data, test_data = train_test_split(data, test_size = train_test_split_test_size)
 
    # Pre-process data (both train and test)
    train_data_full = train_data.copy()
    train_data = train_data.drop(["price"], axis = 1)
    train_labels = train_data_full["price"]
    test_data_full = test_data.copy()
    test_data = test_data.drop(["price"], axis = 1)
    test_labels = test_data_full["price"]

    #Standardize the inputs
    train_mean = train_data.mean()
    train_std = train_data.std()
    train_data = (train_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std
    
    test_data['price'] = test_data_full['price']
    train_data['price'] = train_data_full['price']
    
    return test_data, train_data
    
def testfileformation(test_file):
    test_data_copy = test_file.copy()
    test_data_copy_withoutlabel = test_data_copy.drop(["price"],axis=1)
    test_data_copy_label = test_file["price"]
     
    return test_data_copy_withoutlabel, test_data_copy_label, 

def trainingformation(train_file):
    train_data_copy = train_file.copy()
    train_data_copy_withoutlabel = train_data_copy.drop(["price"],axis=1)
    train_data_copy_label = train_file["price"]

    return train_data_copy_withoutlabel, train_data_copy_label

    
def model(X_train_full, X_train_label, regression):
    if(regression =="LinearRegression"):
        from sklearn.linear_model import LinearRegression
        baseline = LinearRegression()
        baseline.fit(X_train_full, X_train_label)
        return baseline
    elif(regression == "KNNRegression"):
        from sklearn.neighbors import KNeighborsRegressor
        baseline = KNeighborsRegressor(n_neighbors= 3)
        baseline.fit(X_train_full, X_train_label)
        return baseline
    elif(regression == "RidgeRegression"):
        from sklearn.linear_model import Ridge
        baseline = Ridge()
        baseline.fit(X_train_full, X_train_label)
        return baseline
    elif(regression == "DecisionTreeRegression"):
        from sklearn.tree import DecisionTreeRegressor
        baseline = DecisionTreeRegressor(max_depth=3)
        baseline.fit(X_train_full, X_train_label)
    elif (regression  == "RandomForestRegression"):
        from sklearn.ensemble import RandomForestRegressor
        baseline = RandomForestRegressor()
        baseline.fit(X_train_full, X_train_label)
    elif (regression == "GradientBoostingRegression"):
        from sklearn.ensemble import GradientBoostingRegressor
        baseline = GradientBoostingRegressor(max_depth=3,n_estimators=200)
        baseline.fit(X_train_full, X_train_label)
    elif (regression == "SGDRegression"):
        from sklearn.linear_model import SGDRegressor
        baseline = SGDRegressor()
        baseline.fit(X_train_full, X_train_label)
    elif (regression == "SVMRegression"):
        from sklearn.svm import SVR
        baseline = SVR()
        baseline.fit(X_train_full, X_train_label)
    elif(regression== "MLPRegression"):
        from sklearn.neural_network import MLPRegressor
        baseline = MLPRegressor(early_stopping=True,learning_rate_init=0.2)
        baseline.fit(X_train_full, X_train_label)
    elif (regression == "LinearSVR"):
        from sklearn.svm import LinearSVR
        baseline = LinearSVR()
        baseline.fit(X_train_full, X_train_label)
    else:
        print("Technique not found:"+ regression)
    
    

    return baseline  
    

if __name__ == '__main__':
    file = load_data()
    #print(file)
    #preprocessing of the file
    p = preprocess(file)
    #standardises the data
    train,test = splitandstandardise(p)
    #print(p)
    #print(test)
    
    
    test_label, test_without_label = testfileformation(test)
    train_without_label, training_label = trainingformation(train)
    #print(test_label)
    start_time = datetime.datetime.now()  # starting time
    
    #10 different regression methods 
    regression_technique ="LinearRegression";
    print(regression_technique + ":") 
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique ="KNNRegression";
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "RidgeRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "DecisionTreeRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "RandomForestRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
   
    print('----------------------------------------------------------')
    
    regression_technique = "GradientBoostingRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    
    regression_technique = "SGDRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "SVMRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "MLPRegression"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
    
    regression_technique = "LinearSVR"
    print(regression_technique + ":")
    linear = model(train_without_label, training_label,regression_technique)
    y_pred = linear.predict(test_label)
    print("MSE:", round(metrics.mean_squared_error(test_without_label, y_pred),2))
    print("RMSE:",round(np.sqrt(metrics.mean_squared_error(test_without_label, y_pred)),2))
    print("R-squared:", round(metrics.r2_score(test_without_label, y_pred),2))
    print("MAE:", round(metrics.mean_absolute_error(test_without_label, y_pred),2))
    end_time = datetime.datetime.now()  
    execution_time = (end_time - start_time).total_seconds()  # execution time
    print("Execution time:", execution_time)
    print('----------------------------------------------------------')
