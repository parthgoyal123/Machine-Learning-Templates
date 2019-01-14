#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:54:54 2019

@author: parthgoyal123
"""

''' ---------- Multivariate Linear Regression ------------ '''

# ====== Preprocessing ====== #

# Importing the required libraries
# ---> Numpy arrays are the most convinient way to work on Machine Learning models
# ---> Matplotlib allows us to visualise our model in form of various plots/figures
# ---> Pandas allows us to import the dataset efficiently
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset using pandas library
# ---> pandas create a dataframe of the dataset
# ---> iloc : It locates the column by its index. In other words, using ’iloc’ allows us to take columns by just taking their index.
# ---> .values : It returns the values of the column (by their index) inside a Numpy array(way more efficient than a list)
dataset = pd.read_csv("Filename.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""
# Fixing the missing values from the dataset using sklearn.impute
# ---> Importing the SimpleImputer class from the sklearn.impute library
# ---> .fit : The fit part is used to extract some info of the data on which the object is applied
# ---> .transform : the transform part is used to apply some transformation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, [1,2]] = imputer.fit_transform(X[:, [1,2]])
"""

# Encoding categorical data (optional part, depends on the dataset)
# ---> LabelEncoder encodes the categorical data to [0, n_values]
# ---> OneHotEncoder seperates the LabelEncoded values to different columns, appended leftmost of the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
labelencoder_y = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features= [-1])
X[:, -1] = labelencoder_x.fit_transform(X[:, -1])
X = onehotencoder.fit_transform(X).toarray()

# When the categorical types >= 3, we need to avoid the dummy variable trap
X = X[:, 1:]

""" # Dividing the dataset to training and test set
# ---> train_test_split : function of the sklearn.model_selection library that splits the dataset to training and test sets
# ---> random_state : this parameter is essential to get the same training and test splits(when improving the model time by time)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling the data(if required)
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)"""

# ====== Regression ====== #

# For multivariate linear regression, we need to append a column of ones to the leftmost of our data
X = np.append(arr = np.ones([len(X), 1]).astype(int), values = X, axis = 1)

# Fitting the multivariate linear regressor to the dataset as it is (without enhancing the model to compare the results afterwards)
# ---> .fit : fits linear regressor to the data
# ---> .predict : method that predicts the value of the dependent variable as per the Linear Regression model
from sklearn.linear_model import LinearRegression
multi_linear_regressor = LinearRegression()
multi_linear_regressor.fit(X, y)
y_pred_multi_linear = multi_linear_regressor.predict(X)

# Applying k-fold cross validation
# ---> estimator : object to fit the data
# ---> cv : decides cross_validation splitting strategy
from sklearn.model_selection import cross_val_score
accuracies_before = cross_val_score(estimator = multi_linear_regressor, X = X, y = y, cv = 5)
print('Mean Accuracy before enhancing the model :', accuracies_before.mean())
print('Standard Deviation of Accuracy before enhancing the model :', accuracies_before.std())

# ====== Enhancing the results ====== #

# ---> For the given dataset, in the above model, we are assuming that every independant variable has influence on the dependant variable, but that sometimes might not be true
# ---> We need to remove those independant variables that have very less/no affect on our result

# Automatic Backward Elimination with p-values and AdjR  values
# ---> The backward elimination process returns the independant variables that influences the most the results.
# ---> Read about p-values and adjR values from internet and then view ahead
import statsmodels.formula.api as sm
def backwardElimination_Rsquared(X, y, sl):
    numVars = len(X[0])
    temp = np.zeros((len(X),len(X[0]))).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > sl:
            for j in range(0, numVars -  i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = X[:, j]
                    # X = np.delete(X, j, 1)
                    tmp_regressor = sm.OLS(y, np.delete(X, j, 1)).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if(adjR_before >= adjR_after):
                        # x_rollback = np.hstack((X, temp[:, [0,j]]))
                        # x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return X
                    else:
                        X = np.delete(X, j, 1)
                        continue
    print(regressor_OLS.summary())
    return X

SL = 0.05
X_enhanced = backwardElimination_Rsquared(X, y, SL)

# Fitting the new X_enhanced to our Linear Model and comparing the results
from sklearn.linear_model import LinearRegression
multi_linear_regressor_enhanced = LinearRegression()
multi_linear_regressor_enhanced.fit(X_enhanced, y)
y_pred_multi_linear_enhanced = multi_linear_regressor_enhanced.predict(X_enhanced)

# Applying k-fold cross validation
# ---> estimator : object to fit the data
# ---> cv : decides cross_validation splitting strategy
from sklearn.model_selection import cross_val_score
accuracies_after = cross_val_score(estimator = multi_linear_regressor_enhanced, X = X_enhanced, y = y, cv = 5)
print('Mean Accuracy after enhancing the model :', accuracies_after.mean())
print('Standard Deviation of Accuracy after enhancing the model :', accuracies_after.std())

# ---> We observe that after the model enhancement, both mean and std have improved(that was our major goal)

# ============== Multivariate Linear Regression Complete =================== #