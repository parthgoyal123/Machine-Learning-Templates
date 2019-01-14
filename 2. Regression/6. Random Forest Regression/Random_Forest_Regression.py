#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:45:58 2019

@author: parthgoyal123
"""

''' ---------- Support Vector Regression ------------ '''

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
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

"""
# Fixing the missing values from the dataset using sklearn.impute
# ---> Importing the SimpleImputer class from the sklearn.impute library
# ---> .fit : The fit part is used to extract some info of the data on which the object is applied
# ---> .transform : the transform part is used to apply some transformation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, [1,2]] = imputer.fit_transform(X[:, [1,2]])

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

# Dividing the dataset to training and test set
# ---> train_test_split : function of the sklearn.model_selection library that splits the dataset to training and test sets
# ---> random_state : this parameter is essential to get the same training and test splits(when improving the model time by time)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
"""

# Feature Scaling the data --->> For Support Vector Regression, feature scaling is must
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
scale_y = StandardScaler()
X = scale_X.fit_transform(X)
y = scale_y.fit_transform(y.reshape(-1,1))
y = y.ravel()

# ====== Regression ====== #

# Fitting the Random Forest regressor(forest) to the dataset as it is (without enhancing the model to compare the results afterwards)
# ---> .fit : fits linear regressor to the data
# ---> .predict : method that predicts the value of the dependent variable
from sklearn.ensemble import RandomForestRegressor
forest_regressor = RandomForestRegressor(n_estimators = 10, criterion = 'mse', random_state = 1)
forest_regressor.fit(X, y)
y_pred_forest = forest_regressor.predict(X)

# Formation of a new grid for better visualisation of plot
X_grid_scaled = np.arange(min(X), max(X), 0.01)
X_grid_scaled = X_grid_scaled.reshape(len(X_grid_scaled), 1)

# Visualising Naive Support Vector Regression
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X, y, color = 'red', s = 100, marker = '*')
plt.plot(X_grid_scaled, forest_regressor.predict(X_grid_scaled), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('Random_Forest_Regression_Naive.png')

"""
# Getting the mean squared error of our naive model
from sklearn.metrics import mean_squared_error
mean_error_before = mean_squared_error(y, y_pred_forest)
mean_error_best = mean_error_before
best_degree = 2
"""

# ====== Enhancing the results ====== #

# --- K-Fold Cross Validation --- #
from sklearn.model_selection import cross_val_score
accuracies_before = cross_val_score(estimator = forest_regressor, X = X, y = y, cv = 3)
print('Accuracy Mean before enhancing the model =', accuracies_before.mean())
print('Accuracy Standard Deviation before enhancing the model =', accuracies_before.std())

# Getting the best fit model for our dataset, keeping in mind that we don't overfit the data
# --- Grid-Search --- #
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators' : [5,10,50,100,200], 'criterion': ['mse', 'mae'], 
               'min_samples_split' : [2,3,4,5,6,7]}]
grid_search = GridSearchCV(estimator = forest_regressor, param_grid = parameters, n_jobs = -1, cv = 3)
grid_search = grid_search.fit(X, y)
best_estimator =  grid_search.best_estimator_
best_params = grid_search.best_params_

# Fitting the best estimator to our dataset
best_estimator.fit(X, y)
y_pred_forest_best = best_estimator.predict(X)

# Visualising Best Support Vector Regression
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X, y, color = 'red', s = 100, marker = '*')
plt.plot(X_grid_scaled, best_estimator.predict(X_grid_scaled), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('Support_Vector_Regression_Best.png')

# --- K-Fold Cross Validation (Best Model) --- #
from sklearn.model_selection import cross_val_score
accuracies_before = cross_val_score(estimator = best_estimator, X = X, y = y, cv = 3)
print('Accuracy Mean after enhancing the model =', accuracies_before.mean())
print('Accuracy Standard Deviation after enhancing the model =', accuracies_before.std())

"""
# Getting the mean squared error of our best model
from sklearn.metrics import mean_squared_error
mean_error_after = mean_squared_error(y, y_pred_forest_new)
"""

# Getting the Actual data back
X = scale_X.inverse_transform(X)

# Getting the Actual y_pred(Original)
y = scale_y.inverse_transform(y)
y_pred_best_not_scaled = scale_y.inverse_transform(best_estimator.predict(scale_X.transform(X)))

# ============== Random Forest Regression Complete =================== #