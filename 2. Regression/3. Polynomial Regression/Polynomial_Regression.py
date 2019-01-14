#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:26:58 2019

@author: parthgoyal123
"""


''' ---------- Polynomial Regression ------------ '''

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

# Feature Scaling the data(if required)
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)"""

# ====== Regression ====== #

# Generating new polynomic dataset to apply linear regression on
# ---> In order to perform polynomial regression, we need to make new X_poly and apply linear regression on that new X_poly
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)

# Fitting the polynomial regressor to the dataset as it is (without enhancing the model to compare the results afterwards)
# ---> .fit : fits linear regressor to the data
# ---> .predict : method that predicts the value of the dependent variable as per the Polynomial Regression model
from sklearn.linear_model import LinearRegression
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)
y_pred_poly = poly_regressor.predict(X_poly)

# For predicting a new value using polynomial regression, transform it using the poly object and then apply the poly_linear_regressor
new_value = np.array([6.5])
new_value = new_value.reshape(-1,1)
y_pred_new = poly_regressor.predict(poly.fit_transform(new_value))

# Formation of a new grid for better visualisation of plot
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

# Visualising Polynomial Regression
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X, y, color = 'red', s = 100, marker = '*')
plt.plot(X_grid, poly_regressor.predict(poly.fit_transform(X_grid)), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('Polynomial_Regression.png')

# Getting the mean squared error of our naive model
from sklearn.metrics import mean_squared_error
mean_error_before = mean_squared_error(y, y_pred_poly)
mean_error_best = mean_error_before
best_degree = 2

# ====== Enhancing the results ====== #

# Getting the best fit model for our dataset, keeping in mind that we don't overfit the data
from sklearn.model_selection import GridSearchCV
for i in [1, 2, 3, 4, 5, 6, 7]:
    poly_new = PolynomialFeatures(degree = i)
    X_poly_new = poly_new.fit_transform(X)
    poly_regressor_new = LinearRegression()
    poly_regressor_new.fit(X_poly_new, y)
    y_pred_poly_new = poly_regressor_new.predict(X_poly_new)
    mean_error_after = mean_squared_error(y, y_pred_poly_new)
    if(mean_error_after < mean_error_best):
        mean_error_best = mean_error_after
        best_degree = i
        print('degree =', i)
        plt.scatter(X, y, color = 'red', s = 100, marker = '*')
        plt.plot(X_grid, poly_regressor_new.predict(poly_new.fit_transform(X_grid)), color = 'blue')
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.show()

# We observe from the plots that polynomial regressoin with degree 4 and higher seem to overfit the data
poly_new = PolynomialFeatures(degree = 3)
X_poly_new = poly_new.fit_transform(X)
poly_regressor_new = LinearRegression()
poly_regressor_new.fit(X_poly_new, y)
y_pred_poly_new = poly_regressor_new.predict(X_poly_new)

# Visualising Polynomial Regression with best degree
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X, y, color = 'red', s = 100, marker = '*')
plt.plot(X_grid, poly_regressor_new.predict(poly_new.fit_transform(X_grid)), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('Polynomial_Regression_best.png')

# Getting the mean squared error of our best model
from sklearn.metrics import mean_squared_error
mean_error_after = mean_squared_error(y, y_pred_poly_new)

# ---> We observe that after the model enhancement, mean squared error has reduced

# ============== Polynomial Regression Complete =================== #