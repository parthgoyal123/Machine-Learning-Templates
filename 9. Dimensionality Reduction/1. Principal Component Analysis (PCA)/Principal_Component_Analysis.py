#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:23:11 2019

@author: parthgoyal123
"""

''' ----------- Principal Component Analysis (PCA) ------------ '''

"""
# --> PCA is mathematically defined as an orthogonal linear transformation that transforms the data
 to a new coordinate system such that the greatest variance by some projection of the data comes to 
 lie on the first coordinate (called the first principal component), the second greatest variance on 
 the second coordinate, and so on
 """

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
dataset = pd.read_csv("Wine.csv")
X = dataset.iloc[:, :-1].values.astype(float)
y = dataset.iloc[:, -1].values

# Dividing the dataset to training and test set
# ---> train_test_split : function of the sklearn.model_selection library that splits the dataset to training and test sets
# ---> random_state : this parameter is essential to get the same training and test splits(when improving the model time by time)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling the data
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)

# ======= Applying PCA ========= #

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
variance = pca.explained_variance_ratio_

# Fitting the classifier to a NEW DIMENSIONTIONALLY REDUCED X_train
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr')
classifier.fit(X_train, y_train)

# Predicting results for the test set
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
fig = plt.figure(dpi = 100, figsize= (8,6))
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j, marker = '.', s = 50)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PD1')
plt.ylabel('PD2')
plt.legend()
plt.show()
fig.savefig('PCA_training.png')

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
fig = plt.figure(dpi = 100, figsize= (8,6))
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j, marker = '.', s = 75)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PD1')
plt.ylabel('PD2')
plt.legend()
plt.show()
fig.savefig('PCA_test.png')

# ============ PCA Complete ============ #