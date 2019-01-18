#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:02:45 2019

@author: parthgoyal123
"""

''' ----------- Grid-Search for Best Model Selection ------------ '''

# ---> The k-fold cross validation technique is used to test the performance of our model, by dividing the training set itself into two parts, and testing the test part with the trained model

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
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values.astype(float)
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
"""

# Dividing the dataset to training and test set
# ---> train_test_split : function of the sklearn.model_selection library that splits the dataset to training and test sets
# ---> random_state : this parameter is essential to get the same training and test splits(when improving the model time by time)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling the data --->> For Support Vector Regression, feature scaling is must
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)

# ======== SVM Classification ========= #

# Fitting the kernelSVM to the training set and predicting results for test set
from sklearn.svm import SVC
kernelSVM_classifier = SVC(kernel = 'linear', random_state = 1)
kernelSVM_classifier.fit(X_train, y_train)
y_pred = kernelSVM_classifier.predict(X_test)

# Making the confusion matrix to know how our model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm_kernelSVM = confusion_matrix(y_test, y_pred)
accuracy_before = accuracy_score(y_test, y_pred)*100

# Visualising the Training set results (Naive Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kernelSVM_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('KernelSVM (Training set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("KernelSVM_training_Naive.png")

# Visualising the Test set results (Naive Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kernelSVM_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('KernelSVM (Test set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("KernelSVM_test_Naive.png")

# ===== Enhancing the model using K-fold Cross Validation and Grid-Search Technique ===== #

# --- K-Fold Cross Validation --- #
from sklearn.model_selection import cross_val_score
accuracies_before = cross_val_score(estimator = kernelSVM_classifier, X = X_train, y = y_train, cv = 10, scoring = 'accuracy')
print('Accuracy Mean before enhancing the model =', accuracies_before.mean())
accuracy_mean_before = accuracies_before.mean()
print('Accuracy Standard Deviation before enhancing the model =', accuracies_before.std())
accuracy_std_before = accuracies_before.std()

# --- Grid-Search --- #
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [0.1,0.5,1,2,5,10,100], 'kernel' : ['linear']}, 
               {'C' : [0.1,0.5,1,2,5,10,100], 'kernel' : ['rbf', 'sigmoid'], 'gamma' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]},
               {'C' : [0.1,0.5,1,2,5,10,100], 'kernel' : ['poly'], 'degree': [2,3,4], 'gamma' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}]
grid_search = GridSearchCV(estimator = kernelSVM_classifier, param_grid= parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_estimator = grid_search.best_estimator_
best_params = grid_search.best_params_

# Fitting the best estimator classifier to the training set and predicting results for test set
best_estimator.fit(X_train, y_train)
y_pred_best = best_estimator.predict(X_test)

# Making the confusion matrix to know how our ENHANCED model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm_kernelSVM_best = confusion_matrix(y_test, y_pred_best)
accuracy_after = accuracy_score(y_test, y_pred_best)*100

# --- K-Fold Cross Validation (Best Model) --- #
from sklearn.model_selection import cross_val_score
accuracies_after = cross_val_score(estimator = best_estimator, X = X_train, y = y_train, cv = 10, scoring = 'accuracy')
print('Accuracy Mean after enhancing the model =', accuracies_after.mean())
accuracy_mean_after = accuracies_after.mean()
print('Accuracy Standard Deviation after enhancing the model =', accuracies_after.std())
accuracy_std_after = accuracies_after.std()

# Visualising the Training set results (Best kernelSVM Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, best_estimator.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('KernelSVM (Training set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("KernelSVM_training_Best.png")

# Visualising the Test set results (Best kernelSVM Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, best_estimator.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('KernelSVM (Test set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("KernelSVM_test_Best.png")

# =============== K-Fold Cross Validation and Grid-Search Complete ================= #
