#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 02:03:47 2019

@author: parthgoyal123
"""

''' ----------- Naive-Bayes Classification ------------ '''

# ---> In classification, feature scaling is must and splitting the data to train and test set is also a must

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

# Feature Scaling the data
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)

# ======== Classification ========= #

# Fitting the naiveBayes to the training set and predicting results for test set
from sklearn.naive_bayes import GaussianNB
naiveBayes_classifier = GaussianNB()
naiveBayes_classifier.fit(X_train, y_train)
y_pred = naiveBayes_classifier.predict(X_test)

# Making the confusion matrix to know how our model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm_naiveBayes = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)*100

# Visualising the Training set results (Naive Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, naiveBayes_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('NaiveBayes (Training set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("NaiveBayes_training.png")

# Visualising the Test set results (Naive Model)
from matplotlib.colors import ListedColormap
fig = plt.figure(dpi = 100, figsize = (8,6))
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, naiveBayes_classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j, marker = '.', s = 75)
plt.title('NaiveBayes (Test set)')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig("NaiveBayes_test.png")

# --- K-Fold Cross Validation --- #
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = naiveBayes_classifier, X = X_train, y = y_train, cv = 10, scoring = 'accuracy')
print('Accuracy Mean before enhancing the model =', accuracies.mean())
accuracy_mean = accuracies.mean()
print('Accuracy Standard Deviation before enhancing the model =', accuracies.std())
accuracy_std = accuracies.std()

# ---> Since there are no arguments to pass in the GaussianNB Class, therefore as such we cannot apply our Grid-Seach Algo to get the best model

# =============== NaiveBayes Complete ================= #