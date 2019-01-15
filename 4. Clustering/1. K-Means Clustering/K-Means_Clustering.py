#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 10:58:52 2019

@author: parthgoyal123
"""

''' --------- K-Means Clustering ----------- '''

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
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values.astype(float)

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
X_train, X_test = train_test_split(X, test_size = 0.1, random_state = 1)

"""
# Feature Scaling the data
# ---> StandardScaler : no parameters are to be passed to this class, only an object is to be made
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.fit_transform(X_test)
"""

# ========= Clustering ========= #

# ---- K-Means Clustering ----- #

# Using the Elbow method to get the correct number of clusters
# ---> wcss : within cluster sum of squares
# ---> init : 'kmeans++' is an algorithm for initialising the n_init clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    k_means = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300)
    k_means.fit(X_train)
    wcss.append(k_means.inertia_)

# Plotting the Elbow curve to get the optimal number of clusters
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.plot(range(1,11), wcss, color = 'blue')
plt.title('Elbow Mehtod for Optimal number of clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
fig.savefig('Elbow_Method.png')

# ---> From elbow method, we observe that the major difference occurs at Number of clusters = 5
# Applying K-means with correct number of clusters
k_means = KMeans(n_clusters = 5, init = 'k-means++', n_init = 10, max_iter = 300)
k_means.fit(X)
y_kmeans = k_means.predict(X_train)
y_pred_test = k_means.predict(X_test)

# Visualising the Clusters of Training set 
fig =plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X_train[y_kmeans == 0, 0], X_train[y_kmeans == 0, 1], color = 'red', s=30, marker = '*', label = 'Cluster 1')
plt.scatter(X_train[y_kmeans == 1, 0], X_train[y_kmeans == 1, 1], color = 'blue', s=30, marker = '*', label = 'Cluster 2')
plt.scatter(X_train[y_kmeans == 2, 0], X_train[y_kmeans == 2, 1], color = 'green', s=30, marker = '*', label = 'Cluster 3')
plt.scatter(X_train[y_kmeans == 3, 0], X_train[y_kmeans == 3, 1], color = 'magenta', s=30, marker = '*', label = 'Cluster 4')
plt.scatter(X_train[y_kmeans == 4, 0], X_train[y_kmeans == 4, 1], color = 'cyan', s=30, marker = '*', label = 'Cluster 5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], color = 'yellow', s=100, label = 'Centroid')
plt.title('K-Means Clustering')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('K-Means_training.png')

# Visualising the Clusters of Test set 
fig =plt.figure(dpi = 100, figsize = (8,6))
plt.scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], color = 'red', s=30, marker = '*', label = 'Cluster 1')
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], color = 'blue', s=30, marker = '*', label = 'Cluster 2')
plt.scatter(X_test[y_pred_test == 2, 0], X_test[y_pred_test == 2, 1], color = 'green', s=30, marker = '*', label = 'Cluster 3')
plt.scatter(X_test[y_pred_test == 3, 0], X_test[y_pred_test == 3, 1], color = 'magenta', s=30, marker = '*', label = 'Cluster 4')
plt.scatter(X_test[y_pred_test == 4, 0], X_test[y_pred_test == 4, 1], color = 'cyan', s=30, marker = '*', label = 'Cluster 5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], color = 'yellow', s=100, label = 'Centroid')
plt.title('K-Means Clustering')
plt.xlabel('')
plt.ylabel('')
plt.legend()
plt.show()
fig.savefig('K-Means_test.png')

# ============ K-Means Clustering Complete ============ #