#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:39:19 2019

@author: parthgoyal123
"""

""" -------- Natural Language Processing --------- """

# ---> Reinforcement Learning: It is about taking suitable action to maximize reward in a particular situation.
# ---> Random Selection: Selecting any random action to be taken

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
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)

# New reviews
corpus = []

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
for i in range(0, dataset.shape[0]):
    # ---> Using re.sub method to remove punctuation marks || keeping only the alphabets
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # ---> Obtaining lower case of review
    review = review.lower()
    # ---> Splitting the review to obtain all the distinct words
    review = review.split()
    # ---> Making an object of the PorterStemmer class to stem the words i.e loved, loving --> love
    ps = PorterStemmer()
    # ---> Stemming the word and removing the stopwords like --> articles, prepositions, etc.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # ---> Joining the list of review to again form a string review
    review = ' '.join(review)
    corpus.append(review)

# Creating a bag of Words Model

# ---> A matrix containing a lot of zeroes is called a Sparse Matrix ---> We would like to reduce Sparasity
# ---> We will make this sparse matrix with the process of Tokenization
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting the dataset to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ---> Now that we have created our dependent and independent set, the task is now of CLASSIFICATION
# ---> From observation, it has been known that Naive Bayes Classification gives the best results for NLP

"""
Determining the performance of a classification model
TP = True Positives
TN = True Negatives
FP = False Positives
FN = False Negatives
Accuracy = (TP+TN)/(TP + TN + FP + FN)
Precision = TP/(TP+FP)
Recall = TP/(TP + FN)
F1 Score = 2*Precision*Recall/(Precision + Recall) 
from sklearn.metrics import precision_recall_fscore_support
"""

# ======= Classification ========= #

# ---- Naive Bayes Classification ----- #

# Fitting the naiveBayes to the training set and predicting results for test set
from sklearn.naive_bayes import GaussianNB
naiveBayes_classifier = GaussianNB()
naiveBayes_classifier.fit(X_train, y_train)
y_pred = naiveBayes_classifier.predict(X_test)

# Making the confusion matrix to know how our model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
cm_naiveBayes = confusion_matrix(y_test, y_pred)
accuracy_naivebayes = accuracy_score(y_test, y_pred)*100
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred)
# print(precision_recall_fscore_support(y_test, y_pred))
# ---> Since there are very few reviews, we are not getting an average accuracy of 73%

# ----- Decision Tree ---- #

# Fitting the dectree to the training set and predicting results for test set
from sklearn.tree import DecisionTreeClassifier
dectree_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 1)

# Grid-Search for best parameters
from sklearn.model_selection import GridSearchCV
parameters_dectree = [{'criterion' : ['gini', 'entropy'], 'min_samples_split' : [2,3,4,5]}]
grid_search_dectree = GridSearchCV(estimator = dectree_classifier, param_grid= parameters_dectree, scoring = 'accuracy', cv = 10)
grid_search_dectree = grid_search_dectree.fit(X_train, y_train)
best_estimator_dectree = grid_search_dectree.best_estimator_
best_params_dectree = grid_search_dectree.best_params_

# Fitting the best estimator classifier to the training set and predicting results for test set
best_estimator_dectree.fit(X_train, y_train)
y_pred_dectree = best_estimator_dectree.predict(X_test)

# Making the confusion matrix to know how our ENHANCED model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm_dectree_best = confusion_matrix(y_test, y_pred_dectree)
accuracy_dectree = accuracy_score(y_test, y_pred_dectree)*100

# ------ Random Forest ------- #

# Fitting the rndForest to the training set and predicting results for test set
from sklearn.ensemble import RandomForestClassifier
rndForest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1)

# Grid-Search for best parameters
from sklearn.model_selection import GridSearchCV
parameters_rndForest = [{'criterion' : ['gini', 'entropy'], 'n_estimators' : [5,10,20,100,200,500]}]
grid_search_rndForest = GridSearchCV(estimator = rndForest_classifier, param_grid= parameters_rndForest, scoring = 'accuracy', cv = 10)
grid_search_rndForest = grid_search_rndForest.fit(X_train, y_train)
best_estimator_rndForest = grid_search_rndForest.best_estimator_
best_params_rndForest = grid_search_rndForest.best_params_

# Fitting the best estimator classifier to the training set and predicting results for test set
best_estimator_rndForest.fit(X_train, y_train)
y_pred_rndForest = best_estimator_rndForest.predict(X_test)

# Making the confusion matrix to know how our ENHANCED model did
# ---> trace of the confusion_matrix represents the correct predictions, else all are wrong predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm_rndForest = confusion_matrix(y_test, y_pred_rndForest)
accuracy_rndForest = accuracy_score(y_test, y_pred_rndForest)*100

# ---> From the accuracy obtained, we observe that naive bayes gave us the best results for this small dataset

# ======== Natural Language Processing Complete ========== #