#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:59:54 2019

@author: parthgoyal123
"""

""" ------- Eclat Algorithm for Association Rule Learning ------ """

# ---> Association Rule Learning : Person who bought X also bought Y. We need to find these X's and corresponding Y's
# ---> It focuses mainly only the support unlink the apriori algo that focuses on support, confidence and lift`

# ====== Preprocessing ====== #

# Importing the required libraries 
# ---> Numpy arrays are the most convinient way to work on Machine Learning models
# ---> Matplotlib allows us to visualise our model in form of various plots/figures
# ---> Pandas allows us to import the dataset efficiently
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset using pandas library --> Here there is no header, so a slight modification in pd.read_csv
# ---> pandas create a dataframe of the dataset
# ---> iloc : It locates the column by its index. In other words, using ’iloc’ allows us to take columns by just taking their index.
# ---> .values : It returns the values of the column (by their index) inside a Numpy array(way more efficient than a list)
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)

# ---> Since the Eclat depends only on the support, we need to know the total types of products available and their dependancy on other products

# Making the transactions list, neglecting 'nan' for all the observations
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1]) 
    if str(dataset.values[i, j]) != 'nan'])

# Generate a list of unique items
unique_items = []
for data in transactions:
    for item in data:
        if not (item in unique_items):
            unique_items.append(item)

# Generate a list of pairs of items with relevant support value
# [[(item_a, item_b) , support_value]]
# support_value is initialized to 0 for all pairs
eclat = []
for itemX in range(0, len(unique_items)):
    for itemY in range(itemX+1, len(unique_items)):
        eclat.append([(unique_items[itemX], unique_items[itemY]), 0])

# Compute support value for each pair by looking for transactions with both items
for pair in eclat:
    for data in transactions:
        if(pair[0][0] in data) and (pair[0][1] in data):
            pair[1] += 1
    pair[1] = pair[1]/len(transactions)

# Converting eclat in sorted DataFrame to be visualized in variable explorer
learned_rules_df = pd.DataFrame(eclat, columns = ['rule', 'support']).sort_values(by = 'support', ascending = False)

# ========== Eclat Algorithm Complete =========== #