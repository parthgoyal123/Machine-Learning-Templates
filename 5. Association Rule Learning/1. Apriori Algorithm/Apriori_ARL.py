#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:28:56 2019

@author: parthgoyal123
"""

""" ------- Apriori Algorithm for Association Rule Learning ------ """

# ---> Association Rule Learning : Person who bought X also bought Y. We need to find these X's and corresponding Y's

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

# ---> For the apriori algorithm we need list of list of strings, so we cannot directly apply that to the dataset

# Making list of transactions(also a list)
transactions = []
for i in range(dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1])])
    
# ---> Now that we have the list of list of transactions , we can apply the apriori algorithm to the transactions we made

# Applying Apriori Algorithm
from apyori_class import apriori
rules = apriori(transactions= transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# ---> In Python, the rules are already sorted wrt to Min_lift

# Visualising the Rules
results = list(rules)
the_rules = []
for result in results:
    the_rules.append({'rule': ','.join(result.items),
                      'support': result.support,
                      'confidence':result.ordered_statistics[0].confidence,
                      'lift': result.ordered_statistics[0].lift})

learned_rules_df = pd.DataFrame(the_rules, columns = ['rule', 'support', 'confidence', 'lift'])

# ============ Apriori Algorithm Complete =============== #