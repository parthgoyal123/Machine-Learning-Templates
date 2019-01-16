#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 23:51:33 2019

@author: parthgoyal123
"""

""" -------- Upper Confidence Bound(UCB) Reinforcement Learning --------- """

# ---> Reinforcement Learning: It is about taking suitable action to maximize reward in a particular situation.

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
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

# Upper Confidence Bound(UCB)
import math
N = 10000
d = 10
ads_selected = []
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    # For the first d rounds, all the variables will be tested, such that number of selections of each become atleast one
    ad = 0
    max_upperbound = 0
    for i in range(0, d):
        if(number_of_selections[i] != 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta = math.sqrt(3/2 * math.log(n+1) / number_of_selections[i])
            upper_confidence = average_reward + delta
        else:
            upper_confidence = 1e400
        if(upper_confidence > max_upperbound):
            max_upperbound = upper_confidence
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward += reward
    sum_of_rewards[ad] += reward
    number_of_selections[ad] += 1

# Visualising the number of selections of each graph
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
fig.savefig('Upper_Confidence_Bound.png')

# ============= Upper Confidence Bound (UCB) =============== #