#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:15:43 2019

@author: parthgoyal123
"""

""" -------- Thompson Sampling Reinforcement Learning --------- """

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

# Implementing Thompson Sampling
import random
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward

# Visualising the number of selections of each graph
fig = plt.figure(dpi = 100, figsize = (8,6))
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
fig.savefig('Thompson_Sampling.png')

# ============== Thompson Sampling =============== #
