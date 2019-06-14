# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

"""RANDOM SELECTION
# Implementing random selection
import random
N = 10000
d = 10
ad_selected = []
total_reward = 0
for i in range(0,N):
    ad = random.randrange(d)
    ad_selected.append(ad)
    reward = dataset.values[i, ad]
    total_reward = reward + total_reward
    
# Visualising the results - Histogram
plt.hist(ad_selected)
plt.title('Histograms of ad selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

UPPER CONFIDENCE BOUND """
# Implementing the UCB
import math
N = 10000
d = 10
total_reward = 0
ad_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
for n in range(0, N):
    max_upper_bound = 0
    ad = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0) :
            average_reward = sum_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            ad = i
    ad_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ad_selected)
plt.title('Histograms of ad selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            
