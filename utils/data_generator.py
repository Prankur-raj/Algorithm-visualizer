## This File generates data for App

import numpy as np
from sklearn.datasets import make_blobs,make_regression
from config.settings import DEFAULT_ARRAY_SIZE
# def generate_random_array(size):

#     l1=[]
#     count=0

#     while count< size:

#         l1.append(np.random.randint(1,100))
#         count+=1


#     return l1

def generate_random_array(size= DEFAULT_ARRAY_SIZE):
    return np.random.randint(1, 100, size).tolist()


def generate_cluster_data(n_samples=200, n_clusters=3):
    x, y = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=2)
    return x, y
     
      
def generate_regression_data(n_samples=100):
    x, y = make_regression(n_samples=n_samples, n_features=1, noise=15)
    return x, y
    

