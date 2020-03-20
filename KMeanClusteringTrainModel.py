# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:51:30 2020

@author: Santosh Sah
"""

from sklearn.cluster import KMeans
from KMeanClusteringUtils import (saveKMeanClusteringMeans, readKMeanClusteringDataset, saveKMeanClustering)

"""
Train KMeanClustering model 
"""
def trainKMeanClusteringModel():
    
    X = readKMeanClusteringDataset()
    
    kMeanClustering = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)
    kMeanClustering.fit(X)
    saveKMeanClustering(kMeanClustering)
    
    kMeanClusteringMeans = kMeanClustering.predict(X)
    saveKMeanClusteringMeans(kMeanClusteringMeans)

if __name__ == "__main__":
    trainKMeanClusteringModel()