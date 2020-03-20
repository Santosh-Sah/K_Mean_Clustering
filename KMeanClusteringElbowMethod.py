# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:47:24 2020

@author: Santosh Sah
"""
from sklearn.cluster import KMeans
from KMeanClusteringUtils import readKMeanClusteringDataset, saveKMeanClusteringWCSS

def kMeanClusteringElbow():
    
    X = readKMeanClusteringDataset()
    kMeanClusteringWCSS = []
    
    for i in range(1, 11):
        kMeanClustering = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kMeanClustering.fit(X)
        kMeanClusteringWCSS.append(kMeanClustering.inertia_)
    
    saveKMeanClusteringWCSS(kMeanClusteringWCSS)

if __name__ == "__main__":
    kMeanClusteringElbow()
