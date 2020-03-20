# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:29:04 2020

@author: Santosh Sah
"""
from sklearn.preprocessing import MinMaxScaler
from KMeanClusteringUtils import (importKMeanClusteringDataset, saveKMeanClusteringDataset, saveKMeanClusteringMinMaxScaler)

def preprocess():
    
    X = importKMeanClusteringDataset("K_Mean_Clustering_Mall_Customers.csv")
    
    kMeanClusteringMinMaxScaler = MinMaxScaler()
    kMeanClusteringMinMaxScaler.fit(X)
    saveKMeanClusteringMinMaxScaler(kMeanClusteringMinMaxScaler)
    
    X = kMeanClusteringMinMaxScaler.transform(X)
    saveKMeanClusteringDataset(X)
    

if __name__ == "__main__":
    preprocess()