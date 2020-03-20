# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:55:55 2020

@author: Santosh Sah
"""

import pandas as pd
import pickle

"""
Import dataset and read specific column.
"""
def importKMeanClusteringDataset(kMeanClusteringDatasetFileName):
    
    kMeanClusteringDataset = pd.read_csv(kMeanClusteringDatasetFileName)
    X = kMeanClusteringDataset.iloc[:, [2, 3]].values
    
    return X

def saveKMeanClusteringDataset(X):
    
    #Write X in a picke file
    with open("X.pkl",'wb') as X_Pickle:
        pickle.dump(X, X_Pickle, protocol = 2)

"""
read X from pickle file
"""
def readKMeanClusteringDataset():
    
    #load X
    with open("X.pkl","rb") as X_pickle:
        X = pickle.load(X_pickle)
    
    return X

"""
Save KMeanClusteringMeans as a pickle file.
"""
def saveKMeanClusteringMeans(kMeanClusteringMeans):
    
    #Write KMeanClusteringModel as a picke file
    with open("KMeanClusteringMeans.pkl",'wb') as KMeanClusteringMeans_Pickle:
        pickle.dump(kMeanClusteringMeans, KMeanClusteringMeans_Pickle, protocol = 2)

"""
read KMeanClusteringMeans from pickle file
"""
def readKMeanClusteringMeans():
    
    #load KMeanClusteringMeans model
    with open("KMeanClusteringMeans.pkl","rb") as KMeanClusteringMeans:
        kMeanClusteringMeans = pickle.load(KMeanClusteringMeans)
    
    return kMeanClusteringMeans

"""
Save KMeanClusteringWCSS as a pickle file.
"""
def saveKMeanClusteringWCSS(kMeanClusteringWCSS):
    
    #Write KMeanClusteringWCSS as a picke file
    with open("KMeanClusteringWCSS.pkl",'wb') as KMeanClusteringWCSS_Pickle:
        pickle.dump(kMeanClusteringWCSS, KMeanClusteringWCSS_Pickle, protocol = 2)

"""
read KMeanClusteringWCSS from pickle file
"""
def readKMeanClusteringWCSS():
    
    #load KMeanClusteringWCSS model
    with open("KMeanClusteringWCSS.pkl","rb") as KMeanClusteringWCSS:
        kMeanClusteringWCSS = pickle.load(KMeanClusteringWCSS)
    
    return kMeanClusteringWCSS

"""
Save KMeanClustering as a pickle file.
"""
def saveKMeanClustering(kMeanClustering):
    
    #Write KMeanClustering as a picke file
    with open("KMeanClustering.pkl",'wb') as KMeanClustering_Pickle:
        pickle.dump(kMeanClustering, KMeanClustering_Pickle, protocol = 2)

"""
read KMeanClustering from pickle file
"""
def readKMeanClustering():
    
    #load KMeanClustering model
    with open("KMeanClustering.pkl","rb") as KMeanClustering:
        kMeanClustering = pickle.load(KMeanClustering)
    
    return kMeanClustering

"""
Save KMeanClusteringMinMaxScaler as a pickle file.
"""
def saveKMeanClusteringMinMaxScaler(kMeanClusteringMinMaxScaler):
    
    #Write KMeanClusteringMinMaxScaler as a picke file
    with open("KMeanClusteringMinMaxScaler.pkl",'wb') as KMeanClusteringMinMaxScaler_Pickle:
        pickle.dump(kMeanClusteringMinMaxScaler, KMeanClusteringMinMaxScaler_Pickle, protocol = 2)

"""
read KMeanClusteringMinMaxScaler from pickle file
"""
def readKMeanClusteringMinMaxScaler():
    
    #load KMeanClusteringMinMaxScaler model
    with open("KMeanClusteringMinMaxScaler.pkl","rb") as KMeanClusteringMinMaxScaler:
        kMeanClusteringMinMaxScaler = pickle.load(KMeanClusteringMinMaxScaler)
    
    return kMeanClusteringMinMaxScaler
