# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:58:38 2020

@author: Santosh Sah
"""
import matplotlib.pyplot as plt
from KMeanClusteringUtils import readKMeanClusteringDataset, readKMeanClusteringWCSS, readKMeanClusteringMeans, readKMeanClustering

def kMeanClusteringVisualizeElbow():
    
    kMeanClusteringWCSS = readKMeanClusteringWCSS()
    plt.plot(range(1, 11), kMeanClusteringWCSS)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    
    plt.savefig("k_means_clustering_elbow.png")
    
    plt.show()

def kMeanClusteringVisualizeCluster():
    
    X = readKMeanClusteringDataset()
    kMeanClusteringMeans = readKMeanClusteringMeans()
    
    kMeanClustering = readKMeanClustering()
    
    # Visualising the clusters
    plt.scatter(X[kMeanClusteringMeans == 0, 0], X[kMeanClusteringMeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[kMeanClusteringMeans == 1, 0], X[kMeanClusteringMeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[kMeanClusteringMeans == 2, 0], X[kMeanClusteringMeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[kMeanClusteringMeans == 3, 0], X[kMeanClusteringMeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[kMeanClusteringMeans == 4, 0], X[kMeanClusteringMeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(kMeanClustering.cluster_centers_[:, 0], kMeanClustering.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
    plt.savefig("k_means_clustering_cluster.png")
    
    plt.show()

if __name__ == "__main__":
    #kMeanClusteringVisualizeElbow()
    kMeanClusteringVisualizeCluster()