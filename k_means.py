import time
import numpy as np
import time
import sklearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
%matplotlib inline
from sklearn.datasets import make_blobs
import random
import networkx as nx
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from random import randint
import math

def init_centroids(k, data):
    centroids = []
    i = 0
    while i < k:
        centroid = random.choice(data)
        centroids.append(centroid)
        i+=1
    return centroids

def distance(X, Y):
    #X and Y are both arrays
    dist = np.square(np.sum((X-Y)**2))
    return dist

def get_dist(pair):
    return pair[0]

def mean_tuple(tuple_list):
    
    avg_x = sum(map(lambda x: x[0], tuple_list)) / len(tuple_list)
    avg_y = sum(map(lambda x: x[1], tuple_list)) / len(tuple_list)
    res = (avg_x, avg_y)
    return res
    
def create_cluster_lists(k):
    i = 0
    clustersclusters = []
    while i < k:
        cluster = []
        clustersclusters.append(cluster)
        i += 1
    return clustersclusters

def random_list_of_points_generator(n_tuples, val_limit):
    our_list = []
    for x in range(n_tuples):
        r1 = random.randint(1,val_limit)
        r2 = random.randint(1,val_limit)
        point = (r1, r2)
        our_list.append(point)
    return our_list

def visualize_clusters(c_lists):
    
    x = [None]*len(c_lists)
    y = [None]*len(c_lists)
    
    for i in range(len(c_lists)):
        
        xv = []
        yv = []
        
        for j in range(len(c_lists[i])):     
            
            xv.append(c_lists[i][j][0])
            yv.append(c_lists[i][j][1])
            
        x[i] = xv
        y[i] = yv
        
        plt.scatter(xv,yv)
    
    plt.show()
    
    #print(x)
    return x

def k_means(point_list, k):
    #need to randomly assign he centroids at the beginning
    #out point_list is like [(x1,y1), (x2,y2), ... (xn,yn)]
    centroids = []
    centroids += random.sample(point_list, k)
    
    #ok i have some centroids now
    #we need to make a list called assignments for assigned cluster for each point
    #each index represents the same index point in point_list i.e. [...(x4,y4)...] will have assignment at asn[3].
    lst = [None]*(len(point_list))
    
    #i think here is where we do a while loop a couple times to assign/reassign clusters and centroids so... here goes
    iterations = 0
    start_time = time.time()
    while iterations < 10:

        #kk that list is now a thing
        #use euclidean distance to calculate distance from each point to each centroid
        for i in range(len(point_list)):
            point = point_list[i]
            point_dists = []
            label = 0
            
            for centroid in centroids:

                a = point[0]-centroid[0]
                b = point[1]-centroid[1]
                sum = a**2 + b**2
                dist = math.sqrt(sum)
                point_dists.append((dist, label))
                label += 1

            sorted_by_distance = sorted(point_dists, key = get_dist)
            #now we have a list of length k where the shortest distance is at the 0th index, with label of k closest centroid
            #need to assign the point to the appropriate centroid in lst
            lst[i] = sorted_by_distance[0][1]
        #cool so all of the points are now assigned to an initial cluster
        #now we need to calculate the mean distance between all the points in our clusters and update the centroid to be that new point
        #keep doing all this a certain number of times for now (eventually until the distance between old_c and new_c becomes very small i.e. convergent)
        cluster_lists = create_cluster_lists(k)
        lmao = k
        lamo = 0
        #need a loopdeeloop to assign points to their clusters for mean analysis based on lst value of index
        for i in lst:
            cluster_lists[i].append(point_list[lamo])
            lamo += 1

        if iterations == 9:
            visualize_clusters(cluster_lists)
            
        for x in range(k):
            centroids[x] = mean_tuple(cluster_lists[x])

        #print(centroids)
        iterations += 1
    
    end_time = time.time()
    run_time = end_time - start_time
    print(run_time)
    return cluster_lists