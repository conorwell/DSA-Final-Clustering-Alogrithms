import random
import networkx as nx
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

def big_square_tuples_list(n):
    
    tuples = []
    for i in range (n):
        a = random.randrange(0, 1000)
        b = random.randrange(0, 1000)
        tuples.append((a,b))
        
    return tuples
    
def random_tuple_groups_list(nodes, groups):
    
    tuples = []
    for j in range(0,groups):
        for i in range(int(nodes/groups)):
            a = random.randrange(j*groups*100 -350,j*groups*100)
            b = random.randrange(j*groups*100 -350,j*groups*100)
            tuples.append((a,b))
            
    return tuples
            
def circle_in_circle_tuples_list(nodes):
    tuples = []
    
    X, Y = datasets.make_circles(
    n_samples=nodes, factor=0.5, noise=0.05, random_state=30
    )
    
    
    for x in X:
        x[0] = x[0]*750
        x[1] = x[1]*750
        tuples.append((x[0],x[1]))
    return tuples

def double_horseshoes_tuples_list(nodes):
    tuples = []
    
    X,Y = datasets.make_moons(
    n_samples=nodes, noise=0.05, random_state=30
    )
    
    for x in X:
        x[0] = x[0]*750
        x[1] = x[1]*750
        tuples.append((x[0],x[1]))
        
    return tuples
        
    
def lines_tuples_list(nodes):
    points = []
    
    for i in range(0,nodes,10):
        points.append((i,i))
        points.append((i+100,i-100))
        points.append((i+200, i-200))
        points.append((i+300, i-300))
        points.append((i+400, i-400))
        
    return points

    
