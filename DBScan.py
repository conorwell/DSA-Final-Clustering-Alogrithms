#imports
import random
import networkx as nx
import numpy as np
import time

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

#methods used in DBScan Method
def findNeighbors(graph, new_Node):
    neighbors = []
    for node in graph.nodes():
        distance = np.sqrt(np.square(new_Node[0] - node[0]) + np.square(new_Node[1] - node[1]))
        if distance < 50 and distance != 0:
            neighbors.append(node)
    return neighbors

def buildNeighborhood(graph, node, neighborhood_Counter, neighborhood, visited):
    neighborhood[node] = neighborhood_Counter
    neighbors = findNeighbors(graph, node)
    if len(neighbors) > 3:
        for neighbor in neighbors:
            if (node,neighbor) not in graph.edges():
                graph.add_edge(node, neighbor)
                buildNeighborhood(graph, neighbor, neighborhood_Counter, neighborhood, visited)
    visited[node] = 1
    
class myQueue():
    
    def __init__(self):
        self.items = []
        
    def get(self):
        return self.items.pop(0)
    
    def add(self, item):
        self.items.append(item)

#DBScan Method
def DBScan(graph,bag):
    #add all nodes to a bag
    for node in graph.nodes():
        bag.add(node)
    #mark all nodes unvisited
    visited = {node:0 for node in graph.nodes() }
    neighborhood = {node:0 for node in graph.nodes() }
    neighborhood_counter = 0
    
    while len(bag.items) > 0:
        #remove node from bag
        new_Node= bag.get()
        if visited[new_Node] == 0:
            #mark node as visited
            visited[new_Node] = 1
            
            #search list of nodes for nodes within given radius
            neighbors = findNeighbors(graph, new_Node)
                #if neighbor count is high enough:
            if len(neighbors) > 3:
                # define a new neighborhood:
                neighborhood_counter += 1
                neighborhood[new_Node] = neighborhood_counter
                
                #make new edges connecting node to neighbors
                for node in neighbors:
                    graph.add_edge(new_Node,node)
                    buildNeighborhood(graph, node, neighborhood_counter, neighborhood, visited) 
                    
    for edge in graph.edges():
        if neighborhood[edge[0]] != neighborhood[edge[1]]:
            graph.remove_edge(edge[0],edge[1])
                    
    return neighborhood 
 
