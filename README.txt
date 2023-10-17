Readme for hierarchical clustering algs
Tucker Hale

The hierarchy_cluster.ipynb file contains two methods which perform hierarchical clustering on a list of tuples. 
The first method, hierarchical_cluster, has O(n^3) time complexity, as shown by the pseudocode and recurrence relation below.
The second implementation reduces the time complexity to O(n^2) by reducing matrix calculations/queries, but is currently not operational. It is described in this paper: https://sites.cs.ucsb.edu/~veronika/MAE/summary_SLINK_Sibson72.pdf
This relationship is demonstr


hierarchy_cluster_(list of tuples)
1. Treat each data point as its own cluster
2. Calculate a pairwise distance matrix 
3. Find the smallest distance and merge the two clusters together
4. Record the new cluster and update  the matrix
5. Loop steps 3 & 4 until one cluster remains
6. Draw the “dendrogram”

Recurrence Relation:

T(n)=t(n-1) + n^2
T(1)=1
T(n-1) = t(n-2) + (n-1)^2
T(n) = (t(n-2) + (n-1)^2) + n^2
T(n) = T(1) + n(n+1)(2n+1)/6
O(n^3)


hierarchy_cluster_dissimilarity(list of tuples)
1. Treat all data points as unique clusters
2. Calculate a pairwise distance matrix, as well as arrays populated with a) each nodes’ closest relative and b) the distances between them
3. Find the smallest element in the  distance array and merge the closest clusters, updating all data structures.
4. Use the matrix to determine the new values of the distance and closest neighbor arrays
5. Repeat steps 3-4 until one cluster remains
6. Draw the dendrogram

Recurrence Relation:

T(n)=t(n-1) + ~n
T(1)=1
T(n-1) = t(n-2) + n
T(n) = (t(n-2) + n) + n
T(n) = T(1) + n*n
O(n^2)

