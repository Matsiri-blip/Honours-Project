
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
def adjacency_matrix(n,k):
  def binary_matrix(n,k):
    m = (n**2-n)/2
    binary = int(bin(k)[2:])
    binary_vec = [int(d) for d in str(binary)]
    l = len(binary_vec)
    if m!=len(binary_vec):
        horizontal_matrix = np.concatenate((np.zeros(int(m-l)),binary_vec))
    else:
        horizontal_matrix = np.array([int(i) for i in str(binary)])
    ##this returns the binary matrix
    return horizontal_matrix
## put the vector into an adjacency matrix
  ##conditions for the matrix
  ##for putting vector into the matrix
  ##first define the pattern in which it is going to be put numbers for the z
  ##binary vector
  pattern = [0]
  difference = n-1
  for _ in range(1, n): ##this loop is to return the list of the pattern
      next_number = pattern[-1] + difference ##here if n=3, the next number should be
      pattern.append(next_number) ##putting each number into a list
      difference = difference - 1 ## ensure that the difference increase like the matrix
  ##initialize the matrices
  A = np.zeros((n,n))
  B = binary_matrix(n,k)
  ##this loops through the A-matrix and binary vector
  ##putting each element on the matrix is as we defined our labeling
  for i in range(n-1):
    A[i][1+i:] = B[pattern[i]:pattern[i+1]]

  for l in range(n):
      for j in range(n):
          A[j,l] = A[l,j]
  return A
def Graph(n,k):
  def vertices(n):
    l =[]
    m =[]
    for i in range(n):
      l.append(float(np.cos(-2*np.pi*i/n+np.pi))) 	             ##x = cos(2pi/n)
      m.append(float(np.sin(-2*np.pi*i/n+np.pi)))                ##y = sin(2pi/n) both divide by n for number of vertices
    return l,m
  x = vertices(n)[0]
  y = vertices(n)[1]
  def edges(n,k):                                                              ##using the vertice function defined
    for i in range(n):                                                              ##looping through the vertices
      for j in range(n):                                                            ##looping through the vertices
        if n <2:
          return [0]
        elif adjacency_matrix(n,k)[i][j]==1:                                          ## check if there is an edge between the edges using adjacency matrix defined                                               ## if there is an edge add the vertices into a list
          plt.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)
          plt.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)
          plt.axis('off')
          plt.text(x[i]-0.04,y[i]-0.04,f'$v_{i+1}$',fontsize = 15,color = 'white')
          plt.text(0,0,f'{k}',fontsize = 20)
        else:
          plt.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)
          plt.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)
          plt.axis('off')
          plt.text(x[i]-0.04,y[i]-0.04,f'$v_{i+1}$',fontsize = 15,color = 'white')
          plt.text(0,0,f'{k}',fontsize = 20)
    plt.show()
  return edges(n,k)


r1 = 3
r2 = 3   #verifying for r
n = 5   # starting with 4 vertices
m = int((n**2-n)/2)  ##total number of graphs
ll = np.zeros(2**m) ## this is for identifying whether a graph has a clique
red_clique1 = np.zeros((r1,r1), int)    ## for the independent set
blue_clique1 = np.ones((r1,r1), int)
np.fill_diagonal(blue_clique1,0)        ##for the monochromatic clique


red_clique2 = np.zeros((r2,r2), int)    ## for the independent set
blue_clique2 = np.ones((r2,r2), int)
np.fill_diagonal(blue_clique2,0)        ##for the monochromatic clique     ##for the monochromatic clique
for k in range(2**m):  ##iterating over total number of graphs
    current_graph_matrix = adjacency_matrix(n, k)

    clique_found = False    ##intialize for when a clique exists in the graph
    ##combination of different was to choose vertices takes n choose r
    comb1 = list(combinations(range(n), r1))  ##this depends on r1
    comb2 = list(combinations(range(n), r2))  ##this ones depends on r2
    # Check for monochromatic subgraphs of size r1
    for vertices in comb1:
        sub_matrix = current_graph_matrix[np.ix_(vertices, vertices)]    ##chooses the combinations from the nxn matrix
        if np.array_equal(sub_matrix, red_clique1) or np.array_equal(sub_matrix, blue_clique1):  ##if a clique or an empty graph exists 
            clique_found = True                                                                ##then the variable clique_found updates to True
            break   
    if clique_found:                      ##if graph has clique update the ll vector to one 
        ll[k] = 1
        continue  
    # Check for monochromatic subgraphs of size r2
    for vertices in comb2:
        sub_matrix = current_graph_matrix[np.ix_(vertices, vertices)]      
        if np.array_equal(sub_matrix, red_clique2) or np.array_equal(sub_matrix, blue_clique2):
            clique_found = True
            break

    if clique_found:    ## this one says if there is a clique of r2xr2 then update the ll vector to one
        ll[k] = 1
for i in np.where(ll == 0)[0]:
  Graph(n,i)
