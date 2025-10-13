import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import os
def total_edges(n):                         #total edges function,
  return int((n**2-n)/2)                    # returns total edges of a graph

def binary_vector(n,graph_number):          ##define binary matrix 
    m = total_edges(n)                      #recall tota edges function
    binary_vec = np.zeros(m)
    binary = bin(graph_number)[2:].zfill(m) # Use zfill to pad with leading zeros
    for i in range(m):
        binary_vec[i] = int(binary[i])
    return binary_vec                       #returns binary string of a number with m total bits
  
def  adjacency_matrix(n,graph_number):                 #define adjacency matrix, using binary matrix to fill in upper triangular terms

  m =total_edges(n)                         ##call total edges function
  A_matrix = np.zeros((n,n))                       #initialize binary matrix
  b_vector = binary_vector(n,graph_number)                    #recall binary matrix
  start = 0                                 #start at index 0 from the binary matrix

  for i in range(n-1):                         ##this loop is to return the list of the pattern
      length_cut = n-1-i                        ##length of cutting matrix matching with elements from binary vector
      end = start + length_cut                  ##end of the length cut,
      A_matrix[i][i + 1:] = b_vector[start:end]
      start = end                              ##to ensure that the start restarts to the previous end
##loop for ensuring symmetric
  for l in range(n):
      for j in range(n):
          A_matrix[j,l] = A_matrix[l,j]
  return A_matrix

##draw graph function, returns a plot of a graph on n vertices
def Graph(n,graph_number):

  # Create a new figure and axes
  fig, ax = plt.subplots()
  Adj_matrix =adjacency_matrix(n,graph_number)                                 ##recall adjacency matrix function
  x = -np.cos(2 * np.pi * np.arange(n) / n+np.pi/2)        ##x points of the vertices
  y = np.sin(2 * np.pi * np.arange(n) / n+np.pi/2)         ##y-points of the vertices, these points are around a circle

  for i in range(n):
    ax.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
    ax.text(x[i],y[i],"$v_{{{}}}$".format(i+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center') #labelling vertices


  for i in range(n):                                                          ##looping through the vertices
    for j in range(n):
      if Adj_matrix[i][j]==1:                                                 ##if there is an edge between vertices
        ax.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)  ##add a blue line between the vertices with edges

      else:
        ax.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)    ##if a clique does not exist add a red edge
  
  ax.text(0,0,f'{graph_number}',fontsize = 20, horizontalalignment='center', verticalalignment='center')   ##label graphs
  ax.axis('off')
##return a graph with n vertices, labeled 'graph_number'

#def save graph function
def save_graph(n, graph_number, folder_path):
    if not os.path.exists(folder_path):                ##create a folder if it does not exist
        os.makedirs(folder_path)        

    fig= Graph(n, graph_number)                        ##recall Graph function
    
    image_name = f"counter_example_for_{n}_graph_{graph_number}.png"    ##name the graph counter example
    
    image_path = os.path.join(folder_path, image_name)                 

    plt.savefig(image_path, dpi=300, bbox_inches='tight')        

    
    return plt.close(fig)

  
def ramsey_numbers(r,b):
  folder_path = f"graphs/R({r},{b})"
  n = 2
  while True:
    m = total_edges(n)  ##total number of edges
    N = 2**m            ##total number of graphs
    #calculate clique indices for the current n
    edges = list(combinations(range(n), 2))                       ##define edges in graph 
    edge_to_index = {edge: i for i, edge in enumerate(edges)}     ##map each edge to a number
    
##define ways of selecting vertices
    r_clique_vertices = list(combinations(range(n), r))           ##choose a way to select r vertices using combinations
    r_clique_indices = []                                     
    for vertices in r_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))                    ##select r vertices to verify if they have a clique
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      r_clique_indices.append(indices)                                    #append to r_clique_indices list

    b_clique_vertices = list(combinations(range(n), b))
    b_clique_indices = []
    for vertices in b_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      b_clique_indices.append(indices)                                    #append to r_clique_indices list
    
    is_ramsey_number = True

    start_time = time.time()                     ##define start of algorithm        

    for graph in range(N):  ##iterating over total number of graphs
      
      current_graph_label = binary_vector(n, graph)
      clique_found = False    ##intialize for when a clique exists in the graph
      # Check for blue cliques
      for indices in b_clique_indices:
        if all(current_graph_label[idx] == 1 for idx in indices):   ##search for red cliques
          clique_found = True
          break                                                     ##if blue clique found go to the next graph
      ##if a blue clique not found then,
      # Check for red cliques
      for indices in r_clique_indices:
        if all(current_graph_label[idx] == 0  for idx in indices):   ##search for red cliques
          clique_found = True         
          break                                                     ##if a red clique found go to the next graph        


      if not clique_found:                          ##if a clique of either size r or size b is not found, n is not Ramsey number
        is_ramsey_number = False
        print(f'{n} is not ramsey number R{(r,b)}, because graph {graph} has no clique')
        save_graph(n,graph,folder_path)            ##save the current graph
        
        break  
    final_time = time.time()              
    change_in_time =   final_time-start_time
    if is_ramsey_number:
        final_time = time.time()
        change_in_time = final_time - start_time
        print(f"All graphs with n = {n} vertices have a monochromatic clique.")
        print(f"{n} = R{(r,b)} and time taken to find R{(r,b)} was {change_in_time} seconds.")
        break  # Exit the while loop
    
    n += 1                                    ##increase n if one graph does not contain a clique
