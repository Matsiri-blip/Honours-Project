import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import os
def total_edges(n):
  return int((n**2-n)/2)
def binary_vector(n,k):
    m = total_edges(n)
    binary_vec = np.zeros(m)
    binary = bin(k)[2:].zfill(m) # Use zfill to pad with leading zeros
    for i in range(m):
        binary_vec[i] = int(binary[i])
    return binary_vec
def  adjacency_matrix(n,k):
  ##conditions for the matrix
  ##for putting vector into the matrix
  ##first define the pattern in which it is going to be put numbers for the z
  ##binary vector
  m =total_edges(n)
  start = 0
  A = np.zeros((n,n))
  B = binary_vector(n,k)
  for i in range(n-1): ##this loop is to return the list of the pattern
      length_cut = n-1-i
      end = start + length_cut
      A[i][i + 1:] = B[start:end]
      start = end

  for l in range(n):
      for j in range(n):
          A[j,l] = A[l,j]
  return A
def Graph(n,k):

  # Create a new figure and axes
  fig, ax = plt.subplots()
  A =adjacency_matrix(n,k)
  x = -np.cos(2 * np.pi * np.arange(n) / n+np.pi/2)        ##points where the vertices will be placed
  y = np.sin(2 * np.pi * np.arange(n) / n+np.pi/2)         ## these points are around a circle
  if n < 2:
      ax.scatter([0],[0], s=400, color='black', zorder=2)
      ax.text(0, 0, f'$v_1$', fontsize=15, color='white', horizontalalignment='center', verticalalignment='center')
      ax.axis('off')
      ax.show()
      return
  for i in range(n):
    ax.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
    ax.text(x[i],y[i],"$v_{{{}}}$".format(i+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center')

  A=adjacency_matrix(n,k)
  for i in range(n):                                                          ##looping through the vertices
    for j in range(n):
      if A[i][j]==1:                                                           ## check if there is an edge between the edges using adjacency matrix defined                                               ## if there is an edge add the vertices into a list
        ax.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)

      else:
        ax.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)
  
  ax.text(0,0,f'{k}',fontsize = 20, horizontalalignment='center', verticalalignment='center')
  ax.axis('off')
  return fig
def save_graph(n, k, folder_path, r_s=None):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

    fig= Graph(n, k)
    
    if r_s:
        image_name = f"counter_example_R({r_s}).png"
    else:
        image_name = f"counter_example_for_{n}_graph_{k}.png"
    
    image_path = os.path.join(folder_path, image_name)
    
    try:
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {image_path}")
    except Exception as e:
        print(f"Error saving {image_path}: {e}")
    
    return plt.close(fig)

  
def ramsey_numbers(s,r):
  folder_path = f"graphs/R({s},{r})"
  n = 2
  while True:
    m = total_edges(n)  ##total number of edges possible
    #calculate clique indices for the current n
    edges = list(combinations(range(n), 2))                       ##define edges in graph
    edge_to_index = {edge: i for i, edge in enumerate(edges)}

    r_clique_vertices = list(combinations(range(n), r))
    r_clique_indices = []
    for vertices in r_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      r_clique_indices.append(indices)

    s_clique_vertices = list(combinations(range(n), s))
    s_clique_indices = []
    for vertices in s_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      s_clique_indices.append(indices)
    
    is_ramsey_number = True

    start_time = time.time()

    for k in range(2**m):  ##iterating over total number of graphs
      current_graph_matrix = binary_vector(n, k)
      clique_found = False    ##intialize for when a clique exists in the graph
      # Check for monochromatic subgraphs of size r1
      # Check for red r-cliques
      for indices in r_clique_indices:
        if all(current_graph_matrix[idx] ==0  for idx in indices):
          clique_found = True
          break
      # Check for red r-cliques
      for indices in s_clique_indices:
        if all(current_graph_matrix[idx] == 1 for idx in indices):
          clique_found = True
          break

      if not clique_found:
        is_ramsey_number = False
        print(f'{n} is not ramsey number R{(s,r)}, because graph {k} has no clique')
        save_graph(n,k,folder_path)
        
        
        break
    final_time = time.time()
    change_in_time =   final_time-start_time
    if is_ramsey_number:
        final_time = time.time()
        change_in_time = final_time - start_time
        print(f"All graphs with n = {n} vertices have a monochromatic clique.")
        print(f"{n} is R{(s,r)} and time taken to find R{(s,r)} was {change_in_time} seconds.")
        break  # Exit the while loop
    
    n += 1
ramsey_numbers(2, 2)
ramsey_numbers(2, 3)
ramsey_numbers(2, 4)
ramsey_numbers(2, 5)
ramsey_numbers(2, 6)
ramsey_numbers(2, 7)
ramsey_numbers(2, 8)
ramsey_numbers(2, 9)
ramsey_numbers(2, 10)
ramsey_numbers(3, 3)
ramsey_numbers(3, 4)
ramsey_numbers(3, 5)
ramsey_numbers(3, 6)
ramsey_numbers(3, 7)
ramsey_numbers(3, 8)
ramsey_numbers(3, 9)
ramsey_numbers(4, 4)
ramsey_numbers(4, 5)

