import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import os
def total_edges(n):
  return int((n**2-n)/2)
def  adjacency_matrix(n,k):
  ##conditions for the matrix
  ##for putting vector into the matrix
  ##first define the pattern in which it is going to be put numbers for the z
  ##binary vector
  m =total_edges(n)
  binary_vec = np.zeros(m)
  binary = bin(k)[2:]
  for i in range(len(binary)):
    binary_vec[m-i-1] = int(binary[::-1][i])
  start = 0
  A = np.zeros((n,n))
  B = binary_vec
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
  return plt.close(fig)
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
  folder_path = f"graphs_1/R({s},{r})"
  n = 2
  while True:
    m = total_edges(n)  ##total number of edges possible
    red_clique = np.zeros((r,r), int)    ## for the independent set
    
    blue_clique = np.ones((s,s))-np.eye(s)       ##for the monochromatic clique
    
    is_ramsey_number = True

    start_time = time.time()

    for k in range(2**m):  ##iterating over total number of graphs
      current_graph_matrix = adjacency_matrix(n, k)
      clique_found = False    ##intialize for when a clique exists in the graph
      ##combination of different was to choose vertices takes n choose r
      comb1 = list(combinations(range(n), r))  ##this depends on r
      comb2 = list(combinations(range(n), s))  ##this ones depends on s
      # Check for monochromatic subgraphs of size r1
      for vertices in comb1:
          sub_matrix = current_graph_matrix[np.ix_(vertices, vertices)]    ##chooses the combinations from the nxn matrix
          if (np.array_equal(sub_matrix, red_clique)):                                               ##checks if a red clique  exists 
              clique_found = True                                                                ##then the variable clique_found updates to True
              break
      for vertices in comb2:
        sub_matrix = current_graph_matrix[np.ix_(vertices, vertices)]    ##chooses the combinations from the nxn matrix
        if (np.array_equal(sub_matrix, blue_clique)):                                                ##if a blue clique exists 
          clique_found = True                                                                ##then the variable clique_found updates to True
          break   

      if not clique_found:
        is_ramsey_number = False
        print(f'{n} is not ramsey number R{(s,r)}, because graph {k} has no clique')
        print(Graph(n,k))
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

ramsey_numbers(3, 3)
ramsey_numbers(3, 4)
ramsey_numbers(3, 5)
ramsey_numbers(3, 6)
ramsey_numbers(3, 7)
ramsey_numbers(3, 8)
ramsey_numbers(3, 9)
ramsey_numbers(4, 4)
ramsey_numbers(4, 5)

