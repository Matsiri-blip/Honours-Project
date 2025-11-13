import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import time
import os

def edges(n):
  edges = list(combinations(range(n), 2))                   ##from n choose 2 formula returns all the edges
  return edges
def total_edges(num_vertices):                         #total edges function,
  return int((num_vertices**2-num_vertices)/2)                    # returns total edges of a graph

def binary_string(num_vertices,graph_id):          ##define binary string, for inputting into oracle
    num_edges = total_edges(num_vertices)                      #recall total edges function
    binary_string = bin(graph_id)[2:].zfill(num_edges) #fill zeros for each element to match with number of edges
    return binary_string                                   #return binary string

def binary_vector(num_vertices,graph_id):          ##define binary matrix 
    num_edges = total_edges(num_vertices)                      #recall tota edges function
    binary_vec = np.zeros(num_edges)
    binary = bin(graph_id)[2:].zfill(num_edges) # Use zfill to pad with leading zeros
    for edge in range(num_edges):
        binary_vec[edge] = int(binary[edge])
    return binary_vec                       #returns binary string of a number with m total bits
  
def  adjacency_matrix(num_vertices,graph_id):                 #define adjacency matrix, using binary matrix to fill in upper triangular terms

  A_matrix = np.zeros((num_vertices,num_vertices))                       #initialize binary matrix
  b_vector = binary_vector(num_vertices,graph_id)                    #recall binary matrix
  start = 0                                 #start at index 0 from the binary matrix

  for vertex in range(num_vertices-1):                         ##this loop is to return the list of the pattern
      length_cut = num_vertices-1-vertex                        ##length of cutting matrix matching with elements from binary vector
      end = start + length_cut                  ##end of the length cut,
      A_matrix[vertex][vertex + 1:] = b_vector[start:end]
      start = end                              ##to ensure that the start restarts to the previous end
##loop for ensuring symmetric
  for vertex_1 in range(num_vertices):
      for vertex_2 in range(num_vertices):
          A_matrix[vertex_2,vertex_1] = A_matrix[vertex_1,vertex_2]
  return A_matrix

##draw graph function, returns a plot of a graph on n vertices
def graph(num_vertices,graph_id):

  # Create a new figure and axes
  fig, ax = plt.subplots()
  Adj_matrix =adjacency_matrix(num_vertices,graph_id)                                 ##recall adjacency matrix function
  if num_vertices%2 != 0:
     x_vals = -np.cos(2 * np.pi * np.arange(num_vertices) / num_vertices+np.pi/2)       ##points where the vertices will be placed
     y_vals = np.sin(2 * np.pi * np.arange(num_vertices) / num_vertices+np.pi/2)         ## these points are around a circle
  else:
     x_vals = -np.cos(2 * np.pi * np.arange(num_vertices) / num_vertices-np.pi/4)       ##points where the vertices will be placed
     y_vals = np.sin(2 * np.pi * np.arange(num_vertices) / num_vertices-np.pi/4)

  for vertex in range(num_vertices):
    ax.scatter(x_vals[vertex],y_vals[vertex], label =f"{vertex+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
    ax.text(x_vals[vertex],y_vals[vertex],"$v_{{{}}}$".format(vertex+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center') #labelling vertices


  for vertex_1 in range(num_vertices):                                                          ##looping through the vertices
    for vertex_2 in range(num_vertices):
      if Adj_matrix[vertex_1][vertex_2]==1:                                                 ##if there is an edge between vertices
        ax.plot([x_vals[vertex_1],x_vals[vertex_2]],[y_vals[vertex_1],y_vals[vertex_2]],color = 'blue', linewidth = 3,zorder=1)  ##add a blue line between the vertices with edges

      else:
        ax.plot([x_vals[vertex_1],x_vals[vertex_2]],[y_vals[vertex_1],y_vals[vertex_2]],color = 'red',linewidth = 3, zorder=1)    ##if a clique does not exist add a red edge
  
  ax.text(0,0,f'{graph_id}',fontsize = 20, horizontalalignment='center', verticalalignment='center')   ##label graphs
  ax.axis('off')
##return a graph with n vertices, labeled 'graph_id'

#def save graph function
def save_graph(num_vertices, graph_id, folder_path):
    if not os.path.exists(folder_path):                ##create a folder if it does not exist
        os.makedirs(folder_path)        

    fig= graph(num_vertices, graph_id)                        ##recall Graph function
    
    image_name = f"counter_example_for_{num_vertices}_graph_{graph_id}.png"    ##name the graph counter example
    
    image_path = os.path.join(folder_path, image_name)                 

    plt.savefig(image_path, dpi=300, bbox_inches='tight')        

    
    return plt.close(fig)

def verifies_cliques(num_vertices,graph_id,r_clique_size,b_clique_size):                  ##def a clique identifying function
  binary_vec = binary_vector(num_vertices,graph_id)                       ##define binary vector
  edges = list(combinations(range(num_vertices), 2))                       ##define edges in graph
  edge_to_index = {edge: i for i, edge in enumerate(edges)}               ##map each edge to a num
  b_clique_vertices = list(combinations(range(num_vertices),b_clique_size))   ##choose different way of selecting subgraphs of size 'b_clique_size'
  b_clique_indices = []                                                     ##initialize clique_indices list
  r_clique_vertices = list(combinations(range(num_vertices),r_clique_size))   ##choose different way of selecting subgraphs of size 'r_clique_size'
  r_clique_indices = []                                                     ##initialize clique_indices list
  for vertices in r_clique_vertices:                                        ##for each subgraph of size 'r_clique_size'
    subgraph_edges = list(combinations(vertices, 2))                       ##select different ways to arrange edges from subgraph
    indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges]    #choose indices from defined subgraph_edges
    r_clique_indices.append(indices)
  for vertices in b_clique_vertices:                                        ##for each subgraph of size 'r_clique_size'
    subgraph_edges = list(combinations(vertices, 2))                       ##select different ways to arrange edges from subgraph
    indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges]    #choose indices from defined subgraph_edges
    r_clique_indices.append(indices)                                            
  for indices in r_clique_indices:                                           ##for indices in clique size edges 
    if all(binary_vec[idx] == 0 for idx in indices):                       #verifies if all edges are connected
      return 0                                                             ##returns        
  for indices in b_clique_indices:                                           ##for indices in clique size edges 
    if all(binary_vec[idx] == 1 for idx in indices):                       #verifies if all edges are connected
      return 0                                                             ##returns        
  return 1


def get_clique_edge_indices(num_vertices,a_sized_clique):
  num_qubits = total_edges(num_vertices)
  edges = list(combinations(range(num_vertices), 2))                       ##define edges in graph
  edge_to_index = {edge: i for i, edge in enumerate(edges)}               ##map each edge to a number
  a_cliques = list(combinations(range(num_vertices),a_sized_clique))   ##choose different way of selecting subgraphs of size 'a_clique_size'
  a_clique_indices = []                                                     ##initialize clique_indices list
  for vertices in a_cliques:
    subgraph_edges = list(combinations(vertices, 2))
    indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges]
    a_clique_indices.append(indices)
  return a_clique_indices
def ramsey_numbers(r_sized_clique,b_sized_clique):
  folder_path = f"graphs/R({r_sized_clique},{b_sized_clique})"
  num_vertices = 2
  while True:
    num_edges = total_edges(num_vertices)  ##total number of edges
    total_number_graphs = 2**num_edges            ##total number of graphs
    #calculate clique indices for the current n
    edges = list(combinations(range(num_vertices), 2))                       ##define edges in graph 
    edge_to_index = {edge: i for i, edge in enumerate(edges)}     ##map each edge to a number
    
##define ways of selecting vertices
    r_clique_vertices = list(combinations(range(num_vertices), r_sized_clique))           ##choose a way to select r vertices using combinations
    r_clique_indices = []                                     
    for vertices in r_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))                    ##select r vertices to verify if they have a clique
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      r_clique_indices.append(indices)                                    #append to r_clique_indices list

    b_clique_vertices = list(combinations(range(num_vertices), b_sized_clique))
    b_clique_indices = []
    for vertices in b_clique_vertices:
      subgraph_edges = list(combinations(vertices, 2))
      indices = [edge_to_index[tuple(sorted(e))] for e in subgraph_edges] # Ensure consistent edge representation
      b_clique_indices.append(indices)                                    #append to r_clique_indices list
    
    is_ramsey_number = True

    start_time = time.time()                     ##define start of algorithm
    start_k = 0
    for graph in range(start_k,total_number_graphs):  ##iterating over total number of graphs
      
      current_graph_label = binary_vector(num_vertices, graph)
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
        print(f'{num_vertices} is not ramsey number R{(r_sized_clique,b_sized_clique)}, because graph {graph} has no clique')
        save_graph(num_vertices,graph,folder_path)            ##save the current graph
        
        break  
    final_time = time.time()              
    change_in_time =   final_time-start_time
    if is_ramsey_number:
        final_time = time.time()
        change_in_time = final_time - start_time
        print(f"All graphs with n = {num_vertices} vertices have a monochromatic clique.")
        print(f"{num_vertices} = R{(r_sized_clique,b_sized_clique)} and time taken to find R{(r_sized_clique,b_sized_clique)} was {change_in_time} seconds.")
        break  # Exit the while loop
    start_k = graph
    num_vertices += 1                                    ##increase n if one graph does not contain a clique


def edge_values(n,k):
  binary_vec = binary_vector(n, k)    ##call binary vector function
  edge_list = edges(n)     ##define map edges
  edge_values = {edge: int(binary_vec[idx]) for idx, edge in enumerate(edge_list)}     ##enumerate each edge to the binary matrix
  return edge_values                    ##returns edges and their values using binary matrix


## create a function a function that counts the total cliques of a graph.
def energy(n, b, r, graph_number):
    binary_vec = binary_vector(n,graph_number)                                        ##define binary vector
    graph_edges = edges(n)                                                        ##define the total edges
    edge_value = edge_values(n,graph_number)                                                ##define map edges
    total_cliques = 0                                                               ##initialize objective
    penalty = 1

    # Count the total number of Blue cliques
    for edge in combinations(range(n), b):
        blue_edges = [(min(i, j), max(i, j)) for i, j in combinations(edge, 2)]      ##selects different ways to arrange b edges from n
        if all(edge_value.get(graph_edge, 0) == 1 for graph_edge in blue_edges):     ##if the edges selected are equal to 1 then add penalty to objective
            total_cliques += penalty                                                 ##add 1 per clique found

    # Count the total number of Red cliques
    for edge in combinations(range(n), r):
        red_edges = [(min(i, j), max(i, j)) for i, j in combinations(edge, 2)]
        if all(edge_value.get(graph_edge, 0) == 0 for graph_edge in red_edges):      ##if the edges selected are equal to 0 then add penalty to objective
            total_cliques += penalty                                                 ##add 1 per clique found

    return total_cliques # Return the total_cliques
  
def plot_energy(n,s,r):
    fig, ax = plt.subplots()
    m = total_edges(n)
    graph_numbers = np.arange(2**m)
    energy_values = np.array([energy(n, s, r, i) for i in range(2**m)])
    ax.bar(graph_numbers,energy_values, clip_on=False,color = 'maroon')
    ax.scatter(graph_numbers,energy_values, clip_on=False,color = 'black')
    ax.set_title(f"Total cliques found in graphs of order {n}")
    ax.set_xlabel('Graphs')
    ax.set_ylabel("Clique count")
    ax.set_xlim(right=2**m-1)
    ax.set_yticks(np.arange(0,max(energy_values) , 2))
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylim(top=max(energy_values))
    ax.grid('True')
    ax.set_ylim(bottom=0)
    plt.savefig(f"clique_distribution_{n}.png")
