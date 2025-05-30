import numpy as np
import matplotlib.pyplot as plt
import os
folder_path = "graphs/order_4"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created folder: {folder_path}")
def Graph(n,k):
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
  def  adjacency_matrix(n,k):
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
    ###define a function for vertices
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
          #plt.text(0,0,f'{k}',fontsize = 20)
        else:
          #plt.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)  
          plt.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2) 
          plt.axis('off')
          plt.text(x[i]-0.04,y[i]-0.04,f'$v_{i+1}$',fontsize = 15,color = 'white')
    plt.show()
  def save_figures(n, colorings):
        m = int((n**2 - n) / 2)
        for idx, k in enumerate(colorings):
            # Create a new figure for each coloring
            edges(n, k)  # Generate the plot
            # Save the figure
            image_name = f"k{n}_coloring_{k}.png"
            image_path = os.path.join(folder_path, image_name)
            try:
                plt.savefig(image_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {image_path}")
            except Exception as e:
                print(f"Error saving {image_path}: {e}")
            # Close the plot to free memory
            plt.close()

    # Call save_figures with a list containing the single k value
  return edges(n,k)
n = 4
k=27
Graph(n,27)
#[1+4+32+64+512]
