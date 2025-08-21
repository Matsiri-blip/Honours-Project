import numpy as np
import matplotlib.pyplot as plt

def Graph(n,k):

## put the vector into an adjacency matrix
  def  adjacency_matrix(n,k):
    ##conditions for the matrix
    ##for putting vector into the matrix
    ##first define the pattern in which it is going to be put numbers for the z
    ##binary vector
    m =int((n**2-n)/2)
    binary_vec = np.zeros(m)
    binary = bin(k)[2:]
    for i in range(len(binary)):
      binary_vec[m-i-1] = int(binary[::-1][i])
    start = 0
    difference = n-1
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
    ###define a function for vertices
  x = -np.cos(2 * np.pi * np.arange(n) / n+np.pi/2)
  y = np.sin(2 * np.pi * np.arange(n) / n+np.pi/2)
  if n < 2:
      plt.scatter([0],[0], s=400, color='black', zorder=2)
      plt.text(0, 0, f'$v_1$', fontsize=15, color='white', horizontalalignment='center', verticalalignment='center')
      plt.axis('off')
      plt.show()
      return
  for i in range(n):
    plt.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
    plt.text(x[i],y[i],"$v_{{{}}}$".format(i+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center')
    plt.text(0,0,f'{k}',fontsize = 20, horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
  A=adjacency_matrix(n,k)
  for i in range(n):                                                          ##looping through the vertices
    for j in range(n):
      if A[i][j]==1:                                                           ## check if there is an edge between the edges using adjacency matrix defined                                               ## if there is an edge add the vertices into a list
        plt.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)

      else:
        plt.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)


  return plt.show()


#[1+4+32+64+512]
