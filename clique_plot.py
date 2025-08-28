import numpy as np
import matplotlib.pyplot as plt
def plot_clique(A):
  fig, ax = plt.subplots()
  n = len(A)
  x = -np.cos(2 * np.pi * np.arange(n) / n+np.pi/2)        ##points where the vertices will be placed
  y = np.sin(2 * np.pi * np.arange(n) / n+np.pi/2)
  for i in range(n):
    plt.scatter(x[i],y[i], label =f"{i+1}",s=400, color = 'black', zorder=2)  ##drawing vertices
    plt.text(x[i],y[i],"$v_{{{}}}$".format(i+1),fontsize = 15,color = 'white', horizontalalignment='center', verticalalignment='center')

  for i in range(n):                                                          ##looping through the vertices
    for j in range(n):
      if A[i,j]==1:                                                           ## check if there is an edge between the edges using adjacency matrix defined                                               ## if there is an edge add the vertices into a list
        plt.plot([x[i],x[j]],[y[i],y[j]],color = 'blue', linewidth = 3,zorder=1)

      else:
        plt.plot([x[i],x[j]],[y[i],y[j]],color = 'red',linewidth = 3, zorder=1)
  
  plt.text(0,0,'clique',fontsize = 20, horizontalalignment='center', verticalalignment='center')
  plt.axis('off')
  return plt.show()
def red_clique(r):
    return np.zeros((r,r), int)    ## for the independent set
def blue_clique(s):
    return np.ones((s,s))-np.eye(s)  
