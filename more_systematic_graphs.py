import numpy as np
import matplotlib.pyplot as plt

def binary_matrix(n,k):
    m = (n**2-n)/2
    binary = int(bin(k)[2:])
    binary_vec = [int(d) for d in str(binary)]
    l = len(binary_vec)
    if m!=len(binary_vec):
        horizontal_matrix = np.concatenate((np.zeros(int(m-l)),binary_vec))
    else:
        horizontal_matrix = np.array([int(i) for i in str(binary)])
    ##here this returns the binary matrix
    return horizontal_matrix

##define adjacency matrix
def  adjacency_matrix(n,k):
  ##conditions for the matrix
  ##for putting vector into the matrix
  ##first define the pattern in which it is going to be put numbers for the 
  ##binary vector
  pattern = [0]
  difference = n-1
  for _ in range(1, n): ##this loop is to return the list of the pattern
      next_number = pattern[-1] + difference 
      pattern.append(next_number)              ##putting each number into a list
      difference = difference - 1             ## ensure that the difference increase like the matrix
  ##initialize the matrices
  A = np.zeros((n,n))
  B = binary_matrix(n,k) 
  ##this loops through the A-matrix and binary vector
  ##putting each element on the matrix is as we defined our labeling
  for i in range(n-1):
    A[i][1+i:] = B[pattern[i]:pattern[i+1]]

  for l in range(n):                              ##this ensures that the matrix is symmetric
      for j in range(n):
          A[j,l] = A[l,j]
  return A
###define a function for vertices
def vertices(n):
  l =[]
  m =[]
  for i in range(n):
    l.append(float(np.cos(2*np.pi*i/n))) 	        ##x = cos(2pi/n), puts in the x values in the list l.
    m.append(float(np.sin(2*np.pi*i/n)))                ##y = sin(2pi/n) both divide by n for number of vertices
  return l,m 					
