import numpy as np
import matplotlib.pyplot as plt


##Create a random adjacent matrix
## creating a random matrix
n = 4
a  = np.random.randint(2,size = (n,n))
np.random.seed(42)

##ensure that the matrix is symmetric 
for i in range(n):
    for j in range(n):
        a[i,j]=a[j,i]

##diagonals are 0
np.fill_diagonal(a,0)



### IDEA: Turn a number into binary, then turn the binary into a vector"


##firstly, i define the number of elements in the upper triangular matrix

def binary_matrix(n,k):
    #define the number of elements in the upper triangular matrix.
    m = (n**2-n)/2
    #use the bin function turning numbers into binary
    binary = int(bin(k)[2:])
    binary_vec = [int(d) for d in str(binary)]
    l = len(binary_vec)
    if m!=len(binary_vec): 
        horizontal_matrix = np.concatenate((np.zeros(int(m-l)),binary_vec))
    else:
        horizontal_matrix = np.array([int(i) for i in str(binary)])
    ##here this returns the binary matrix
    return horizontal_matrix
## put the binary vector onto the adjacency matrix
def  adjacency_matrix(n,k):
    ##initialize the matrix
    A = np.zeros((n,n))
    B = binary_matrix(n,k)
    ##this loops through the A-matrix and binary vector
    ## making sure that each element on the matrix is as we defined our labeling
    for i in range(n-1):
        if i==0: 
            A[i][1+i:] = B[:n-i-1]
        else:
            A[i][1+i:] = B[n-i-1:n-i-1+len(A[i][1+i:])]
        for l in range(n):
            for j in range(n):
                A[j,l] = A[l,j]
        
    return A
##Drawing graphs

#putting the vertex at (0,0)
print("n = 1")
x0 = np.array([0])
y0 = np.array([0])
plt.plot(x0,y0,'o')
plt.axis('off')
plt.title('GRAPH OF 1 VERTEX')
plt.show()

##for n = 2
n = 2


x1 =np.array([0,1])
y1 = np.array([0,0])
print("n = 2")
plt.scatter(x1,y1)
plt.text(-0.01,0.01,'1')
plt.text(0.99,0.01,'2')
plt.text(0.5,-0.01,'0',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 2 VERTEX')
print(binary_matrix(2,0))
print(adjacency_matrix(n,0))
plt.show()




plt.title('GRAPH OF 2 VERTEX')
plt.scatter(x1,y1)
plt.text(-0.01,0.01,'1')
plt.text(0.99,0.01,'2')
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95) 
plt.text(0.5,-0.01,'1',fontsize = 15)
plt.axis('off')
print(binary_matrix(2,1))
print(adjacency_matrix(n,1))
plt.show()

n = 3
##N = 3
print("n = 3")
x2 =np.array([0,1,2])
y2 = np.array([0,2,0])
##conditions for diagonal lines
x12 = np.array([0,1])
y12 = np.array([0,2])
x23 = np.array([1,2])
y23 = np.array([2,0])

plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.text(1,1,'0',fontsize = 15)
plt.title('GRAPH OF 3 VERTEX')
print(adjacency_matrix(n,0))
print(binary_matrix(3,0))
plt.show()


plt.axhline(y = 0, xmin = 0.05, xmax = 0.95) ###between vertices 1,3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.98,0.05,'3')
plt.text(1,1,'1',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(adjacency_matrix(n,1))
print(binary_matrix(3,1))
plt.show()



plt.plot(x23,y23)##edge between 2&3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(1,1,'2',fontsize = 15)
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(adjacency_matrix(n,2))
print(binary_matrix(3,2))
plt.show()


plt.plot(x23,y23)##edge between 2&3 
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.scatter(x2,y2)
plt.text(1,1,'3',fontsize = 15)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(adjacency_matrix(n,2))
print(binary_matrix(3,3))
plt.show()



plt.plot(x12,y12)##edge between 1,2
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1.99,0.05,'3')
plt.text(1,1,'4',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(adjacency_matrix(n,4))
print(binary_matrix(3,4))
plt.show()



plt.plot(x12,y12)##edge between 1,2
plt.plot(x23,y23)##edge between 2&3 
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'5',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(binary_matrix(3,5))
plt.show()



plt.plot(x12,y12)##edge between 1,2
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'6',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
print(binary_matrix(3,6))
plt.show()



plt.plot(x12,y12)##edge between 1,2
plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)##and also 1&3
plt.plot(x23,y23)##edge between 2&3 
plt.scatter(x2,y2)
plt.text(-0.01,0.01,'1')
plt.text(0.99,2.05,'2')
plt.text(1,1,'7',fontsize = 15)
plt.text(1.99,0.05,'3')
plt.axis('off')
plt.title('GRAPH OF 3 VERTEX')
plt.show()
print(binary_matrix(3,7))




##N = 4

print('n=4')

##here i define the edges' coordinates
x3 =np.array([0,0,1,1])
y3 = np.array([0,1,1,0])
##conditions for diagonal lines
x23 = np.array([0,1])
y23 = np.array([0,1])
x14 = np.array([0,1])
y14 = np.array([1,0])


##define the edges
## to save time so that each plot is recalled later
def plot_12():
    return plt.axhline(y = 1, xmin = 0.05, xmax = 0.95)
def plot_13():
    return plt.axvline(x = 0, ymin = 0.05, ymax = 0.95) 
def plot_14():
    return plt.plot(x14,y14)
def plot_23():
    return plt.plot(x23,y23)
def plot_24():
    return plt.axvline(x = 1, ymin = 0.05, ymax = 0.95) 
def plot_34():
    return plt.axhline(y = 0, xmin = 0.05, xmax = 0.95)
## using this as reference [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
n = 4
#0
plt.scatter(x3,y3)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.01,0.05,'3')
plt.text(0.99,0.05,'4')
plt.axis('off')
plt.text(0.5,0.5,'0',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,0))


#1
plt.scatter(x3,y3)
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.01,0.05,'3')
plt.text(0.99,0.05,'4')
plt.text(0.5,0.5,'1',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,1))


#2
plt.scatter(x3,y3)
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'2',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,2))


#3

plt.scatter(x3,y3)
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'3',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,3))


#4
plt.scatter(x3,y3)
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'4',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,4))

#5
plt.scatter(x3,y3)
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'5',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,5))


#6
plt.scatter(x3,y3)
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'6',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,6))

#7
plt.scatter(x3,y3)
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'7',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,7))


#8
plt.scatter(x3,y3)
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'8',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,8))
#9
plt.scatter(x3,y3)
plot_34()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'9',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,9))
#10
plt.scatter(x3,y3)
plot_14()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'10',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,10))

#11
plt.scatter(x3,y3)
plot_14()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'11',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,11))


#12
plt.scatter(x3,y3)
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'12',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,12))

#13
plt.scatter(x3,y3)
plot_14()
plot_34()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'13',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,13))


#14
plt.scatter(x3,y3)
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'14',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,14))


#15
plt.scatter(x3,y3)
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'15',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,15))


#16
plt.scatter(x3,y3)
plot_13()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'16',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,16))

#17
plt.scatter(x3,y3)
plot_13()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'17',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,17))


#18
plt.scatter(x3,y3)
plot_13()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'18',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,18))

#19
plt.scatter(x3,y3)
plot_13()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'19',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,19))

#20
plt.scatter(x3,y3)
plot_13()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'20',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,20))


#21
plt.scatter(x3,y3)
plot_13()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'21',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,21))


#22
plt.scatter(x3,y3)
plot_13()
plot_23()
plot_24()
plt.text(0.5,0.5,'22',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,22))

#23
plt.scatter(x3,y3)
plot_13()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'23',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,23))

#24
plt.scatter(x3,y3)
plot_13()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'24',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,24))


#25

plt.scatter(x3,y3)
plot_13()
plot_14()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'25',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,25))

#26
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'26',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,26))


#27
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'27',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,27))


#28
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'28',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,28))



#29
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'29',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,29))



#30
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'30',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,30))



#31
plt.scatter(x3,y3)
plot_13()
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'31',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,31))


#32

plt.scatter(x3,y3)
plot_12()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'32',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,32))



#33
plt.scatter(x3,y3)
plot_12()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'33',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,33))


#34
plt.scatter(x3,y3)
plot_12()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'34',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,34))


#35
plt.scatter(x3,y3)
plot_12()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'35',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,35))


#36
plt.scatter(x3,y3)
plot_12()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.text(0.5,0.5,'36',fontsize = 15)
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,36))

#37
plt.scatter(x3,y3)
plot_12()
plot_34()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'37',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,37))

#38
plt.scatter(x3,y3)
plot_12()
plot_24()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'38',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,38))


#39

plt.scatter(x3,y3)
plot_12()
plot_23()
plot_24()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.text(0.5,0.5,'39',fontsize = 15)
plt.text(0.5,0.5,'2',fontsize = 15)
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,39))

#40

plt.scatter(x3,y3)
plot_12()
plot_14()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(0.5,0.5,'40',fontsize = 15)
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,40))

#41
plt.scatter(x3,y3)
plot_12()
plot_14()
plot_34()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'41',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,41))


#42
plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(0.5,0.5,'42',fontsize = 15)
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,42))


#43
plt.scatter(x3,y3)
plot_12()
plot_14()
plot_34()
plot_24()
plt.text(-0.01,1.01,'1')
plt.text(0.5,0.5,'43',fontsize = 15)
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,43))


#44

plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plt.text(0.5,0.5,'44',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,44))

#45

plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_34()
plt.text(0.5,0.5,'45',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,45))


#46

plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_24()
plt.text(0.5,0.5,'46',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,46))

#47


plt.scatter(x3,y3)
plot_12()
plot_14()
plot_23()
plot_24()
plot_34()
plt.text(0.5,0.5,'47',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,47))

#48

plt.scatter(x3,y3)
plot_12()
plot_13()
plt.text(0.5,0.5,'48',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,48))

#49
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_34()
plt.text(0.5,0.5,'49',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,49))


#50
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_24()
plt.text(0.5,0.5,'50',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,50))


#51

plt.scatter(x3,y3)
plot_12()
plot_13()
plot_34()
plot_24()
plt.text(0.5,0.5,'51',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,51))

#52

plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plt.text(0.5,0.5,'51',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,52))

#53
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_34()
plt.text(0.5,0.5,'53',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,53))


#54
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_24()
plt.text(0.5,0.5,'54',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,54))


#55
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_23()
plot_34()
plot_24()
plt.text(0.5,0.5,'55',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,55))


#56

plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plt.text(0.5,0.5,'56',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,56))

#57
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_34()
plt.text(0.5,0.5,'57',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,57))


#58
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_24()
plt.text(0.5,0.5,'58',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,58))


#59

plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_24()
plot_34()
plt.text(0.5,0.5,'59',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,59))

#60
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plt.text(0.5,0.5,'60',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,60))


#61
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_34()
plt.text(0.5,0.5,'61',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,61))


#62
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_24()
plt.text(0.5,0.5,'62',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,62))



#63
plt.scatter(x3,y3)
plot_12()
plot_13()
plot_14()
plot_23()
plot_34()
plot_24()
plt.text(0.5,0.5,'63',fontsize = 15)
plt.text(-0.01,1.01,'1')
plt.text(0.99,1.05,'2')
plt.text(-0.03,0.03,'3')
plt.text(1.01,0,'4')
plt.axis('off')
plt.title('GRAPH OF 4 VERTEX')
plt.show()
print(binary_matrix(4,63))
