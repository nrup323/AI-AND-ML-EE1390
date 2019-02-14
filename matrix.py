import numpy as np
import matplotlib.pyplot as plt

#given centroid is the origin
G = np.array([0,0])

#finding point of intesection for given normal vectors and y intercepts
def intersect(n1, n2, c1, c2):
    N = np.vstack((n1,n2))
    C = np.array([c1,c2])
    return np.matmul(np.linalg.inv(N), C)

dvec = np.array([-1,1])    	 #1*2
omat = np.array([[0,-1],[1,0]])  #2*2


#(1,1)*x  = 2
#n1 is the normal vector of given line
n1 = (np.array([1,1]))

# n2 is the normal vector of the line passing through origin and  perpendicular to n1
n2 = np.matmul(omat,n1)  #2*1

#x-int and y-int made by intersection of the line with the axes
nx = np.array([0,1])            #normal to x-axis
ny = np.array([1,0])            #normal to y-axis
B = intersect(n1,nx,2,0)	#x-int
C = intersect(n1,ny,2,0)	#y-int

#point of intesection of line from centroid and the given line
X = intersect(n1, n2, 2, 0)
print('X = ', X)

#distance between centroid and the given base line
dst  = np.linalg.norm(G-X)
print('distance GX = ', dst)

#finding the side length using trigonometry
a = 2*3**(0.5)*dst
print('side length = ', a)
#finding the area 
area = ((3**(0.5))/4)*a**(2)
print('area of the triagle is ', area)

#finding the third vertex using section formula
# G = (1*A + 2*X)/(2+1)
A = 3*G - 2*X

len = 10
lam = np.linspace(0, 1, len)

x_BC = np.zeros((2,len))
x_AX = np.zeros((2,len))

for i in range(len):	
	temp2 = B + lam[i]*(C-B)
	x_BC[:,i] = temp2.T
	temp4 = A + lam[i]*(X-A)
	x_AX[:,i] = temp4.T

#plottiing the the given line lying between the axes
plt.plot(x_BC[0,:],x_BC[1,:], label = '$BC = x + y = 2$')

#plotting the median line passing through the origin
plt.plot(x_AX[0,:],x_AX[1,:], label = '$AX$')

#labeling the points
plt.plot(A[0], A[1], 'o')
plt.text(A[0]*(1+0.05), A[1]*(1+0.05), 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0]*(1+0.02), B[1]*(1+0.02), 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0]*(1+0.05), C[1]*(1+0.02), 'C')
#labeling the foot of the perpendicular
plt.plot(X[0], X[1], 'o')
plt.text(X[0]*(1+0.05), X[1]*(1+0.02), 'X')
#labeling the origin/centroid
plt.plot(G[0], G[1], 'o')
plt.text(G[0]*(1+0.05), G[1]*(1-0.5), 'G')


plt.grid()
plt.show()






