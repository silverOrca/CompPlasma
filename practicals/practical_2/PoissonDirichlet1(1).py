#Imports
from scipy.sparse import lil_matrix
from numpy import zeros,linspace,exp,abs,dot
import matplotlib.pyplot as plt
from time import time

#Parameters
length = 1.0 ; nx = 10000
xval=linspace(-length/2,length/2,num=nx)
dx = xval[1]-xval[0]

#Do we use a dense approach?
dense = True #Make this False to use sparse approach

#Create matrix operator
if not dense:
    M = lil_matrix((nx,nx))
else:
    M = zeros((nx,nx))

#Create right hand side vector
rho = zeros(nx)

#Set elements, -d^2/dx^2  -- Skip boundaries
rho = exp(-((xval)**2)/1.0e-2) #Gaussian rho
for i in range(1,nx-1):
    M[i,i-1] = -1.0/dx**2
    M[i,i]   =  2.0/dx**2
    M[i,i+1] = -1.0/dx**2
    
#Boundaries
#/Lower -- Dirichlet:0
M[0,0] = 1.0 ; rho[0] = 0.0
#/Upper -- Dirichlet:0.05
M[-1,-1] = 1.0 ; rho[-1] = 0.05

#Final setup and get solve routine
if not dense:
    M = M.tocsr()
    from scipy.sparse.linalg import spsolve as solve
else:
    from scipy.linalg import solve

#Solve (and time)
t1 = time() ; sol = solve(M,rho) ; t2 = time()
print("Time for solve: {tt}s".format(tt=t2-t1))

#Substitute solution back in to calculate RHS and print error
if not dense:
    rc=M*sol
else:
    rc=dot(M,sol)
print("Max absolute error is {num}".format(num=abs(rc[1:-1]-rho[1:-1]).max()))
#Plot solution
plt.plot(xval,sol,"-") ; plt.xlabel(r"$x$")
plt.ylabel(r"$\phi$") ; plt.show()
