#Imports
from scipy.linalg import solve
from numpy import zeros,linspace,exp,abs
import matplotlib.pyplot as plt

#Parameters
length = 1.0 ; nx = 100
xval=linspace(-length/2,length/2,num=nx)
dx = xval[1]-xval[0]

#Create matrix operator and rhs vector
M = zeros((nx,nx))

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

#Solve
sol = solve(M,rho)

#Substitute solution back in to calculate RHS
rc=M.dot(sol)
#Print error
print("Max absolute error is {num}".format(num=abs(rc[1:-1]-rho[1:-1]).max()))
#Plot solution
plt.plot(xval,sol,"-") ; plt.xlabel(r"$x$")
plt.ylabel(r"$\phi$") ; plt.show()
