# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:27:31 2025

@author: ciara

Trying to finite difference solve for a different equation:
    
-d^phi/dx^2 + dphi/dx = 1

Just need to add 2nd part to matrix

"""

#import a sparse solver instead for sparse matrix
from scipy.sparse.linalg import spsolve

#added new import
from scipy.sparse import lil_matrix

from numpy import zeros, linspace, exp, abs
import matplotlib.pyplot as plt

from time import time



#Parameters
#Length of whole thing we are integrating over
length = 1.0
#Number of points
nx = 100

#Set of interval numbers (x values) in between the length (-L/2 and L/2)
xval = linspace(-length/2, length/2, num=nx)

#Setting the spacing of element we are integrating over
#Each spacing between the x values is the same, so just using first 2
dx = xval[1]-xval[0]


#Create the (here, square) empty (sparse type) matrix operator M of dimensions nx
M = lil_matrix((nx, nx))


#Define the rhs vector, rho.
#We want to compare our solution to this?
rho = zeros(nx)+1.

#Set the elements in the matrix but skip boundary conditions
#Elements are determined by the second order difference formula
#Size of matrix
for i in range(1, nx-1):
    #Only set diagonal elements
    #sets previous value (A) 
    M[i, i-1] = -1.0/dx**2 - 0.5/dx
    #Sets current value (B)
    M[i, i] = 2.0/dx**2
    #Sets following value (C)
    M[i, i+1] = -1.0/dx**2 + 0.5/dx

#true for dirichlet, false for neumann
dirichlet = False

#analytic solution






#Lower -- Dirichlet:0 (we always want to use zero-Dirichlet for the lower boundary)
M[0, 0] = 1.0 ; rho[0] = 0.0

if dirichlet:
    #Upper -- Dirichlet
    #Use constant value, same as lower bounds
    M[-1, -1] = 1.0
else:
    #Use Neumann boundary for lower, meaning a constant gradient (constant rho value)
    #Neumann upper boundary uses forward difference formula instead of backward difference.
    #This means signs are swapped
    M[-1, -1] = 1./dx
    M[-1, -2] = -1./dx
  
#This is what it would be for both Dirichlet and Neumann
rho[-1] = 1.0



#include times before and after so we can find the difference

#change the format of M from lil to CSR so 
M = M.tocsr()
t1 = time()
    
#Solve for rho!
solution = spsolve(M, rho)
t2 = time()

print('Time taken to solve: '+ str(t2-t1))

#Substitute solution back in to calculate rhs vector (rho)
rhocheck = M.dot(solution)

#Print the error - use abs for full value and find difference between check (calculated) value and defined value
print('Max absolute error is {num}'.format(num=abs(rhocheck[1:-1]-rho[1:-1]).max()))

#Plot the solution
plt.plot(xval, solution, '-')
plt.xlabel(r'$x$') ; plt.ylabel(r'$\phi$')
plt.show()