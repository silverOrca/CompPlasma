# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:33:04 2025

@author: ciara

Task 1 of practical 2: Modify the starting code to use sparse matrices,
then use time.time to time the solution step and compare to time for dense n_x = 10,000

Sparse is a method to use to speed up computations of pretty sparse matrices, such as what we
are working with for finite difference solving, in which we only have values down the diagnonal
and the rest of the matrix is populated by zeros.
"""
#Could move these to their respective if statements

from scipy.linalg import solve

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
nx = 10000

#Set of interval numbers (x values) in between the length (-L/2 and L/2)
xval = linspace(-length/2, length/2, num=nx)

#Setting the spacing of element we are integrating over
#Each spacing between the x values is the same, so just using first 2
dx = xval[1]-xval[0]

dense = False

if not dense:
    #Create the (here, square) empty (sparse type) matrix operator M of dimensions nx
    M = lil_matrix((nx, nx))
else:
    #create regular full (dense) matrix
    M = zeros((nx, nx))

#Define the rhs vector, rho. Gaussian in this case.
#We want to compare our solution to this?
#(If not using an array in definition, use zeros(nx) and then whatever you want to add to it)
rho = exp(-((xval)**2)/1.0e-2)

#Set the elements in the matrix but skip boundary conditions
#Elements are determined by the second order difference formula
#Size of matrix
for i in range(1, nx-1):
    #Only set diagonal elements
    #sets previous value (A) 
    M[i, i-1] = -1.0/dx**2
    #Sets current value (B)
    M[i, i] = 2.0/dx**2
    #Sets following value (C)
    M[i, i+1] = -1.0/dx**2
    

#Set boundaries - want fixed value so use Dirichlet, in which corner values 
#are 1 and rho values are 0. This is to make phi = a where a is rho, to give
#us a solution for rho for the edges which we can use to solve generally

#Lower -- Dirichlet:0
M[0,0] = 1.0 ; rho[0] = 0.0

#Upper -- Dirichlet:0.05
M[-1, -1] = 1.0 ; rho[-1] = 0.05


#include times before and after so we can find the difference
if not dense:
    #change the format of M from lil to CSR so 
    M = M.tocsr()
    t1 = time()
    #Solve for rho!
    solution = spsolve(M, rho)
    t2 = time()
else:
    t1=time()
    solution = solve(M, rho)
    t2=time()

print('Time taken to solve: '+ str(t2-t1))

#Substitute solution back in to calculate rhs vector (rho)
rhocheck = M.dot(solution)

#Print the error - use abs for full value and find difference between check (calculated) value and defined value
print('Max absolute error is {num}'.format(num=abs(rhocheck[1:-1]-rho[1:-1]).max()))

#Plot the solution
plt.plot(xval, solution, '-')
plt.xlabel(r'$x$') ; plt.ylabel(r'$\phi$')
plt.show()