# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:56:38 2025

@author: ciara

Solving a simple equation of the form -d^2phi/dx^2 = rho(x)

Writing it in the form M phi = rho, so need to specify what M and rho are
To then calculate phi, the function (sometimes labelled f)

Uses a dense matrix, M
"""

from scipy.linalg import solve
from numpy import zeros, linspace, exp, abs
import matplotlib.pyplot as plt

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

#Create the (here, square) empty matrix operator M of dimensions nx
M = zeros((nx, nx))

#Define the rhs vector, rho. Gaussian in this case.
#We want to compare our solution to this?
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


#Solve for rho!
solution = solve(M, rho)

#Substitute solution back in to calculate rhs vector (rho)
rc = M.dot(solution)

#Print the error - use abs to calculate this
print('Max absolute error is {num}'.format(num=abs(rc[1:-1]-rho[1:-1]).max()))

#Plot the solution
plt.plot(xval, solution, '-')
plt.xlabel(r'$x$') ; plt.ylabel(r'$\phi$')
plt.show()