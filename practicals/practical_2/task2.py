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

from numpy import zeros, linspace, exp, abs, sqrt, log, array, mean, polyfit
import matplotlib.pyplot as plt

from time import time


#analytic solution
def anaSol(xVal, aVal, dirichlet=False):
    #computes the coefficients depending on dirichlet or neumann boundary conditions
    #I had to figure out c1 and c2 with pen and paper
    #Because length = 1, and goes from -0.5 to 0.5, these are the boundary x values
    #Boundary condition for lower x boundary is always done with Dirichlet, and phi=0, hence why c2 is the same
    #While for upper x boundary, uses aVal (whichever chosen value), where for Dirichlet, phi=aVal, but for 
    #Neumann, dphi/dx = Aval
    if dirichlet:
        c1 = (aVal - 1)/(exp(0.5)-exp(-0.5))
    else:
        c1 = (aVal - 1)/exp(0.5)    
        
    c2 = 0.5 - c1 * exp(-0.5)    

    #this is the given analytic solution
    anaSolution = c1*exp(xVal) + c2 + xVal
    
    
    return(anaSolution)


#numerical solution
def numSol(dx, M, rho, dirichlet=False, secondOrder=True):
    #Boundaries
    #Lower -- Dirichlet:0 (we always want to use zero-Dirichlet for the lower boundary)
    M[0, 0] = 1.0
    rho[0] = 0.0

    if dirichlet:
        #Upper -- Dirichlet
        #Use constant value, same as lower bounds
        M[-1, -1] = 1.0
    else:
        #Use Neumann boundary for lower, meaning a constant gradient (constant rho value)
        #Neumann upper boundary uses forward difference formula instead of backward difference.
        #This means signs are swapped
        if secondOrder:
            M[-1, -1] = 1.5/dx
            M[-1, -2] = -2/dx
            M[-1, -3] = 0.5/dx
        else:
            M[-1, -1] = 1./dx
            M[-1, -2] = -1./dx
        

    #change the format of M from lil to CSR so spsolve can work
    M = M.tocsr()
    
    #include times before and after so we can find the difference
    t1 = time()
        
    #Solve for rho!
    solution = spsolve(M, rho)
    t2 = time()
    
    print('Time taken to solve: '+ str(t2-t1))
    #Substitute solution back in to calculate rhs vector (rho)
    rhocheck = M.dot(solution)

    #Print the error - use abs for full value and find difference between check (calculated) value and defined value
    print('Max absolute error is {num}'.format(num=abs(rhocheck[1:-1]-rho[1:-1]).max()))
    
    
    return solution



#Parameters, run solve funcs, and error between numerical and analytic
def solve(nx, aVal=0.05, dirichlet=False):
    #Length of whole thing we are integrating over
    length = 1.0
    #Number of points is passed in as nx
    #Set of interval numbers (x values) in between the length (-L/2 and L/2)
    xVal = linspace(-length/2, length/2, num=nx)
    #Setting the spacing of element we are integrating over
    #Each spacing between the x values is the same, so just using first one
    dx = xVal[1] - xVal[0]

    #Create the (here, square) empty (sparse type) matrix operator M of dimensions nx
    M = lil_matrix((nx, nx))

    #Define the rhs vector, rho. We want to compare our solution to this
    rho = zeros(nx) + 1.0

    #Set the elements in the matrix but skip boundary conditions
    #Elements are determined by the second order difference formula
    for i in range(1, nx-1):
        # sets previous value (A)
        M[i, i-1] = -1.0/dx**2 - 0.5/dx
        # Sets current value (B)
        M[i, i]   =  2.0/dx**2
        # Sets following value (C)
        M[i, i+1] = -1.0/dx**2 + 0.5/dx

    #true for dirichlet, false for neumann (controlled by caller)
    dirichlet = False
    secondOrder = True
    
    #upper boundary condition value is passed as aVal
    rho[-1] = aVal

    #Solve for numeric and analytic solutions
    numeric = numSol(dx, M, rho, dirichlet, secondOrder)
    analytic = anaSol(xVal, aVal, dirichlet)
    
    plt.plot(xVal, numeric, '-')
    plt.plot(xVal, analytic, '+')
    plt.show()

    #calculate the rms error (use mean because otherwise it's absolute pointwise error)
    rms = sqrt(mean((analytic - numeric)**2))
    
    
    return xVal, numeric, analytic, rms



#Want to plot rms error vs nx to show how error changes for bigger grid sizes (split into more points)
nx_values = array([10,20,30,35,40,45,50,100,500,1000,5000,10000])
errors = []
for nx in nx_values:
    #solve for each mesh size
    x, numericSolution, analyticSolution, rms = solve(int(nx), aVal=0.05, dirichlet=False)
    errors.append(rms)


#Plot errors
plt.figure()
plt.loglog(nx_values, errors, '-o')
plt.xlabel('nx')
plt.ylabel('rms error')
plt.title('Error over larger grid size')
plt.grid(True, which='both', ls=':')
plt.show()

#Fit log to get error over grid sizes in number
coeffs = polyfit(log(nx_values), log(errors), 1)
print('Error scales with order ~', coeffs[0])