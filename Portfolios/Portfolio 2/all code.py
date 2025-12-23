# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:40:11 2025

@author: zks524

Portfolio 2

- need to put the data into a sparse matrix
- solving for rhs vector, rho


Plots:
    - For no R, v=0.2 and v=-0.2 (can use same code and have R set to 0)
    - For v=0.2, varying R

"""
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import linspace
import numpy as np
from numpy.linalg import det

import matplotlib.pyplot as plt


#length of box
L_x = 1.0

#Number of points
nx = 10000

#Set of interval numbers (x values) in between the length (0 and L_x)
xval = linspace(0, L_x, num=nx)

#Setting the spacing of element we are integrating over
#Each spacing between the x values is the same, so just using first 2
dx = xval[1]-xval[0]

D = 0.2     #diffusion coefficient

v = [0.2, -0.2]     #advection flow

S = 100 * (1 - np.tanh(-5*(xval - 0.75)))       #net source

R = [0, -0.1, -0.2]     #reaction coefficient

#rhs vector, equal to what the 2nd order difference equals
rho = -S

#returns a sparse matrix of the input x grid
#also given advection speed v and reaction coefficient R
#has neumann lower boundary and dirichlet upper boundary
def return_sparse_matrix(aVal, v, R):
    #create the empty sparse matrix
    M = lil_matrix((nx, nx))
    
    #set the values in middle of matrix (2nd order diff)
    for i in range(1, nx-1):
        # sets previous value (A)
        M[i, i-1] = (D/dx**2) - ((0.5*v)/dx)
        # Sets current value (B)
        M[i, i]   =  -(2.0*D)/dx**2 + R
        # Sets following value (C)
        M[i, i+1] = (D/dx**2) + ((0.5*v)/dx)
    
    
    #set boundaries
    #lower boundaries - 2nd order neumann
    M[0,0] = -1.5/dx
    M[0,1] = 2/dx
    M[0,2] = -0.5/dx
    rho[0] = aVal
    
    
    #upper boundary - dirichlet (zero-Dirichlet means M= 1)
    M[-1,-1] = 1.0
    rho[-1] = aVal
    
    return M


def main():
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    
    for idx, value in enumerate(v):
        axPlot = ax[idx]
        
        for co in R:
    
            M = return_sparse_matrix(aVal=0.0, v=value, R=co)
        
            #change the format of M from lil to CSR so spsolve can work
            M = M.tocsr()
             
            solution = spsolve(M, rho)
            
            rhocheck = M.dot(solution)
            
            #Print the error - use abs for full value and find difference between check (calculated) value and defined value
            print('Max absolute error is {num}'.format(num=abs(rhocheck[1:-1]-rho[1:-1]).max()))
            
            #Plot the solution
            axPlot.plot(xval, solution, '-', label=f'R = {co}')

        axPlot.legend(loc='lower left')
        axPlot.set_title(fr'Plot for advection speed, $v={value}$')
        axPlot.grid(alpha=0.3)
        
    plt.xlabel(r'$x$') ; plt.ylabel(r'\$')
    fig.text(0.5, 0.04, r'$x$', fontsize=8, ha='center')
    fig.text(0.04, 0.5, 'P', va='center', rotation='vertical')
    fig.suptitle('Pressure against position')
    
    plt.show()



if __name__ == '__main__':
    main()



