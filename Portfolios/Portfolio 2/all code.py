# -*- coding: utf-8 -*-
"""

Created on Tue Nov 25 13:40:11 2025

@author: zks524

Portfolio 2

- Solves the advection-diffusion equation for a plasma
- need to put the data into a sparse matrix to solve as a linear equation
- solving for rhs vector, rho


Plots:
    - For v=0.2, varying R
    - For v=-0.2, varying R, to compare to values for v=0.2

"""
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import linspace
import numpy as np

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


#calls the code to solve the equation and plots
def main():
    #figure to compare varying R for both v values
    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    
    
    #iterate over how many v values
    for idx, value in enumerate(v):
        #2 v values for 2 plots
        axPlot = ax[idx]
        #iterate over R values to plot together
        for co in R:
            #get the matrix with boundaries and values satisfying the advection-diffusion equation
            M = return_sparse_matrix(aVal=0.0, v=value, R=co)
        
            #change the format of M from lil to CSR so spsolve can work
            M = M.tocsr()
            
            #spsolve solves the sparse linear system: M f = rho for the function f (advection-diffusion eq.)
            solution = spsolve(M, rho)
            
            #get the error
            rhocheck = M.dot(solution)
            
            #Print the error - use abs for full value and find difference between check (calculated) value and defined value
            print('Max absolute error is {num}'.format(num=abs(rhocheck[1:-1]-rho[1:-1]).max()))
            
            #Plot the solution
            axPlot.plot(xval, solution, '-', label=f'R = {co}')
                

        #plotting for Fig 1 of varying R
        edge_idx = np.argmax(S)
        axPlot.axvline(x=xval[edge_idx], color='black', linestyle='--', alpha=0.4, label='Tokamak edge')
        center_idx = np.argmin(S)
        axPlot.axvline(x=xval[center_idx], color='red', linestyle='--', alpha=0.4, label='Plasma center')
        axPlot.axvspan(xval[center_idx], xval[edge_idx], color='yellow', alpha=0.1, label='Plasma')
        axPlot.legend(loc='lower center')
        axPlot.grid(alpha=0.3)
        axPlot.set_title(fr'Plot for advection speed, $v={value}$')
           
        
    fig.text(0.5, 0.04, r'Distance from plasma centre, $x$', fontsize=12, ha='center')
    fig.text(0.04, 0.5, 'Pressure, P', fontsize=12, va='center', rotation='vertical')
    fig.text(0.5,0.0, r'Fig. 1: Steady state pressure profile for a plasma in a tokamak given by equation: $D\frac{\partial^2P}{\partial x^2} + v\frac{\partial P}{\partial x} + S + RP = 0$', ha='center', fontsize=14)
    fig.suptitle('Pressure against distance from plasma centre for a plasma in a tokamak.', fontsize=16)
    
    plt.show()


#run the code
if __name__ == '__main__':
    main()



