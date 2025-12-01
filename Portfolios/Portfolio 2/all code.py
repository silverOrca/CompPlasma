# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 13:40:11 2025

@author: zks524

Portfolio 2

- need to put the data into a sparse matrix
- solving for rhs vector, rho


"""
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import zeros, linspace
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

d = 0.2

v = [0.2, -0.2]
r = [0, -0.1, -0.2]

s = 100 * (1 - np.tanh(-5*(xval - 0.75)))


#steady state transport equation



#returns a sparse matrix of the input x grid
#also given diffusion coefficient, D, and advection speed, v
#has neumann lower boundary and dirichlet upper boundary

def return_sparse_matrix(aVal):
    #create the empty sparse matrix
    M = lil_matrix((nx, nx))
    
    #set boundaries
    #lower boundaries - 2nd order neumann
    M[0,0] = -1.5/dx
    M[0,1] = 2/dx
    M[0,3] = -0.5/dx
    
    #upper boundary - dirichlet
    M[-1,-1] = aVal
    
    return M



M = return_sparse_matrix(aVal=0.0)

#change the format of M from lil to CSR so spsolve can work
M = M.tocsr()
    
    
    