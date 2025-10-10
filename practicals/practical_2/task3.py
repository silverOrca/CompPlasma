# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:11:27 2025

@author: ciara

Task 3: solving a 2d heat equation

Finite difference approx. uses: ix, iy, ix+1, ix-1, iy+1, iy-1

Equation is k delta^2 u - q = 0
With k = 1, this rearranges to: delta^2 u = q
"""

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import zeros, linspace

import matplotlib.pyplot as plt

k = 1.0

nx = 100
ny = 100
nz = nx * ny

#need q to be an array because we want to change boundary values
q = zeros(nz)

#length is 1 by 1 by 1, so square box
length = 1.0

xVal = yVal = linspace(0, length, nx)


dx = dy = xVal[0] - xVal[1]
denom = dx**2

#create the matrix of z dimensions
M = lil_matrix((nz, nz))

#Function to map from (x,y) to (z)
def xyToz(ix,iy,nx,ny):
    #It inputs the current x, current y, nx and ny
    #It outputs those in terms of current z
    return ix*ny + iy


#fill the diagonal of the 2d matrix
for x in range(1,nx-1):
    for y in range(1,ny-1):
        #first need the current x and y in terms of z so use mapping func
        curZ = xyToz(x, y, nx, ny)
        #fda also uses x+1, x-1, y+1, y-1 so need these as z too
        zforXplus1 = xyToz(x+1, y, nx, ny)
        zforXminus1 = xyToz(x-1, y, nx, ny)
        zforYplus1 = xyToz(x, y+1, nx, ny)
        zforYminus1 = xyToz(x, y-1, nx, ny)
        
        #now fill the z matrix from the central difference formula (collecting terms)
        M[curZ, curZ] = -4/denom
        M[curZ, zforXplus1] = 1/denom
        M[curZ, zforXminus1] = 1/denom
        M[curZ, zforYplus1] = 1/denom
        M[curZ, zforYminus1] = 1/denom
        
        
#set the boundary values using dirichlet method.
#need to loop again because we have the boundaries in x and y so need to convert to z.
#need to loop twice as well because we are going through 2d values
#Loop over the x boundaries for all y values, q=1, and dirichlet value=1.0
for x in [0, nx-1]:
    for y in range(ny):
        curZ = xyToz(x, y, nx, ny)
        q[curZ] = 1.0
        M[curZ, curZ] = 1.0
        
#loop over all the y boundaries for all y values, dirichlet=1.0
for y in [0, ny-1]:
    for x in range(nx):
        curZ = xyToz(x, y, nx, ny)
        M[curZ, curZ] = 1.0
        
#Sparse matrix so need to convert to CSR and then solve
M = M.tocsr()
solution = spsolve(M, q)

#Now to plot.

#Need to convert from 1d solution back into 2d using reshape   
solution2d = solution.reshape((nx, ny))

#contour plot
plt.contourf(xVal, yVal, solution2d, 256)
plt.show()



















