from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve as solve
from numpy import zeros,linspace,exp,polyfit,log,sqrt,mean,array
import matplotlib.pyplot as plt
from time import time

#Define problem size
nx = ny = 100
length = 1.0 #Assume square box for simplicity
xval = yval = linspace(0,length,nx)
dx = dy = xval[1]
#Setup "flattened" problem
nz = nx * ny
M = lil_matrix((nz,nz))
def xyToZ(ix,iy,nx,ny):
    return ix*ny+iy

#Now create rhs
q = zeros(nz)

#Populate centre of matrix
t0=time()
for ix in range(1,nx-1):
    for iy in range(1,ny-1):
        #Find indices, this point and ix+/-1, iy+/-1
        indx = xyToZ(ix,iy,nx,ny)
        indx_xp1 = xyToZ(ix+1,iy,nx,ny)
        indx_xm1 = xyToZ(ix-1,iy,nx,ny)     
        indx_yp1 = xyToZ(ix,iy+1,nx,ny)
        indx_ym1 = xyToZ(ix,iy-1,nx,ny)
        
        #Set values
        M[indx,indx] = -4.0 / dx
        M[indx,indx_xp1] = 1.0/dx
        M[indx,indx_xm1] = 1.0/dx
        M[indx,indx_yp1] = 1.0/dx
        M[indx,indx_ym1] = 1.0/dx
        
#Set boundary values
for ix in [0,nx-1]: #Loop over x-boundaries
    for iy in range(ny): #All y values
        indx = xyToZ(ix,iy,nx,ny)
        q[indx] = 1.0
        M[indx,indx] = 1.0
for iy in [0,ny-1]: #Loop over y-boundaries
    for ix in range(nx): #All x values
        indx = xyToZ(ix,iy,nx,ny)
        M[indx,indx] = 1.0
t1=time()
#Convert and solve
M=M.tocsr() ; t2=time() ; sol = solve(M,q); t3=time()
#Make solution 2D
solView = sol.reshape((nx,ny))
#Plot
plt.contourf(xval,yval,solView,64)
plt.xlabel(r"$x$") ; plt.ylabel(r"$y$")
plt.colorbar(label="Temperature") ; plt.show()
#Print timing data
print("Time to populate matrix : {tim}s".format(tim=t1-t0))
print("Time to solve           : {tim}s".format(tim=t3-t2))
