#
# Demonstrates how to solve a 2D Laplacian equation using
# SciPy's sparse matrix libraries
#
# d2/dx2 + d2/dy2 = 0 
#
# with X boundaries set to 1, and Y boundaries to 0
#
# Equation is discretized using Finite Differences as a matrix equation
#
#  A x = b
#
# where b is zero in the domain and sets the value of the boundaries,
# and x is the solution
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, bicg
from numpy import ones, zeros, arange

# Size of the domain
nx = 159
ny = 101

# Create a sparse matrix
N = nx * ny              # Number of points
A = lil_matrix((N, N))   # Linked list format. Useful for creating matrices

# Function to map from (x,y) location to row/column index
def indx(i,j):
    return i*ny + j
    
# Fixed boundaries, so set everything to the identity matrix
A.setdiag(ones(N))
    
# Loop over the core of the domain
for x in range(1,nx-1):
    for y in range(1,ny-1):
        i = indx(x,y)
        # d2/dx2 + d2/dy2
        A[i, i]           = -4.
        A[i, indx(x+1,y)] = 1.
        A[i, indx(x-1,y)] = 1.
        A[i, indx(x,y+1)] = 1.
        A[i, indx(x,y-1)] = 1.
        
# Convert to Compressed Sparse Row (CSR) format for efficient solving
A = A.tocsr()
    
# Create a vector for b, containing all zeros
b = zeros(N)
    
# Set the values on x=0 and x=nx-1 boundaries to 1
b[indx(0, arange(ny))] = 1.
b[indx(nx-1, arange(ny))] = 1.

# Direct solve sparse matrix
x = spsolve(A, b)
    
# View X as a 2D array. This avoids copying the array
xv = x.view()
xv.shape = (nx, ny)

import matplotlib.pyplot as plt  # Plotting library
import matplotlib.cm as cm       # For the color maps

# Create a filled contour plot, note we transpose to make x be on the x-axis
im = plt.imshow(xv.T, interpolation='bilinear', origin='lower', cmap=cm.viridis)
plt.xlabel('x') ; plt.ylabel('y')
plt.colorbar() # Add a scale
plt.show() # Display interactively

