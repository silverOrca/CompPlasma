from scipy.sparse import lil_matrix
length = 1.0
# Size of box
N = 10
# Number of grid points
dx = length / (N -1)
# Size of step
A = lil_matrix (( N , N ))
# Create sparse matrix ( lil )
# Set the middle of matrix ( no BC )
for i in range (1 ,N -1):
    A [i ,i -1] = 1.0/ dx **2 + 1.0/(2* dx ) # Acts on f_i -1
    A [i , i ] = -2.0/ dx **2
    # Acts on f_i
    A [i , i +1] = 1.0/ dx **2 - 1.0/(2* dx ) # Acts on f_i +1

from numpy import zeros
rhs = zeros ( N ) -3 # Creates a vector of length N
                     # with all elements set to -3

# Set Dirichlet boundary , fixed to 0
A [0 ,0] = 1.0
A [N -1 ,N -1] = 1.0
rhs [0] = 0.0
rhs [N -1] = 0.0

# Convert from LIL to CSR format
A = A.tocsr()
# Use a sparse solve routine from scipy
from scipy.sparse.linalg import spsolve
# Solve
solution = spsolve (A , rhs )

# Calculate A . solution to recalculate rhs
rhs_check = A.dot ( solution )
# Print each value of rhs and rhs_check
# they should be the same if solution is correct
for i in range ( N ):
    print ( str ( i )+ " : " + str ( rhs [ i ])+ " " + str ( rhs_check [ i ]))

import matplotlib.pyplot as plt
from numpy import linspace
plt.plot ( linspace (0.0 , length , N ) , solution )
plt.show ()
