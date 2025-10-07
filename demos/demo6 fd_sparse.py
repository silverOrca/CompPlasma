#
# Example of a Finite Difference solver
#

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy import e, exp

def solve(x, a, b, c, d, boundary=True):
    N = len(x)
    # Create a matrix A which is NxN
    A = lil_matrix((N, N))
    rhs = d.copy()
    
    # Central differencing
    for i in range(1, N-1):
        dx = 0.5*(x[i+1] - x[i-1])  # Assuming approximately uniform grid
        A[i, i-1] = a[i]/dx**2 - 0.5*b[i]/dx
        A[i, i]   = -2.*a[i]/dx**2 + c[i]
        A[i, i+1] = a[i]/dx**2 + 0.5*b[i]/dx
    # Inner boundary fixed to zero (Dirichlet)
    A[0,0] = 1.
    rhs[0] = 0.  # Set value to zero
    
    # Outer boundary set to zero gradient (Neumann)
    if boundary:
        # Use second-order accuracy on the boundary
        A[N-1, N-3] = 0.25 / dx
        A[N-1, N-2] = -1.0 / dx
        A[N-1, N-1] = 0.75 / dx
    else:
        # First order accuracy on the boundary
        A[N-1,N-1] = 1./dx
        A[N-1,N-2] = -1./dx
    rhs[N-1] = 0. # Set gradient to 0
    
    # Convert to Compressed Sparse Row (CSR) format for efficient solving
    A = A.tocsr()
    
    # Direct solve sparse matrix
    y = spsolve(A, rhs)
    return y

if __name__ == "__main__":
    from numpy import linspace, zeros, array, log
    import matplotlib.pyplot as plt
    
    boundary = True # Use 2nd-order accurate boundary conditions?

    x = linspace(0, 1, 40)
    y = -exp(x-1.) + 1./e + x # exact solution
    exact = y[-1] # Exact final value
    plt.plot(x, y, label="Exact solution")

    error = []
    nxvals = array([3,5,10,20])
    for nx in nxvals:
        x = linspace(0, 1, nx)
        z = zeros([nx]) # convenient to have an empty array of size nx
    
        # Set the coefficients
        a = z - 1.
        b = z + 1.
        c = z
        d = z + 1.
    
        # Create matrix and solve
        y = solve(x, a, b, c, d, boundary=boundary)
        error.append(abs(y[-1] - exact))
        plt.plot(x, y, label="nx = %d" % (nx))
        
    # Use the last two values to estimate the order
    order = log(error[-1]/error[-2]) / log((nxvals[-2]-1.)/(nxvals[-1]-1.))
    print("Order of accuracy = %.2f" % (order))
    plt.legend(loc='lower right')
    plt.xlabel("X")
    plt.ylabel("Y")

    # Create a small sub-figure for accuracy
    a = plt.axes([0.2, 0.5, 0.15, 0.35])
    plt.plot(nxvals-1, error,marker='x')
    plt.xlabel("Nx")
    plt.ylabel("Error")
    plt.yscale('log')
    plt.xscale('log')
    plt.text(nxvals[1], error[0], "Order %.1f" % order)
    plt.show()
    
