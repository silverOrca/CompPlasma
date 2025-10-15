#
# Lax-Wendroff 2nd order method for hyperbolic problems
#
# Solving  dy/dt = -v dy/dx
#
# with fixed v and uniform, periodic 1D grid
#

def step(y, t, dx, dt, v):
    """ Take a single step """
    n = len(y)
    result = y.copy() # An array for the result
    
    # Loop over grid points
    for i in range(n):
        im = (i - 1 + n) % n   # Left index, wrap around
        ip = (i + 1) % n
        result[i] += -0.5*(dt/dx) * v * (y[ip] - y[im]) + 0.5*(dt/dx * v)**2 * ( y[ip] - 2.*y[i] + y[im] )
    return result

from numpy import zeros

def solve(method, y0, dx, v, tarray, factor=0.8):
    """ Call a stepping method until desired output times are reached """
    y = y0.copy() # Make a copy of the starting values
    time = 0.     # Starting time
    maxdt = factor* dx / v
    
    nt = len(tarray)
    nx = len(y0)
    result = zeros([nt, nx])
    for i, t in enumerate(tarray):
        # Take steps until we reach time t
        while time < t:
            dt = min([t - time, maxdt])    # Choose the time step
            y = step(y, time, dx, dt, v)   # Take a single step
            time += dt
        result[i,:] = y
    return result

from numpy import exp, linspace, zeros
import matplotlib.pyplot as plt

nx = 100
v = 1.0

x = linspace(0,1,nx, endpoint=False) # Periodic box of length 1
dx = x[1] - x[0]

y0 = exp( -((x - 0.3) / 0.1)**2 )  # Gaussian shape
#y0 = zeros([nx])
y0[(6*nx//8):(7*nx//8)] = 1.0   # Top-hat function

tarray = [0,0.1,1]

y = solve(step, y0, dx, v, tarray)

for i,t in enumerate(tarray):
    plt.plot(x, y[i,:], label="t = %.2f" % (t))

plt.xlabel("x") ; plt.ylabel("f")
plt.legend(loc='upper center')
plt.show()
