#
# 1D illustration of limiter methods for hyperbolic problems
# 
# Solving  dy/dt = -v dy/dx
#
# with fixed v and uniform, periodic grid
#

from numpy import isfinite

# These linear limiters reduce the method to a simpler scheme

def upwind(theta):
    """ Reverts to 1st order upwind method """
    return 0.0

def laxwendroff(theta):
    """ Revert to 2nd order Lax-Wendroff method """
    return 1.0

def beamwarming(theta):
    """ Revert to 2nd order Beam-Warming upwind method """
    if not isfinite(theta):
        return 0.0;
    return theta

# Remaining are nonlinear limiters. All are Total Variation Diminishing (TVD)
#gets us around Godunov's theorem

def vanleer(theta):
    if not isfinite(theta):
        return 0.0;
    return (theta + abs(theta)) / (1. + abs(theta))

def minmod(theta):
    if not isfinite(theta):
        return 0.0;
    if theta <= 0.:
        return 0.
    return min([theta, 1])

def superbee(theta):
    if not isfinite(theta):
        return 0.0;
    return max([ 0.0, min([1., 2.*theta]), min([2., theta]) ])

def MC(theta):
    if not isfinite(theta):
        return 0.0;
    return max([ 0.0, min([ (1.+theta)/2., 2., 2.*theta ]) ])

def step(y, t, dx, dt, v, limit=vanleer):
    """ Take a single step using a limited scheme """
    n = len(y)
    result = y.copy() # An array for the result
    
    s = v * dt/dx
    
    # Loop over grid points
    for i in range(n):
        im  = (i - 1 + n) % n   # Left index, wrap around
        imm = (i - 2 + n) % n
        ip  = (i + 1) % n
        ipp = (i + 2) % n
        
        #direction of flow
        if v > 0.:
            theta_m = (y[im] - y[imm]) / (y[i] - y[im])
            theta_p = (y[i] - y[im]) / (y[ip] - y[i])       #below, pass into limiter
            result[i] += -s*(y[i] - y[im]) - 0.5*s*(1.-s) * (limit(theta_p)*(y[ip] - y[i]) - limit(theta_m)*(y[i] - y[im]) )
        else:
            theta_m = (y[ip] - y[i]) / (y[i] - y[im])
            theta_p = (y[ipp] - y[ip]) / (y[ip] - y[i])
            result[i] += -s*(y[ip] - y[i]) + 0.5*s*(1.+s) * (limit(theta_p)*(y[ip] - y[i]) - limit(theta_m)*(y[i] - y[im]) )
    return result

from numpy import zeros

def solve(method, y0, dx, v, tarray, factor=0.8, limit=vanleer):
    """ Call a stepping method until desired output times are reached """
    y = y0.copy() # Make a copy of the starting values
    time = 0.     # Starting time
    maxdt = factor* dx / abs(v)
    
    nt = len(tarray)
    nx = len(y0)
    result = zeros([nt, nx])
    for i, t in enumerate(tarray):
        # Take steps until we reach time t
        while time < t:
            dt = min([t - time, maxdt])    # Choose the time step
            y = step(y, time, dx, dt, v, limit=limit)   # Take a single step
            time += dt
        result[i,:] = y
    return result

from numpy import exp, linspace, zeros
import matplotlib.pyplot as plt

nx = 100            # Number of spatial grid points
v = 1.0             # Velocity
tarray = [0,0.1,1]  # Output times
limit = vanleer     # Choose the limiter method

x = linspace(0,1,nx, endpoint=False) # Periodic box of length 1
dx = x[1] - x[0]

y0 = exp( -((x - 0.3) / 0.1)**2 )  # Gaussian shape
y0[(6*nx//8):(7*nx//8)] = 1.0   # Top-hat function

y = solve(step, y0, dx, v, tarray, limit=limit)

for i,t in enumerate(tarray):
    plt.plot(x, y[i,:], label="t = %.2f" % (t))

plt.xlabel("x") ; plt.ylabel("f")
plt.legend(loc='upper center')
plt.show()
