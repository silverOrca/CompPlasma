"""
Created on Weds Oct 22 12:42:01 2025

@author: zks524

Task 3: Including Time Evolution in the Poisson Solver
- charge density rho = rho(x,t)
- where rho(x) = sin(2 pi x / L)
- equation for rho evolution: d rho(x,t)^2/dt^2 = -phi

To do:
- Use solve_ivp to evolve rho from initial conditions: rho(x,0) = sin(2 pi x / L), 
- And d rho / dt = 0 
- From t=0.0 to t=200.0, in nx=100 steps
"""



import matplotlib.pyplot as plt 
from scipy.fft import fft, fftfreq, ifft
from numpy import linspace, sin, pi, zeros, sqrt, mean, zeros_like
from scipy.integrate import solve_ivp

#equation for use of solve_ivp to evolve rho in time
#defines the time evolution of rho
#include boundary conditions here too
def dfdt(curT, curF):
    #already passed in initial condition rho(x,0) into solve_ivp
    #initial condtion for d rho / dt = 0 = v
    rho = curF[:nx]
    v = curF[nx:]

    drho_dt = v

    phi = compSolve(x, rho, curF)

    dv_dt = - phi  #need to get phi from current rho


    return [drho_dt, dv_dt]






#computationally solve for phi doing fourier
def compSolve(x, rho, f0):
    dx = x[1]-x[0]

    #1/lambda
    lamb = fftfreq(nx, dx)

    #now can calculate k
    k = 2 * pi * lamb

    #with initial condition rho = rho(x,0)
    
    
    #fourier transform to get rho_hat
    rho_hat = fft(rho)

    #now phi_hat = rho_hat / k^2 epsilon_0

    #Because of zero at start of k we need to do a fiddle to avoid this part
    #Start where k=0 will just stay as 0
    #Need to make this complex or get error of casting complex values to real
    phi_hat = zeros(len(rho_hat), dtype='complex')

    #Divide all except the first element
    phi_hat[1:] = rho_hat[1:] / k[1:]**2  


    #now inverse fourier transform to get phi from phi_hat
    phi = ifft(phi_hat)
    
    return phi

#analytically solve for phi - this is just the first term in fourier eqn
def anaSolve(x):
    phiAn = sin(2*pi*x)/(2*pi)**2

    return phiAn

#define parameters
length = 1.0
nx = 100

time = linspace(0.0, 200.0, nx)

#spatial grid to pass in
x = linspace(0, length, nx, endpoint=False)

#define rho (rhs vector/function)
rho = sin(2 * pi * x / length)
drho_dt = zeros_like(rho)

#initial conditions
#
f0 = [rho, drho_dt]

solution = solve_ivp(dfdt, [0, time[-1]], f0, t_eval=time)#.y[:,0]
print(solution)

#do the computational solution and plot
#compSol = compSolve(x, rho, f0)
#plt.plot(x, compSol)

#get the analytical solution and plot
#anaSol = anaSolve(x)
#plt.plot(x, anaSol, '+')
#plt.show()

#calculate rms error
#err = anaSol-compSol
#rmsErr = sqrt(mean(err*err.conjugate()))
#print("The rms error is {err}".format(err=rmsErr))
