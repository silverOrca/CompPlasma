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
from scipy.fft import fft, fftfreq, ifft, fftshift
from numpy import linspace, sin, pi, zeros, sqrt, mean, zeros_like, concatenate
from scipy.integrate import solve_ivp



#computationally solve for phi doing fourier
def compSolve(x, rho):
    dx = x[1]-x[0]
    nx = len(x)

    #1/lambda
    lamb = fftfreq(nx, dx)

    #now can calculate k
    k = 2 * pi * lamb

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
    
    return phi.real



#equation for use of solve_ivp to evolve rho in time
#defines the time evolution of rho
#include boundary conditions here too
def dfdt(curT, curF, x):
    nx = len(x)

    #set out variables from flattened initial state function f0
    rho = curF[:nx]
    v = curF[nx:]

    phi = compSolve(x, rho)

    drho_dt = v
    #dv_dt = - phi = d^2 rho / dt^2
    
    dv_dt = - phi  #need to get phi from current rho

    #returns drho/dt and -phi = d^2 rho / dt^2
    return concatenate([drho_dt, dv_dt])



#analytically solve for phi - this is just the first term in fourier eqn
def anaSolve(x):
    phiAn = sin(2*pi*x)/(2*pi)**2

    return phiAn


#define parameters
length = 1.0
nx=100

x = linspace(0.0,length,nx,endpoint=False)
dx = x[1]-x[0]

nt = 100
time = linspace(0.0, 200.0, nt)

#spatial grid to pass in
x = linspace(0, length, nx, endpoint=False)

#define rho (rhs vector/function)
rho0 = sin(2 * pi * x / length)
drho_dt0 = zeros_like(rho0)

#initial conditions
#needs to be 1d
f0 = concatenate([rho0, drho_dt0])

solution = solve_ivp(dfdt, [0, time[-1]], f0, t_eval=time, args=(x,))#.y[:,0]
#print(solution)


#extract final values from solution - all times and first nx points
drho_dt_final = solution.y[:nx, :].T
#this is the -phi = d^2 rho / dt^2
dv_dt_final = solution.y[:,nx:]

plt.contourf(x,time,drho_dt_final,64)
plt.xlabel(r'$x$') ; plt.ylabel(r'$t$')
plt.colorbar(label=r'$\rho$')
plt.show()


#Look at FFT coefficients
rhoHat = (1.0/nx)*(fft(drho_dt_final,axis=1).imag)
invLamb = fftfreq(nx,dx)
plt.figure(2)
plt.contourf(fftshift(invLamb),time,fftshift(rhoHat,axes=1),64)
plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$t$')
plt.colorbar(label=r'$\hat{\rho}$')
plt.show()

#get the analytical solution and plot
#anaSol = anaSolve(x)
#plt.plot(x, anaSol, '+')
#plt.show()

#calculate rms error
#err = anaSol-compSol
#rmsErr = sqrt(mean(err*err.conjugate()))
#print("The rms error is {err}".format(err=rmsErr))
