# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 12:11:11 2025

@author: ciara
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:08:43 2025

@author: zks524
https://youjunhu.github.io/research_notes/particle_simulation/particle_simulationsu24.html


1: solve Poisson's eq. using fft methods (use .real for return variable)
    - dE/dx = rho, and rho = 1 - n_e



"""

    
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq, ifft, fftshift
import numpy as np
from numpy import linspace, sin, pi, zeros, arange, concatenate
import matplotlib.pyplot as plt


#solves the Poisson equation for the E field using the electron density
#using fourier transforms to solve with respect to x
def solvePoisson(x, ne):
    #define nx and dx
    dx = x[1]-x[0]
    nx = len(x)

    #1/lambda
    freq = fftfreq(nx, dx)

    #now convert to wavenumber k = 2*pi*freq
    k = 2 * np.pi * freq
    
    #compute charge density in physical space first: rho = 1 - n_e
    rho = 1 - ne
    
    #get fourier transform of charge density
    rho_line = fft(rho)
    
    #get array of zeros to put electric field into
    #also sets 1st value to 0
    eLine = zeros(len(rho_line), dtype='complex')
    
    #what the fourier transform of E is equal to
    #use only from 2nd element onwards so 1st element, zero-wavenumber, stays 0
    #dE/dx = rho, so in Fourier space: ik*E = rho, thus E = rho/(ik)
    #1j is the imaginary number
    eLine[1:] = rho_line[1:] / (1j*k[1:])
    
    #reverse fourier transform to get electric field
    eField = ifft(eLine).real
    
    
    return eField


def fourier_derivative ( field , xx ):
    """ This function calculates d field / dx using a Fourier approach """
    #calculates wavenumber using fftfreq and convert to k = 2*pi*freq
    nx = len(xx)
    dx = xx[1] - xx[0]
    wavenumbers = fftfreq(nx, dx)
    k = 2 * np.pi * wavenumbers
    #returns inverse fourier of i * k * fft(field)
    #this is the fourier derivative because in fourier space, d/dx = i * k
    return ifft(complex(0, 1) * k * fft(field)).real



x = np.linspace(0, 1, 101, endpoint = False)

#electron density equation     
ne = 1 + np.sin(2*np.pi*x)

# Expected dne/dx
dne_expected = 2*np.pi*np.cos(2*np.pi*x)

# Test the derivative function
dne_computed = fourier_derivative(ne, x)

print(f"Expected dne/dx max: {np.max(np.abs(dne_expected))}")
print(f"Computed dne/dx max: {np.max(np.abs(dne_computed))}")
print(f"Ratio: {np.max(np.abs(dne_expected)) / np.max(np.abs(dne_computed))}")

eField = solvePoisson(x, ne)

dEdx = fourier_derivative(eField, x)

# After calculating, add:
error = np.max(np.abs((1 - dEdx) - ne))
print(f"Maximum error: {error}")
print(f"dEdx min: {dEdx.min()}, max: {dEdx.max()}")
print(f"1 - dEdx min: {(1-dEdx).min()}, max: {(1-dEdx).max()}")
print(f"ne min: {ne.min()}, max: {ne.max()}")
if error > 1e-10:
    print("Warning: 1 - dE/dx ≠ n_e!")

plt.figure(figsize=(10, 6))
plt.plot(x, ne, 'b-', label=r'$\hat{n}_e$')
plt.plot(x, eField, 'r-', label=r'$\hat{E}_x$')
plt.plot(x, 1-dEdx, 'g--', label=r'$1 - d\hat{E}_x/d\hat{x}$')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Normalized values')
plt.legend(loc='best')
plt.title('Task 3: Poisson solver verification')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()





def initial_ne(xx, kx, amp):
    from numpy import cos, pi
    nx = len(xx)
    ne0 = 1 + amp * cos(2 * pi * xx * kx)
    ne0[:nx//4] = 1.0
    ne0[3*nx//4:] = 1.0
    return ne0

def initial_Ue(xx):
    return xx * 0.0


#evolves electron density and electron flow with time
def evolveFunc(curT, curF, x, q_e, m_e, T_e):
    nx = len(x)
    
    #set out the variables
    ne = curF[:nx]
    Ue = curF[nx:]
    
    #solve for electric field
    eField = solvePoisson(x, ne)
    
    #use fourier derivatives to get dne/dx and dUe/dx
    dne_dx = fourier_derivative(ne, x)
    dUe_dx = fourier_derivative(Ue, x)
    
    #use normalised equations to solve for Ue and Ne evolving with time
    dne_dt = -Ue*dne_dx - ne*dUe_dx
    dUe_dt = (q_e/m_e)*eField - (T_e/(m_e*ne))*dne_dx
    
    
    return concatenate([dne_dt, dUe_dt])




x2 = linspace(0, 1, 401, endpoint=False)
time = linspace(0, 2*np.pi, 251)

q_e = -1.0
m_e = 1.0
T_e = 0.0

kx = 4
amp = 0.01

nx=len(x2)

initialNe = initial_ne(x2, kx=4, amp=0.01)
initialUe = initial_Ue(x2)

#needs to be concatenated into one array for solve_ivp
f0 = concatenate([initialNe, initialUe])

#solve the system of equations
solution = solve_ivp(evolveFunc, [0, time[-1]], f0, t_eval=time, args=(x2,q_e,m_e,T_e))

#unpack solution
neSol = solution.y[:nx, :].T
UeSol = solution.y[nx:, :].T

# Diagnostic checks
print(f"Solution status: {solution.status}")
print(f"Number of time points: {len(solution.t)}")
print(f"neSol shape: {neSol.shape}")
print(f"neSol min: {neSol.min()}, max: {neSol.max()}")
print(f"neSol mean: {neSol.mean()}, std: {neSol.std()}")

#do contour plot of electron density with x on y axis and t on x axis
X, T = np.meshgrid(x2, solution.t)
plt.figure(figsize=(10, 6))
contour = plt.contourf(T, X, neSol, levels=50, cmap='viridis')
plt.colorbar(contour, label=r'$\hat{n}_e$')
plt.xlabel(r'$\hat{t}$')
plt.ylabel(r'$\hat{x}$')
plt.title('Electron Density Evolution Over Time')
plt.tight_layout()
plt.show()




