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


    
def solvePoisson(x, ne):
    #define nx and dx
    dx = x[1]-x[0]
    nx = len(x)

    #1/lambda
    lamb = fftfreq(nx, dx)

    #now can calculate k
    k = 2 * np.pi * lamb
    
    #get fourier transform of electron density
    ne_line = fft(ne)
    
    #get array of zeros to put electric field into
    #also sets 1st value to 0
    eLine = zeros(len(ne_line), dtype='complex')
    
    #what the fourier transform of E is equal to
    #use only from 2nd element onwards so 1st element, zero-wavenumber, stays 0
    #1j is the imaginary number
    eLine[1:] = (1 - ne_line[1:]) / (1j*k[1:])
    
    #reverse fourier transform to get electric field
    eField = ifft(eLine).real
    
    
    return eField



def fourier_derivative ( field , xx ):
    """ This function calculates d field / dx using a Fourier approach """
    wavenumbers = fftfreq(len(xx), xx[1] - xx[0])
    return ifft(complex(0, 1) * wavenumbers * fft ( field )).real



x = np.linspace(0, 1, 101, endpoint = False)

#electron density equation     
ne = 1 + np.sin(2*np.pi*x)



eField = solvePoisson(x, ne)

deFielddx = fourier_derivative(eField, x)

# After calculating, add:
error = np.max(np.abs((1 - deFielddx) - ne))
print(f"Maximum error: {error}")
if error > 1e-10:
    print("Warning: 1 - dE/dx ≠ n_e!")

plt.figure(figsize=(10, 6))
plt.plot(x, ne, 'b-', label=r'$\hat{n}_e$')
plt.plot(x, eField, 'r-', label=r'$\hat{E}_x$')
plt.plot(x, 1-deFielddx, 'g--', label=r'$1 - d\hat{E}_x/d\hat{x}$')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Normalized values')
plt.legend(loc='best')
plt.title('Task 3: Poisson solver verification')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
