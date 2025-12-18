# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:08:43 2025

@author: zks524
https://youjunhu.github.io/research_notes/particle_simulation/particle_simulationsu24.html


1: solve Poisson's eq. using fft methods (use .real for return variable)



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
ne = 1+ np.sin(2*np.pi*x)



eField = solvePoisson(x, ne)

deFielddx = fourier_derivative(eField, x)

plt.plot(x, eField, label='eField')
plt.plot(x, 1-deFielddx, label='1-dE/dx')

plt.legend(loc='best')
plt.show()
