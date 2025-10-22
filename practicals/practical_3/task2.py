# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 12:58:11 2025

@author: zks524

Poisson's equation fourier solver for 1d box

d^2 rho / dx^2 = -rho(x)
"""
#To do:
#   - will need to calculate k at some point: k = 2 pi n/L = 2 pi / lambda
#   - 1/lambda comes from doing fftfreq so will need to do that 
#   - phi_hat = rho_hat / k^2 epsilon_0, so need to find rho_hat first
#   - when have phi_hat, use ifft to get phi
#   - so to get rho_hat from rho, regular fourier transform
#   - and start from some known equation for rho: rho = sin(2 pi x / L)
#   - Normalisations: epsilon_0 = 1, so this disappears

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from numpy import linspace, sin, pi, zeros, sqrt, mean

#computationally solve for phi doing fourier
def compSolve():
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
def anaSolve():
    phiAn = sin(2*pi*x)/(2*pi)**2

    return phiAn

#define parameters
length = 1.0
nx = 100
x = linspace(0, length, nx, endpoint=False)
dx = x[1]-x[0]

#define rho (rhs vector/function)
rho = sin(2 * pi * x / length)

#fourier transform to get rho_hat
rho_hat = fft(rho)

#1/lambda
lamb = fftfreq(nx, dx)

#now can calculate k
k = 2 * pi * lamb


#do the computational solution and plot
compSol = compSolve()
plt.plot(x, compSol)


anaSol = anaSolve()
plt.plot(x, anaSol, '+')
plt.show()

err = anaSol-compSol
rmsErr = sqrt(mean(err*err.conjugate()))
print("The rms error is {err}".format(err=rmsErr))
