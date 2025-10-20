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
#   - when have phi_hat, use ifft to get rho
#   - so to get rho_hat from rho, regular fourier transform
#   - and start from some known equation for rho

import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from numpy import linspace, sin, pi

#define parameters
length = 1.0
nx = 100
x = linspace(0, length, nx, endpoint=False)
dx = x[1]-x[0]


#define rho (rhs vector/function)
rho = sin(2 * pi * x / length)

#fourier transform to get rho_hat
rho_hat = fft(rho)

lamb = fftfreq(rho_hat)









