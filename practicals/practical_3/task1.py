# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:06:14 2025

@author: zks524
"""
#fast fourier transform (fft)
#uses recursive divide and conquer strategy to return the input in frequency domain
#input is the target function, as it already has the periodic basis functions
#output is the fourier transform: x to k_x 
#(time domain into frequency domain - shows individual frequency components of overall signal)

#fftfreq: input is number of spaces and spacing, output is the sample frequencies used in fourier for given spacing
#(to apply to the fourier transform and separate out into the difference frequencies for plotting)
#because they are frequencies, they are 1/lambda (inverse wavelength)

from scipy.fft import fft, fftfreq, fftshift

from numpy import sin, linspace, pi

import matplotlib.pyplot as plt

#first define box size etc
length = 1.0
#number of elements
nx = 100
#make array of x elements
#endpoint=false means that we are not plotting the same value on both ends (it circles back on itself)
x = linspace(0, length, nx, endpoint=False)
#define spacing (evenly spaced so just use difference between first two elements)
dx = x[1] - x[0]

#because of symmetry, need to halve results
#// to do integer division
upLim = nx//2

#define k values (wavenumbers)
#wavelength is L/n where n is 10 or 25
k1 = 2 * pi * 10
k2 = 2 * pi * 25

#define function
f = 0.5 * sin(x * k1) + sin(x * k2)


#bound set to get values up to upper limit (halfway - symmetry, beyond this gives -ve values)
#do fourier transform
fourierspace = fft(f)[:upLim]

#do fftfreq to get frequencies for plotting fourier transform (x-axis)
space = fftfreq(nx, dx)[:upLim]
print(space)

#plot the fourier against the inverse wavelength
#the (2.0/nx) multiplier is the normalisation: 2 for halving, and 1/nx is part of fft definition
plt.plot(space, (2.0/nx)*abs(fourierspace))
#plt.plot(space, (2.0/nx)*(fourierspace))
plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
plt.grid()
plt.show()
