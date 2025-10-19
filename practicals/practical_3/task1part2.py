# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 15:13:38 2025

@author: ciara
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
#number of elements - increase to capture higher frequencies
nx = 400
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
k1 = 2 * pi * 10  # ≈ 62.83
k2 = 2 * pi * 25  # ≈ 157.08

#define function
f = 0.5 * sin(x * k1) + sin(x * k2)


#do fourier transform
fourierspace = fft(f)

#do fftfreq to get frequencies for plotting fourier transform (x-axis)
space = fftfreq(nx, dx)

# Use fftshift to center the frequencies around zero (negative to positive)
space_shifted = fftshift(space)
fourierspace_shifted = fftshift(fourierspace)

print("Frequency range:", space_shifted[0], "to", space_shifted[-1])
print(f"Signal frequencies: k1/(2π) = {k1/(2*pi):.1f}, k2/(2π) = {k2/(2*pi):.1f}")

#plot the fourier against the inverse wavelength
#the (2.0/nx) multiplier is the normalisation: 2 for halving, and 1/nx is part of fft definition
plt.figure(figsize=(15, 5))

# Plot 1: Original (confusing) order
plt.subplot(1, 3, 1)
plt.plot(space, (2.0/nx)*abs(fourierspace))
plt.xlabel(r'$1/\lambda$ (original order)')
plt.ylabel(r'$\hat{f}$')
plt.title('Original FFT order\n(negative freqs on right)')
plt.grid()
plt.xlim(-200, 200)

# Plot 2: Properly shifted to show negative frequencies
plt.subplot(1, 3, 2)
plt.plot(space_shifted, (2.0/nx)*abs(fourierspace_shifted))
plt.xlabel(r'$1/\lambda$ (shifted order)')
plt.ylabel(r'$\hat{f}$')
plt.title('With fftshift\n(negative to positive)')
plt.grid()
plt.xlim(-200, 200)

# Plot 3: Zoomed in to see the peaks clearly
plt.subplot(1, 3, 3)
plt.plot(space_shifted, (2.0/nx)*abs(fourierspace_shifted))
plt.xlabel(r'$1/\lambda$ (shifted order)')
plt.ylabel(r'$\hat{f}$')
plt.title('Zoomed view of peaks')
plt.grid()
plt.xlim(-180, 180)

plt.tight_layout()
plt.show()
