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

from scipy.fft import fft, fftfreq, fftshift, ifft

from numpy import sin, linspace, pi
import numpy as np

import matplotlib.pyplot as plt

#task 1.1
def pos_plot(fourierspace, space):
    #bound set to get values up to upper limit (halfway - symmetry, beyond this gives -ve values)
    fourierspace = fourierspace[:upLim]
    space = space[:upLim]

    #plot the fourier against the inverse wavelength
    #the (2.0/nx) multiplier is the normalisation: 2 for halving, and 1/nx is part of fft definition
    plt.plot(space, (2.0/nx)*abs(fourierspace))
    plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
    plt.grid()
    plt.show()
    
#task 1.2
def neg_plot(fourierspace, space, plot=False):
    
    #can see that without shifting values, the positive print first and negative second (need to switch)
    print(space)
    
    #shifts values so -ve come first, then 0, then +ve
    #must do for both fourierspace and space so that the fourier values match with the 1/lamba
    fourierspace = fftshift(fourierspace)
    space = fftshift(space)
    
    if plot:
        plt.plot(space, (2.0/nx)*abs(fourierspace))
        plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
        plt.grid()
        plt.show()
    
    return fourierspace, space
    
#task 1.3
def inverse_plot(fourierspace, space, f, plot=False, filtered=False):
    inverse = ifft(fourierspace)
    
    #plotting both together shows that they are the same
    plt.plot(x, f, label='Regular plot')
    if filtered:
        plt.plot(x, inverse, linestyle='dashed', label='Plot with '+r'$\lambda=\frac{1}{10}$ '+'filtered out.')
    else:
        plt.plot(x, inverse, linestyle='dashed', label='Fourier and inverse plot')
    plt.xlabel(r'$1/\lambda$') ; plt.ylabel(r'$\hat{f}$')
    plt.legend(loc='best')
    plt.show()
    
    return inverse
  
#task 1.4
def filtered_plot(fourierspace, space, plot=False):
    #need to remove the lambda = 1/10 part from fourier transform then transform back with ifft
    #these points are only one data point each
    #if space is 10 then will return 0 and be smallest value so gets index
    pos_10 = np.argmin(abs(10-space))
    #same for -10
    neg_10 = np.argmin(abs(10+space))
    
    #set the y values for those particular x positions to 0
    fourierspace[pos_10] = 0.
    fourierspace[neg_10] = 0.
    
    inverse_plot(fourierspace, space, f, True, True)

    return fourierspace

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

#do fourier transform
fourierspace = fft(f)

#do fftfreq to get frequencies for plotting fourier transform (x-axis)
space = fftfreq(nx, dx)

#call function to get fourier transform with negative frequencies and in
#correct order
neg = neg_plot(fourierspace, space, False)

#compare = inverse_plot(fourierspace, space, f, False)

#call function to filter the fourier transform and then plot as regular function
filtered = filtered_plot(fourierspace, space, False)

