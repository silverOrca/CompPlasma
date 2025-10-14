# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 15:06:14 2025

@author: zks524
"""
#fast fourier transform
#uses recursive divide and conquer strategy to return the input in frequency domain
#input is the target function, as it already has the periodic basis functions
from scipy import fft

from numpy import sin, linspace, pi

#first define box size etc
length = 1.0
#number of elements
nx = 100
#make array of x elements
x = linspace(0, length, nx)
#define spacing (evenly spaced so just use difference between first two elements)
dx = x[1] - x[1]
