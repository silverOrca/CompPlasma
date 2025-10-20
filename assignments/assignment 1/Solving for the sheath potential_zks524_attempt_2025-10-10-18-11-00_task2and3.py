# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:41:14 2025

@author: ciara

Plasma sheath code
"""
#Need to solve Poisson's equation, by writing it as a 1st order ODE
#dphi/dx = -E, so d^2phi/dx = -dE/dx
#We will calculate dE/dx, which will give us the solution in negative

#dE/dx depends on the electron density, n_e, and ion density, n_i. dE/dx = n_i - n_e
#These values change with phi


from scipy.integrate import solve_ivp
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt


def dfdt(curX, curF, v_s_hat = 1.):
#DD: Goods that you pass in v_s_hat here.
#DD: Generally good variable names, although denom might be better sqrt vi_squared or similar?
    #set out the variables
    phi_hat = curF[0]
    energy_hat = curF[1]
    
    denom = v_s_hat**2 - 2 * phi_hat
    n_i = v_s_hat / sqrt(denom)
    
    n_e = exp(phi_hat)
    dedt = n_i - n_e
    
    dphidt = -energy_hat
    
    
    return [dphidt, dedt]

#distance from plasma into sheath
xPos = linspace(0, 40, 100)

#normalised ion velocity at start of sheath
v_s_hat = 1.


#initial conditions
#DD: Good use of variable names -- it's clear what your initial values are
phi_hat = 0.
energy_hat = 0.001

#define function of values with initial values
f0 = [phi_hat, energy_hat]


result = solve_ivp(dfdt, [xPos[0], xPos[-1]], f0, t_eval = xPos, args = (v_s_hat,))

plt.plot(xPos, result.y[0,:], label = 'Normalised Electrostatic Potential Energy, '+r'$\hat{\phi}$')
plt.plot (xPos, result.y[1,:], label = 'Normalised Ion Energy, ' + r'$\hat{E}$')

plt.title('Normalised energy of plasma sheath against position')
plt.xlabel('Distance into sheath, x'); plt.ylabel('Energy, ' + r'$\hat{\phi}$' + ' and ' + r'$\hat{E}$')
plt.legend(loc='best'); plt.show()


#Task 3
#Plotting normalised current against time
#DD: I'd recommend avoiding magic numbers like 1840. It's much better to use
#a named variable such as mass_ratio or mi_over_me etc .
j = sqrt(1840/(2*pi)) * exp(result.y[0,:]) - 1

plt.plot(xPos, j)
#DD: Good description on the x-axis, could say distance from what though (i.e. from bulk plasma or from wall?) and the units/normalisation.
#DD: You should probably say what the current is normalised to.
plt.xlabel('Distance into sheath, x'); plt.ylabel('Current, ' + r'$\hat{j}$')
plt.title('Normalised current against position')

plt.show()




