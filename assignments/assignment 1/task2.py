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
from numpy import linspace, exp, sqrt
import matplotlib.pyplot as plt


def dfdt(curX, curF, v_s_hat = 1.):
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
phi_hat = 0.
energy_hat = 0.001

#define function of values with initial values
f0 = [phi_hat, energy_hat]


result = solve_ivp(dfdt, [xPos[0], xPos[-1]], f0, t_eval = xPos, args = (v_s_hat,))

plt.plot(xPos, result.y[0,:], label = 'Electrostatic Potential Energy')
plt.plot (xPos, result.y[1,:], label = 'Ion Energy')

plt.xlabel('Position, x'); plt.ylabel('Energy')
plt.legend(loc='best'); plt.show()



