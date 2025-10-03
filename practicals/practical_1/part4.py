# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 14:35:01 2025

@author: ciara

Double pendulum
"""

#Two first order ODEs which interact with each other
#Variables: theta_1, theta_2, p_1, p_2, g, l, m, omega_1, omega_2




from scipy.integrate import solve_ivp
from numpy import linspace, sin, cos
import matplotlib.pyplot as plt


def dfdt(curT, curF, g=9.81, l=2., m=1.):
    
    #set out the variables
    theta1 = curF[0]; theta2 = curF[1]
    p1 = curF[2]; p2 = curF[3]
    
    #variables of difference in theta
    thetaDiff = theta1 - theta2
    
    #denominator in dtheta/dt equation
    dthetaDenom = (m * l**2) * (16 - 9*(cos(thetaDiff)**2))
    
    #numerator in dtheta1/dt equation
    dtheta1Num = 6 * (2*p1 - 3*p2*cos(thetaDiff))
    #full equation for theta1
    dtheta1dt = dtheta1Num / dthetaDenom
    
    #numerator in dtheta2dt equation
    dtheta2Num = 6 * (8*p2 - 3*p1*cos(thetaDiff))
    #full equation for theta2
    dtheta2dt = dtheta2Num / dthetaDenom
    
    #factor for dpdt
    pFactor = (m * l**2)/2
    
    dp1dt = -pFactor * (dtheta1dt * dtheta2dt * sin(thetaDiff) + 3 * (g/l) * sin(theta1))
    
    dp2dt = pFactor * (dtheta1dt * dtheta2dt * sin(thetaDiff) - (g/l) * sin(theta2))
    
    return [dtheta1dt, dtheta2dt, dp1dt, dp2dt]


time = linspace(0, 10, 400)

#define the variables to be passed into dfdt
g = 9.81
l = 2.
m = 1.

#set initial values

theta0_1 = 1.5
theta0_2 = 3.5

p1 = 0.
p2 = 0.

#create the function of changing variables
f0 = [theta0_1, theta0_2, p1, p1]

#controlling error - the smaller the better
#total accuracy value (number of correct decimal places)
totalAcc = 1e-10
#relative accuracy value (number of correct digits)
relativeAcc = 1e-4

result = solve_ivp(dfdt, [time[0], time[-1]], f0, t_eval=time, args=(g, l, m), rtol=relativeAcc, atol=totalAcc)

plt.plot(time, result.y[0,:], label=r'$\theta$ 1'); plt.plot(time, result.y[1,:], label=r'$\theta$ 2') 
plt.plot(time, result.y[2,:], label='Momentum 1'); plt.plot(time, result.y[3,:], label='Momentum 2')

plt.xlabel('Time'); plt.ylabel('Theta and momentum')
plt.legend(loc='best'); plt.show()





