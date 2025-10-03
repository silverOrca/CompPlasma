# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:24:41 2025

@author: ciara

Simple pendulum
"""
#Equation is d^2theta/dt^2 = -g/l sin(theta)
#define first an equation to make this first order
#F = -mgsin(theta) = ma, and a = l d^2theta/dt^2
#So we can define a = dv/dt, and v = l * dtheta/dt, 

#need to pass in variables: g, l, curF, curTheta


from scipy.integrate import solve_ivp
from numpy import linspace, sin
import matplotlib.pyplot as plt

def dfdt(curT, curF, g=9.81, l=2., useSmall=False):
    
    theta = curF[0]
    omega = curF[1]
    
    #define the angular velocity
    dthetadt = omega
    
    #decides whether to use equation for small angle approx. or not
    if useSmall:
        domegadt = (-g / l) * theta
    else:
        domegadt = (-g / l) * sin(theta)
        
    return [dthetadt, domegadt]
    
    
time = linspace(0, 10, 100)   
  
#define the parameters
g = 9.91
l = 2.
  
omega = 0.
startingTheta = [0.01, 0.02, 0.05, 0.1]




for angle in startingTheta:
    f0=[angle, omega]
    result = solve_ivp(dfdt, [time[0], time[-1]], f0, t_eval=time, args=(g, l))
    resultSmall = solve_ivp(dfdt, [time[0], time[-1]], f0, t_eval=time, args=(g, l, True))
    
    
    plt.plot(time, result.y[0,:], label=r'$\theta(t)$ = ' + str(angle) + ' Full')
    plt.plot(time, result.y[0,:], '--x', label='Small angle approx.')
    
    
plt.xlabel('Time'); plt.ylabel(r'$\theta(t)$')
plt.legend(loc='best'); plt.show()

    
    
    
    
    

