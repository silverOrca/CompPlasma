# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 10:07:35 2025

@author: zks524
"""

#solve_ivp is solve initial value problems
#i.e. solves dy/dt given y(0)
from scipy.integrate import solve_ivp

#read documentation
help(solve_ivp)

#need to pass: fun - the function which returns dy/dt
#t_span - two element array with limits of integration
#y0 - an array of initial conditions

#t_eval - stores time history, doesn't control step size just
#gives certain number of values throughout the integration
