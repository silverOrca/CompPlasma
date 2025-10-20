# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 15:08:52 2025

@author: ciara
"""


from scipy.integrate import solve_ivp
#this is the interpolation module
from scipy.interpolate import interp1d
#to find x at a root (y=0)
from scipy.optimize import root_scalar
from numpy import linspace, exp, sqrt, pi
import matplotlib.pyplot as plt


#function of poisson equation
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


#solves for the electrostatic potential energy and ion energy, and plots
def solve_energy(xPos, v_s_hat, phi_hat, energy_hat, f0):
    result = solve_ivp(dfdt, [xPos[0], xPos[-1]], f0, t_eval = xPos, args = (v_s_hat,))
    
    plt.plot(xPos, result.y[0,:], label = 'Normalised Electrostatic Potential Energy, '+r'$\hat{\phi}$'+', for '+r'$\hat{v}_s = $'+str(v_s_hat))
    plt.plot (xPos, result.y[1,:], label = 'Normalised Ion Energy, ' + r'$\hat{E}$'+', for '+r'$\hat{v}_s = $'+str(v_s_hat), linestyle='dashed')
    
    return result
 

#solves for the current and plots 
def solve_current(result, v_s_hat):
    #Task 3
    #Plotting normalised current against time
    j = sqrt(1840/(2*pi)) * exp(result.y[0,:]) - 1
    
    return j
    

#distance from plasma into sheath
xPos = linspace(0, 40, 100)

#normalised ion velocity at start of sheath
#using multiple values to compare
v_s_hat = [1., 1.5, 2.]


#initial conditions
phi_hat = 0.
energy_hat = 0.001

#define function of values with initial values
f0 = [phi_hat, energy_hat]


#array to store any number of results for energy to use in current solution function
energy_result=[]



#calls the energy solve function and shows graph
for i in range(len(v_s_hat)):
    e_result = solve_energy(xPos, v_s_hat[i], phi_hat, energy_hat, f0)
    energy_result.append(e_result) 
    
#caption for energy plot
energy_caption = 'Figure 1: Graph showing how the normalised electrostatic potential energy and normalised ion energy of an ion vary with distance into a plasma sheath, for varying ion velocities.' 
    
plt.title('Normalised energy of plasma sheath against position')
plt.xlabel('Distance into sheath, '+r'$\hat{x}$'); plt.ylabel('Energy, ' + r'$\hat{\phi}$' + ' and ' + r'$\hat{E}$')
plt.legend(loc='best', prop={'size': 7})
plt.figtext(0.5, -0.055, energy_caption, wrap=True, horizontalalignment='center', fontsize=8)
plt.grid()
plt.show()

#store wall placements with corresponding velocities
roots = [[] for i in range(len(v_s_hat))]
#store solutions to use for colouring plot
solves = [[] for i in range(len(v_s_hat))]

#calls the current solve function and shows graph
for i in range(len(v_s_hat)):
    j_result = solve_current(energy_result[i], v_s_hat[i])
    solves.append(j_result)
    
    #create interpolated function of x and j values
    interpolated = interp1d(xPos, j_result)
    #function to return value for use of root finder
    def f(x_val):
        return interpolated(x_val)
    #find which x value corresponds to the first j=0. Used bracket length of full thing
    res = root_scalar(f, bracket=[0,40])
    wall = res.root
    
    #putting roots and corresponding velocities into 2d roots list
    roots[i].append(wall)
    roots[i].append(v_s_hat[i])

    #subtracts extra wall position from x value to line up all plots
    plt.plot(xPos-roots[i][0], j_result, label=r'$\hat{v}_s = $'+str(v_s_hat[i]))
    
    
print(roots)
  
#caption for normalised current figure
j_caption = 'Figure 2: Graph showing normalised current varying with distance into a plasma sheath, for varying ion velocities.'

#to show visually where plasma and sheath wall are
plt.axvspan(-40, 0, facecolor='red', alpha=0.1, label='Plasma')
plt.axvspan(0, 40, facecolor='blue', alpha=0.1, label='Sheath wall')

plt.xlabel('Distance into sheath, ' r'$\hat{x}$'); plt.ylabel('Normalised current, ' + r'$\hat{j}$')
plt.title('Normalised current against position')
plt.legend(loc='best')
plt.figtext(0.5, -0.055, j_caption, wrap=True, horizontalalignment='center', fontsize=8)
plt.grid()
plt.show()




