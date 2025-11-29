# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 11:22:43 2025

@author: ciara

Need to solve equations of motion for a single particle in the x and y directions
"""

import matplotlib.pyplot as plt
from numpy import zeros, linspace, asarray
from scipy.integrate import solve_ivp

 

#define how many spaces I want in x and y
nx = 1001 ; ny = 501 
#define length of x and y
L_x = 100.0 ; L_y = 50.0

#set out the individual points along x and y
xx = linspace(0, L_x, nx)
yy = linspace(0, L_y, ny)

#define an empty array of spatial dimensions to get B for each spatial location
Bmag = zeros([nx, ny])

#define initial parameters
x_0 = L_x / 2
x_c =-L_x
B_0 = 1.0

#define the time limits
time = linspace(0, 500, 1000)

#values for each parameter to iterate over
initial_x = [(3*L_x)/4, L_x/2, L_x/4, L_x/2]
initial_y = [L_y/2, L_y/2, L_y/2, L_y/2]
initial_vx = [1.0, 1.0, 1.0, -1.0]
initial_vy = [0.0, 0.0, 0.0, 0.0]
q = [1.0, -1.0, 1.0, 1.0]
m = [1.0, 1.0, 0.1, 0.1]

#function defining the equation of motion of charged particle
#changing variables: v_x, v_y, B_z, 
def dvdt(curT, curF, q, m, E_x=0, E_y=0):
    
    #set out the variables
    x = curF[0]
    y = curF[1]
    v_x = curF[2]
    v_y = curF[3]
    
    dx_dt = v_x
    dy_dt = v_y
    
    #need to find what the current B_z is
    B_z = B_func(x, y, x_0, x_c, B_0)
    
    
    dvx_dt = (q/m) * (v_y * B_z + E_x)
    
    dvy_dt = (q/m) * ((-v_x * B_z) + E_y)
    
    
    return [dx_dt, dy_dt, dvx_dt, dvy_dt]
    
   

#example of a magnetic field varying with position
def B_func(x, y, x_0, x_c, B_0):

    #doesn't use y position in equation
    B_z = (B_0 * x_0) / (x - x_c)
    
    
    return B_z


#to plot anything needed
def plotting(Bmag):
    plt.figure()
    #create the plot      
    plt.pcolormesh(xx, yy, Bmag.T, shading = 'auto')
    plt.xlabel(r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8) 
    plt.ylabel(r'$\hat{y}$, normalised to thermal Hydrogen gyro-period, $\tau_H$', fontsize=8)
    plt.text(-4, -11, r'Figure 1: Example $B_{func}$ output showing $B_z$ varying with $\hat{x}$ and $\hat{y}$.')
    plt.colorbar()


# #runs the code to output the B_func plot
def run_B_plot(bPlot=False):
    
    #fill the empty array with spatially varying B calculated from B_func
    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            B_z = B_func(x, y, x_0, x_c, B_0)
            Bmag[ix, iy] = B_z
    
    if bPlot:
        plotting(Bmag)
    
    return Bmag


#kinetic energy equation, normalised
def kinetic_energy(m, v_x, v_y):
    e_k = m * (v_x**2 + v_y**2)
    
    return e_k


    
def integrate_and_plot(E_x, E_y, B_mag):
    #solve for background magnetic field
    
    #kinetic energy
    ek_plot = plt.figure()
    #trajectory in x and y
    trajectory_plot = plt.figure()
    
    #create axes
    ek_axis = ek_plot.add_subplot(1,1,1)
    
    trajectory_axis = trajectory_plot.add_subplot(1,1,1)
    
    #choosing B colour to better see the trajectories
    im = trajectory_axis.pcolormesh(xx, yy, B_mag.T, shading = 'auto', cmap='bone')
    

    for i in range(len(initial_x)):
        #define the current initial set of parameters
        f0 = [initial_x[i], initial_y[i], initial_vx[i], initial_vy[i]]
        #integrate the equation to get velocity and trajectory
        solution = solve_ivp(dvdt, [time[0], time[-1]], f0, t_eval=time, args=(q[i], m[i], E_x, E_y))
        
        #set out solutions
        x = solution.y[0,:]
        y = solution.y[1,:]
        v_x = solution.y[2,:]
        v_y = solution.y[3,:]
        
        #solve kinetic energy and plot
        e_k = kinetic_energy(m[i], v_x, v_y)
        
        #do final e_k / initial e_k for each particle
        print('E_k,final/E_k,initial for particle ' + str(i+1) + ': ' + str(e_k[-1]/e_k[0]))
        
        ek_axis.plot(time, e_k, label =r'$E_{x} = $ '+str(E_x)+r', $E_{y} = $'+str(E_y))
        
        #plot the trajectory - y against x
        trajectory_axis.plot(x, y, label='Particle '+str(i+1)+r': Initial $\hat{x}$ = '+str(initial_x[i])+r', initial $\hat{y}$ = '+str(initial_y[i])+r', initial $\hat{v}_x$ = '+str(initial_vx[i])+r', initial $\hat{v}_y$ = '+str(initial_vy[i])+r', charge $(\hat{q})$ = '+str(q[i])+r', mass $(\hat{m})$ = '+str(m[i]))

    #e_k axes labelling
    ek_axis.set_xlabel(r'Time (s), normalised to the thermal Hydrogen gyro-period, $\tau_H$')
    ek_axis.set_ylabel(r'Kinetic energy, $\hat{E}_k$, normalised to')
    ek_axis.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    ek_axis.grid(alpha=0.3)
    ek_axis.set_title('Kinetic energy of particles varying with time')

    #trajectory axes labelling
    trajectory_axis.set_xlabel(r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8)
    trajectory_axis.set_ylabel(r'$\hat{y}$, normalised to thermal Hydrogen gyro-period, $\tau_H$', fontsize=8)
    trajectory_axis.grid(alpha=0.3)
    trajectory_axis.legend(loc='lower center', bbox_to_anchor=(0.5,-0.6))
    
    traj_caption = r'Fig Electric field, $\hat{E}$ (x, y) = ('+str(E_x)+', '+str(E_y)+'), normalised to .\n'
    trajectory_axis.text(-5, -12, traj_caption, fontsize=8)
    trajectory_axis.set_title('Trajectories of particles over the background magnetic field')
    trajectory_plot.colorbar(im, ax=trajectory_axis, label=r'Magnetic field, $\hat{B}$, normalised to')

    plt.show()
    
    
    
    
    
if __name__ == "__main__":#

    B_mag = run_B_plot(bPlot=True)
    
    electric_fields = asarray([[0.0, 0.0], [0.0, 0.01]])
    
    for i in range(len(electric_fields)):
        integrate_and_plot(electric_fields[i][0], electric_fields[i][1], B_mag)
    