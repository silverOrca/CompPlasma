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
    
    #define variables so code easier to understand
    dx_dt = v_x
    dy_dt = v_y
    
    #need to find what the current B_z is
    B_z = B_func(x, y, x_0, x_c, B_0)
    
    #equations for how velocity changes with time
    dvx_dt = (q/m) * (v_y * B_z + E_x)
    
    dvy_dt = (q/m) * ((-v_x * B_z) + E_y)
    
    
    return [dx_dt, dy_dt, dvx_dt, dvy_dt]
    
   

#example of a magnetic field varying with position
def B_func(x, y, x_0, x_c, B_0):

    #doesn't use y position in equation
    B_z = (B_0 * x_0) / (x - x_c)
    
    
    return B_z



#runs the code to output the B_func plot
def run_B_plot():
    
    #fill the empty array with spatially varying B calculated from B_func
    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            B_z = B_func(x, y, x_0, x_c, B_0)
            Bmag[ix, iy] = B_z
    
    #create the plot
    plt.figure()      
    plt.pcolormesh(xx, yy, Bmag.T, shading = 'auto')
    plt.xlabel(r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8) 
    plt.ylabel(r'$\hat{y}$, normalised to thermal Hydrogen gyro-period, $\tau_H$', fontsize=8)
    plt.title(r'Example $B_{func}$ output showing $B_z$ varying with $\hat{x}$ and $\hat{y}$.')
    plt.colorbar(label=r'Magnetic field, $\hat{B}_z(\hat{x}, \hat{y})$, normalised to $B_r$')
    
    return Bmag


#kinetic energy equation, normalised
def kinetic_energy(m, v_x, v_y):
    e_k = m * (v_x**2 + v_y**2)
    
    return e_k


    
def integrate_and_plot(E_x, E_y, B_mag):
    #solve for background magnetic field
    
    #kinetic energy - create fig with large graph of all at top and two zoomed graphs underneath
    ek_fig = plt.figure() 
    ek_axis = ek_fig.add_subplot(1,1,1)
    
    #separate figure for start and end of kinetic energy (2x2 grid)
    startEndEkFig, startEndEkax = plt.subplots(2,2, figsize=(12,8))

    #trajectory in x and y
    trajectory_plot = plt.figure()
    trajectory_axis = trajectory_plot.add_subplot(1,1,1)
    
    #individual trajectories
    indTrajFig, indTrajax = plt.subplots(2,2, figsize=(10,6))
    indTrajax_flat = indTrajax.flatten()
    
    #colours that stand out
    colors = ['dodgerblue','orange','limegreen','red']
    
    #store e_k data for each particle for start/end plotting
    all_e_k = []
    

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
        all_e_k.append(e_k)
        
        #do final e_k / initial e_k for each particle
        print('E_k,final/E_k,initial for particle ' + str(i+1) + ': ' + str(e_k[-1]/e_k[0]))
        print(f'E_k,final - E_k,initial for particle {i+1}: {e_k[-1]-e_k[0]}')
        
        
        #text for legend with all varying parameters
        txt = 'Particle '+str(i+1)+r': Initial $\hat{x}$ = '+str(initial_x[i])+r', initial $\hat{y}$ = '+str(initial_y[i])+r', initial $\hat{v}_x$ = '+str(initial_vx[i])+r', initial $\hat{v}_y$ = '+str(initial_vy[i])+r', charge $(\hat{q})$ = '+str(q[i])+r', mass $(\hat{m})$ = '+str(m[i])
        
        #plot all ek values on same plot
        ek_axis.plot(time, e_k, label = txt, linewidth=0.8, color=colors[i])
        
        #plot the trajectory - y against x
        trajectory_axis.plot(x, y, label=txt, linewidth=0.5, color=colors[i])
        
        
        #Calculate trajectory bounds and zoom in with padding
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_padding = (x_max - x_min) * 0.3
        y_padding = (y_max - y_min) * 0.3
        
        #plot individual trajectory onto its own axis
        indax = indTrajax_flat[i]
        indax.pcolormesh(xx, yy, B_mag.T, shading='auto', cmap='bone')
        indax.plot(x, y, linewidth=0.5, label=txt, color=colors[i])
        indax.set_xlim(x_min - x_padding, x_max + x_padding)
        indax.set_ylim(y_min - y_padding, y_max + y_padding)
        indax.grid(alpha=0.3)
         

    #e_k axes labelling for the main plot
    ek_axis.set_xlabel(r'Time, normalised to the thermal Hydrogen gyro-period, $\tau_H$')
    ek_axis.set_ylabel(r'Kinetic energy, $\hat{E}_k$, normalised to $\frac{1}{2}m_H\nu_{th,H}^2$')
    ek_axis.legend(loc='center right', bbox_to_anchor=(1.2, -0.4), fontsize=8)
    ek_axis.grid(alpha=0.3)
    ek_axis.set_title('Kinetic energy of particles varying with time over a spatially\nvarying magnetic field and constant electric field.')
    

    #plotting start and end of kinetic energy on separate figure to see better
    n_t = len(time)
    ek_initialEnd = int(n_t * 0.05)  # first 5% to see start
    ek_finalStart = int(n_t * 0.95)  # last 5% to see end
    
    #flatten the subplots for easier indexing
    startEndEkax_flat = startEndEkax.flatten()
    
    #particles 0,1 on left column, particles 2,3 on right column
    particle_groups = [(0, 1), (2, 3)]
    
    for col, (p1, p2) in enumerate(particle_groups):
        # top row: start of kinetic energy
        ax_start = startEndEkax[0, col]
        ax_start.plot(time[:ek_initialEnd], all_e_k[p1][:ek_initialEnd], label='Particle '+str(p1+1), linewidth=0.8, color=colors[p1])
        ax_start.plot(time[:ek_initialEnd], all_e_k[p2][:ek_initialEnd], label='Particle '+str(p2+1), linewidth=0.8, color=colors[p2])
        ax_start.set_title('Start of kinetic energy (first 5%)', fontsize=10)
        ax_start.grid(alpha=0.3)
        ax_start.legend(fontsize=8)
        
        # bottom row: end of kinetic energy
        ax_end = startEndEkax[1, col]
        ax_end.plot(time[ek_finalStart:], all_e_k[p1][ek_finalStart:], label='Particle '+str(p1+1), linewidth=0.8, color=colors[p1])
        ax_end.plot(time[ek_finalStart:], all_e_k[p2][ek_finalStart:], label='Particle '+str(p2+1), linewidth=0.8, color=colors[p2])
        ax_end.set_title('End of kinetic energy (last 5%)', fontsize=10)
        ax_end.grid(alpha=0.3)
        ax_end.legend(fontsize=8)
    
    #have same x and y axes for all subplots
    startEndEkFig.text(0.5, 0.04, r'Time, normalised to the thermal Hydrogen gyro-period, $\tau_H$', ha='center')
    startEndEkFig.text(0.04, 0.5, r'Kinetic energy, $\hat{E}_k$, normalised to $\frac{1}{2}m_H\nu_{th,H}^2$', va='center', rotation='vertical')
    startEndEkFig.suptitle(r'Start and End of Kinetic Energy with $\hat{E}$ (x, y) = ('+str(E_x)+', '+str(E_y)+r')')

    #trajectory axes labelling
    trajectory_axis.set_xlabel(r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8)
    trajectory_axis.set_ylabel(r'$\hat{y}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8)
    trajectory_axis.grid(alpha=0.3)
    trajectory_axis.legend(loc='lower center', bbox_to_anchor=(0.5,-0.5), fontsize=8)
    trajectory_axis.set_title(r'Trajectories of particles over the background magnetic field with\nelectric field, $\hat{E}$ (x, y) = ('+str(E_x)+', '+str(E_y)+r'), normalised to $\nu_{th,H}B_r$.')
    #choosing B colour to better see the trajectories
    im = trajectory_axis.pcolormesh(xx, yy, B_mag.T, shading = 'auto', cmap='bone')
    trajectory_plot.colorbar(im, ax=trajectory_axis, label=r'Magnetic field, $\hat{B}_z(\hat{x}, \hat{y})$, normalised to $B_r$')
    
    #individual plots of trajectory labelling
    indTrajFig.suptitle(r'Individual trajectories of particles over the background magnetic field with electric field,\n$\hat{E}$ (x, y) = ('+str(E_x)+', '+str(E_y)+r'), normalised to $\nu_{th,H}B_r$.')
    indTrajFig.text(0.5, 0.04, r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', ha='center')
    indTrajFig.text(0.04, 0.5, r'$\hat{y}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', va='center', rotation='vertical')
    indTrajFig.legend(loc='lower center', bbox_to_anchor=(0.45,-0.15), fontsize=8)
    #uses last ax, all are the same
    im2 = indax.pcolormesh(xx, yy, B_mag.T, shading = 'auto', cmap='bone')
    plt.colorbar(im2, ax=indTrajax, label=r'Magnetic field, $\hat{B}$, normalised to $B_r$')

    plt.show()
    
    
    
    
    
if __name__ == "__main__":#

    #get the background magnetic field values
    B_mag = run_B_plot()
    
    electric_fields = asarray([[0.0, 0.0], [0.0, 0.01]])
    
    #iterate over each electric field
    for i in range(len(electric_fields)):
        integrate_and_plot(electric_fields[i][0], electric_fields[i][1], B_mag)
    