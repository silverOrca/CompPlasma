# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 11:22:43 2025

@author: ciara

Need to solve equations of motion for a single particle in the x and y directions
"""

import matplotlib.pyplot as plt
from numpy import zeros, linspace
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
    figurebmag = plt.figure()
    #create the plot      
    plt.pcolormesh(xx, yy, Bmag.T, shading = 'auto')
    plt.xlabel(r'$\hat{x}$, normalised to thermal Hydrogen gyro-radius, $\rho_H$', fontsize=8) 
    plt.ylabel(r'$\hat{y}$, normalised to thermal Hydrogen gyro-period, $\tau_H$', fontsize=8)
    #plt.title(r"")
    plt.text(-4, -11, r'Figure 1: Example $B_{func}$ output showing $B_z$ varying with $\hat{x}$ and $\hat{y}$.')
    plt.colorbar()


# #runs the code to output the B_func plot
def run_B_plot():
    
    #fill the empty array with spatially varying B calculated from B_func
    for ix, x in enumerate(xx):
        for iy, y in enumerate(yy):
            B_z = (B_0 * x_0) / (x - x_c)  # Calculate B_z directly here
            Bmag[ix, iy] = B_z
    
    #plotting(Bmag)
    
    return Bmag



#kinetic energy equation, normalised
def kinetic_energy(m, v_x, v_y):
    e_k = m * (v_x**2 + v_y**2)
    
    return e_k


    
def main():
    E_x = 0.0 
    E_y = 0.01
        
    #velocity
    figure1 = plt.figure()
    #kinetic energy
    figure2 = plt.figure()
    #trajectory in x and y
    figure3 = plt.figure()
    
    ax1 = figure1.add_subplot(1,1,1)
    ax2 = figure2.add_subplot(1,1,1)
    
    ax3 = figure3.add_subplot(1,1,1)
    ax4=ax3.twinx().twiny()
    
    #ax5 = figure4.add_subplot(1,1,1)
    #ax5=ax5.twinx()
    
    B_mag= run_B_plot()
    im = ax3.pcolormesh(xx, yy, B_mag.T, shading = 'auto', zorder=0)
    ax3.set_xlabel('x hat')
    ax3.set_ylabel('y hat')
    
    for i in range(len(initial_x)):
        f0 = [initial_x[i], initial_y[i], initial_vx[i], initial_vy[i]]
        solution = solve_ivp(dvdt, [time[0], time[-1]], f0, t_eval=time, args=(q[i], m[i], E_x, E_y))
        
        #set out solutions
        x = solution.y[0,:]
        y = solution.y[1,:]
        v_x = solution.y[2,:]
        v_y = solution.y[3,:]
        
        #plot the velocity component
        ax1.plot(time, x, label='dvx_dt '+str(i))
        ax1.plot(time, y, label='dvy_dt '+str(i))
                
        #solve kinetic energy and plot
        e_k = kinetic_energy(m[i], v_x, v_y)
        
        #do final e_k / initial e_k for each particle
        print(e_k[-1]/e_k[0])
        
        
        ax2.plot(time, e_k, label =r'$E_k$ '+str(i))
        
        #plot the trajectory - y against x
        ax4.plot(x, y, zorder=10)

        
         
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Velocity components')
    ax1.set_title('Velocity components of charged particle over time in varying B field')
    ax1.legend(loc='best')
    
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Kinetic energy')
    ax1.set_title('Kinetic energy of particle varying with time')
    
    #solve for background magnetic field
   
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
   
    figure3.colorbar(im, ax=ax4, label='Magnetic field')
    plt.show()
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()