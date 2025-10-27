#solves differential equations
from scipy.integrate import solve_ivp
#this is the interpolation module
from scipy.interpolate import interp1d
#maths modules
from numpy import linspace, exp, sqrt, pi
#plotting graphs
import matplotlib.pyplot as plt
#plotting axes within a graph with a plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#distance from plasma into sheath
xPos = linspace(0, 60, 100)

#normalised ion velocity at start of sheath
#using multiple values to compare
v_s_hat = [1.]

#initial conditions
phi_hat = 0.
electric_field_hat = 0.001
v_i_hat = 1.

#define function of values with initial values
f0 = [phi_hat, electric_field_hat]

g0 = [v_i_hat]


#different lengths before sheath (debye lengths)
lengths = [0.1, 1., 10., 100., 1000., 10000.]

#function of poisson equation
def dfdt(curX, curF, v_s_hat = 1.):
    #set out the variables
    phi_hat = curF[0]
    electric_field_hat = curF[1]
    
    vi_squared = v_s_hat**2 - 2 * phi_hat
    n_i = v_s_hat / sqrt(vi_squared)
    
    n_e = exp(phi_hat)
    dedt = n_i - n_e
    
    dphidt = -electric_field_hat
    
    
    return [dphidt, dedt]


#solves for how the ion velocity changes w.r.t. position to wall
def dgdt(curX, curF, electric_field_hat, length):
    #unpack values
    v_i_hat = curF[0]
    e_field = electric_field_hat(curX)
    
    dvi_dt = (e_field / v_i_hat) - (1/length) * v_i_hat
    
    return dvi_dt


#solves for the electrostatic potential energy and electric field, returns result only
def solve_field(xPos, v_s_hat, phi_hat, electric_field_hat, f0):
    result = solve_ivp(dfdt, [xPos[0], xPos[-1]], f0, t_eval = xPos, args = (v_s_hat,))
    return result
 

#solves for the current
def solve_current(potentials, v_s_hat, mass_ratio):
    #Plotting normalised current against time
    j = sqrt(mass_ratio/(2*pi)) * exp(potentials) - 1
    
    return j
  
    
#solves for the ion velocity 
def solve_ion_velocity(xPos, electric_field_hat, length, v_s_hat):
    
    ion_velocities = solve_ivp(dgdt, [xPos[0], xPos[-1]], g0, t_eval = xPos, args = (electric_field_hat, length))
    return ion_velocities
    

#finds the wall positions and returns them
def find_wall(xPos, j_result):
    #create interpolated function of x and j values
    interpolated = interp1d(j_result, xPos)
    
    #find where the interpolated is = 0
    wall = interpolated(0.0)
    
    return wall

def velocities_at_wall(xPos, velocities, wall_position):
    #get corresponding values for velocity against position
    vel_interp = interp1d(xPos, velocities)
    
    #find velocity at wall
    v_wall = vel_interp(wall_position)
    
    return v_wall
    

  
def plot_field_results(xPos, potentials, electric_fields, v_s_hat):
    # Plot the electric field data
    for i, (potential, field, v_s) in enumerate(zip(potentials, electric_fields, v_s_hat)):
        plt.plot(xPos, potential, label = 'Normalised Electrostatic Potential Energy, '+r'$\hat{\phi}$'+', for '+r'$\hat{v}_s = $'+str(v_s))
        plt.plot(xPos, field, label = 'Normalised Electric Field, ' + r'$\hat{E}$'+', for '+r'$\hat{v}_s = $'+str(v_s), linestyle='dashed')
    
    #caption for electric field plot
    field_caption = 'Figure 1: Graph showing how the normalised electrostatic potential energy and normalised electric field vary with distance through a plasma sheath into a wall, for varying initial ion velocities.' 
        
    plt.title('Normalised potential energy and electric field of plasma sheath against position')
    plt.xlabel('Distance from bulk plasma into sheath, '+r'$\hat{x}$'+', normalised to '+r'$\lambda_D$')
    plt.ylabel('Energy, ' + r'$\hat{\phi}$' + ' and Electric Field, ' + r'$\hat{E}$')
    plt.legend(loc='best', prop={'size': 7})
    plt.figtext(0.5, -0.055, field_caption, wrap=True, horizontalalignment='center', fontsize=8)
    plt.grid()
    plt.show()
  
    
  
def plot_current_results(xPos, wall_pos, current_result, v_s_hat):
    
    #loops over number of indexes within 2d array of results
    for i, (j_result, wall_position, v_s) in enumerate(zip(current_result, wall_pos, v_s_hat)):
        #subtracts extra wall position from x value to line up all plots
        plt.plot(xPos-wall_position, j_result, label=r'$\hat{v}_s = $'+str(v_s))
    
    #caption for normalised current figure
    j_caption = 'Figure 2: Graph showing normalised current varying with distance into a plasma sheath, for varying ion velocities.'

    #to show visually where plasma and sheath wall are
    plt.axvspan(-40, 0, facecolor='red', alpha=0.1, label='Plasma')
    plt.axvspan(0, 40, facecolor='blue', alpha=0.1, label='Wall')

    plt.xlabel('Distance (from bulk plasma into wall), ' +r'$\hat{x} - x_w$'+', normalised to '+r'$\lambda_D$')
    plt.ylabel('Current, ' + r'$\hat{j}$'+', normalised to '+r'$e n_s c_s$')
    plt.title('Normalised current against position')
    plt.legend(loc='best')
    plt.figtext(0.5, -0.055, j_caption, wrap=True, horizontalalignment='center', fontsize=8)
    plt.grid()
    plt.show()
    
  
def plot_velocity_results(xPos, wall_pos, velocity_result, lengths, v_s_hat):
    
    fig, ax = plt.subplots()
    
    for i, (v, length) in enumerate(zip(velocity_result, lengths)):
        ax.plot(xPos-wall_pos, v, label = 'L = '+str(length))
        
        
    ax.set_xlabel('Distance (from bulk plasma into wall), ' +r'$\hat{x} - x_w$'+', normalised to '+r'$\lambda_D$')
    ax.set_ylabel('Ion velocity, ' + r'$\hat{v}_i$'+', normalised to '+r'$c_s$')
    
    ax.grid(True, alpha=0.4)
    ax.set_title('Ion Velocities for Different Collision Lengths')
    
    #find velocities at wall
    wall_velocities = []
    
    #iterates over results to find velocity at wall for each
    for v in velocity_result:
        v_wall = velocities_at_wall(xPos, v, wall_pos)
        wall_velocities.append(v_wall)
    
    
    #inset plot of velocities at wall
    axins = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=3)
    
    #set log scale
    axins.semilogx(lengths, wall_velocities, '-o', markersize=3)
    
    axins.set_xlabel('Collision Length, L, normalised to '+r'$\lambda_D$', fontsize=7)
    axins.set_ylabel(r'$\hat{v}_i$ at wall', fontsize=7)
    axins.grid(True, alpha=0.4)
    axins.set_title('Velocity at Wall vs L', fontsize=8)
    
    #set axis limits for better visualisation
    
    #go slightly above where the x axis begins and ends
    axins.set_xlim([0.05, 20000])
    
    #need to find how high y axis goes
    v_min, v_max = min(wall_velocities), max(wall_velocities)
    #need to go slightly above axis again
    v_range = v_max - v_min
    axins.set_ylim([v_min - 0.1*v_range, v_max + 0.1*v_range])
    
    ax.legend(loc='upper right', prop={'size': 8})
    
    
    plt.show()
  
    
  
    
    
def main():
    
    #arrays to store results of potential and field
    potentials = []
    electric_fields = []

    #calls the electric field solve function and shows graph
    for i in range(len(v_s_hat)):
        e_result = solve_field(xPos, v_s_hat[i], phi_hat, electric_field_hat, f0)
        
        #unpack the results
        potentials.append(e_result.y[0,:])

        electric_fields.append(e_result.y[1,:])
        
        
    plot_field_results(xPos, potentials, electric_fields, v_s_hat)
    

    #store current results
    current_result = []
    #store wall placements with corresponding velocities
    wall_pos = []

    #calls the current solve function and shows graph
    for i in range(len(v_s_hat)):
        j_result = solve_current(potentials[i], v_s_hat[i], 1840)
        current_result.append(j_result)
        
        #find the current wall positions
        cur_wall = find_wall(xPos, j_result)
        
        #putting roots into list of wall locs for each velocity
        wall_pos.append(cur_wall)
        
    plot_current_results(xPos, wall_pos, current_result, v_s_hat)

    

    ##Assignment 3
    
    
    
    velocity_result = []
    #interpolate electric field to enable plotting
    e_interp = interp1d(xPos, electric_fields)
    
    for i in range(len(v_s_hat)):
        for length in lengths:
            
            ion_velocity = solve_ion_velocity(xPos, e_interp, length, v_s_hat[i])
            
            #unpack and store results
            velocity_result.append(ion_velocity.y[0,:])
    print(velocity_result) 
    plot_velocity_results(xPos, wall_pos, velocity_result, lengths, v_s_hat)



if __name__ == "__main__":
    main()

