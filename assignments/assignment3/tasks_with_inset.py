from scipy.integrate import solve_ivp
#this is the interpolation module
from scipy.interpolate import interp1d
#to find x at a root (y=0)
from scipy.optimize import root_scalar
from numpy import linspace, exp, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Add numerical safety
    if vi_squared <= 0:
        print(f"Warning: vi_squared = {vi_squared} at x = {curX}")
        n_i = 0.0
    else:
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
    try:
        # Check if current crosses zero
        if not ((j_result.min() <= 0) and (j_result.max() >= 0)):
            print(f"Warning: Current doesn't cross zero. Range: [{j_result.min():.3f}, {j_result.max():.3f}]")
            return None
        
        # Create interpolated function j(x) - FIXED: correct order!
        interpolated = interp1d(xPos, j_result, bounds_error=False, fill_value='extrapolate')
        
        # Find where j(x) = 0 using root finding
        result = root_scalar(interpolated, bracket=[xPos[0], xPos[-1]], method='brentq')
        
        if result.converged:
            return result.root
        else:
            print("Warning: Root finding did not converge")
            return None
            
    except Exception as e:
        print(f"Error in find_wall: {e}")
        return None

  
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
    

def extract_velocity_at_wall(xPos, velocity_array, wall_position):
    """Extract ion velocity at the wall position using interpolation"""
    try:
        # Create interpolation function for velocity vs position
        vel_interp = interp1d(xPos, velocity_array, bounds_error=False, fill_value='extrapolate')
        
        # Get velocity at wall position
        v_wall = vel_interp(wall_position)
        return v_wall
    except Exception as e:
        print(f"Error extracting velocity at wall: {e}")
        return np.nan
    
  
def plot_velocity_results(xPos, wall_pos, velocity_result, lengths, v_s_hat):
    
    # Create main figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot velocity profiles
    for i, (v, length) in enumerate(zip(velocity_result, lengths)):
        ax.plot(xPos-wall_pos[0], v, label = 'L = '+str(length), linewidth=2)
    
    ax.set_xlabel('Distance (from bulk plasma into wall), ' +r'$\hat{x} - x_w$'+', normalised to '+r'$\lambda_D$')
    ax.set_ylabel('Ion velocity, ' + r'$\hat{v}_i$'+', normalised to '+r'$c_s$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('Ion Velocity Profiles for Different Damping Lengths')
    
    # Extract velocities at wall for inset plot
    velocities_at_wall = []
    for i, v in enumerate(velocity_result):
        v_wall = extract_velocity_at_wall(xPos, v, wall_pos[0])
        velocities_at_wall.append(v_wall)
    
    # Create inset plot
    axins = inset_axes(ax, width="40%", height="40%", loc='center left', borderpad=3)
    
    # Plot velocity at wall vs L on log scale
    axins.semilogx(lengths, velocities_at_wall, 'ro-', linewidth=2, markersize=6)
    axins.set_xlabel('Damping Length, L', fontsize=10)
    axins.set_ylabel(r'$\hat{v}_i$ at wall', fontsize=10)
    axins.grid(True, alpha=0.3)
    axins.set_title('Velocity at Wall vs L', fontsize=10)
    
    # Set axis limits for better visualization
    axins.set_xlim([0.05, 20000])  # Slightly wider than data range
    if velocities_at_wall:
        v_min, v_max = min(velocities_at_wall), max(velocities_at_wall)
        v_range = v_max - v_min
        axins.set_ylim([v_min - 0.1*v_range, v_max + 0.1*v_range])
    
    plt.tight_layout()
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
        j_result = solve_current(potentials[i], v_s_hat[i], 1840)  # FIXED: use potentials[i]
        current_result.append(j_result)
        
        #find the current wall positions
        cur_wall = find_wall(xPos, j_result)
        
        #putting roots into list of wall locs for each velocity
        wall_pos.append(cur_wall)
        
    plot_current_results(xPos, wall_pos, current_result, v_s_hat)

    ##Assignment 3
    
    velocity_result = []
    #interpolate electric field to enable plotting - FIXED: use correct field for single v_s_hat
    e_interp = interp1d(xPos, electric_fields[0], bounds_error=False, fill_value='extrapolate')
    
    for i in range(len(v_s_hat)):
        for length in lengths:
            
            ion_velocity = solve_ion_velocity(xPos, e_interp, length, v_s_hat[i])
            
            #unpack and store results
            velocity_result.append(ion_velocity.y[0,:])
    
    print(f"Computed {len(velocity_result)} velocity profiles")
    print(f"Wall position: {wall_pos[0]:.3f}")
    
    plot_velocity_results(xPos, wall_pos, velocity_result, lengths, v_s_hat)


if __name__ == "__main__":
    main()