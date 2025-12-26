# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 12:03:44 2025

@author: ciara
"""

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from numpy import linspace
import numpy as np

import matplotlib.pyplot as plt


#length of box
L_x = 1.0

#Number of points
nx = 10000

#Set of interval numbers (x values) in between the length (0 and L_x)
xval = linspace(0, L_x, num=nx)

#Setting the spacing of element we are integrating over
#Each spacing between the x values is the same, so just using first 2
dx = xval[1]-xval[0]

D = 0.2     #diffusion coefficient

v = [0.2, -0.2]     #advection flow

S = 100 * (1 - np.tanh(-5*(xval - 0.75)))       #net source

R = [0, -0.1, -0.2]     #reaction coefficient

#rhs vector, equal to what the 2nd order difference equals
rho = -S

#returns a sparse matrix of the input x grid
#also given advection speed v and reaction coefficient R
#has neumann lower boundary and dirichlet upper boundary
def return_sparse_matrix(aVal, v, R):
    #create the empty sparse matrix
    M = lil_matrix((nx, nx))
    
    #set the values in middle of matrix (2nd order diff)
    for i in range(1, nx-1):
        # sets previous value (A)
        M[i, i-1] = (D/dx**2) - ((0.5*v)/dx)
        # Sets current value (B)
        M[i, i]   =  -(2.0*D)/dx**2 + R
        # Sets following value (C)
        M[i, i+1] = (D/dx**2) + ((0.5*v)/dx)
    
    
    #set boundaries
    #lower boundaries - 2nd order neumann
    M[0,0] = -1.5/dx
    M[0,1] = 2/dx
    M[0,2] = -0.5/dx
    rho[0] = aVal
    
    
    #upper boundary - dirichlet (zero-Dirichlet means M= 1)
    M[-1,-1] = 1.0
    rho[-1] = aVal
    
    return M

def main():
    # Create figure with better layout
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.3], hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax_info = fig.add_subplot(gs[2])
    ax_info.axis('off')  # For text information
    
    # Create color map for R values
    norm = plt.Normalize(min(R), max(R))
    cmap = plt.cm.plasma
    
    plots = []
    labels = []
    
    for idx, value in enumerate(v):
        axPlot = ax1 if idx == 0 else ax2
        
        for co in R:
            M = return_sparse_matrix(aVal=0.0, v=value, R=co)
            M = M.tocsr()
            solution = spsolve(M, rho)
            
            # Plot with color based on R value
            line, = axPlot.plot(xval, solution, 
                               color=cmap(norm(co)),
                               linewidth=2.5,
                               alpha=0.9,
                               label=f'R = {co}')
            
            if idx == 0:
                plots.append(line)
                labels.append(f'R = {co}')
        
        # Add vertical line at source peak
        source_peak_idx = np.argmax(S)
        axPlot.axvline(x=xval[source_peak_idx], 
                      color='red', 
                      linestyle='--', 
                      alpha=0.4,
                      label='Source peak' if idx == 0 else "")
        
        # Customize plot
        axPlot.set_ylabel(f'Pressure\n(v = {value})', fontsize=12)
        axPlot.grid(True, alpha=0.2, linestyle='--')
        axPlot.legend(loc='upper right', fontsize=10)
        
        # Add value at boundaries
        axPlot.text(0.02, 0.05, f'P(0) = {solution[0]:.3f}', 
                   transform=axPlot.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axPlot.text(0.85, 0.05, f'P(L) = {solution[-1]:.3f}', 
                   transform=axPlot.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create custom colorbar for R values
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Reaction Coefficient (R)', fontsize=12)
    
    # Add information panel
    info_text = f"""
    Parameters:
    • Domain length: L = {L_x}
    • Grid points: N = {nx} (Δx = {dx:.2e})
    • Diffusion coefficient: D = {D}
    • Advection speeds: v = {v}
    • Reaction coefficients: R = {R}
    • Boundary conditions: Neumann at x=0, Dirichlet at x=L
    """
    ax_info.text(0.02, 0.5, info_text, fontsize=11, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Final touches
    ax2.set_xlabel('Position (x)', fontsize=14, fontweight='bold')
    fig.suptitle('Advection-Reaction-Diffusion System: Pressure Distribution', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.show()
    
    
#run the code
if __name__ == '__main__':
    main()


