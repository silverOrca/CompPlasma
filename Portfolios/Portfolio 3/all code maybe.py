# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 13:08:43 2025

@author: zks524
https://youjunhu.github.io/research_notes/particle_simulation/particle_simulationsu24.html


1: solve Poisson's eq. using fft methods (use .real for return variable)
    - dE/dx = rho, and rho = 1 - n_e



"""

    
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq, ifft, fftshift
import numpy as np
from numpy import linspace, sin, pi, zeros, arange, concatenate
import matplotlib.pyplot as plt


#solves the Poisson equation for the E field using the electron density
#using fourier transforms to solve with respect to x
def solvePoisson(x, ne):
    #define nx and dx
    dx = x[1]-x[0]
    nx = len(x)

    #1/lambda
    lamb = fftfreq(nx, dx)

    #now can calculate k - it is 1/lambda because of normalisation, so no 2 pi
    k = lamb
    
    #get fourier transform of rho = 1 - electron density
    rho_hat = fft(1-ne)
    
    #get array of zeros to put electric field into
    #also sets 1st value to 0
    eHat = zeros(len(rho_hat), dtype='complex')
    
    #what the fourier transform of E is equal to
    #use only from 2nd element onwards so 1st element, zero-wavenumber, stays 0
    #1j is the imaginary number
    eHat[1:] = (rho_hat[1:]) / (1j*k[1:])
    
    #reverse fourier transform to get electric field
    eField = ifft(eHat).real
    
    
    return eField


def fourier_derivative ( field , xx ):
    """ This function calculates d field / dx using a Fourier approach """
    #calculates 1/lambda using fftfreq of the nx and dx and uses this as k
    wavenumbers = fftfreq(len(xx), xx[1] - xx[0])
    #returns inverse fourier of i * k * fft(field)
    #this is the fourier derivative because in fourier space, d/dx = i * k
    return ifft(complex(0, 1) * wavenumbers * fft ( field )).real







def initial_ne(xx, kx, amp):
    from numpy import cos, pi
    nx = len(xx)
    ne0 = 1 + amp * cos(2 * pi * xx * kx)
    ne0[:nx//4] = 1.0
    ne0[3*nx//4:] = 1.0
    return ne0

def initial_Ue(xx):
    return xx * 0.0


#evolves electron density and electron flow with time
def evolveFunc(curT, curF, x, q_e, m_e, T_e):
    nx = len(x)
    
    #set out the variables
    ne = curF[:nx] # electron density
    Ue = curF[nx:] # electron flow
    
    #solve for electric field
    eField = solvePoisson(x, ne)
    
    #use fourier derivatives to get dne/dx and dUe/dx
    dne_dx = fourier_derivative(ne, x)
    dUe_dx = fourier_derivative(Ue, x)
    
    #use normalised equations to solve for Ue and Ne evolving with time
    dne_dt = -Ue*dne_dx - ne*dUe_dx
    dUe_dt = (q_e/m_e)*eField - (T_e/(m_e*ne))*dne_dx
    
    
    return concatenate([dne_dt, dUe_dt])



def staticPoissonSolver():
    x = np.linspace(0, 1, 101, endpoint = False)

    #electron density equation     
    ne = 1 + np.sin(2*np.pi*x)


    eField = solvePoisson(x, ne)

    dEdx = fourier_derivative(eField, x)


    plt.figure(figsize=(10, 6))
    plt.plot(x, ne, color='green', label=r'$\hat{n}_e$')
    plt.plot(x, eField, 'r-', label=r'$\hat{E}_x$')
    plt.plot(x, 1-dEdx, '--', color='orange', label=r'$1 - d\hat{E}_x/d\hat{x}$')
    plt.xlabel(r'Position, $\hat{x}$')
    plt.ylabel('Normalised values')
    plt.legend(loc='best')
    plt.title('Poisson solver verification, for constant electron density and flow.')
    plt.text(0.5, -2.0, 'Fig. 1: Plot of electron density, $\hat{n}_e$, normalised to the plasma density, $n_s$; electric field, $\hat{E}_x$, normalised to the Debye length over plasma\ntemperature, $\lambda_D / T_p$, and 1 - the rate of change of electric field with position, all against position, $\hat{x}$, normalised to the Debye length, $\lambda_D$.\nValues are for a 1-dimensional periodic plasma modelled as a fluid with no net flow and no magnetic field.', ha='center')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




def dynamicPoissonSolver(integrationMethod = 'RK45'):

    x = linspace(0, 1, 401, endpoint=False)
    time = linspace(0, 2*np.pi, 251)
    
    q_e = -1.0
    m_e = 1.0
    T_e = [0.0, 1.0]
    
    nx=len(x)
    
    initialNe = initial_ne(x, kx=4, amp=0.01)
    initialUe = initial_Ue(x)
    
    #needs to be concatenated into one array for solve_ivp
    f0 = concatenate([initialNe, initialUe])
    
    #store solutions for different temperatures
    neSolutions = []
    UeSolutions = []
    
    for i, temp in enumerate(T_e):
        #solve the system of equations
        solution = solve_ivp(evolveFunc, [0, time[-1]], f0, method = integrationMethod, t_eval=time, args=(x,q_e,m_e,T_e[i]))
        
        #unpack solution
        neSol = solution.y[:nx, :].T
        UeSol = solution.y[:,nx:]
        
        
        #do contour plot of electron density with x on y axis and t on x axis
        #need to meshgrid x and t for contourf (map them as coords)
        #use time from solution.t to ensure matching dimensions
        X, T = np.meshgrid(x, solution.t)
        plt.figure(figsize=(10, 6))
        contour = plt.contourf(T, X, neSol, levels=50, cmap='viridis')
        plt.colorbar(contour, label=r'Electron density, $\hat{n}_e$, normalised to the plasma density, $n_s$')
        plt.xlabel(r'Time, $\hat{t}$, normalised to the electron plasma frequency, $\omega_{pe}$')
        plt.ylabel(r'Position, $\hat{x}$, normalised to the Debye length, $\lambda_D$')
        plt.title('Electron Density Evolution Over Time for $\hat{T}_e =$ '+str(temp))
        
        plt.tight_layout()
        plt.show()
        
        #print the highest value and lowest value of electron density overall
        print(f'Integration method: {integrationMethod}')
        print(f'For Te = {temp}, max ne = {np.max(neSol)}, min ne = {np.min(neSol)}')
        

        # #plot contour graph of electron flow against position and time
        # X, T = np.meshgrid(x, solution.t)
        # plt.figure(figsize=(10, 6))
        # contour = plt.contourf(T, X, UeSol, levels=50, cmap='viridis')
        # plt.colorbar(contour, label=r'Electron flow, $\hat{U}_e$, normalised to the electron thermal velocity, $v_{Te}$')
        # plt.xlabel(r'Time, $\hat{t}$, normalised to the electron plasma frequency, $\omega_{pe}$')
        # plt.ylabel(r'Position, $\hat{x}$, normalised to the Debye length, $\lambda_D$')
        # plt.title('Electron Flow Evolution Over Time for $\hat{T}_e =$ '+str(temp))

        # plt.tight_layout()
        # plt.show()

        #append solutions to lists to return
        neSolutions.append(neSol)
        UeSolutions.append(UeSol)

    return neSolutions
        
#compare the final electron density for the two methods and two temperatures on same figure        
def compareMethods(neSolutions):
    #splits the values up into separate variables for clarity
    #variables are arrays of electron density with time and position with shape (time, position)
    Te0Ne_RK45 = neSolutions[0][0]  # Te=0.0 solution for RK45
    Te0Ne_Radau = neSolutions[1][0]  # Te=0.0 solution for Radau
    Te1Ne_RK45 = neSolutions[0][1]  # Te=1.0 solution for RK45
    Te1Ne_Radau = neSolutions[1][1]  # Te=1.0 solution for Radau 
    #need to define x again
    x = linspace(0, 1, 401, endpoint=False)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    
    coldAx = ax[0]
    #plots last value of the electron density for all positions
    coldAx.plot(x, Te0Ne_RK45[-1, :], label='RK45', linestyle='-')
    coldAx.plot(x, Te0Ne_Radau[-1, :], label='Radau', linestyle='--')
    coldAx.set_title('Comparison of Final Electron Density Profiles for Different Integration Methods (Te=0.0)')
    coldAx.grid(alpha=0.3)
    coldAx.legend()
    
    warmAx = ax[1]
    #does the same for Te=1.0
    warmAx.plot(x, Te1Ne_RK45[-1, :], label='RK45', linestyle='-')
    warmAx.plot(x, Te1Ne_Radau[-1, :], label='Radau', linestyle='--')
    warmAx.set_title('Comparison of Final Electron Density Profiles for Different Integration Methods (Te=1.0)')
    warmAx.grid(alpha=0.3)
    warmAx.legend()
    
    fig.text(0.5, -0.01, r'Position, $\hat{x}$, normalised to the Debye length, $\lambda_D$', ha='center')
    fig.text(-0.02, 0.5, r'Final Electron Density, $\hat{n}_e$, normalised to the plasma density, $n_s$', va='center', rotation='vertical')
    
    plt.tight_layout()
    plt.show()

    print(np.shape(Te0Ne_RK45), np.shape(Te0Ne_Radau))
    


if __name__ == '__main__':
    
    staticPoissonSolver()
    
    methods = ['RK45', 'Radau']

    #store solutions for different methods
    #2d arrays, first index is method, second is temperature
    neSolutions = []
    
    #use different integration methods for solve_ivp to compare results
    for method in methods:
        neSol = dynamicPoissonSolver(integrationMethod = method)
        #appends different temperature solutions for each method
        neSolutions.append(neSol)

    
    compareMethods(neSolutions)


