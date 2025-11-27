#!/usr/bin/env python3
#
# Electrostatic PIC code in a 1D cyclic domain
from genericpath import isfile
import os
import time
from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max
import numpy as np
from scipy.signal import argrelextrema, find_peaks

import matplotlib.pyplot as plt # Matplotlib plotting library

try:
    import matplotlib.gridspec as gridspec  # For plot layout grid
    got_gridspec = True
except:
    got_gridspec = False

# Need an FFT routine, either from SciPy or NumPy
try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

#-----------------------------------------------------------------------------#

#rk4 method - does time integration
#used to get positions and velocities at future times (over an amount of time)
def rk4step(f, y0, dt, args=()):
    """ Takes a single step using RK4 method """
    #f = rk4step(pic, f, dt, args=(ncells, L)) is when this is called
        #f in calling is: f = concatenate( (pos, vel) ), the starting state
    
    k1 = f(y0, *args)
    k2 = f(y0 + 0.5*dt*k1, *args)
    k3 = f(y0 + 0.5*dt*k2, *args)
    k4 = f(y0 + dt*k3, *args)

    #4th order runga-kutta integration formula
    return y0 + (k1 + 2.*k2 + 2.*k3 + k4)*dt / 6.

#-----------------------------------------------------------------------------#

def calc_density(position, ncells, L):
    """ Calculate charge density given particle positions
    
    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient
    
    #creates empty array of size number of cells
    density = zeros([ncells])
    #defines number of particles as number of elements in position array
    nparticles = len(position)
    
    dx = L / ncells       # Uniform cell spacing
    
    #for each particle, 
    for p in position / dx:    # Loop over all the particles, converting position into a cell number
        plower = int(p)        # Cell to the left (rounding down)
        offset = p - plower    # Offset from the left
        #adds 1 - the offset to the lefthand density (offset goes to current density)
        density[plower] += 1. - offset
        #current density modulus number of cells (remainder) add offset
        density[(plower + 1) % ncells] += offset
        
    # nparticles now distributed amongst ncells
    #density constant throughout if the distribution is even
    density *= float(ncells) / float(nparticles)  # Make average density equal to 1
    
    return density

#-----------------------------------------------------------------------------#

def periodic_interp(y, x):
    """
    Linear interpolation of a periodic array y at index x
    
    Input

    y - Array of values to be interpolated
    x - Index where result required. Can be an array of values
    
    Output
    
    y[x] with non-integer x
    """
    ny = len(y)
    if len(x) > 1:
        y = array(y) # Make sure it's a NumPy array for array indexing
    xl = floor(x).astype(int) # Left index
    dx = x - xl
    xl = ((xl % ny) + ny) % ny  # Ensures between 0 and ny-1 inclusive
    return y[xl]*(1. - dx) + y[(xl+1)%ny]*dx

#-----------------------------------------------------------------------------#

#used for calculating the electric field from a rho calculation
def fft_integrate(y):
    """ Integrate a periodic function using FFTs
    """
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )
    k = 2.*pi*k/n
    
    # Modify frequencies by dividing by ik
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f).real # Reverse Fourier Transform
 
#-----------------------------------------------------------------------------#

def pic(f, ncells, L):
    """ f contains the position and velocity of all particles
    """
    nparticles = len(f) // 2     # Two values for each particle
    pos = f[0:nparticles] # Position of each particle
    vel = f[nparticles:]      # Velocity of each particle

    dx = L / float(ncells)    # Cell spacing

    # Ensure that pos is between 0 and L
    pos = ((pos % L) + L) % L
    
    # Calculate number density, normalised so 1 when uniform
    density = calc_density(pos, ncells, L)
    
    # Subtract ion density to get total charge density
    #Ion density normalised to 1 (large so assuming neutral background)
    rho = density - 1.
    
    # Calculate electric field
    E = -fft_integrate(rho)*dx
    
    # Interpolate E field at particle locations
    #Because of normalisation, the rate of change of velocity (acceleration) is given by the -ve of E
    accel = -periodic_interp(E, pos/dx)

    # Put back into a single array
    return concatenate( (vel, accel) )

####################################################################

def run(pos, vel, L, ncells=None, out=[], output_times=linspace(0,20,100), cfl=0.5):
    
    if ncells == None:
        ncells = int(sqrt(len(pos))) # A sensible default

    dx = L / float(ncells)
    
    f = concatenate( (pos, vel) )   # Starting state
    nparticles = len(pos)
    
    time = 0.0
    for tnext in output_times:
        # Advance to tnext
        stepping = True
        while stepping:
            # Maximum distance a particle can move is one cell
            dt = cfl * dx / max(abs(vel))
            if time + dt >= tnext:
                # Next time will hit or exceed required output time
                stepping = False
                dt = tnext - time
            f = rk4step(pic, f, dt, args=(ncells, L))
            time += dt
            
        # Extract position and velocities
        pos = ((f[0:nparticles] % L) + L) % L
        vel = f[nparticles:]
        
        # Send to output functions
        for func in out:
            func(pos, vel, ncells, L, time)
        
    return pos, vel

####################################################################
# 
# Output functions and classes
#

class Plot:
    """
    Displays three plots: phase space, charge density, and velocity distribution
    """
    def __init__(self, pos, vel, ncells, L):
        
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        # Plot initial positions
        if got_gridspec:
            self.fig = plt.figure()
            self.gs = gridspec.GridSpec(4, 4)
            ax = self.fig.add_subplot(self.gs[0:3,0:3])
            self.phase_plot = ax.plot(pos, vel, '.')[0]
            ax.set_title("Phase space")
            
            ax = self.fig.add_subplot(self.gs[3,0:3])
            self.density_plot = ax.plot(linspace(0, L, ncells), d)[0]
            
            ax = self.fig.add_subplot(self.gs[0:3,3])
            self.vel_plot = ax.plot(vhist, vbins)[0]
        else:
            self.fig = plt.figure()
            self.phase_plot = plt.plot(pos, vel, '.')[0]
            
            self.fig = plt.figure()
            self.density_plot = plt.plot(linspace(0, L, ncells), d)[0]
            
            self.fig = plt.figure()
            self.vel_plot = plt.plot(vhist, vbins)[0]
        plt.ion()
        plt.show()
        
    def __call__(self, pos, vel, ncells, L, t):
        d = calc_density(pos, ncells, L)
        vhist, bins  = histogram(vel, int(sqrt(len(vel))))
        vbins = 0.5*(bins[1:]+bins[:-1])
        
        self.phase_plot.set_data(pos, vel) # Update the plot
        self.density_plot.set_data(linspace(0, L, ncells), d)
        self.vel_plot.set_data(vhist, vbins)
        plt.draw()
        plt.pause(0.05)
        

class Summary:
    def __init__(self):
        self.t = []
        self.firstharmonic = []
        
    def __call__(self, pos, vel, ncells, L, t):
        # Calculate the charge density
        d = calc_density(pos, ncells, L)
        
        # Amplitude of the first harmonic
        fh = 2.*abs(fft(d)[1]) / float(ncells)
        
        #commented out printing because data is saved in file instead - speed code up
        #print(f"Time: {t} First: {fh}")
        
        self.t.append(t)
        self.firstharmonic.append(fh)

####################################################################
# 
# Functions to create the initial conditions
#

def landau(npart, L, alpha=0.2):
    """
    Creates the initial conditions for Landau damping
    
    """
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    pos0 = pos.copy()
    k = 2.*pi / L
    for i in range(10): # Adjust distribution using Newton iterations
        pos -= ( pos + alpha*sin(k*pos)/k - pos0 ) / ( 1. + alpha*cos(k*pos) )
        
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    return pos, vel

def twostream(npart, L, vbeam=2):
    # Start with a uniform distribution of positions
    pos = random.uniform(0., L, npart)
    # Normal velocity distribution
    vel = random.normal(0.0, 1.0, npart)
    
    np2 = int(npart / 2)
    vel[:np2] += vbeam  # Half the particles moving one way
    vel[np2:] -= vbeam  # and half the other
    
    return pos,vel


####################################################################
#My own functions

def separateNoise(peaks, noisePeak, amplitudes, times):
    
    #find minima values to use for removing the noise peak at minima
    minima, _ = find_peaks(-amplitudes)
    
    #find the minima just before the noisy peak
    #it will have the same location in the array as the peak index
    indexMin = np.where(peaks == noisePeak)
    
    removeMinima = minima[indexMin]
    #need to get value because is currently as array
    removeMinima = removeMinima[0]
    
    #now get all amplitude and time data from this point onwards
    noiseData = amplitudes[removeMinima:]
    noiseTime = times[removeMinima:]
    
    #get all good data from start until this point
    goodData = amplitudes[0:removeMinima]
    goodTime = times[0:removeMinima]
    
    #for use in finding frequency - list of good peaks
    indexMin = indexMin[0][0]
    goodPeaks = peaks[0:indexMin]
    
    #iterate over noisy peaks to take rms for finding noise amplitude level
    totalNoiseSquared = 0.
    for peak in peaks:
        totalNoiseSquared += (amplitudes[peak])**2
    averageNoise = totalNoiseSquared/len(peaks)
    rms = sqrt(averageNoise)
    print(rms)
    
    return noiseData, goodData, noiseTime, goodTime, goodPeaks



def locateNoise(peak_values, peaks):

    #iterates over peaks to find the first one that is greater than the last one
    for i in range(len(peak_values)):
        if i != 0:
            if peak_values[i] >= peak_values[i-1]:
                noisePeakValue = peak_values[i]
                noisePeak = peaks[i]
                break

    return noisePeakValue, noisePeak


#finds each peak in the first harmonic data
def findPeaks(amplitudes, times):

    peaks, _ = find_peaks(amplitudes)
    
        
    peak_values=amplitudes[peaks]
    time_values=times[peaks]
    
    #returns the amplitudes of each peak with its corresponding time value
    return peak_values, time_values, peaks


#find the angular frequency of the peaks (using only good non-noise-dominated data)
#need the values of the goodTime values at each good peak - take average
#get error from variance between the period (slightly different between each peak)
def getFrequency(goodPeaks, goodTime):
    periods = []
    #first find the average period, then inverse for frequency
    #don't calculate for first peak because there isn't a peak before it
    for i, peak in enumerate(goodPeaks[1:]):
        prevPeak = goodPeaks[i]
        periods.append(goodTime[peak] - goodTime[prevPeak])


    avgPeriod = sum(periods) / len(periods)
    #in radians because of time normalisation
    avgFreq = 1 / avgPeriod
    
    #calculate the variance for the s.d. of the mean for uncertainty
    variance=0.
    for p in periods:
        variance += (p-avgPeriod)**2

    #s.d. equation
    sigmaPeriod = sqrt(variance/(len(periods)-1))
    
    #convert to frequency error  
    sigmaFreq = sigmaPeriod / avgPeriod**2


    print(f"Frequency (radians): {avgFreq} +/- {sigmaFreq}")
    
    
#plot a line across the good peaks to get damping
def dampingEq(m, a, x, b):
    return m * np.exp(-a * x) + b

def getDamping():
    #initial values
    m = 0.25
    a = 0.1
    b = 
    
    


def plotData(filename, ):
        #Make a semilog plot to see exponential damping, with peaks
        plt.figure()
        
        if plotAll:
            #plot the raw data
            plt.plot(times, amplitudes, color='black', label='Raw values')
            
        if plotPeaks:
            #plot x's where the peaks are
            plt.plot(time_values, peak_values, 'x', color='peru', label='Peaks')
            
        if plotNoise:
            #plot the noise-dominated data in a red dashed line
            plt.plot(noiseTime, noiseData, '--', color='red', label='Noise-dominated data')
            
        if plotGood:
            #plot the useful data in a green dashed line
            plt.plot(goodTime, goodData, '--', color='green', label='Useful data')
        
        if dampingLine:
            poly, coeffs = getDamping(goodPeaks, goodData, goodTime)
            plt.plot(goodTime, poly(goodTime), label='fit')


        plt.xlabel(r"Time [Normalised to ${\omega_p}^{-1}$]")
        plt.ylabel(r"First harmonic amplitude [Normalised to $\lambda_D$]")
        #plt.yscale('log')
        
        plt.title('Figure 6: Plot of normalised first harmonic amplitude against normalised time,\n for an electric field wave propogating through a plasma.')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.ioff() # This so that the windows stay open - disables interactive mode
        plt.show()
        
        

def getData(file, dampingLine=True):
    #load in data from file and plot it
    file=False
    times = []
    amplitudes = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                p = line.split()
                times.append(float(p[0]))
                amplitudes.append(float(p[1]))
        file = True
    except:
        print('Failed to find and open file.')
    
    if file:
        times=np.asarray(times)
        amplitudes=np.asarray(amplitudes)
        #first, find the peaks (peaks variable is indices of peaks)
        peak_values, time_values, peaks = findPeaks(amplitudes, times)
        
        #next, find where the next peak is greater than last peak
        noisePeakValue, noisePeak = locateNoise(peak_values, peaks)

        #now separate the two sides of the data
        #should separate the whole peak, so find the minima point before the noise peak
        noiseData, goodData, noiseTime, goodTime, goodPeaks = separateNoise(peaks, noisePeak, amplitudes, times)
    
    return goodPeaks, goodData, goodTime

        
def saveData(L, ncells, npart, s):
    #save code in a file
    #filename to include: L - length of box (physical dependent variable), Number of cells, Number of particles, Number of repeats
    #for which run of the code, make it so if a file exists with the other variables the same, it saves it with a number on end (e.g. 2 for 2nd go)
    #L, ncells, npart
    filename = 'L'+str(L)+'_ncells'+str(ncells)+' npart'+str(npart)+'run'
    path = "C:\\Users\\zks524\\compPlasma-repo\\_Lab\\"+filename+".txt"
    if os.path.isfile(filename):
        #get number on end of filename
        run_number = 2
        while isfile(filename+str(run_number)):
            run_number += 1
        filename = filename + str(run_number)
        
        
    with open(filename, "w") as f:
        for times, amplitudes in zip(s.t, s.firstharmonic):
            f.write(f"{times}\t{amplitudes}\n")
            
            
    #add latest filename to file of all filenames     
    with open('filenames.txt', 'a') as filenames:
        filenames.write('\n'+str(filename))
        
    return filename
      
        
#gets the position and velocity data (amplitude)
def generate_data():
    # Generate initial condition
    # 
    t1=time.time()
    npart = 1000   
    if False:
        # 2-stream instability
        L = 100
        ncells = 20
        pos, vel = twostream(npart, L, 3.) # Might require more npart than Landau!
    else:
        # Landau damping
        L = 4.*pi
        ncells = 20
        pos, vel = landau(npart, L)
    
    # Create some output classes
    #p = Plot(pos, vel, ncells, L) # This displays an animated figure - Slow!
    s = Summary()                 # Calculates, stores and prints summary info
    # Summary stores an array of the first-harmonic amplitude

    diagnostics_to_run = [s]   # Remove p to get much faster code! (the plotting)
    
    # Run the simulation
    pos, vel = run(pos, vel, L, ncells, 
                   out = diagnostics_to_run,        # These are called each output step
                   output_times=linspace(0.,20,50)) # The times to output
    
    filename = saveData(L, ncells, npart, s)
    
    
    t2=time.time()
    
    print('Time taken: '+str(t2-t1))
    
    
    

####################################################################

if __name__ == "__main__":

    #generate_data()
    
    with open('filenames.txt', 'r') as filenames:
        for file in (filenames.readlines() [-1:]): #remove the -1 if I want to use all files
            print(file)
            goodPeaks, goodData, goodTime = getData(file, dampingLine=True)
            getFrequency(goodPeaks, goodTime)
            

    
    
    
    
