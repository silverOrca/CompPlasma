#
# Illustrates how to use FFTs to solve a differential equation
#
# Shows a single step in which we calculate the solution, g, to
# dg/dx = y = 2 * cos(2 x)
#
from numpy import arange, concatenate

try:
    from scipy.fftpack import fft, ifft
except:
    # No SciPy FFT routine. Import NumPy routine instead
    from numpy.fft import fft, ifft

def fft_integrate(y, dx):
    n = len(y) # Get the length of y
    
    f = fft(y) # Take FFT
    # Result is in standard layout with positive frequencies first then negative
    # n even: [ f(0), f(1), ... f(n/2), f(1-n/2) ... f(-1) ]
    # n odd:  [ f(0), f(1), ... f((n-1)/2), f(-(n-1)/2) ... f(-1) ]
    # Note we could use fftfreq for this.
    if n % 2 == 0: # If an even number of points
        k = concatenate( (arange(0, n/2+1), arange(1-n/2, 0)) )
    else:
        k = concatenate( (arange(0, (n-1)/2+1), arange( -(n-1)/2, 0)) )

    # Rescale the wavenumbers here. Note n * dx = L (where L is the length of the domain) 
    k = 2.*pi*k / (n * dx)
    
    # Modify frequencies by dividing by ik
    #1j means i in python
    f[1:] /= (1j * k[1:]) 
    f[0] = 0. # Set the arbitrary zero-frequency term to zero
    
    return ifft(f) # Reverse Fourier Transform

if __name__ == "__main__":
    from numpy import linspace, pi, sin, cos
    import matplotlib.pyplot as plt

    # Plot the exact solution
    x = linspace(0, 4*pi, 100, endpoint=False)
    exact = sin(2.*x)
    plt.plot(x, exact, label="Exact solution")
    
    # Calculate an approximate solution
    
    for nx in [5,8,20,32]:
        x = linspace(0, 4*pi, nx, endpoint=False)
        dx = x[1] - x[0]
        
        y = 2.*cos(2.*x) # Function to integrate
        
        inty = fft_integrate(y, dx).real # Only keep real part
        
        plt.plot(x, inty, marker='o', label="FFT n = %d" % (nx))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Integration of periodic function using FFT")
    plt.legend()
    plt.show()


#less points results in aliasing