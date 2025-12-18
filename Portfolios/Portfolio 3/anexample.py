# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 20:49:38 2025

@author: ciara
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft

def solvePoisson(x, ne):
    """
    Solve: dE/dx = ρ = 1 - n_e
    Returns E(x) with zero mean
    """
    dx = x[1] - x[0]
    nx = len(x)
    
    # Get wavenumbers (angular frequencies)
    k = 2 * np.pi * fftfreq(nx, dx)
    
    # Charge density: ρ = 1 - n_e
    rho = 1.0 - ne
    
    # Fourier transform
    rho_hat = fft(rho)
    
    # Solve in Fourier space: E_hat = ρ_hat / (ik)
    E_hat = np.zeros_like(rho_hat, dtype=complex)
    
    # Handle division by zero
    mask = k != 0
    E_hat[mask] = rho_hat[mask] / (1j * k[mask])
    # k=0 component remains 0 (zero mean E)
    
    # Inverse transform
    E = ifft(E_hat).real
    
    # Ensure zero mean
    E = E - np.mean(E)
    
    return E

def fourier_derivative(field, xx):
    """CORRECT: Calculate d(field)/dx using Fourier method"""
    dx = xx[1] - xx[0]
    nx = len(xx)
    k = 2 * np.pi * fftfreq(nx, dx)  # Proper wavenumbers
    
    field_hat = fft(field)
    derivative_hat = 1j * k * field_hat
    derivative = ifft(derivative_hat).real
    
    return derivative

# Test with the given parameters
x = np.linspace(0, 1, 101, endpoint=False)
ne = 1 + np.sin(2*np.pi*x)

E = solvePoisson(x, ne)
dEdx = fourier_derivative(E, x)

plt.plot(x, E)
plt.plot(x, dEdx)
plt.show()

# Calculate and display error
error = np.max(np.abs((1 - dEdx) - ne))
print("="*60)
print(f"Maximum error |(1 - dE/dx) - n_e| = {error:.2e}")
print(f"Expected E amplitude: {1/(2*np.pi):.3f}")
print(f"Actual E amplitude: {np.max(np.abs(E)):.3f}")
print(f"E mean: {np.mean(E):.2e} (should be ~0)")
print("="*60)

if error < 1e-10:
    print("✓ SUCCESS: Poisson solver is correct!")
else:
    print("✗ ERROR: Something is wrong")

# Plot
plt.figure(figsize=(12, 8))

# Plot 1: n_e and E
plt.subplot(2, 2, 1)
plt.plot(x, ne, 'b-', linewidth=2, label=r'$\hat{n}_e = 1 + \sin(2\pi x)$')
plt.plot(x, E, 'r-', linewidth=2, label=r'$\hat{E}_x$')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Value')
plt.title('Electron density and Electric field')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Plot 2: ρ and dE/dx
plt.subplot(2, 2, 2)
plt.plot(x, 1-ne, 'm-', linewidth=2, label=r'$\rho = 1 - \hat{n}_e$')
plt.plot(x, dEdx, 'g-', linewidth=2, label=r'$d\hat{E}_x/d\hat{x}$')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Value')
plt.title('Charge density and E derivative')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Plot 3: Verification
plt.subplot(2, 2, 3)
plt.plot(x, 1-dEdx, 'k-', linewidth=3, label=r'$1 - d\hat{E}_x/d\hat{x}$')
plt.plot(x, ne, 'b--', linewidth=1, alpha=0.7, label=r'$\hat{n}_e$ (original)')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Value')
plt.title(f'Verification (error = {error:.2e})')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Plot 4: Error distribution
plt.subplot(2, 2, 4)
error_dist = (1 - dEdx) - ne
plt.plot(x, error_dist, 'r-', linewidth=2)
plt.fill_between(x, 0, error_dist, alpha=0.3, color='red')
plt.xlabel(r'$\hat{x}$')
plt.ylabel('Error')
plt.title('Error distribution')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)

plt.tight_layout()
plt.show()

# Test the derivative function independently
print("\n" + "="*60)
print("Testing fourier_derivative function:")
test_x = np.linspace(0, 2*np.pi, 100, endpoint=False)
test_f = np.sin(test_x)
test_dfdx = fourier_derivative(test_f, test_x)
expected = np.cos(test_x)
deriv_error = np.max(np.abs(test_dfdx - expected))
print(f"Derivative test error: {deriv_error:.2e}")
print("(For sin(x), derivative should be cos(x))")
print("="*60)