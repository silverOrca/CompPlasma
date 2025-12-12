# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 18:33:04 2025

@author: ciara
"""

def get_growth_rate_simple(peak_times, peak_values):
    """
    SIMPLE and ROBUST method to get growth rate
    """
    # 1. Sort by time
    idx = np.argsort(peak_times)
    t = np.array(peak_times)[idx]
    A = np.array(peak_values)[idx]
    
    # 2. Use FIRST 6 peaks (or fewer if not enough)
    # This assumes early peaks are in exponential growth phase
    n_use = min(6, len(t))
    t_growth = t[:n_use]
    A_growth = A[:n_use]
    
    print(f"Using first {n_use} peaks for growth rate calculation")
    print(f"Time range: {t_growth[0]:.2f} to {t_growth[-1]:.2f}")
    
    # 3. Fit exponential: ln(A) = ln(A0) + γ*t
    # Add tiny number to avoid log(0)
    A_pos = A_growth + 1e-10
    logA = np.log(A_pos)
    
    # Simple linear fit without covariance matrix
    coeffs = np.polyfit(t_growth, logA, 1)
    gamma = coeffs[0]  # Growth rate!
    ln_A0 = coeffs[1]
    A0 = np.exp(ln_A0)
    
    # 4. Calculate error manually
    n = len(t_growth)
    if n > 2:
        # Predicted values
        logA_pred = ln_A0 + gamma * t_growth
        residuals = logA - logA_pred
        
        # Standard error of slope
        SS_res = np.sum(residuals**2)
        SS_x = np.sum((t_growth - np.mean(t_growth))**2)
        if SS_x > 0:
            gamma_err = np.sqrt(SS_res / ((n - 2) * SS_x))
        else:
            gamma_err = abs(gamma) * 0.1  # Rough estimate
    else:
        gamma_err = abs(gamma) * 0.2  # Larger error for few points
    
    # 5. Calculate R²
    A_pred = A0 * np.exp(gamma * t_growth)
    ss_res = np.sum((A_growth - A_pred)**2)
    ss_tot = np.sum((A_growth - np.mean(A_growth))**2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = 0
    
    print(f"\nRESULTS:")
    print(f"  Growth rate γ = {gamma:.4f} ± {gamma_err:.4f}")
    print(f"  Initial amplitude A0 = {A0:.4f}")
    print(f"  R² = {r2:.4f}")
    print(f"  Points used: {n_use}")
    
    # 6. Generate fitted curve for plotting
    t_fit = np.linspace(min(t_growth), max(t), 100)
    A_fit = A0 * np.exp(gamma * t_fit)
    
    return gamma, gamma_err, t_growth, A_growth, t_fit, A_fit


def growthRateCalc(peak_values, time_values, amplitudes):
    #first find tallest peak
    max_peak = max(peak_values)

    #mask all data past tallest peak
    mask = peak_values <= max_peak
    filtered_peaks = peak_values[mask]
    filtered_times = time_values[mask]
    filtered_amplitudes = amplitudes[mask]
    
    plt.plot(filtered_times, filtered_peaks)
    plt.show()
    
    #put the data in reverse to use damping function
    rev_peaks = filtered_peaks[::-1]
    rev_times = filtered_times[::-1]

    #use damping function to fit to the data
    x_fitting, y_fitted, dampingCo, growthErr = getDamping(rev_times, rev_peaks, filtered_amplitudes)

    growthRate = -dampingCo

    return x_fitting, y_fitted, growthRate, growthErr