import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import warnings


# Configuration constants
SHEATH_LENGTH = 40
NUM_POINTS = 100
MASS_RATIO = 1840  # Proton to electron mass ratio

# Distance from plasma into sheath
xPos = np.linspace(0, SHEATH_LENGTH, NUM_POINTS)

# Normalised ion velocity at start of sheath
v_s_hat = np.array([1.0, 1.5, 2.0])

# Initial conditions
phi_hat = 0.0
energy_hat = 0.001
f0 = np.array([phi_hat, energy_hat])

def dfdt(curX: float, curF: np.ndarray, v_s_hat: float = 1.0) -> List[float]:
    """
    Poisson equation for plasma sheath physics.
    
    Args:
        curX: Current position
        curF: Array containing [phi_hat, energy_hat]
        v_s_hat: Normalised ion velocity at sheath entrance
    
    Returns:
        Derivatives [dphidt, dedt]
    """
    phi_hat, energy_hat = curF
    
    vi_squared = v_s_hat**2 - 2 * phi_hat
    
    # Avoid division by zero or negative square root
    if vi_squared <= 0:
        return [0, 0]
    
    n_i = v_s_hat / np.sqrt(vi_squared)
    n_e = np.exp(phi_hat)
    dedt = n_i - n_e
    dphidt = -energy_hat
    
    return [dphidt, dedt]


def solve_energy(xPos: np.ndarray, v_s_hat: float, f0: np.ndarray) -> Optional[object]:
    """
    Solves for electrostatic potential energy and ion energy.
    
    Args:
        xPos: Position array
        v_s_hat: Normalised ion velocity
        f0: Initial conditions [phi_hat, energy_hat]
    
    Returns:
        ODE solution result or None if failed
    """
    try:
        result = solve_ivp(
            dfdt, 
            [xPos[0], xPos[-1]], 
            f0, 
            t_eval=xPos, 
            args=(v_s_hat,),
            rtol=1e-8,
            atol=1e-10
        )
        return result if result.success else None
    except Exception as e:
        warnings.warn(f"Integration failed for v_s_hat={v_s_hat}: {e}")
        return None


def solve_current(result: object, v_s_hat: float, mass_ratio: float = MASS_RATIO) -> np.ndarray:
    """
    Calculates normalised current from ODE solution.
    
    Args:
        result: ODE solution result
        v_s_hat: Normalised ion velocity
        mass_ratio: Ion to electron mass ratio
    
    Returns:
        Normalised current array
    """
    j = np.sqrt(mass_ratio / (2 * np.pi)) * np.exp(result.y[0, :]) - 1
    return j
def find_wall_position(xPos: np.ndarray, j_result: np.ndarray) -> Optional[float]:
    """
    Find wall position where current goes to zero.
    
    Args:
        xPos: Position array
        j_result: Current array
    
    Returns:
        Wall position or None if not found
    """
    try:
        # Check if current crosses zero
        if not (np.min(j_result) <= 0 <= np.max(j_result)):
            return None
            
        interpolated = interp1d(xPos, j_result, bounds_error=False, fill_value='extrapolate')
        res = root_scalar(interpolated, bracket=[0, SHEATH_LENGTH])
        return res.root if res.converged else None
    except Exception as e:
        warnings.warn(f"Root finding failed: {e}")
        return None


def plot_energy_results(xPos: np.ndarray, energy_results: List, v_s_hat: np.ndarray) -> None:
    """Plot energy results for all velocities."""
    plt.figure(figsize=(10, 6))
    
    for i, (result, v_s) in enumerate(zip(energy_results, v_s_hat)):
        if result is not None:
            plt.plot(xPos, result.y[0, :], 
                    label=f'Electrostatic Potential $\\hat{{\\phi}}$, $\\hat{{v}}_s = {v_s}$',
                    linewidth=2)
            plt.plot(xPos, result.y[1, :], 
                    label=f'Ion Energy $\\hat{{E}}$, $\\hat{{v}}_s = {v_s}$',
                    linestyle='dashed', linewidth=2)
    
    plt.title('Normalised Energy of Plasma Sheath vs Position', fontsize=14)
    plt.xlabel('Distance from bulk plasma into sheath, $\\hat{x}$', fontsize=12)
    plt.ylabel('Energy, $\\hat{\\phi}$ and $\\hat{E}$', fontsize=12)
    plt.legend(loc='best', prop={'size': 9})
    plt.grid(True, alpha=0.3)
    
    caption = ('Figure 1: Normalised electrostatic potential energy and ion energy '
              'variation with distance into plasma sheath for different ion velocities.')
    plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_current_results(xPos: np.ndarray, current_results: List, wall_positions: List, 
                        v_s_hat: np.ndarray) -> None:
    """Plot current results with wall positions aligned."""
    plt.figure(figsize=(10, 6))
    
    for i, (j_result, wall_pos, v_s) in enumerate(zip(current_results, wall_positions, v_s_hat)):
        if j_result is not None and wall_pos is not None:
            adjusted_x = xPos - wall_pos
            plt.plot(adjusted_x, j_result, 
                    label=f'$\\hat{{v}}_s = {v_s}$', linewidth=2)
    
    # Visual indicators for plasma and wall regions
    plt.axvspan(-SHEATH_LENGTH, 0, facecolor='red', alpha=0.1, label='Plasma')
    plt.axvspan(0, SHEATH_LENGTH, facecolor='blue', alpha=0.1, label='Wall')
    
    plt.title('Normalised Current vs Position', fontsize=14)
    plt.xlabel('Distance from bulk plasma into wall, $\\hat{x}$', fontsize=12)
    plt.ylabel('Current, $\\hat{j}$, normalised to $e n_s c_s$', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    caption = ('Figure 2: Normalised current variation with distance into plasma sheath '
              'for different ion velocities.')
    plt.figtext(0.5, 0.02, caption, wrap=True, horizontalalignment='center', fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    """Main computation and plotting routine."""
    # Compute energy solutions for all velocities
    energy_results = []
    for v_s in v_s_hat:
        result = solve_energy(xPos, v_s, f0)
        energy_results.append(result)
    
    # Plot energy results
    plot_energy_results(xPos, energy_results, v_s_hat)
    
    # Compute current and wall positions
    current_results = []
    wall_positions = []
    
    for i, (result, v_s) in enumerate(zip(energy_results, v_s_hat)):
        if result is not None:
            j_result = solve_current(result, v_s)
            wall_pos = find_wall_position(xPos, j_result)
            
            current_results.append(j_result)
            wall_positions.append(wall_pos)
            
            if wall_pos is not None:
                print(f"Wall position for v_s_hat = {v_s}: {wall_pos:.3f}")
            else:
                print(f"Warning: Could not find wall position for v_s_hat = {v_s}")
        else:
            current_results.append(None)
            wall_positions.append(None)
            print(f"Error: Integration failed for v_s_hat = {v_s}")
    
    # Plot current results
    plot_current_results(xPos, current_results, wall_positions, v_s_hat)


if __name__ == "__main__":
    main()





