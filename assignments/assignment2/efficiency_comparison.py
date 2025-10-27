"""
Efficiency comparison of different iteration methods
"""
import numpy as np
import time
from typing import List

# Mock data for testing
n = 1000
current_results = [np.random.rand(100) for _ in range(n)]
wall_positions = [np.random.rand() for _ in range(n)]
v_s_hat = np.random.rand(n)

def method_1_zip_enumerate():
    """Current method: zip with enumerate"""
    results = []
    for i, (j_result, wall_pos, v_s) in enumerate(zip(current_results, wall_positions, v_s_hat)):
        if j_result is not None and wall_pos is not None:
            results.append((i, j_result, wall_pos, v_s))
    return results

def method_2_direct_indexing():
    """Alternative: Direct indexing"""
    results = []
    for i in range(len(current_results)):
        j_result = current_results[i]
        wall_pos = wall_positions[i] 
        v_s = v_s_hat[i]
        if j_result is not None and wall_pos is not None:
            results.append((i, j_result, wall_pos, v_s))
    return results

def method_3_zip_no_enumerate():
    """Alternative: zip without enumerate (if index not needed)"""
    results = []
    for j_result, wall_pos, v_s in zip(current_results, wall_positions, v_s_hat):
        if j_result is not None and wall_pos is not None:
            results.append((j_result, wall_pos, v_s))
    return results

def method_4_numpy_vectorized():
    """Vectorized approach using numpy (where applicable)"""
    # This works best when operations can be vectorized
    current_array = np.array(current_results, dtype=object)
    wall_array = np.array(wall_positions)
    v_s_array = np.array(v_s_hat)
    
    # Create mask for valid data
    valid_mask = np.array([cr is not None for cr in current_results]) & \
                 np.array([wp is not None for wp in wall_positions])
    
    valid_indices = np.where(valid_mask)[0]
    return [(i, current_array[i], wall_array[i], v_s_array[i]) for i in valid_indices]

def benchmark_methods():
    """Benchmark different methods"""
    methods = [
        ("zip + enumerate", method_1_zip_enumerate),
        ("direct indexing", method_2_direct_indexing), 
        ("zip only", method_3_zip_no_enumerate),
        ("numpy vectorized", method_4_numpy_vectorized)
    ]
    
    for name, method in methods:
        start_time = time.perf_counter()
        for _ in range(100):  # Run multiple times for better measurement
            result = method()
        end_time = time.perf_counter()
        
        print(f"{name:20}: {(end_time - start_time)*1000:.3f} ms")

if __name__ == "__main__":
    benchmark_methods()