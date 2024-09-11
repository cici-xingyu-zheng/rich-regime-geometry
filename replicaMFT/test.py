import os
import time
import pandas as pd
import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr

def analyze_manifold(file_name, kappa=0, n_t=200):
    loaded_list = np.load(file_name, allow_pickle=True)
    capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(loaded_list, kappa, n_t)
    avg_capacity = 1 / np.mean(1 / capacity_all)
    avg_radius = np.mean(radius_all)
    avg_dimension = np.mean(dimension_all)
    return avg_capacity, avg_radius, avg_dimension

file_path = './activations/example.npy'
start_time = time.time()  # Record the start time
avg_capacity, avg_radius, avg_dimension = analyze_manifold(file_path)
print(f"Capacity:{avg_capacity}")
end_time = time.time()  # Record the end time
execution_time = end_time - start_time  # Calculate the execution time
execution_time_minutes = execution_time / 60
print(f"Execution time for {file_path}: {execution_time_minutes:.6f} minutes \n")            
