import os
import time
import pandas as pd
import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr


kappa = 0
n_t = 200

def analyze_manifold(file_name, kappa=0, n_t=200):
    loaded_list = np.load(file_name, allow_pickle=True)
    capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(loaded_list, kappa, n_t)
    avg_capacity = 1 / np.mean(1 / capacity_all)
    avg_radius = np.mean(radius_all)
    avg_dimension = np.mean(dimension_all)
    return avg_capacity, avg_radius, avg_dimension, center_correlation, K 

def analyze_folder(folder_path, kappa=0, n_t=200):
    results = []
    file_names = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            print(f"Calculating for: {file_name}...")
            file_path = os.path.join(folder_path, file_name)
            start_time = time.time()  # Record the start time
            avg_capacity, avg_radius, avg_dimension, center_correlation, K = analyze_manifold(file_path, kappa, n_t)
            results.append([avg_capacity, avg_radius, avg_dimension, center_correlation, K])
            file_names.append(os.path.splitext(file_name)[0])  # Get the file name without the extension
            print(f"Capacity:{avg_capacity}")
            end_time = time.time()  # Record the end time
            execution_time = end_time - start_time  # Calculate the execution time
            execution_time_minutes = execution_time / 60
            print(f"Execution time for {file_name}: {execution_time_minutes:.6f} minutes \n")            

    results_array = np.array(results)
    results_df = pd.DataFrame(results_array, index=file_names, columns=['avg_capacity', 'avg_radius', 'avg_dimension', 'center_correlation', 'K'])
    return results_df


grand_parent_dir = '/data/cici/Geometry/new/activations'
parent_dirs = [os.path.join(grand_parent_dir, d) for d in os.listdir(grand_parent_dir)]
dirs =  [d for d in os.listdir(grand_parent_dir)]

output_parent_dir = '../output/capacity_measures'

parent_dir = parent_dirs[-4]
dir = dirs[-4]
print('Running activation in:', parent_dir)

print('Output in:', dir)
print()

for subfolder in sorted(os.listdir(parent_dir)):
    subfolder_path = os.path.join(parent_dir, subfolder)
    # Check if the current item is a directory
    if os.path.isdir(subfolder_path):
        print("folder's", subfolder_path, '\n')
        
        # Analyze the subfolder
        results_df = analyze_folder(subfolder_path, kappa, n_t)
        
        # Save the results to a CSV file
        output_dir = os.path.join(output_parent_dir, dir)
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        results_df.to_csv(os.path.join(output_dir, f'{subfolder}_results.csv'))