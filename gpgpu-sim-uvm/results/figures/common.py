#!/usr/bin/python3
import sys
import subprocess
import os
import time
import fnmatch

# enter dir
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# List of high-level directory names (uvm-*)
uvm_dirs = [
     "uvm-2dconv",
    # "uvm-add_vectors",
     "uvm-atax",
    # "uvm-backprop",
    # "uvm-bfs",
    # "uvm-fdtd-2d",
    # "uvm-hotspot",
    # "uvm-pathfinder",
    # "uvm-ra",
    # "uvm-srad_v2",
    # "uvm-sssp",
    # "uvm-stencil",
    # "uvm-stream_triad"
]

# List of A100 configuration subdirectories
a100_configs = [
    "A100",
    "A100collect",
    "A100oversub",
    "A100oversubcollect"
]

sim_log_str = "86b30a5ada0d3c0446ed64d4b09d4a4ef08c5ee3_modified_2.0.o"

result_txt = [
    "fig-access.txt",       # Figure 2
    "fig-access_count.txt", # Figure 1,6
    "fig-cycle_pe_num.txt", # Figure 3,5
    "fig-cycle_pf_num.txt", # Figure 3,5
    "fig-cycle_valid_pages.txt", # Figure 4
    "fig-page_access_stats.txt", # Figure 7
    "fig-page_rw.txt" # Figure 9
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
light_colors = [
    '#7aaed2', # moderate blue
    '#ff9966', # moderate orange
    '#77c678', # moderate green
    '#ab82c5', # moderate purple
]
color2=['#5861ac',
        '#ff6100',
        '#f28080',
        '#70b48f',
        '#ffe0c1',
        '#e84445',

        '#cac8ef',
        '#9bdcfc',
        '#c9efbe',
        '#f0cfea',
        
        '#2c6344',
        '#61496d',
        '#c74d26',
        '#308192',
        '#5f9c61',
        '#b092b6',
        '#e38d26',
        '#5ea7b8',
        '#a4c97c',
        '#cac1d4',
        '#f1cc74',
        '#aed2e2'
        ]

commit_id = "86b30a5ada0d3c0446ed64d4b09d4a4ef08c5ee3"
root_dir = "/Users/yangyang/Desktop/gpgpu-sim/gpgpu-sim-v4_2-uvm-accelwattch/results"  # assuming the script is inside figures-unmod directory

def get_results_file(app,filename,commit_ids):
    results = []
    # Define the root directory (modify this path if necessary)
    #root_dir = "/Users/yangyang/Desktop/gpgpu-sim/gpgpu-sim-v4_2-uvm-accelwattch/results"  # assuming the script is inside figures-unmod directory
    
    # Iterate through the high-level directories

    uvm_dir = app
    dataset_dir = os.path.join(root_dir, uvm_dir)
    if os.path.isdir(dataset_dir):
        for dataset_name in os.listdir(dataset_dir):
            dataset_path = os.path.join(dataset_dir, dataset_name)
            if os.path.isdir(dataset_path):
                # Construct the path to the A100 subdirectory
                a100_dir = os.path.join(dataset_path, a100_configs[1])
                if os.path.isdir(a100_dir):                        
                    
                    result_path = os.path.join(a100_dir, filename)
                    if os.path.isfile(result_path):
                        print(f"Found {filename} in {a100_dir}")
                        abs_result_path = os.path.abspath(result_path)
                        results.append(abs_result_path)
                        #print(abs_result_path)
                    else:
                        print(f"{filename} not found in {a100_dir}")

                    for sim_log in os.listdir(a100_dir):
                        if fnmatch.fnmatch(sim_log, f"*{commit_ids}*.o*"):
                            print(f"Found file with commit ID: {sim_log}")
                            commit_file_path = os.path.abspath(os.path.join(a100_dir, sim_log))
                            results.append(commit_file_path)
                            #print(commit_file_path)
                            break
                    else:
                        print(f"No file with commit ID found in {a100_dir}")

    return results             