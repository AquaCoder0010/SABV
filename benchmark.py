from SABV import SABV


import os
import time
import glob
import timeit
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


PE_FILES_DIR = os.path.join(os.getcwd(), "PE-files")
# Pattern to match PE files (e.g., all .exe files)
FILE_PATTERN = "*.exe" 

# The number of times to run each benchmark for better statistical accuracy
N_REPEATS = 5

# --- Benchmark Logic ---
def run_benchmark_for_files(pe_files):
    """
    Benchmarks the processing time of SABV with and without FIS for multiple files.
    """
    
    # Initialize the two SABV instances once
    try:
        # NOTE: If your SABV implementation is not in this file, you must 
        #       import it and replace the Mock SABV definition above.
        sabv_no_fis = SABV(FIS_ENABLED=False)
        sabv_with_fis = SABV(FIS_ENABLED=True, N=3, sample=0.05, FIS_THREADING_ENABLED=True)
    except NameError:
        print("ERROR: SABV class not found. Ensure it is imported or defined.")
        return None

    results = defaultdict(lambda: defaultdict(list))
    
    print(f"🔬 Starting benchmark with {len(pe_files)} files, {N_REPEATS} repeats per file...")

    for file_path in pe_files:
        file_name = os.path.basename(file_path)
        print(f"\n--- Benchmarking: {file_name} ---")

        # --- 1. Benchmarking Without FIS ---
        def task_no_fis():
            sabv_no_fis.process_file(file_path)
            
        times_no_fis = timeit.repeat(
            task_no_fis,
            repeat=N_REPEATS,
            number=1,
            globals=globals()
        )
        
        # --- 2. Benchmarking With FIS ---
        def task_with_fis():
            sabv_with_fis.process_file(file_path)
            
        times_with_fis = timeit.repeat(
            task_with_fis,
            repeat=N_REPEATS,
            number=1,
            globals=globals()
        )
        
        # Store results
        results["No FIS"]["times"].extend(times_no_fis)
        results["With FIS"]["times"].extend(times_with_fis)
        
        # Print per-file results
        print(f"  No FIS:   Min Time: {min(times_no_fis):.4f}s, Avg Time: {np.mean(times_no_fis):.4f}s")
        print(f"  With FIS: Min Time: {min(times_with_fis):.4f}s, Avg Time: {np.mean(times_with_fis):.4f}s")
        
    return results

# --- Plotting Results ---

def plot_benchmark_results(timing_results):
    """
    Generates and displays a comparison histogram of the benchmark results.
    """
    
    times_no_fis = timing_results["No FIS"]["times"]
    times_with_fis = timing_results["With FIS"]["times"]

    if not times_no_fis or not times_with_fis:
        print("No valid timing data to plot.")
        return

    # Calculate overall averages
    avg_no_fis = np.mean(times_no_fis)
    avg_with_fis = np.mean(times_with_fis)
    
    all_times = [times_no_fis, times_with_fis]
    labels = [
        f"No FIS (Avg: {avg_no_fis:.4f}s)",
        f"With FIS (Avg: {avg_with_fis:.4f}s)"
    ]

    # Histogram Setup
    fig, ax = plt.subplots(figsize=(10, 6))
    
    min_time = min(np.min(t) for t in all_times)
    max_time = max(np.max(t) for t in all_times)
    
    # Create bins across the full range of data
    bins = np.linspace(min_time * 0.9, max_time * 1.1, 25) 

    ax.hist(
        all_times,
        bins=bins,
        label=labels,
        histtype='bar',
        edgecolor='black',
        alpha=0.7 
    )

    # Add labels and title
    ax.set_title('SABV Execution Time Comparison: FIS vs. No FIS', fontsize=16)
    ax.set_xlabel('Execution Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Count of Runs)', fontsize=12)
    ax.legend(title="Processing Mode")
    ax.grid(axis='y', alpha=0.5)

    # Add vertical lines for the means
    ax.axvline(avg_no_fis, color='blue', linestyle='--', linewidth=1, label=f'Mean No FIS: {avg_no_fis:.4f}s')
    ax.axvline(avg_with_fis, color='red', linestyle='--', linewidth=1, label=f'Mean With FIS: {avg_with_fis:.4f}s')
    ax.legend()


    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Discover all PE files
    search_path = os.path.join(PE_FILES_DIR, FILE_PATTERN)
    pe_files = glob.glob(search_path)
    
    if not pe_files:
        print(f"⚠️ Warning: No files found matching '{FILE_PATTERN}' in '{PE_FILES_DIR}'.")
        print("Please ensure the directory and file pattern are correct.")
    else:
        # 2. Run the actual benchmark
        timing_results = run_benchmark_for_files(pe_files)
        
        # 3. Plot the aggregated results
        if timing_results:
            print("\n📈 Aggregating and Rendering Histogram...")
            plot_benchmark_results(timing_results)
