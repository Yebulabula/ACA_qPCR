#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.optimize as opt
import os
import multiprocessing
import matplotlib.pyplot as plt

# Directories
dir_data_2025 = r'C:\Users\louis_kreitmann\DEEP_ACA_7plex_cluster\data\data_2025'
dir_out = r'C:\Users\louis_kreitmann\DEEP_ACA_7plex_cluster\out\BYOL_CDAN'

# Ensure output directory exists
os.makedirs(dir_out, exist_ok=True)

# Define sigmoid models
def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    """5-parameter sigmoid model"""
    return Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb

def sigmoid6_un(x, Fm, Fb, Sc, Cs, As, Bs):
    """6-parameter sigmoid model"""
    return Fm / ((1. + np.exp(-(x - Cs) * Sc))**As)**Bs + Fb

# Define residual function
def res_fun(param, x, y_obs):
    return sigmoid5_un(x, *param) - y_obs

# Define a LS loss function
def Loss(param, x, y_obs):
    residual = res_fun(param, x, y_obs)
    return 0.5*np.sum(residual**2)

def fitDF_5(base_df, NMETA, start, param_bounds, sigmoid):
    """
    Fits each amplification curve to a sigmoid function using curve_fit
    
    Args:
        base_df: DataFrame containing amplification curves with metadata
        NMETA: Number of metadata columns
        start: Initial parameter values (p0)
        param_bounds: Bounds for parameters
        sigmoid: Sigmoid function to use for fitting
        
    Returns:
        DataFrame with fitted parameters and metadata
    """
    current_df = base_df.copy()
    sample_num = current_df.shape[0]
    x = np.arange(current_df.shape[1] - NMETA)  # Create x-values based on number of data points
    
    # creates a new dataframe to store generated parameters from each curve 
    param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'mse'])

    # looping through every single amplification curve 
    for i in range(sample_num):
        y = current_df.iloc[i, NMETA:].values  # Extract y values for current curve
        print(f"Fitting curve {i+1} of {sample_num} total curves")
        
        # fits each amplification curve to the sigmoid function
        try:
            popt, pcov = opt.curve_fit(sigmoid, x, y, p0=start, bounds=param_bounds, method='trf')
        except Exception as e:
            print(f'Cannot find optimal solutions for curve #{i+1}: {e}')
            popt = param_bounds[1]  # Use upper bounds as fallback
            
        mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
        param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mean_square_error]

    # Update indexes
    param_df.index = base_df.index
    
    # Combine metadata with parameters
    if NMETA > 0:
        param_df_with_meta = pd.concat([base_df.iloc[:,:NMETA], param_df], axis=1)
    else:
        param_df_with_meta = param_df
        
    return param_df_with_meta

def fitDF_gb(base_df, NMETA, start, param_bounds):
    """
    Fits curves using global optimization (dual annealing)
    
    Args:
        base_df: DataFrame containing amplification curves with metadata
        NMETA: Number of metadata columns
        start: Initial parameter values (p0)
        param_bounds: Bounds for parameters
        
    Returns:
        DataFrame with fitted parameters and metadata
    """
    current_df = base_df.copy()
    sample_num = current_df.shape[0]
    
    # Get x values (column names or indices)
    if isinstance(current_df.columns[NMETA:], pd.RangeIndex):
        # If columns are integers
        x = np.array(current_df.columns[NMETA:])
    else:
        # If columns are strings, try to convert to float
        x = pd.to_numeric(current_df.columns[NMETA:], errors='coerce')
    
    # creates a new dataframe to store generated parameters from each curve 
    param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'mse'])

    # looping through every single amplification curve 
    for i in range(sample_num):
        y = current_df.iloc[i, NMETA:].values  # Extract y values
        print(f"Globally optimizing curve {i+1} of {sample_num}")
        
        # Fits each amplification curve using dual annealing
        try:
            result_obj = opt.dual_annealing(Loss, bounds=np.array(param_bounds).T, args=(x, y))
            popt = result_obj.x
            mean_square_error = np.mean((y-sigmoid5_un(x, *popt))**2)
        except Exception as e:
            print(f'Error in global optimization for curve #{i+1}: {e}')
            popt = param_bounds[1]  # Use upper bounds as fallback
            mean_square_error = np.nan
            
        param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mean_square_error]

    # Update indexes
    param_df.index = base_df.index
    
    # Combine metadata with parameters
    if NMETA > 0:
        param_df_with_meta = pd.concat([base_df.iloc[:,:NMETA], param_df], axis=1)
    else:
        param_df_with_meta = param_df
        
    return param_df_with_meta

def visualize_fits(original_df, param_df, NMETA, num_samples=9):
    """Visualize the original curves and their fits"""
    param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
    
    # Combine original data with parameters
    if original_df.shape[0] > param_df.shape[0]:
        # If original_df has more rows, filter to match param_df
        df_tmp = pd.concat([original_df.loc[param_df.index], param_df[param_set]], axis=1)
    else:
        df_tmp = pd.concat([original_df, param_df[param_set]], axis=1)
    
    # Limit to requested number of samples
    df_tmp = df_tmp.iloc[:num_samples]
    
    # Create subplot grid
    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), dpi=100)
    ax = ax.flatten()
    
    # Plot each curve
    for i in range(len(df_tmp)):
        # Plot original data
        x_data = np.arange(df_tmp.shape[1] - NMETA - len(param_set))
        y_data = df_tmp.iloc[i, NMETA:NMETA+len(x_data)].values
        ax[i].plot(x_data, y_data, 'k-', label='Original')
        
        # Plot fit
        Fm = df_tmp.iloc[i]['Fm']
        Fb = df_tmp.iloc[i]['Fb']
        Sc = df_tmp.iloc[i]['Sc']
        Cs = df_tmp.iloc[i]['Cs']
        As = df_tmp.iloc[i]['As']
        y_fit = sigmoid5_un(x_data, Fm, Fb, Sc, Cs, As)
        ax[i].plot(x_data, y_fit, 'r-', label='Fit')
        
        # Add labels and title
        ax[i].set_xlabel('Cycle')
        ax[i].set_ylabel('Fluorescence')
        if 'Target' in df_tmp.columns:
            ax[i].set_title(f"Target: {df_tmp.iloc[i]['Target']}")
        else:
            ax[i].set_title(f"Curve #{i+1}")
            
        # Add legend
        ax[i].legend(loc='upper left', fontsize=8)
    
    # Hide empty subplots
    for i in range(len(df_tmp), len(ax)):
        ax[i].axis('off')
        
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Load data
    df = pd.read_csv(os.path.join(dir_data_2025, 'df_dPCR_GB_2025.csv'))
    
    # Check for metadata columns - adjust NMETA based on your data
    metadata_cols = [col for col in df.columns if not col.replace('.', '').isdigit()]
    NMETA = len(metadata_cols)
    print(f"Detected {NMETA} metadata columns: {metadata_cols}")
    
    # Keep only necessary columns
    if NMETA > 0:
        numeric_cols = [col for col in df.columns if col not in metadata_cols]
        data_df = df[metadata_cols + numeric_cols]
    else:
        data_df = df.filter(regex=r'\d+\.?\d*')  # Keep only numerical columns
    
    # FITTING - INITIAL SUBSET - 5 PARAMETERS
    print("\n=== Fitting initial subset with 5-parameter sigmoid ===")
    # Take a small sample for initial fitting
    sample_df = data_df.sample(frac=0.01, random_state=9)
    
    # Split dataframe for parallel computing
    n_jobs = multiprocessing.cpu_count()
    split_df = np.array_split(sample_df, n_jobs)
    
    # Define initial parameters and bounds
    start_5 = (1, 0, 0.5, 20, 1)  # p0 for Fm, Fb, Sc, Cs, As
    param_bounds_5 = ((0, -1, 0, 0, 0), (200, 1, 2, 50, 10))
    
    # Fit the sample in parallel
    results_obj = []
    pool = multiprocessing.Pool()
    for df_chunk in split_df:
        results_obj.append(pool.apply_async(fitDF_5, 
                                           args=(df_chunk, NMETA, start_5, param_bounds_5, sigmoid5_un)))
    print("=== Starting parallel fitting of sample ===")
    pool.close()
    pool.join()
    print("=== Parallel fitting of sample complete ===")
    
    # Collect results
    sample_list = []
    for obj in results_obj:
        sample_list.append(obj.get())
    sample_df_param_5 = pd.concat(sample_list)
    
    # Save sample results
    sample_df_param_5.to_csv(os.path.join(dir_out, 'sample_df_param_5.csv'))
    print(f"Sample parameters saved to {os.path.join(dir_out, 'sample_df_param_5.csv')}")
    
    # Visualize sample fits
    fig_sample = visualize_fits(sample_df, sample_df_param_5, NMETA)
    plt.savefig(os.path.join(dir_out, 'sample_fits.png'))
    plt.close(fig_sample)
    
    # Plot MSE histogram for sample
    plt.figure(figsize=(8, 6))
    plt.hist(sample_df_param_5['mse'], bins=50)
    plt.title('Histogram of MSE for Sample Fits')
    plt.xlabel('Mean Square Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(dir_out, 'sample_mse_hist.png'))
    plt.close()
    
    # FITTING - FULL DATASET - 5 PARAMETERS
    print("\n=== Fitting full dataset with 5-parameter sigmoid ===")
    # Use median parameters from sample as starting point
    param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As']
    start_5 = np.array(sample_df_param_5[param_set].median())
    
    # Adjust bounds based on sample results
    param_bounds_5 = ((0, -5, 0, 5, 0), (100, 5, 4, 30, 10))
    
    # Split full dataset for parallel processing
    split_df = np.array_split(data_df, n_jobs)
    
    # Fit the full dataset in parallel
    results_obj = []
    pool = multiprocessing.Pool()
    for df_chunk in split_df:
        results_obj.append(pool.apply_async(fitDF_5, 
                                           args=(df_chunk, NMETA, start_5, param_bounds_5, sigmoid5_un)))
    print("=== Starting parallel fitting of full dataset ===")
    pool.close()
    pool.join()
    print("=== Parallel fitting of full dataset complete ===")
    
    # Collect results
    fitted_list = []
    for obj in results_obj:
        fitted_list.append(obj.get())
    param_df_5 = pd.concat(fitted_list)
    
    # Save full dataset results
    param_df_5.to_csv(os.path.join(dir_out, 'param_df_5.csv'))
    print(f"Full dataset parameters saved to {os.path.join(dir_out, 'param_df_5.csv')}")
    
    # Visualize full dataset fits (sample of 9)
    random_indices = np.random.choice(param_df_5.index, size=min(9, len(param_df_5)), replace=False)
    fig_full = visualize_fits(data_df.loc[random_indices], param_df_5.loc[random_indices], NMETA)
    plt.savefig(os.path.join(dir_out, 'full_dataset_fits.png'))
    plt.close(fig_full)
    
    # Plot MSE histogram for full dataset
    plt.figure(figsize=(8, 6))
    plt.hist(param_df_5['mse'].clip(upper=5), bins=50)  # Clip to range for better visualization
    plt.title('Histogram of MSE for Full Dataset Fits')
    plt.xlabel('Mean Square Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(dir_out, 'full_dataset_mse_hist.png'))
    
    print("\n=== All processing complete! ===")