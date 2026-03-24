import numpy as np
import pandas as pd
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt
import multiprocessing
import warnings
from datetime import datetime

# Create timestamp for output file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Define sigmoid models
def sigmoid5_un(x, Fm, Fb, Sc, Cs, As):
    """5-parameter sigmoid model for PCR amplification curves"""
    return Fm / (1. + np.exp(-(x-Cs)*Sc))**As + Fb

def sigmoid6_un(x, Fm, Fb, Sc, Cs, As, Bs):
    """6-parameter sigmoid model for PCR amplification curves"""
    return Fm / ((1. + np.exp(-(x - Cs) * Sc))**As)**Bs + Fb

# Define residual function for 6-parameter model
def res_fun_6(param, x, y_obs):
    """Calculate residual between 6-parameter model and observed data"""
    return sigmoid6_un(x, *param) - y_obs

# Define a LS loss function for 6-parameter model
def Loss_6(param, x, y_obs):
    """Loss function for 6-parameter optimization"""
    residual = res_fun_6(param, x, y_obs)
    return 0.5*np.sum(residual**2)

def fitDF_6(df_curves, start, param_bounds, sigmoid):
    """
    Fit 6-parameter sigmoid curves to PCR amplification data
    
    Args:
        df_curves: DataFrame containing PCR curves (each row is a curve)
        start: Initial parameter values (6 parameters)
        param_bounds: Parameter boundaries (lower, upper)
        sigmoid: Sigmoid function to use
        
    Returns:
        DataFrame containing fitted parameters
    """
    # Fixed x-values for all curves (1-60)
    x = np.arange(1, df_curves.shape[1] + 1, 1)
    
    # Create DataFrame to store parameters including Bs
    param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'Bs', 'mse'])

    # Fit each amplification curve
    for i in df_curves.index:
        try:
            print(f'Fitting curve #{i}')
            y = df_curves.loc[i, :].to_numpy()
            
            # Fit curve to sigmoid function
            try:
                popt, pcov = opt.curve_fit(sigmoid, x, y, p0=start, bounds=param_bounds, method='trf')
                mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
            except Exception as e:
                print(f'Cannot find optimal solutions for curve #{i}: {str(e)}')
                popt = param_bounds[1]  # Use upper bounds as fallback
                mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
                
            # Store parameters
            param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], mean_square_error]
        except Exception as e:
            print(f"Error processing curve #{i}: {str(e)}")
            continue
    
    # Preserve index
    param_df.index = df_curves.index
    
    return param_df

def visualize_fits(df_curves, param_df, n_samples=9, sigmoid_fn=sigmoid6_un):
    """
    Visualize the original curves and their fits
    
    Args:
        df_curves: DataFrame with original curves
        param_df: DataFrame with fitted parameters
        n_samples: Number of curves to visualize
        sigmoid_fn: Sigmoid function to use for visualization
    """
    # Parameter columns for 6-parameter model
    param_cols = ['Fm', 'Fb', 'Sc', 'Cs', 'As', 'Bs']
    
    # Get common indices
    common_indices = df_curves.index.intersection(param_df.index)
    
    # Sample indices for visualization
    indices_to_plot = common_indices[:min(n_samples, len(common_indices))]
    
    # Create plots
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=100)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices_to_plot):
        if i >= len(axes):
            break
            
        # Get original curve
        y_orig = df_curves.loc[idx].values
        x = np.arange(1, len(y_orig) + 1)
        
        # Get parameters
        params = param_df.loc[idx, param_cols].values
        
        # Plot original curve
        axes[i].plot(x, y_orig, 'k-', label='Original', linewidth=2)
        
        # Plot fitted curve
        y_fit = sigmoid_fn(x, *params)
        axes[i].plot(x, y_fit, 'r-', label='Fitted', linewidth=2)
        
        # Add metadata if available
        title = f"Curve #{idx}"
        axes[i].set_title(title)
        
        # Add labels
        axes[i].set_xlabel('Cycle', fontsize=10)
        axes[i].set_ylabel('Fluorescence', fontsize=10)
        
        # Add MSE to the plot
        mse = param_df.loc[idx, 'mse']
        axes[i].text(0.05, 0.95, f'MSE: {mse:.4f}', transform=axes[i].transAxes, 
                     fontsize=9, verticalalignment='top')
        
        axes[i].legend(loc='upper left', fontsize=9)
    
    # Hide unused subplots
    for i in range(len(indices_to_plot), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def fit_dataset(df_curves, start_params, param_bounds, sigmoid_fn, sample_frac=0.01):
    """
    Main function to handle fitting a dataset, with optional sampling
    
    Args:
        df_curves: DataFrame with PCR curves
        start_params: Initial parameter values
        param_bounds: Parameter boundaries
        sigmoid_fn: Sigmoid function to use
        sample_frac: Fraction to sample (0 means no sampling)
        
    Returns:
        DataFrame with fitted parameters
    """
    # Sample if needed
    if sample_frac > 0:
        df_to_fit = df_curves.sample(frac=sample_frac, random_state=9)
        print(f"Sampled {len(df_to_fit)} curves from {len(df_curves)} total curves")
    else:
        df_to_fit = df_curves
    
    # Number of CPU cores for multiprocessing
    n_jobs = min(multiprocessing.cpu_count(), len(df_to_fit))
    
    # Calculate chunk size
    chunk_size = len(df_to_fit) // n_jobs + (1 if len(df_to_fit) % n_jobs else 0)
    
    # Split dataframe into chunks using list comprehension to avoid FutureWarning
    df_chunks = [df_to_fit.iloc[i:i+chunk_size] for i in range(0, len(df_to_fit), chunk_size)]
    print(f"Split data into {len(df_chunks)} chunks for parallel processing")
    
    # Determine which fitting function to use based on parameter count
    fit_function = fitDF_6 if len(start_params) == 6 else fitDF_5
    
    # Multiprocessing execution
    results_obj = []
    with multiprocessing.Pool(processes=n_jobs) as pool:
        for df_chunk in df_chunks:
            results_obj.append(pool.apply_async(fit_function, (df_chunk, start_params, param_bounds, sigmoid_fn)))
        
        print("Starting parallel processing...")
        pool.close()
        pool.join()
    
    # Collect results
    result_dfs = []
    for obj in results_obj:
        try:
            result = obj.get()
            if result is not None and not result.empty:
                result_dfs.append(result)
        except Exception as e:
            print(f"Error retrieving results: {str(e)}")
    
    # Combine results
    if result_dfs:
        combined_df = pd.concat(result_dfs, axis=0)
        print(f"Successfully fitted {len(combined_df)} curves")
        return combined_df
    else:
        print("Error: No valid results obtained")
        return pd.DataFrame()

def fitDF_5(df_curves, start, param_bounds, sigmoid):
    """
    Legacy function for 5-parameter model fitting, included for completeness
    """
    # Fixed x-values for all curves (1-60)
    x = np.arange(1, df_curves.shape[1] + 1, 1)
    
    # Create DataFrame to store parameters
    param_df = pd.DataFrame(columns=['Fm', 'Fb', 'Sc', 'Cs', 'As', 'mse'])

    # Fit each amplification curve
    for i in df_curves.index:
        try:
            print(f'Fitting curve #{i}')
            y = df_curves.loc[i, :].to_numpy()
            
            # Fit curve to sigmoid function
            try:
                popt, pcov = opt.curve_fit(sigmoid, x, y, p0=start, bounds=param_bounds, method='trf')
                mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
            except Exception as e:
                print(f'Cannot find optimal solutions for curve #{i}: {str(e)}')
                popt = param_bounds[1]  # Use upper bounds as fallback
                mean_square_error = np.mean((y-sigmoid(x, *popt))**2)
                
            # Store parameters
            param_df.loc[i] = [popt[0], popt[1], popt[2], popt[3], popt[4], mean_square_error]
        except Exception as e:
            print(f"Error processing curve #{i}: {str(e)}")
            continue
    
    # Preserve index
    param_df.index = df_curves.index
    
    return param_df

if __name__ == '__main__':
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Suppress FutureWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
    
    # Directories
    dir_data_2025 = r'C:\Users\louis_kreitmann\DEEP_ACA_7plex_cluster\data\data_2025'
    dir_out = r'C:\Users\louis_kreitmann\DEEP_ACA_7plex_cluster\out\BYOL_CDAN'
    
    # Create output directory if it doesn't exist
    os.makedirs(dir_out, exist_ok=True)
    
    print("=== PCR Curve Fitting Script (6-Parameter Sigmoid) ===")
    print(f"Output directory: {dir_out}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(os.path.join(dir_data, 'df_dPCR_SP_2025.csv'))
    
    # Check for metadata columns
    numeric_cols = [col for col in df.columns if col.replace('.', '').isdigit()]
    metadata_cols = [col for col in df.columns if col not in numeric_cols]
    
    if metadata_cols:
        print(f"Found metadata columns: {metadata_cols}")
        # Keep only numeric columns for curve fitting
        df_curves = df[numeric_cols].copy()
        
        # Keep track of metadata
        has_metadata = True
        NMETA = len(metadata_cols)
    else:
        print("No metadata columns detected")
        df_curves = df.filter(regex=r'\d+\.?\d*')
        has_metadata = False
        NMETA = 0
    
    print(f"Loaded {len(df_curves)} curves with {df_curves.shape[1]} data points each")
    
    # PART 1: FIT SAMPLE TO ESTABLISH PARAMETERS
    print("\n=== Fitting Initial Sample of Curves with 6-Parameter Model ===")
    
    # Initial parameter guesses for 6-parameter model
    # Format: (Fm, Fb, Sc, Cs, As, Bs)
    start_6 = (1, 0, 0.5, 20, 1, 1)
    
    # Parameter bounds for 6-parameter model
    # Format: ((Fm_min, Fb_min, Sc_min, Cs_min, As_min, Bs_min), (Fm_max, Fb_max, Sc_max, Cs_max, As_max, Bs_max))
    param_bounds_6 = ((0, -1, 0, 0, 0, 0.1), (200, 1, 2, 50, 10, 10))
    
    # Fit a sample of the data
    sample_df_param_6 = fit_dataset(df_curves, start_6, param_bounds_6, sigmoid6_un, sample_frac=0.01)
    
    if not sample_df_param_6.empty:
        # Save sample parameters
        sample_output_path = os.path.join(dir_out, f'sample_df_param_6_{timestamp}.csv')
        sample_df_param_6.to_csv(sample_output_path)
        print(f"Saved sample parameters to: {sample_output_path}")
        
        # Visualize sample fits
        print("Generating visualizations...")
        fig_sample = visualize_fits(df_curves, sample_df_param_6, sigmoid_fn=sigmoid6_un)
        sample_plot_path = os.path.join(dir_out, f'sample_fits_6param_{timestamp}.png')
        fig_sample.savefig(sample_plot_path)
        plt.close(fig_sample)
        print(f"Saved sample visualization to: {sample_plot_path}")
        
        # Plot MSE histogram
        plt.figure(figsize=(10, 6))
        plt.hist(sample_df_param_6['mse'].clip(upper=0.05), bins=50)
        plt.title('Histogram of MSE for 6-Parameter Sample Fits')
        plt.xlabel('Mean Square Error')
        plt.ylabel('Count')
        mse_hist_path = os.path.join(dir_out, f'sample_mse_hist_6param_{timestamp}.png')
        plt.savefig(mse_hist_path)
        plt.close()
        print(f"Saved MSE histogram to: {mse_hist_path}")
        
        # Compare MSEs with 5-parameter vs 6-parameter
        print("\n=== Parameter Analysis ===")
        print(f"Median MSE for 6-parameter model: {sample_df_param_6['mse'].median():.6f}")
        
        # Display parameter statistics
        param_stats = sample_df_param_6[['Fm', 'Fb', 'Sc', 'Cs', 'As', 'Bs']].describe()
        print("\nParameter Statistics:")
        print(param_stats)
        
        # PART 2: FIT FULL DATASET
        print("\n=== Fitting Full Dataset with 6-Parameter Model ===")
        
        # Use median parameters from sample as starting point
        param_set = ['Fm', 'Fb', 'Sc', 'Cs', 'As', 'Bs']
        start_6_full = sample_df_param_6[param_set].median().values
        print(f"Using median parameters from sample: {start_6_full}")
        
        # Adjust bounds based on sample results
        param_bounds_6_full = ((0, -5, 0, 5, 0, 0.1), (100, 5, 4, 30, 10, 10))
        
        # Ask for confirmation before proceeding with full dataset
        proceed = input("Proceed with fitting the full dataset? (y/n): ").lower().strip()
        
        if proceed == 'y':
            # Fit the full dataset
            param_df_6 = fit_dataset(df_curves, start_6_full, param_bounds_6_full, sigmoid6_un, sample_frac=0)
            
            if not param_df_6.empty:
                # Save full dataset parameters
                full_output_path = os.path.join(dir_out, f'param_df_6_{timestamp}.csv')
                param_df_6.to_csv(full_output_path)
                print(f"Saved full dataset parameters to: {full_output_path}")
                
                # Visualize random subset of full dataset fits
                print("Generating full dataset visualizations...")
                
                # Sample a subset for visualization
                if len(param_df_6) > 9:
                    indices = np.random.choice(param_df_6.index, size=9, replace=False)
                    param_df_subset = param_df_6.loc[indices]
                else:
                    param_df_subset = param_df_6
                
                fig_full = visualize_fits(df_curves, param_df_subset, sigmoid_fn=sigmoid6_un)
                full_plot_path = os.path.join(dir_out, f'full_dataset_fits_6param_{timestamp}.png')
                fig_full.savefig(full_plot_path)
                plt.close(fig_full)
                print(f"Saved full dataset visualization to: {full_plot_path}")
                
                # Plot MSE histogram for full dataset
                plt.figure(figsize=(10, 6))
                plt.hist(param_df_6['mse'].clip(upper=.05), bins=50)
                plt.title('Histogram of MSE for 6-Parameter Full Dataset Fits')
                plt.xlabel('Mean Square Error')
                plt.ylabel('Count')
                full_mse_hist_path = os.path.join(dir_out, f'full_dataset_mse_hist_6param_{timestamp}.png')
                plt.savefig(full_mse_hist_path)
                plt.close()
                print(f"Saved full dataset MSE histogram to: {full_mse_hist_path}")
        else:
            print("Skipping full dataset fitting")
    
    print("\n=== Processing Complete ===")