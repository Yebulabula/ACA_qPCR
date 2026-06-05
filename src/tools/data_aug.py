import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import random

# Directories
dir_data_2025 = '7_plex'
dir_out = '7_plex_output'

# Load data
df = pd.read_csv(os.path.join(dir_data_2025, 'df_dPCR_GB_2025.csv'))  # Adjust filename if needed
df = df.filter(regex=r'\d+\.?\d*')  # Keep only numerical columns
curves = df.to_numpy()
curve = curves[10]  # Select the first amplification curve

# Function: Add Gaussian noise
def add_gaussian_noise(curve, mean=0, std_factor=0.005):
    std = std_factor * (np.max(curve) - np.min(curve))  # Scale noise level
    noise = np.random.normal(mean, std, curve.shape)
    return curve + noise

def time_warp(curve, sigma=.00001):
    orig_steps = np.linspace(0, 1, len(curve))
    random_warp = np.cumsum(np.random.normal(0, sigma, len(curve)))
    random_warp = (random_warp - np.min(random_warp)) / (np.max(random_warp) - np.min(random_warp))
    
    f = interp1d(orig_steps, curve, kind='linear', fill_value="extrapolate")
    return f(random_warp)

def phase_shift(curve, shift_range=(-5, 5)):
    shift = np.random.randint(*shift_range)
    return np.roll(curve, shift)

def intensity_scaling(curve, scale_range=(0.6, 1.4)):
    scale_factor = np.random.uniform(*scale_range)
    return curve * scale_factor

def smooth_warping(curve, sigma=0.001):
    x = np.arange(len(curve))
    noise = np.random.normal(0, sigma, size=len(curve))
    cs = CubicSpline(x, curve + noise)
    return cs(x)

def gp_warp_curve(curve, warp_strength=10):
    """Create warped versions of amplification curves using Gaussian processes"""
    x = np.arange(len(curve))
    
    # Define GP kernel - controls smoothness of warping
    kernel = 1.0 * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=0)
    
    # Generate random warping function
    warp = np.zeros_like(x, dtype=float)
    gp.fit(x.reshape(-1, 1), warp)
    warp_func = gp.sample_y(x.reshape(-1, 1), 1).flatten() * warp_strength * len(curve)
    
    # Create new x-coordinates with warping
    x_warped = x + warp_func
    
    # Interpolate curve at new coordinates
    from scipy.interpolate import interp1d
    f = interp1d(x, curve, bounds_error=False, fill_value=(curve[0], curve[-1]))
    
    # Generate warped curve
    warped_curve = f(x_warped)
    
    return warped_curve

def perturb_pcr_curve(curve, phase_shift_range=(-3, 3), amp_scale_range=(0.6, 1.4)):
    """
    Perturb PCR curve with realistic phase shifts and amplitude scaling
    
    Args:
        curve: Original amplification curve
        phase_shift_range: Range for random horizontal shift
        amp_scale_range: Range for random amplitude scaling
    """
    # Generate random phase shift and amplitude scaling
    phase_shift = random.uniform(*phase_shift_range)
    amp_scale = random.uniform(*amp_scale_range)
    
    # Create x-coordinates for interpolation
    x = np.arange(len(curve))
    x_shifted = x - phase_shift  # Shift curve left/right
    
    # Interpolate to get values at shifted positions
    from scipy.interpolate import interp1d
    f = interp1d(x, curve, bounds_error=False, fill_value=(curve[0], curve[-1]))
    
    # Apply phase shift and amplitude scaling
    new_curve = f(x_shifted) * amp_scale
    
    return new_curve

def pcr_model_augmentation(curve, efficiency_variation=0.05, baseline_noise=0.02):
    """
    Generate variations based on PCR kinetics model
    
    Args:
        curve: Original amplification curve
        efficiency_variation: Variation in amplification efficiency
        baseline_noise: Noise level in baseline region
    """
    # Estimate baseline and plateau
    baseline = np.mean(curve[:5])
    plateau = np.mean(curve[-5:])
    
    # Find inflection point (Ct value approximation)
    half_height = baseline + (plateau - baseline) / 2
    inflection_idx = np.argmin(np.abs(curve - half_height))
    
    # Generate new curve using 4-parameter logistic model with variations
    x = np.arange(len(curve))
    efficiency = 2.0 * (1 + random.uniform(-efficiency_variation, efficiency_variation))
    inflection_point = inflection_idx * (1 + random.uniform(-0.1, 0.1))
    slope_factor = 1.0 * (1 + random.uniform(-0.2, 0.2))
    
    # 4PL model for PCR
    new_curve = baseline + (plateau - baseline) / (1 + np.exp(-(x - inflection_point) * slope_factor))
    
    # Add realistic baseline noise
    new_curve[:inflection_idx] += np.random.normal(0, baseline_noise, size=inflection_idx)
    
    return new_curve

def bootstrap_curve_variation(curves_dataset, smooth_factor=0.7):
    """
    Create new curves by bootstrapping segments from a dataset of real curves
    
    Args:
        curves_dataset: Collection of real amplification curves
        smooth_factor: Controls smoothness of transitions between segments
    """
    # Number of segments to divide the curve into
    n_segments = 4
    
    # Create new synthetic curve by combining segments from real curves
    result_curve = np.zeros(60)  # Assuming 60 points like in your original function
    segment_length = 60 // n_segments
    
    for i in range(n_segments):
        # Randomly select a curve from dataset
        source_curve = random.choice(curves_dataset)
        
        # Extract corresponding segment
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        segment = source_curve[start_idx:end_idx]
        
        # Add segment to result
        result_curve[start_idx:end_idx] = segment
    
    # Smooth transitions between segments
    from scipy.ndimage import gaussian_filter1d
    result_curve = gaussian_filter1d(result_curve, sigma=smooth_factor)
    
    return result_curve

# Generate augmented curve
augmented_curve = pcr_model_augmentation(curve)
augmented_curve = bootstrap_curve_variation(curves)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(curve, label="Original Curve", color='blue', linewidth=2)
plt.plot(augmented_curve, label="Augmented (Gaussian Noise)", color='red', linestyle='dashed', alpha=1)
plt.xlabel("PCR Cycle")
plt.ylabel("Fluorescence Intensity")
plt.title("Gaussian Noise Augmentation in PCR Amplification Curve")
plt.legend()
plt.grid(True)
plt.show()



