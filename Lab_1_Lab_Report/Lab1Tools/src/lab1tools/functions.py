import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy.signal import correlate
from scipy.stats import norm

def freq_from_filename(fname):
    match = re.search(r"([\d\.]+)Hz", fname)
    if match is None:
        raise ValueError(f"Could not extract frequency from {fname}")
    return float(match.group(1))  # Hz

def compute_period(y, fs):
    y = y - np.mean(y)  # remove DC
    # Find zero crossings (sign change)
    zero_crossings = np.where(np.diff(np.sign(y)) > 0)[0]  # positive-going
    if len(zero_crossings) < 2:
        return None  # cannot compute period
    # Compute periods in samples
    periods_samples = np.diff(zero_crossings)
    # Convert to time in seconds
    periods_sec = periods_samples / fs
    return np.mean(periods_sec)

def compute_lags(n, fs):
    return np.arange(-n+1, n)/fs

def compute_fwhm(x, y):
    """
    Compute FWHM of a peak in y(x) using linear interpolation.
    Returns NaN if no half-max points are found.
    """
    y = np.array(y)
    x = np.array(x)
    
    y_max = np.max(y)
    half_max = y_max / 2

    # Indices where y crosses half-max
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return np.nan  # not enough points to define FWHM

    left_idx = indices[0]
    right_idx = indices[-1]

    # Linear interpolation for left half-max
    if left_idx == 0:
        x_left = x[0]
    else:
        x1, x2 = x[left_idx-1], x[left_idx]
        y1, y2 = y[left_idx-1], y[left_idx]
        x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    # Linear interpolation for right half-max
    if right_idx == len(y)-1:
        x_right = x[-1]
    else:
        x1, x2 = x[right_idx], x[right_idx+1]
        y1, y2 = y[right_idx], y[right_idx+1]
        x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)

    return x_right - x_left

def gaussian(x, A, sigma):
    return A * np.exp(-0.5*(x/sigma)**2)

