# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:23:55 2024

@author: Nico Reinders
"""

from matplotlib import pyplot as plt
import numpy as np
import Data_analysis_and_transforms as lib
import time
from scipy.signal import find_peaks
from numba import njit


def get_threshholds(trace, debugging=False): #get the threshholds of a given trace  
    
    start_params = [0.35, 0.015, 0.001, 0.43, 0.05, 0.001, 0] # mus will be reset automatically later 
    a1, mu1, s1, a2, mu2, s2, o = start_params
    
    
    bounds_double_gaussian = ([0, 0.01, 1e-5, 0, 0.0477, 1e-5, 0],[1, 0.032, 0.04, 1, 0.0659, 0.04, 100]) # bounds for a, mu and sigma will be reset automatically later
    lba1, lbmu1, lbs1, lba2, lbmu2, lbs2, lbo = bounds_double_gaussian[0]
    
    uba1, ubmu1, ubs1, uba2, ubmu2, ubs2, ubo = bounds_double_gaussian[1]


    m = 3/7 #parameters that work for maximum of snr of ~ 9
    b = -6/7    
    n_bins = 150
    
    #make a histogramm of the trace value distribution to determine high and low signal
    hist, bins = np.histogram(trace, bins=n_bins, density=False)
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    hist = hist/np.max(hist) #normalize histogramm 
    hist_smoothed = lib.moving_average(hist, 5) # smooth histogram data, might have to be adjusted depending on nb of bins
    peaks, heights = find_peaks(hist_smoothed, prominence=0.003)
    #print(peaks)
    
    #determine wether approximations of fitting parameters can be set
    if len(peaks) <= 1: 
        return np.nan, np.nan, np.nan #if there is only one peak in the distribution the signal is too noisy
    elif len(peaks) == 2: #if two peaks were identified, use them to set initial fitting parameters 
        mu1, mu2 = bin_centers[peaks]
        lbmu1 = mu1 - (mu2-mu1)/5
        ubmu1 = mu1 + (mu2-mu1)/5
        
        lbmu2 = mu2 - (mu2-mu1)/5
        ubmu2= mu2 + (mu2-mu1)/5
        a1, a2 = hist[peaks]
       
    else: 
        mu1 = bin_centers[peaks[0]]
        mu2 = bin_centers[peaks[-1]]
        
        lbmu1 = mu1 - (mu2-mu1)/5
        ubmu1 = mu1 + (mu2-mu1)/5
        
        lbmu2 = mu2 - (mu2-mu1)/5
        ubmu2= mu2 + (mu2-mu1)/5
        
        
    start_params = [a1, mu1, s1, a2, mu2, s2, o]
    bounds_double_gaussian = [lba1, lbmu1, lbs1, lba2, lbmu2, lbs2, lbo], [uba1, ubmu1, ubs1, uba2, ubmu2, ubs2, ubo]
    
    params, cov = lib.fit_double_gaussian(bin_centers, hist_smoothed, start_params, bounds_double_gaussian) 
    if params[0] == 0 or params[-2] == 0: #if one peak is completely flat, discard measurement 
        params[:] = np.nan
    snr = lib.snr_calculation(params)
    a = lib.det_a(snr, m, b) 
    
    #determine thresholds
    thresh_lower = params[1]+a*params[2] 
    thresh_upper = params[4]-a*params[5]
    
    #in debugging mode display the histogram with fits and print the fit parameters 
    if debugging: 
        print(params)
        fig, ax = plt.subplots(1, 1)
        if params.all() != 0 and len(peaks) >= 1:   
            ax.scatter(bin_centers,hist_smoothed/1e4, s=0.5)
            ax.plot(bin_centers, lib.double_gaussian(bin_centers, *params)/1e4)
            ax.plot(bin_centers, lib.gaussian(bin_centers, *params[0:3])/1e4)
            ax.plot(bin_centers, lib.gaussian(bin_centers, *params[3:6])/1e4)
            ax.axvline(thresh_lower, color='g')
            ax.axvline(thresh_upper, color='r')
        else: ax.scatter(bin_centers,hist_smoothed/1e4, s=0.5)
        
        for peak_index in peaks:
             ax.axvline(x=bin_centers[peak_index])
    
    return snr, thresh_lower, thresh_upper


def get_t_rates(traces, time_axis):
    shape = np.shape(traces)[0:2]
    snr_array = np.zeros(shape)
    gamma_up_array = np.zeros(shape) 
    gamma_down_array = np.zeros(shape)
    
    #fill the arrays with the respective values
    for i, row in enumerate(traces):
        for j, trace in enumerate(row):
            
            if get_threshholds(trace)[0] >= 1: #filter out points with snr < 1 because the tunnel rates cannot be determined reliably at this point
            
                snr_array[i][j], thresh_lower, thresh_upper = get_threshholds(trace)
                
                            
                x_result, diff_result, up_list, down_list, up_time, down_time = lib.detect_events_vec(time_axis, trace, thresh_upper, thresh_lower)
                
                gamma_up = lib.gamma(up_time)[0]
                gamma_down = lib.gamma(down_time)[0]
                
                gamma_up_array[i][j] = gamma_up
                gamma_down_array[i][j] = gamma_down
            
            else:
                gamma_up_array[i][j] = np.nan
                gamma_down_array[i][j] = np.nan
    return snr_array, gamma_up_array, gamma_down_array

# calculate the fourier spectrums for manual denoising later 
# def get_fourier(trace, time_axis):
     
#     fft_signal = np.fft.fft(trace) 
#     fft_signal_shifted = np.fft.fftshift(fft_signal)
#     fft_signal = fft_signal_shifted       
#     original_angles = np.angles(fft_signal_shifted)
            
    
#     frequencies = np.abs(np.fft.fftfreq(len(trace), d=(time_axis[1] - time_axis[0])))
#     frequencies_shifted = np.abs(np.fft.fftshift(frequencies))
#     fft_signal = fft_signal/np.max(fft_signal)
#     return frequencies_shifted, fft_signal, original_angles


# calculate the fourier spectrums for manual denoising later 
def get_fourier(traces, time_axis):
    fft_mean = np.empty(len(traces[0][0]))
    original_angles = np.empty(np.shape(traces))
    fft_signals = np.empty(np.shape(traces), dtype=np.complex128)
    
    for i, row in enumerate(traces):
        for j, trace in enumerate(row):
           fft_signal = np.fft.fft(trace) 
           fft_signal_shifted = np.fft.fftshift(fft_signal)
           fft_signals[i][j] = fft_signal_shifted
          
           original_angles[i][j] = np.angle(fft_signal_shifted)
            
           fft_mean += np.abs(fft_signal_shifted)
            
    
    frequencies = np.abs(np.fft.fftfreq(len(trace), d=(time_axis[1] - time_axis[0])))
    frequencies_shifted = np.abs(np.fft.fftshift(frequencies))
    fft_mean = fft_mean/np.max(fft_mean)
    return frequencies_shifted, fft_mean, fft_signals, original_angles

#%% 
#FFT correction 


def fft_correction_select(frequencies_shifted, fft_mean, fig, axs):
    backup_fft_mean = np.copy(fft_mean)
    @njit()
    def correct_fft_noise(freq, sig, f_low, f_high):
        # Set FFT signal to 0 between f_low and f_high
        sig[(np.abs(freq) > f_low) & (np.abs(freq) < f_high)] = 0
        return sig

    def plot_fft(ax):
        # Initialize the Fourier plot
        ax.set_ylim(0, 0.0125)
        ax.set_xlim(-10, 1000)
        return ax.plot(frequencies_shifted, fft_mean)[0]

    def update_fft_plot(fft_corrected):
        mean_fft_plot.set_ydata(fft_corrected)
        fig.canvas.draw()

    def on_press(event):
        nonlocal press
        if event.button == 1:
            if event.inaxes != axs: return
            contains, _ = vline.contains(event)
            if not contains: return
            x0 = vline.get_xdata()[0]
            press = (x0, event.xdata)
        elif event.button == 3:
            vline.set_xdata(event.xdata)

    def on_motion(event):
        nonlocal press
        if press is None: return
        if event.inaxes != axs: return
        x0, xpress = press
        dx = event.xdata - xpress
        vline.set_xdata([x0 + dx])
        fig.canvas.draw()

    def on_release(event):
        nonlocal press
        press = None
        fig.canvas.draw()

    def on_key_press(event):
        nonlocal fft_mean
        if event.key == 'x':
            x = np.ravel(vline.get_xdata())[0]
            print('x pressed at', x)
            xs.append(axs.axvline(x, color='g', linestyle='--', linewidth=1))
            fft_mean_corr = correct_fft_noise(frequencies_shifted, fft_mean, x-0.1, x+0.1)
            update_fft_plot(fft_mean_corr)
            fft_mean = fft_mean_corr

        elif event.key == 'r':
            print('reset cuts')
            fft_mean_corr = backup_fft_mean.copy()
            update_fft_plot(fft_mean_corr)
            for line in xs:
                line.remove()
            xs.clear()

    #fig, axs = plt.subplots(1, 1, figsize=(10, 8))
    mean_fft_plot = plot_fft(axs)
    vline = axs.axvline(x=[50], color='r', linestyle='--', linewidth=2)
    
    xs = []
    press = None

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    #plt.show(block=True)


def get_cuts(ax):    
    xs = ax.get_lines()[2::]
    cuts = np.array([np.ravel(line.get_xdata())[0] for line in xs])
    return cuts


def fft_correction_apply(traces, cuts, frequencies_shifted, fft_signals, original_angles):
   
    @njit()
    def correct_fft_noise(freq, sig, f_low, f_high):
        # Set FFT signal to 0 between f_low and f_high
        sig[(np.abs(freq) > f_low) & (np.abs(freq) < f_high)] = 0
        return sig

    
    @njit()
    def apply_changes(cuts, fft_signal_shifted_corr, original_angle):
        for x in cuts:
            fft_signal_shifted_corr = correct_fft_noise(frequencies_shifted, fft_signal_shifted_corr, x-0.1, x+0.1)
        modified_magnitude = np.abs(fft_signal_shifted_corr)
        new_fft_data = modified_magnitude * np.exp(1j * original_angle)
        return new_fft_data

    tick = time.perf_counter()
    print('Applying changes...')

    if len(cuts) != 0:
        for i, row in enumerate(traces):
            for j, trace in enumerate(row):
                new_fft_data_unshifted = np.fft.ifftshift(apply_changes(cuts, np.abs(fft_signals[i][j]), original_angles[i][j]))
                traces[i][j] = np.fft.ifft(new_fft_data_unshifted).real

    tock = time.perf_counter()
    print(f'Done \nCalculation time: {tock - tick} s')
    return traces



