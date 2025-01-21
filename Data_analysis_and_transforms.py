import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.fft import fft2, fftfreq, fftshift


def interpolate_2d_results(imag, x, y, grid):

    '''
    Interpolates a 2D dataset (imag) onto a specified grid.

    :param imag: 2D numpy array representing the image or data to interpolate.
    :param x: 2D numpy array of x-coordinates for the original data.
    :param y: 2D numpy array of y-coordinates for the original data.
    :param grid: Tuple of numpy arrays (xg, yg) representing the grid points to interpolate onto.
    :return: 2D numpy array of interpolated image data.
    '''

    mask_array = np.ma.masked_invalid(imag)
    imag = imag[~mask_array.mask].flatten()
    cor = np.array([x.flatten(), y.flatten()]).T
    result = np.array(interpolate.griddata(cor, imag, grid, method='linear', fill_value=0))
    return result


def image_down_sampling(imag, x, y, res_fac=(0.5, 0.5)):

    '''
    downsamples a 2D image (imag) by a specified resolution factor.

    :param imag: 2D numpy array representing the image to downsample.
    :param x: 2D numpy array of x-coordinates for the original image.
    :param y: 2D numpy array of y-coordinates for the original image.
    :param res_fac: Tuple of floats indicating the downscaling factor for the x and y dimensions, respectively.
    :return: Tuple containing the downsampled x-grid, y-grid, and image data.
    '''
    x_num = int(np.shape(imag)[1])
    fac_x = float(res_fac[0])
    y_num = int(np.shape(imag)[0])
    fac_y = float(res_fac[1])
    x_inter = np.linspace(np.min(x), np.max(x), int(x_num * fac_x))
    y_inter = np.linspace(np.min(y), np.max(y), int(y_num * fac_y))
    xg, yg = np.meshgrid(x_inter, y_inter)
    result = interpolate_2d_results(imag, x, y, (xg, yg))
    return xg, yg, result


def two_d_fft_on_data(imag, x, y):

    '''
    Applies a 2D Fast Fourier Transform (FFT) on a given image or 2D data.

    :param imag: 2D numpy array representing the image or data to transform.
    :param x: 2D numpy array of x-coordinates for the original data.
    :param y: 2D numpy array of y-coordinates for the original data.
    :return: Tuple containing the FFT frequency grids for x and y, and the amplitude spectrum of the FFT.
    '''
    x_num = int(np.shape(imag)[1])
    y_num = int(np.shape(imag)[0])
    x_inter, dx = np.linspace(np.min(x), np.max(x), int(x_num), retstep=True)
    y_inter, dy = np.linspace(np.min(y), np.max(y), int(y_num), retstep=True)
    xg, yg = np.meshgrid(x_inter, y_inter)
    data_on_regular_grid = interpolate_2d_results(imag, x, y, (xg, yg))
    fft_amp = fftshift(np.abs(fft2(data_on_regular_grid - np.mean(data_on_regular_grid), norm='ortho')))
    ixg, iyg = np.meshgrid(fftshift(fftfreq(int(x_num), dx)), fftshift(fftfreq(int(y_num), dy)))
    return ixg, iyg, fft_amp


def evaluate_poly_background_2d(x, y, z, order_x, order_y):

    '''
    Evaluates and subtracts a polynomial background from 2D data.

    This function fits a polynomial of specified orders in x and y to the 2D data
    and then subtracts this polynomial background.

    :param x: 2D array of x coordinates.
    :param y: 2D array of y coordinates.
    :param z: 2D array of z values at each (x, y) point.
    :param order_x: Integer specifying the order of the polynomial in the x-direction.
    :param order_y: Integer specifying the order of the polynomial in the y-direction.
    :return: 2D numpy array of the background-subtracted data.
    '''

    assert x.shape == y.shape == z.shape, "x, y, and z must have the same shape"

    # Flatten the x, y, and z arrays for the fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Generate the design matrix for the polynomial terms
    A = np.zeros((x_flat.size, (order_x + 1) * (order_y + 1)))
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            A[:, i * (order_y + 1) + j] = (x_flat**i) * (y_flat**j)

    # Solve the least squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

    # Evaluate the polynomial at the grid points
    background_flat = A @ coeffs
    background = background_flat.reshape(x.shape)

    return background


def correct_median_diff(imag):

    '''
     Corrects an image by subtracting the median difference between consecutive rows.

    This function is useful for removing systematic row-wise variations from an image,
    such as those caused by background drift.

    :param imag: 2D numpy array representing the image to be corrected.
    :return: 2D numpy array of the corrected image.
    '''
    # Difference of the pixel between two consecutive row
    N2 = np.gradient(imag)[1]
    # Take the median of the difference and cumsum them
    C = np.cumsum(np.median(N2, axis=0))
    # Extend the vector to a matrix (row copy)
    D = np.tile(C, (imag.shape[0], 1))
    corrected_data = imag - D
    return corrected_data


def correct_mean_of_lines(img):

    '''
    Corrects an image by substractiong the mean of each line

    :param img: 2D numpy array representing the image to be corrected
    :return: 2D numpy array of the corrected image.
    '''

    BG = np.mean(img, axis=1, keepdims=True)
    return img - BG


def subtract_trace_average(img, n, axis=0, from_end=False, use_filter=False, poly_fit=False,
                           filter_sigma=1.0, poly_fit_order=1):
    '''
    Subtract the average trace from the data.

    :param img: 2D numpy array representing the image to be corrected
    :param n: Number of traces to compute the average from.
    :param axis: The axis to consider for traces (0 or 1).
    :param from_end: If True, use the last `n` traces instead of the first.
    :param use_filter: If True, use filtered traces for subtraction
    :param poly_fit: If True, subtract the fitted polynomial function instead of averaged traces
    :param filter_sigma: Width of the gaussian filter function
    :param poly_fit_order: Order of the fitted Polynomial
    :return: 2D numpy array of the corrected image.
    '''
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (rows) or 1 (columns).")
    if n <= 0 or n > img.shape[axis]:
        raise ValueError("Invalid value for n: must be between 1 and the size of the specified axis.")

    # Select the traces for averaging
    else:
        if from_end:
            if axis == 0:
                traces = img[-n:, :]
            else:
                traces = img[:, -n:]
        else:
            if axis == 0:
                traces = img[:n, :]
            else:
                traces = img[:, :n]

    # Compute the average trace
    avg_trace = np.mean(traces, axis=axis, keepdims=True)
    if use_filter:
        if axis == 0:
            avg_trace = gaussian_filter1d(avg_trace, filter_sigma)
        else:
            avg_trace = gaussian_filter1d(avg_trace, filter_sigma)
    else:
        pass

    if poly_fit:
        x = np.arange(0, np.shape(avg_trace)[0])
        y = np.arange(0, np.shape(avg_trace)[1])
        xg, yg = np.meshgrid(y, x)
        print(np.shape(xg))
        avg_trace = evaluate_poly_background_2d(xg, yg, avg_trace, poly_fit_order, 0)
    else:
        pass
    # Subtract the average trace from the data
    return img - avg_trace


def gradient_5p_stencil(f, *varargs, axis=None, edge_order=1):
    #### 5-point stencil method for gradient calculation, base code adapted from numpy.gradient
    f = np.asanyarray(f)
    N = f.ndim  # number of dimensions

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = np.lib.array_utils.normalize_axis_tupl(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        dx = [1.0] * len_axes
    elif n == 1 and np.ndim(varargs[0]) == 0:
        dx = varargs * len_axes
    elif n == len_axes:
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = np.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError("when 1d, distances must match "
                                 "the length of the corresponding dimension")
            if np.issubdtype(distances.dtype, np.integer):
                distances = distances.astype(np.float64)
            diffx = np.diff(distances)
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    outvals = []

    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N
    slice5 = [slice(None)]*N
    slice6 = [slice(None)]*N

    otype = f.dtype
    if otype.type is np.datetime64:
        otype = np.dtype(otype.name.replace('datetime', 'timedelta'))
        f = f.view(otype)
    elif otype.type is np.timedelta64:
        pass
    elif np.issubdtype(otype, np.inexact):
        pass
    else:
        if np.issubdtype(otype, np.integer):
            f = f.astype(np.float64)
        otype = np.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < 5:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient using five-point stencil, "
                "at least 5 elements are required.")

        out = np.empty_like(f, dtype=otype)

        uniform_spacing = np.ndim(ax_dx) == 0

        slice1[axis] = slice(2, -2)
        slice2[axis] = slice(None, -4)
        slice3[axis] = slice(1, -3)
        slice4[axis] = slice(3, -1)
        slice5[axis] = slice(4, None)

        if uniform_spacing:
            out[tuple(slice1)] = (-f[tuple(slice5)] + 8 * f[tuple(slice4)] - 8 * f[tuple(slice3)] + f[tuple(slice2)]) / (12. * ax_dx)
        else:
            dx1 = ax_dx[1:-2]
            dx2 = ax_dx[2:-1]
            dx3 = ax_dx[3:]
            a = -1 / (12 * dx1)
            b = 8 / (12 * dx2)
            c = -8 / (12 * dx2)
            d = 1 / (12 * dx3)
            shape = np.ones(N, dtype=int)
            shape[axis] = -1
            a.shape = b.shape = c.shape = d.shape = shape
            out[tuple(slice1)] = (a * f[tuple(slice2)] + b * f[tuple(slice3)] - b * f[tuple(slice4)] + d * f[tuple(slice5)])

        # First order difference at edges
        slice1[axis] = 0
        slice2[axis] = 0
        slice3[axis] = 1
        dx_0 = ax_dx if uniform_spacing else ax_dx[0]
        out[tuple(slice1)] = (f[tuple(slice3)] - f[tuple(slice2)]) / dx_0

        slice1[axis] = 1
        slice2[axis] = 0
        slice3[axis] = 2
        out[tuple(slice1)] = (f[tuple(slice3)] - f[tuple(slice2)]) / (ax_dx if uniform_spacing else ax_dx[1])

        slice1[axis] = -2
        slice2[axis] = -3
        slice3[axis] = -1
        out[tuple(slice1)] = (f[tuple(slice3)] - f[tuple(slice2)]) / (ax_dx if uniform_spacing else ax_dx[-2])

        slice1[axis] = -1
        slice2[axis] = -2
        slice3[axis] = -1
        dx_n = ax_dx if uniform_spacing else ax_dx[-1]
        out[tuple(slice1)] = (f[tuple(slice3)] - f[tuple(slice2)]) / dx_n

        outvals.append(out)

        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)
        slice5[axis] = slice(None)
        slice6[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    return tuple(outvals)




### from down here: work in progress

def gaussian(x, a, mu, sigma):
    '''
    :param x: 1d array for x-values
    :param a: amplitude
    :param mu: mean
    :param sigma: standard deviation
    :return: gaussian peak function a * exp( - (x - mu) ** 2 / (2 * sigma ** 2) wo constant offset
    '''
    return a * np.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2))


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, c):
    '''
    :param x: 1d array
    :param a1: amplitude of first gaussian
    :param mu1: mean of first gaussian
    :param sigma1: standard deviation of first gaussian
    :param a2: amplitude of second gaussian
    :param mu2: mean of second gaussian
    :param sigma2: standard deviation of second gaussian
    :param c: constant offset
    :return: sum of two gaussian functions (see gaussian function)
    '''
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2) + c


def snr_calculation(param):
    '''
    :param param: fit parameters of double gaussian in format (a1, mu1, sigma1, a2, mu2, sigma2, c)
    :return: signal-to-noise ratio according to the formula | mu1 - mu2 | / sqrt( sigma1 ** 2 + sigma2 ** 2)
    '''
    # return np.abs(param[1] - param[4]) / np.sqrt(param[3] ** 2 + param[6] ** 2)
    # return np.abs(param[1] - param[4]) / np.sqrt(param[2] ** 2 + param[5] ** 2)
    return np.abs(param[1] - param[4]) / max(param[2], param[5])


def det_a(snr, m, b):
    '''
    :param snr: signal-to-noise
    :param m: empirical slope for scaling parameter a
    :param b: empirical offset for scaling parameter a
    :return:
    '''
    return m * snr + b


def detect_events_vec(x_data, y_data, thresh_upper, thresh_lower):
    '''
    :param x_data: 1d array of times
    :param y_data: 1d array of detector signals
    :param thresh_upper: upper threshold for schmidt trigger
    :param thresh_lower: lower threshold for schmidt trigger
    :return:
    '''
    # make conditions and divide data into points meeting one of those conditions
    above_upper = y_data > thresh_upper
    below_lower = y_data < thresh_lower

    # simplify array to values -1,0,1 for the three conditions
    result = np.zeros_like(y_data)
    result[above_upper] = 1
    result[below_lower] = -1

    # use diff to detect state changes and set conditions for state changes
    x_result = x_data[1:]
    diff_result = np.diff(result)
    diff_events = np.nonzero(diff_result)[0]
    result_events = diff_result[diff_events]

    up_idx = np.where((result_events == 2) | ((result_events == 1) & (np.roll(result_events, -1) == 1)))[0]
    up_list = diff_events[up_idx]

    down_idx = np.where((result_events == -2) | ((result_events == -1) & (np.roll(result_events, -1) == -1)))[0]

    down_list = diff_events[down_idx]
    events_list = np.sort(np.concatenate((up_list, down_list)))

    x_events = x_data[events_list]
    times_list = np.diff(x_events)

    # check whether first event goes up or down and whether the lists contain events
    if len(up_idx) != 0 and len(down_idx) != 0:
        if up_idx[0] < down_idx[0]:
            up_times = times_list[::2]
            down_times = times_list[1::2]
        elif up_idx[0] > down_idx[0]:
            up_times = times_list[1::2]
            down_times = times_list[::2]
    else:
        up_times, down_times = [], []

    return x_result, diff_result, up_list, down_list, up_times, down_times


def detection_param_double_gauss_fit(bin_centers, hist, offset, width_start, bounds_gaussian, bounds_double_gaussian,
                                     peak_finder_param=(100, 30, 200, 10)):
    # instead of getting histograms, modify the function to create histograms.
    hist_smoothed = gaussian_filter1d(hist, len(hist) / 20)
    hist_diff = np.diff(hist_smoothed)
    peaks, _ = find_peaks(hist_smoothed, height=peak_finder_param[0], distance=peak_finder_param[1],
                          prominence=peak_finder_param[2])
    if len(peaks) == 1:
        peaks_diff, _ = find_peaks(np.abs(hist_diff), height=peak_finder_param[0], distance=peak_finder_param[1],
                                   prominence=peak_finder_param[2], width=peak_finder_param[3])
        if len(peaks_diff) == 2:
            snr = 0  # one has to think about this setting
            popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=(hist[peaks][0], bin_centers[peaks][0],
                                                                    width_start), bounds=bounds_gaussian)
        elif len(peaks_diff) == 3:
            a1 = hist[int((peaks_diff[1] + peaks_diff[0]) / 2)]
            a2 = hist[int((peaks_diff[2] + peaks_diff[1]) / 2)]

            mu1 = bin_centers[int((peaks_diff[1] + peaks_diff[0]) / 2)]
            mu2 = bin_centers[int((peaks_diff[2] + peaks_diff[1]) / 2)]

            popt, pcov = curve_fit(double_gaussian, bin_centers, hist,
                                   p0=(a1, mu1, width_start, a2, mu2, width_start, offset),
                                   bounds=bounds_double_gaussian)
            snr = snr_calculation(popt)

    elif len(peaks) == 2:
        a1 = hist[peaks][0]
        a2 = hist[peaks][1]

        mu1 = bin_centers[peaks][0]
        mu2 = bin_centers[peaks][1]

        popt, pcov = curve_fit(double_gaussian, bin_centers, hist,
                               p0=(a1, mu1, width_start, a2, mu2, width_start, offset),
                               bounds=bounds_double_gaussian)
        snr = snr_calculation(popt)

    else:
        return None  # currently I have not better idea, maybe set some default parameters

    return bin_centers, hist, popt, snr


def moving_average(data, window_size):
    half_window = window_size // 2
    smoothed_data = np.zeros_like(data)
    for i in range(half_window, len(data) - half_window):
        smoothed_data[i] = np.mean(data[i - half_window: i + half_window + 1])
    return smoothed_data


def fit_double_gaussian(x_data, y_data, p0, bounds):
    try:
        params, cov = curve_fit(double_gaussian, x_data, y_data, p0, bounds=bounds)

    except RuntimeError:
        params, cov = np.zeros(7), 0

    return params, cov


def gamma(t_list):
    try:
        t_mean = np.mean(t_list)
        t_s = np.std(t_list)
        gamma = 1 / t_mean
        gamma_s = gamma * t_s / t_mean

    except (TypeError, ZeroDivisionError, ValueError):
        print("Error occured")
        gamma = float("NaN")
        gamma_s = float("NaN")

    return gamma, gamma_s



