import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.special import beta, erf
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

def beta_func_shape(x, a, b, scale):
    """
    Computes the scaled beta distribution shape function at a given point x with
    specified parameters a, b, and scale.

    This function calculates the product of the beta function for parameters `a`
    and `b` and the power terms that scale the input. The beta distribution is
    frequently used in probability theory and statistics for modeling random
    variables limited to intervals of finite length.

    :param x: The input value where the beta distribution is evaluated.
    :param a: The alpha parameter of the beta distribution, which controls
        the shape of the curve.
    :param b: The beta parameter of the beta distribution, which controls
        the shape of the curve.
    :param scale: A scaling factor applied to the input value `x`, defining
        the range for the beta distribution.
    :return: The computed value of the scaled beta function for the given
        inputs.
    :rtype: float
    """
    return beta(a, b) * ((scale * x) ** (a - 1)) * ((1 - scale * x) ** (b - 1))

def skewed_gaussian_func_shape(x, x0, sigma, alpha):
    """
    Computes the shape of a skewed Gaussian distribution given the input `x`,
    distribution mean `x0`, standard deviation `sigma`, and skewness `alpha`.
    Combines the Gaussian cumulative and probability density functions to
    achieve skewness in the distribution.

    :param x: The input variable around which the skewed Gaussian shape is
        computed.
    :type x: float or numpy.ndarray
    :param x0: The mean or central value of the distribution.
    :type x0: float
    :param sigma: The standard deviation, represents the spread of the
        distribution.
    :type sigma: float
    :param alpha: The skewness parameter; positive values skew to the right
        while negative values skew to the left.
    :type alpha: float
    :return: The skewed Gaussian value(s) corresponding to the input `x`.
    :rtype: float or numpy.ndarray
    """
    z = (x - x0) / sigma
    gauss_cum = 1/2 * (1 + erf(alpha * z/np.sqrt(2)))
    gauss_dis = 1/np.sqrt(2*np.pi) * np.exp(-z**2/2)
    return 2 / sigma * gauss_dis * gauss_cum

def extract_linecut(x, y, z, start_point, end_point):
    """
    Extract values from an irregular 2D dataset along a line between two points.

    Parameters:
    -----------
    x : numpy array
        X coordinates of the data points (irregular grid)
    y : numpy array
        Y coordinates of the data points (irregular grid)
    z : numpy array
        The data values at each (x,y) point
    start_point : tuple (x0, y0)
        Starting point of the line cut
    end_point : tuple (x1, y1)
        Ending point of the line cut

    Returns:
    --------
    numpy array
        Z values extracted along the line cut
    """
    # Ensure inputs are numpy arrays and flatten if needed
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    z = np.asarray(z).flatten()

    # Extract start and end points
    x0, y0 = start_point
    x1, y1 = end_point

    # Calculate the Euclidean distance between the start and end points
    total_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # Estimate the average grid spacing based on nearest neighbor distances
    if len(x) > 1:
        points = np.column_stack((x, y))
        # Calculate distances between each point and its nearest neighbor
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)  # Find self and nearest neighbor
        avg_grid_spacing = np.mean(distances[:, 1])  # placeholder
    else:
        avg_grid_spacing = 1.0

    # Determine number of points based on the line length and grid spacing
    num_points = int(total_distance / avg_grid_spacing) + 1
    num_points = max(num_points, 2)  # Ensure at least 2 points

    # Create points along the line
    t = np.linspace(0, 1, num_points)
    x_line = x0 + t * (x1 - x0)
    y_line = y0 + t * (y1 - y0)

    # Use griddata to interpolate z values at the line points
    points = np.column_stack((x, y))
    line_points = np.column_stack((x_line, y_line))

    # Interpolate using cubic b-splines
    z_values = interpolate.griddata(points, z, line_points, method='linear')

    return z_values


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


def two_d_fft_on_data(imag, x, y, mode='Amplitude'):

    """
    Performs a 2D Fast Fourier Transform (FFT) operation on irregularly gridded data
    and returns frequency grids along with the transformed data. The input data
    is first interpolated onto a regular grid before applying the FFT. The output
    can be amplitude, phase, or complex values based on the mode selected.

    :param imag:
        A 2D array representing the input data values to be transformed. It should
        match with the dimensions of the x and y input coordinates.

    :param x:
        A 1D array representing the x-coordinates corresponding to the input data.
        Coordinates may be unevenly spaced values.

    :param y:
        A 1D array representing the y-coordinates corresponding to the input data.
        Coordinates may be unevenly spaced values.

    :param mode:
        A string specifying the desired output type of the FFT. Options include:
        - 'Amplitude': Returns the amplitude of the FFT.
        - 'Phase': Returns the phase of the FFT.
        - 'Complex': Returns the full complex FFT result.
        Defaults to 'Amplitude'.

    :return:
        A tuple containing three elements:
        - ixg: The frequency grid corresponding to the x-direction after FFT.
        - iyg: The frequency grid corresponding to the y-direction after FFT.
        - fft_amp: The resulting 2D FFT data, calculated based on the selected mode.
    """
    x_num = int(np.shape(imag)[1])
    y_num = int(np.shape(imag)[0])
    x_inter, dx = np.linspace(np.min(x), np.max(x), int(x_num), retstep=True)
    y_inter, dy = np.linspace(np.min(y), np.max(y), int(y_num), retstep=True)
    xg, yg = np.meshgrid(x_inter, y_inter)
    data_on_regular_grid = interpolate_2d_results(imag, x, y, (xg, yg))
    if mode == 'Amplitude':
        fft_amp = fftshift(np.abs(fft2(data_on_regular_grid - np.mean(data_on_regular_grid), norm='ortho')))
    elif mode == 'Phase':
        fft_amp = fftshift(np.angle(fft2(data_on_regular_grid, norm='ortho')))
    elif mode == 'Complex':
        fft_amp = fftshift(fft2(data_on_regular_grid, norm='ortho'))
    else:
        raise ValueError("Invalid mode. Choose from 'Amplitude', 'Phase', or 'Complex'.")
    ixg, iyg = np.meshgrid(fftshift(fftfreq(int(x_num), dx)), fftshift(fftfreq(int(y_num), dy)))
    
    return ixg, iyg, fft_amp


def two_d_ifft_on_data(fft_data, ixg, iyg, mode='Amplitude'):
    """
    Performs a 2D Inverse Fast Fourier Transform (IFFT) operation on frequency-domain data
    and returns the reconstructed spatial-domain data. The input data is assumed to be
    on a regular frequency grid.

    :param fft_data:
        A 2D array representing the input frequency-domain data to be transformed.

    :param ixg:
        A 2D array representing the frequency grid in the x-direction.

    :param iyg:
        A 2D array representing the frequency grid in the y-direction.

    :param mode:
        A string specifying the desired output type of the IFFT. Options include:
        - 'Amplitude': Returns the amplitude of the reconstructed data.
        - 'Phase': Returns the phase of the reconstructed data.
        - 'Complex': Returns the full complex IFFT result.
        Defaults to 'Amplitude'.

    :return:
        A tuple containing three elements:
        - xg: The spatial grid corresponding to the x-direction after IFFT.
        - yg: The spatial grid corresponding to the y-direction after IFFT.
        - reconstructed_data: The resulting 2D spatial-domain data, calculated based on the selected mode.
    """
    # Perform the inverse FFT
    ifft_result = np.fft.ifft2(np.fft.ifftshift(fft_data), norm='ortho')

    # Extract the desired mode
    if mode == 'Amplitude':
        reconstructed_data = np.abs(ifft_result)
    elif mode == 'Phase':
        reconstructed_data = np.angle(ifft_result)
    elif mode == 'Complex':
        reconstructed_data = ifft_result
    else:
        raise ValueError("Invalid mode. Choose from 'Amplitude', 'Phase', or 'Complex'.")

    # Generate spatial grids based on the frequency grids
    freq_diff_x = ixg[0, 1] - ixg[0, 0]
    freq_diff_y = iyg[1, 0] - iyg[0, 0]

    if freq_diff_x == 0 or freq_diff_y == 0:
        raise ValueError("Frequency grid differences must be non-zero.")

    dx = 1 / freq_diff_x
    dy = 1 / freq_diff_y

    xg = np.linspace(-0.5 * dx, 0.5 * dx, fft_data.shape[1])
    yg = np.linspace(-0.5 * dy, 0.5 * dy, fft_data.shape[0])
    xg, yg = np.meshgrid(xg, yg)

    return xg, yg, reconstructed_data


def evaluate_poly_background_2d(x, y, z, order_x, order_y,
                                       x_range=None, y_range=None):
    '''
    Evaluates a polynomial background from a restricted region and applies it to the full dataset.

    This function fits a polynomial of specified orders to a defined region of data,
    then applies that background model to the entire dataset.

    :param x: 2D array of x coordinates
    :param y: 2D array of y coordinates
    :param z: 2D array of z values at each (x, y) point
    :param order_x: Integer specifying the order of the polynomial in the x-direction
    :param order_y: Integer specifying the order of the polynomial in the y-direction
    :param x_range: Tuple (xmin, xmax) defining the x range for fitting, or None to use all data
    :param y_range: Tuple (ymin, ymax) defining the y range for fitting, or None to use all data
    :return: 2D numpy array of the calculated background for the full dataset
    '''
    assert x.shape == y.shape == z.shape, "x, y, and z must have the same shape"

    # Flatten the arrays
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Create mask for the region of interest
    mask = np.ones_like(x_flat, dtype=bool)

    if x_range is not None:
        x_min, x_max = x_range
        mask &= (x_flat >= x_min) & (x_flat <= x_max)

    if y_range is not None:
        y_min, y_max = y_range
        mask &= (y_flat >= y_min) & (y_flat <= y_max)

    # Filter points to only those in the region of interest
    x_roi = x_flat[mask]
    y_roi = y_flat[mask]
    z_roi = z_flat[mask]

    if len(x_roi) == 0:
        raise ValueError("No data points in the specified region!")

    # Generate the design matrix for the polynomial terms (for ROI only)
    A_roi = np.zeros((x_roi.size, (order_x + 1) * (order_y + 1)))
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            A_roi[:, i * (order_y + 1) + j] = (x_roi ** i) * (y_roi ** j)

    # Solve the least squares problem for the ROI
    coeffs, residuals, rank, s = np.linalg.lstsq(A_roi, z_roi, rcond=None)

    # Now generate design matrix for the ENTIRE dataset
    A_full = np.zeros((x_flat.size, (order_x + 1) * (order_y + 1)))
    for i in range(order_x + 1):
        for j in range(order_y + 1):
            A_full[:, i * (order_y + 1) + j] = (x_flat ** i) * (y_flat ** j)

    # Apply the coefficients to the entire dataset
    background_flat = A_full @ coeffs
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
    """
    Computes the numerical gradient of a multidimensional array using the
    5-point stencil method. This method calculates the gradient along the
    specified axes of a given array using a finite difference approximation.
    It supports both uniformly and non-uniformly spaced grids for input data.
    The output is a numerical approximation of the gradient, with accuracy
    dependent on the `edge_order` and spacing.

    :param f: Input array for which the gradient will be computed.
    :type f: numpy.ndarray
    :param varargs: Variable parameter(s) for spacing between points along each
        axis. Supports scalar values for uniform spacing, or 1-D arrays with
        explicitly defined distances for non-uniform spacing.
    :type varargs: list, tuple, or numpy.ndarray
    :param axis: Axis or axes along which the gradient is computed. Defaults to
        None, in which case the gradient is computed along all axes.
    :type axis: int, tuple, or None
    :param edge_order: Order of the finite difference approximation at the edges.
        Acceptable values are 1 or 2, with higher values producing more accurate
        derivative estimates near boundaries. Default is 1.
    :type edge_order: int
    :return: Numerical gradient computed along the specified axes. Returns a
        single numpy.ndarray if the gradient is computed along a single axis, or
        a tuple of numpy.ndarrays for multiple axes.
    :rtype: numpy.ndarray or tuple of numpy.ndarray
    :raises ValueError: If the shape of the input array along a specified axis is
        less than 5, or if distances in `varargs` are improperly formatted,
        mismatched with the input array dimensions, or are not supported.
    :raises TypeError: If the number of elements in `varargs` does not match the
        number of axes along which the gradient is calculated.
    """
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


def cut_data_range(x_grid, y_grid, z_data, x_range, y_range):
    """
    Filters and extracts a specific range of data from given 2D grids and associated data array.

    This function takes two 2D grids (x and y) and a 2D data array, alongside specified ranges
    for x and y dimensions. It computes masks to filter the given grids and data array to only
    include elements within the specified range. The filtered subsets of the grids and data array
    are returned.

    :param x_grid: 2D numpy array representing the x-coordinate grid.
    :param y_grid: 2D numpy array representing the y-coordinate grid.
    :param z_data: 2D numpy array representing the data array associated with the grids.
    :param x_range: Tuple containing the lower and upper bounds for the x-coordinate (inclusive).
    :param y_range: Tuple containing the lower and upper bounds for the y-coordinate (inclusive).
    :return: A tuple containing the filtered x-grid, y-grid, and data array.
    :rtype: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    x_mask = (x_grid[0, :] >= x_range[0]) & (x_grid[0, :] <= x_range[1])
    y_mask = (y_grid[:, 0] >= y_range[0]) & (y_grid[:, 0] <= y_range[1])

    x_grid_cut = x_grid[y_mask, :][:, x_mask]
    y_grid_cut = y_grid[y_mask, :][:, x_mask]
    z_data_cut = z_data[y_mask, :][:, x_mask]

    return x_grid_cut, y_grid_cut, z_data_cut

def trace_wise_min_max_scaling(img):
    return (img - np.min(img, axis=-1, keepdims=True)) / (np.max(img, axis=-1, keepdims=True) - np.min(img, axis=-1, keepdims=True))
### from down here: work in progress (some parts are already refactored into gamma maps)

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



