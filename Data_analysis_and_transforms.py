import numpy as np
from scipy import interpolate
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

