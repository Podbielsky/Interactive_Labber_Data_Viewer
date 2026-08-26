import time
import os
import threading
import queue
import tkinter as tk
from collections import deque
from contextlib import contextmanager
from tkinter import messagebox
from tkinter import filedialog
import ttkbootstrap as ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import rc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.colors as colors
from matplotlib.path import Path
from matplotlib.transforms import TransformedPath
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from Data_analysis_and_transforms import (image_down_sampling, two_d_fft_on_data, two_d_ifft_on_data, evaluate_poly_background_2d,
                                          correct_median_diff, correct_mean_of_lines, gradient_5p_stencil,
                                          subtract_trace_average, cut_data_range, extract_linecut,
                                          get_linecut_pixel_normal,
                                          cumulative_integral, skewed_gaussian_func_shape,
                                          beta_func_shape, trace_wise_min_max_scaling)
from gamma_map import (get_t_rates, get_fourier, fft_correction_select, fft_correction_apply, get_cuts)
from custom_cmap import make_neon_cyclic_colormap, make_bi_colormap, make_half_red_map, make_half_blue_map
from fitting_tools import (
    DEFAULT_FIT_EXPRESSION,
    DEFAULT_MAXFEV,
    FIT_EXPRESSION_EXAMPLE,
    FitModelDefinition,
    default_initial_value,
    format_fit_result,
)
from plot_style import (
    AVAILABLE_COLORMAPS,
    COLOR_CYCLE_OPTIONS,
    DEFAULT_PLOT_STYLE,
    configure_transparent_matplotlib_canvas,
    get_color_cycle,
    normalize_plot_style,
    open_plot_style_dialog,
    resolve_plot_color,
)
neon_cmap = make_neon_cyclic_colormap()
bi_map = make_bi_colormap() # take out
half_red_map = make_half_red_map()
half_blue_map = make_half_blue_map()
plt.register_cmap(name='BiMap', cmap=bi_map)
plt.register_cmap(name='RedMap', cmap=half_red_map)
plt.register_cmap(name='BlueMap', cmap=half_blue_map)
plt.register_cmap(name='NeonPiCy', cmap=neon_cmap)
rc('pdf', fonttype=42)


TRACE_SAMPLE_AXIS_ROLE = 'trace_samples'


def format_hdf5_label(label):
    """Return a readable label for byte or string HDF5 metadata."""
    if isinstance(label, (bytes, np.bytes_)):
        return label.decode('utf-8', errors='replace')
    return str(label)


def _normalized_grid_variation(grid, axis):
    """Measure coordinate variation along one grid axis independent of units."""
    finite_values = np.asarray(grid, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size < 2:
        return 0.0
    value_range = np.ptp(finite_values)
    if not np.isfinite(value_range) or value_range == 0:
        return 0.0
    differences = np.abs(np.diff(np.asarray(grid, dtype=float), axis=axis))
    if not np.any(np.isfinite(differences)):
        return 0.0
    return float(np.nanmedian(differences) / value_range)


def canonicalize_plot_grid(x_grid, y_grid, data_grid):
    """Orient a 2D coordinate grid so X varies by column and Y by row."""
    x_array = np.asarray(x_grid)
    y_array = np.asarray(y_grid)
    data_array = np.asarray(data_grid)
    if x_array.ndim != 2 or y_array.ndim != 2 or data_array.ndim != 2:
        raise ValueError('X, Y, and displayed data must all be two-dimensional.')
    if x_array.shape != y_array.shape or x_array.shape != data_array.shape:
        raise ValueError(
            'X, Y, and displayed data must have matching shapes; got '
            f'{x_array.shape}, {y_array.shape}, and {data_array.shape}.'
        )

    x_column_variation = _normalized_grid_variation(x_array, axis=1)
    x_row_variation = _normalized_grid_variation(x_array, axis=0)
    y_column_variation = _normalized_grid_variation(y_array, axis=1)
    y_row_variation = _normalized_grid_variation(y_array, axis=0)

    # Judge the two coordinates together. Summing their variations makes a
    # serpentine X grid look transposed because alternate rows place opposite
    # X endpoints beside one another. The Jacobian-like products below still
    # identify Y as the row coordinate when it is constant within each row.
    standard_score = x_column_variation * y_row_variation
    transposed_score = x_row_variation * y_column_variation
    if max(standard_score, transposed_score) <= 1e-15:
        # Degenerate grids (for example, a singleton coordinate) do not carry
        # enough two-axis information for the product test. Retain the older
        # one-coordinate fallback for those cases.
        standard_score = x_column_variation + y_row_variation
        transposed_score = x_row_variation + y_column_variation

    if transposed_score > standard_score + 1e-15:
        return x_array.T, y_array.T, data_array.T
    return x_array, y_array, data_array


def slice_scan_grid_for_axes(
    grid,
    x_axis_index,
    y_axis_index,
    fixed_axis_indices,
):
    """Slice an N-D scan grid and return its selected plane in [Y, X] order."""
    grid_array = np.asarray(grid)
    axis_count = grid_array.ndim
    x_axis_index = int(x_axis_index)
    y_axis_index = int(y_axis_index)
    if x_axis_index == y_axis_index:
        raise ValueError('The displayed X and Y scan axes must be different.')
    if not (
        0 <= x_axis_index < axis_count
        and 0 <= y_axis_index < axis_count
    ):
        raise ValueError('A displayed scan-axis index is out of range.')

    fixed_axis_indices = dict(fixed_axis_indices)
    expected_fixed_axes = set(range(axis_count)) - {
        x_axis_index,
        y_axis_index,
    }
    if set(fixed_axis_indices) != expected_fixed_axes:
        raise ValueError(
            'Every scan axis other than X and Y must have one selected index.'
        )

    selection = [slice(None)] * axis_count
    for axis_index, selected_index in fixed_axis_indices.items():
        selected_index = int(selected_index)
        if not 0 <= selected_index < grid_array.shape[axis_index]:
            raise IndexError(
                f'Selected index {selected_index} is outside scan axis '
                f'{axis_index} with size {grid_array.shape[axis_index]}.'
            )
        selection[axis_index] = selected_index

    sliced_grid = grid_array[tuple(selection)]
    remaining_axes = [
        axis_index
        for axis_index in range(axis_count)
        if axis_index not in expected_fixed_axes
    ]
    y_position = remaining_axes.index(y_axis_index)
    x_position = remaining_axes.index(x_axis_index)
    return np.transpose(sliced_grid, axes=(y_position, x_position))


def slice_scan_vector_for_axis(
    grid,
    varying_axis_index,
    fixed_axis_indices,
):
    """Slice an N-D scan grid while retaining one selected dimension."""
    grid_array = np.asarray(grid)
    axis_count = grid_array.ndim
    varying_axis_index = int(varying_axis_index)
    if not 0 <= varying_axis_index < axis_count:
        raise ValueError('The varying scan-axis index is out of range.')

    fixed_axis_indices = dict(fixed_axis_indices)
    expected_fixed_axes = set(range(axis_count)) - {varying_axis_index}
    if set(fixed_axis_indices) != expected_fixed_axes:
        raise ValueError(
            'Every scan axis other than the varying axis must have one '
            'selected index.'
        )

    selection = [slice(None)] * axis_count
    for axis_index, selected_index in fixed_axis_indices.items():
        selected_index = int(selected_index)
        if not 0 <= selected_index < grid_array.shape[axis_index]:
            raise IndexError(
                f'Selected index {selected_index} is outside scan axis '
                f'{axis_index} with size {grid_array.shape[axis_index]}.'
            )
        selection[axis_index] = selected_index
    return np.asarray(grid_array[tuple(selection)])


def reshape_trace_order_to_scan_grid(trace_order, measure_dim):
    """Convert Labber's flattened trace order into logical scan-axis order."""
    dimensions = tuple(int(dimension) for dimension in measure_dim)
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise ValueError('Trace scan dimensions must be positive.')
    trace_order_array = np.asarray(trace_order)
    if trace_order_array.size != int(np.prod(dimensions)):
        raise ValueError(
            'The trace count does not match the product of the scan dimensions.'
        )

    flattened_scan_shape = (
        dimensions[0],
        int(np.prod(dimensions[1:])),
    )
    if trace_order_array.shape == flattened_scan_shape:
        trace_order_array = trace_order_array.swapaxes(0, 1)
    return trace_order_array.reshape(tuple(reversed(dimensions)))


def load_selected_trace_matrix(trace_reference, trace_indices):
    """Read selected Labber traces and return one trace per matrix row."""
    trace_indices = np.asarray(trace_indices, dtype=int).ravel()
    if trace_indices.size == 0:
        raise ValueError('At least one trace must be selected for the map.')
    trace_count = int(trace_reference.shape[2])
    if np.any(trace_indices < 0) or np.any(trace_indices >= trace_count):
        raise IndexError('A selected trace index is outside the trace dataset.')

    # h5py requires increasing indices for fancy dataset selection. Reading
    # sorted unique traces also avoids loading unrelated traces from large files.
    unique_indices, inverse_indices = np.unique(
        trace_indices,
        return_inverse=True,
    )
    selected_traces = np.asarray(
        trace_reference[:, 0, unique_indices]
    )
    if selected_traces.ndim == 1:
        selected_traces = selected_traces[:, np.newaxis]
    return selected_traces[:, inverse_indices].T


def build_trace_axis_map(
    trace_reference,
    trace_order_grid,
    trace_x_values,
    y_coordinate_grid,
    y_axis_index,
    fixed_axis_indices,
):
    """Build a [scan Y, trace X] map whose values are trace amplitudes."""
    trace_indices = slice_scan_vector_for_axis(
        trace_order_grid,
        y_axis_index,
        fixed_axis_indices,
    )
    y_values = slice_scan_vector_for_axis(
        y_coordinate_grid,
        y_axis_index,
        fixed_axis_indices,
    )
    trace_matrix = load_selected_trace_matrix(
        trace_reference,
        trace_indices,
    )
    trace_x_values = np.asarray(trace_x_values)
    if trace_x_values.ndim != 1:
        trace_x_values = np.ravel(trace_x_values)
    if trace_matrix.shape[1] != trace_x_values.size:
        raise ValueError(
            'The trace X axis length does not match the stored trace length.'
        )

    map_shape = trace_matrix.shape
    x_grid = np.broadcast_to(trace_x_values, map_shape)
    y_grid = np.broadcast_to(np.asarray(y_values)[:, np.newaxis], map_shape)
    return x_grid, y_grid, trace_matrix


def _row_direction(values):
    """Return the predominant direction of finite values in a grid row."""
    row = np.asarray(values, dtype=float)
    finite_values = row[np.isfinite(row)]
    if finite_values.size < 2:
        return 0

    differences = np.diff(finite_values)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if differences.size == 0:
        return 0
    return int(np.sign(np.nanmedian(differences)))


def reverse_alternating_rows(array):
    """Return a copy with the acquisition order of every second row reversed."""
    corrected = np.array(array, copy=True)
    if corrected.ndim != 2:
        raise ValueError('Alternating sweep correction requires a 2-D array.')
    corrected[1::2, :] = corrected[1::2, ::-1]
    return corrected


def correct_alternating_x_sweep(
    x_grid,
    y_grid,
    data_grid,
    acquisition_row_indices=None,
):
    """Correct maps acquired with X reversing direction on every second row.

    Labber data may contain coordinate rows in their physical sweep order, or
    it may store an already regular X grid while keeping the signal in
    acquisition order. Signal rows are therefore always reversed. Coordinate
    rows are reversed only when their X direction is opposite to the first
    usable row, avoiding a folded coordinate mesh for regular stored grids.
    """
    x_array = np.asarray(x_grid)
    y_array = np.asarray(y_grid)
    data_array = np.asarray(data_grid)
    if x_array.ndim != 2 or y_array.ndim != 2 or data_array.ndim != 2:
        raise ValueError('X, Y, and displayed data must all be two-dimensional.')
    if x_array.shape != y_array.shape or x_array.shape != data_array.shape:
        raise ValueError(
            'X, Y, and displayed data must have matching shapes; got '
            f'{x_array.shape}, {y_array.shape}, and {data_array.shape}.'
        )

    if acquisition_row_indices is None:
        acquisition_rows = np.arange(x_array.shape[0], dtype=int)
    else:
        acquisition_rows = np.asarray(acquisition_row_indices, dtype=int)
        if acquisition_rows.ndim != 1 or acquisition_rows.size != x_array.shape[0]:
            raise ValueError(
                'Acquisition row indices must contain one entry per map row.'
            )

    corrected_x = np.array(x_array, copy=True)
    corrected_y = np.array(y_array, copy=True)
    corrected_data = np.array(data_array, copy=True)
    reversed_row_positions = np.flatnonzero(acquisition_rows % 2 == 1)
    corrected_data[reversed_row_positions, :] = corrected_data[
        reversed_row_positions, ::-1
    ]

    reference_direction = 0
    for row_position, row in enumerate(corrected_x):
        reference_direction = _row_direction(row)
        if reference_direction != 0:
            if acquisition_rows[row_position] % 2 == 1:
                reference_direction *= -1
            break

    if reference_direction != 0:
        for row_position in reversed_row_positions:
            if _row_direction(corrected_x[row_position]) == -reference_direction:
                corrected_x[row_position] = corrected_x[row_position, ::-1]
                corrected_y[row_position] = corrected_y[row_position, ::-1]

    return corrected_x, corrected_y, corrected_data


def nearest_grid_indices(x_grid, y_grid, x_value, y_value):
    """Return the nearest (row, column) using both 2D coordinates."""
    x_array = np.asarray(x_grid, dtype=float)
    y_array = np.asarray(y_grid, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y_array)
    if not np.any(valid):
        return None

    x_range = np.ptp(x_array[valid])
    y_range = np.ptp(y_array[valid])
    x_scale = x_range if np.isfinite(x_range) and x_range > 0 else 1.0
    y_scale = y_range if np.isfinite(y_range) and y_range > 0 else 1.0
    distances = np.full(x_array.shape, np.inf, dtype=float)
    distances[valid] = (
        ((x_array[valid] - x_value) / x_scale) ** 2
        + ((y_array[valid] - y_value) / y_scale) ** 2
    )
    return tuple(int(index) for index in np.unravel_index(
        np.argmin(distances),
        distances.shape,
    ))




class InteractiveSlicePlotter:
    # class is depracted and should not be used in further code but is left as an example
    def __init__(self, root, data):
        self.root = root
        self.root.title("Interactive Slice through 3D Array")

        self.data = data
        self.current_slice_index = 0

        self.create_widgets()

    def create_widgets(self):
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        configure_transparent_matplotlib_canvas(self.fig, self.canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.slice_slider = ttk.Scale(self.root, from_=0, to=self.data.shape[0] - 1, orient=tk.HORIZONTAL,
                                      command=self.update_slice, length=200)
        self.slice_slider.pack(pady=10)
        self.slice_slider.set(0)

        # Add buttons for navigation
        self.prev_button = ttk.Button(self.root, text="Previous", command=self.prev_slice)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button = ttk.Button(self.root, text="Next", command=self.next_slice)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.update_plot()

    def update_slice(self, value):
        self.current_slice_index = int(float(value))  # Explicit conversion to integer
        self.update_plot()

    def prev_slice(self):
        self.current_slice_index = max(0, self.current_slice_index - 1)
        self.update_plot()

    def next_slice(self):
        self.current_slice_index = min(self.data.shape[0] - 1, self.current_slice_index + 1)
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        slice_data = self.data[self.current_slice_index, :, :]

        self.ax.imshow(slice_data, cmap='viridis', origin='lower', aspect='gouraud')
        self.ax.set_title(f"Slice {self.current_slice_index}")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")

        self.canvas.draw_idle()


class InteractiveHistogramPlotter:
    # class is depracted and should not be used in further code but is left as an example
    def __init__(self, root, data, nbins):
        self.root = root
        self.root.title("Interactive Histogram Plotter")

        self.data = data
        self.nbins = nbins
        self.current_row = tk.StringVar()
        self.current_col = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        configure_transparent_matplotlib_canvas(self.fig, self.canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.row_label = ttk.Label(self.root, text="Selected Row:")
        self.row_label.pack()

        # Combobox for row selection
        self.current_row.set(0)
        self.row_combobox = ttk.Combobox(self.root, textvariable=self.current_row, state="readonly")
        self.row_combobox["values"] = list(range(self.data.shape[0]))
        self.row_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        self.row_combobox.pack()

        self.col_label = ttk.Label(self.root, text="Selected Column:")
        self.col_label.pack()

        # Combobox for column selection
        self.current_col.set(0)
        self.col_combobox = ttk.Combobox(self.root, textvariable=self.current_col, state="readonly")
        self.col_combobox["values"] = list(range(self.data.shape[1]))
        self.col_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        self.col_combobox.pack()

        self.update_plot()

    def update_plot(self, event=None):
        self.ax.clear()

        # Get selected row and column indices
        selected_row = int(self.current_row.get())
        selected_col = int(self.current_col.get())

        # Select data for the current row and column
        selected_data = self.data[selected_row, selected_col, :]

        # Plot 1D histogram
        self.ax.hist(selected_data, bins=self.nbins, edgecolor='black')
        self.ax.set_title(f"Histogram - Row {selected_row}, Column {selected_col}")
        self.ax.set_xlabel("Value")
        self.ax.set_ylabel("Frequency")

        self.row_label.config(text=f"Selected Row: {selected_row}")
        self.col_label.config(text=f"Selected Column: {selected_col}")

        self.canvas.draw()


class InteractiveArrayPlotter:
    """
    Class for interactive plotting and analysis of multidimensional array data using a graphical user interface.

    The InteractiveArrayPlotter class is designed to enable visualization and interaction with data stored in
    an HDF5 format. The class integrates several functionalities, including data manipulation, analysis, and
    rendering plots using Matplotlib. Users can utilize features such as ROI-based data selection, Gaussian
    filtering, background subtraction, and 2D FFT transformations directly within the interface. The class
    also supports interactive crosshair visualization, interpolation settings, and trace-specific processes
    if the dataset contains trace information.

    """
    supports_trace_axis_map = False

    def __init__(
        self,
        root,
        hdf5data,
        figure=None,
        ax=None,
        plot_style=None,
        plot_style_change_callback=None,
    ):
        self.root = root
        self.root.title("Interactive Array Plotter")
        self.plot_style = normalize_plot_style(plot_style)
        self.plot_style_change_callback = plot_style_change_callback

        # Initialize attributes
        self.data = hdf5data
        #### Created by Nico Reinders  for trace loading validation
        self.contains_traces = hdf5data.has_traces()
        self.trace_axis_map_mode = False
        self._data_selection_before_trace_axis_map = None
        ####
        self.name_data_z = ''
        self.name_data_y_axis = ''
        self.name_data_x_axis = ''
        measurement_axis_count = (
            0 if self.data.measure_axis is None else len(self.data.measure_axis)
        )
        if measurement_axis_count == 0:
            raise ValueError('The measurement does not contain a sweep axis.')
        if len(self.data.measure_dim) != measurement_axis_count:
            raise ValueError(
                'The number of scan-axis arrays does not match the scan shape.'
            )
        self.scan_axis_count = measurement_axis_count
        self.scan_shape = tuple(
            int(dimension) for dimension in reversed(self.data.measure_dim)
        )
        self.scan_axis_names = [
            format_hdf5_label(axis_name)
            for axis_name in reversed(self.data.name_axis)
        ]
        self.x_scan_axis_index = self.scan_axis_count - 1
        self.y_scan_axis_index = (
            self.scan_axis_count - 2
            if self.scan_axis_count >= 2
            else None
        )
        self.additional_scan_axis_indices = [
            axis_index
            for axis_index in range(self.scan_axis_count)
            if axis_index not in (
                self.x_scan_axis_index,
                self.y_scan_axis_index,
            )
        ]
        self.single_axis_measurement = measurement_axis_count == 1
        self.num_dimensions = len(self.additional_scan_axis_indices)
        print(self.num_dimensions)
        self.name_data = [
            format_hdf5_label(label) for label in self.data.name_data
        ]
        if self.contains_traces:
            self.name_data.append("Tunneling rates in")
            self.name_data.append("Tunneling rates out")
        self.x_index = 0
        self.y_index = 0
        self.nan_mask = np.array([])
        self.loaded = False # rename to be more discriptiv
        self.calculated = False
        self.auto_scale_factor = 2.5
        self.data_operation_history = deque(maxlen=5)

        # ROI selection and its lightweight, coalesced canvas preview.
        self.roi_mode = False
        self.roi_corners = []  # Store corners of the ROI as [(x1, y1), (x2, y2)]
        self.current_roi_patch = None  # Current ROI rectangle
        self.roi_preview_item = None
        self._pending_roi_preview = None
        self._roi_preview_after_id = None
        self._overlay_update_interval_ms = 33

        # Lever-Arm attributes
        self.lever_arm_mode = 'Double'
        self.lever_arm_points_list = []
        self.sd_bias = None
        self.lever_arm_coefficients = None


        # Create Menu Bar
        self.menubar = ttk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # Create File Menu
        self.file_menu = ttk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(
            label="Undo",
            command=self.undo_last_data_operation,
            state=tk.DISABLED
        )
        self.undo_menu_index = self.file_menu.index('end')
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save whole data as NumPy array", command=self.save_file)
        self.file_menu.add_command(label="Save displayed data as NumPy array", command=self.save_data)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # Create Data Display Menu
        self.data_menu = ttk.Menu(self.menubar, tearoff=0)
        self.data_menu.add_command(label="Interpolation Settings", command=self.open_interpolation_window)
        self.data_menu.add_command(label="Gaussian Filter", command=self.open_gaussian_filter_window)
        self.data_menu.add_command(label="Cut Data to ROI", command=self.open_roi_data_cut_window)
        self.data_menu.add_command(label="Background Subtraction", command=self.open_background_subtraction_window)
        self.data_menu.add_command(label="Rename and Scale Data and Axis", command=self.open_data_axis_transform)
        self.data_menu.add_separator()
        self.data_menu.add_command(
            label='Re-arrange Scan Axes…',
            command=self.open_rearrange_scan_axes_window,
            state=(
                tk.NORMAL
                if self._scan_axis_rearrangement_available()
                else tk.DISABLED
            ),
        )
        self.rearrange_scan_axes_menu_index = self.data_menu.index('end')
        self.menubar.add_cascade(label="Displayed Data", menu=self.data_menu)

        # Create Tool Menu
        self.tool_menu = ttk.Menu(self.menubar, tearoff=0)
        self.tool_menu.add_command(label="Derivative along Axis", command=self.open_derivative_window)
        self.tool_menu.add_command(label="Savitzky-Golay Filter", command=self.open_savitzky_golay_filter_window)
        self.tool_menu.add_command(label="Norm of Gradient", command=self.apply_sum_of_gradient)
        self.tool_menu.add_command(label="2-D FFT on Data", command=self.apply_2d_fft)
        self.tool_menu.add_command(label="Draw Lines", command=self.open_draw_lines_window)
        if self.contains_traces:
            self.tool_menu.add_command(label="Correct Trace Fourier spectrum", command=self.open_fft_trace_correction_window)
            self.tool_menu.add_command(label="Fit Traces", command=self.fit_traces)
        self.menubar.add_cascade(label="Tools", menu=self.tool_menu)
        self.tool_menu.add_command(label="2-D FFT Filter", command=self.open_2d_fft_filter)

        self.style_menu = ttk.Menu(self.menubar, tearoff=0)
        self.style_menu.add_command(
            label='Plot Colors…',
            command=self.open_plot_style_settings,
        )
        self.menubar.add_cascade(label='Style', menu=self.style_menu)

        # Create Help Menu
        self.help_menu = ttk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        # Create a figure and axis for plotting
        if figure is None or ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.figure, self.ax = figure, ax

        # The crosshair linecuts use independent figures and Tk canvases. This
        # keeps their frequent redraws away from the comparatively expensive
        # color-map canvas.
        self.horizontal_linecut_figure = Figure(figsize=(6, 1.2), dpi=100)
        self.ax_hline = self.horizontal_linecut_figure.add_subplot(111)
        self.vertical_linecut_figure = Figure(figsize=(1.6, 5), dpi=100)
        self.ax_vline = self.vertical_linecut_figure.add_subplot(111)
        self.ax_vline.set_visible(False)
        self.ax_hline.set_visible(False)

        self.plot_area_frame = ttk.Frame(self.root)
        self.horizontal_linecut_frame = ttk.Frame(
            self.plot_area_frame,
            height=120,
        )
        self.map_frame = ttk.Frame(self.plot_area_frame)
        self.vertical_linecut_frame = ttk.Frame(
            self.plot_area_frame,
            width=160,
        )
        self.plot_area_frame.columnconfigure(0, weight=1)
        self.plot_area_frame.columnconfigure(1, weight=0)
        self.plot_area_frame.rowconfigure(0, weight=0)
        self.plot_area_frame.rowconfigure(1, weight=1)
        self.horizontal_linecut_frame.grid(
            row=0,
            column=0,
            sticky=tk.EW,
        )
        self.map_frame.grid(row=1, column=0, sticky=tk.NSEW)
        self.vertical_linecut_frame.grid(row=1, column=1, sticky=tk.NS)
        self.horizontal_linecut_frame.grid_remove()
        self.vertical_linecut_frame.grid_remove()

        # Create a new figure for the histogram
        self.histogram_fig, self.histogram_ax = plt.subplots(figsize=(3.5, 1.5))
        self.histogram_ax.set_yticklabels([])
        self.histogram_ax.set_xticklabels([])
        self.picked_line = None

        # Define interactive button options
        self.colormaps = list(AVAILABLE_COLORMAPS)
        self.bg_methods = ['Polynomial', 'Median Difference', 'Mean of Lines', 'Relation Parameters',
                           'Subtract Trace Average']
        self.relation_parameter_entry_list = []
        self.roi_cut_entry_list = []
        self.drawn_lines_list = []
        self.linecut_settings_list = []
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.map_frame)
        self.horizontal_linecut_canvas = FigureCanvasTkAgg(
            self.horizontal_linecut_figure,
            master=self.horizontal_linecut_frame,
        )
        self.vertical_linecut_canvas = FigureCanvasTkAgg(
            self.vertical_linecut_figure,
            master=self.vertical_linecut_frame,
        )
        configure_transparent_matplotlib_canvas(self.figure, self.canvas)
        configure_transparent_matplotlib_canvas(
            self.horizontal_linecut_figure,
            self.horizontal_linecut_canvas,
            opaque_for_blitting=True,
        )
        configure_transparent_matplotlib_canvas(
            self.vertical_linecut_figure,
            self.vertical_linecut_canvas,
            opaque_for_blitting=True,
        )
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.update()
        self.crosshair_button = ttk.Button(self.toolbar, text='Crosshair', command=self.toggle_crosshair, bootstyle='info outline')
        self.crosshair_button.pack(side=tk.LEFT)
        self.fast_crosshair_var = tk.BooleanVar(master=self.root, value=False)
        self.interpol_button = ttk.Button(self.toolbar, text='Interpolation', command=self.toggle_interpolation, bootstyle='info outline')
        self.interpol_button.pack(side=tk.LEFT)
        self.roi_button = ttk.Button(self.toolbar, text='ROI', command=self.toggle_roi, bootstyle='info outline')
        self.roi_button.pack(side=tk.LEFT)
        self.crosshair_horizontal_shadow_item = None
        self.crosshair_vertical_shadow_item = None
        self.crosshair_horizontal_item = None
        self.crosshair_vertical_item = None
        self._crosshair_overlay_visible = False
        self._last_crosshair_canvas_coordinates = None
        self._last_crosshair_indices = None
        self._pending_crosshair_position = None
        self._crosshair_update_after_id = None
        self._crosshair_update_interval_ms = 33
        self._display_resize_after_id = None
        self._last_display_sampling_signature = None
        self._pending_linecut_request = None
        self._current_linecut_request = None
        self._linecut_update_after_id = None
        self._linecut_update_interval_ms = 33
        self._linecut_axes_signature = None
        self._horizontal_linecut_background = None
        self._vertical_linecut_background = None
        self.horizontal_linecut_artist = None
        self.horizontal_linecut_cursor = None
        self.vertical_linecut_artist = None
        self.vertical_linecut_cursor = None
        self.auto_scale_var = tk.BooleanVar(value=True)  # Default to True (auto-scaling
        self.crosshair_enabled = False
        self.interpolation_enabled = False
        self.invert_enabled = False
        self.x_sweep_mode = tk.StringVar(master=self.root, value='normal')
        self.freeze_linecut = False
        self.linecut_position = None
        self.drawing_line = False
        self.current_line = None
        self.line_preview_item = None
        self._pending_line_preview = None
        self._line_preview_after_id = None
        self.click_cid = None
        self.move_cid = None
        self.motion_cid = None
        self.crosshair_motion_cid = None
        self.release_cid = None
        self.start_point = None # Added by Nico Reinders for error handling in 2D FFT filter

        # Create a custom style for label frames to make them smaller
        small_label_frame_style = ttk.Style()
        small_label_frame_style.configure('Small.TLabelframe', font=('Arial', 8))  # Adjust font size here
        self.parameter_frame = ttk.Frame(self.root)
        self.parameter_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.parameter_labels = []
        self.parameter_values = []
        self.parameter_comboboxes = []
        self.parameter_selector_frames = []
        self.display_values_list = []
        self._rebuild_parameter_selectors()
        self.frame2 = ttk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT, padx=5, pady=5)

        # Create a combobox for colormap selection
        self.colormap_combobox = ttk.Combobox(self.frame2, values=self.colormaps, state='readonly', width=10)
        self.colormap_combobox.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.colormap_combobox.set(self.plot_style['preferred_colormap'])
        self.colormap_combobox.bind(
            '<<ComboboxSelected>>',
            self._on_preferred_colormap_selected,
        )

        # Create a combobox for data selection
        self.data_combobox = ttk.Combobox(self.frame2, values=self.name_data, state='readonly', width=20)
        self.data_combobox.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.data_combobox.set(self.name_data[0])  # Set the default colormap

        # Create a Frame for the "Plot" buttons
        self.button_frame = ttk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT, padx=5, pady=5)

        self.auto_scale_check = ttk.Checkbutton(
            self.frame2,
            text="Auto-scale plot bounds",
            variable=self.auto_scale_var
        )
        self.auto_scale_check.pack(side=tk.TOP, pady=2)

        self.fast_crosshair_checkbutton = ttk.Checkbutton(
            self.frame2,
            text='Fast crosshair',
            variable=self.fast_crosshair_var,
            command=self._on_fast_crosshair_changed,
        )
        self.fast_crosshair_checkbutton.pack(
            side=tk.TOP,
            anchor=tk.W,
            pady=2,
        )

        self.x_sweep_frame = ttk.LabelFrame(
            self.frame2,
            text='X sweep direction',
        )
        self.x_sweep_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        self.normal_x_sweep_radio = ttk.Radiobutton(
            self.x_sweep_frame,
            text='Normal',
            variable=self.x_sweep_mode,
            value='normal',
            command=self._on_x_sweep_mode_changed,
        )
        self.normal_x_sweep_radio.pack(anchor=tk.W, padx=4)
        self.alternating_x_sweep_radio = ttk.Radiobutton(
            self.x_sweep_frame,
            text='Alternating',
            variable=self.x_sweep_mode,
            value='alternating',
            command=self._on_x_sweep_mode_changed,
            state=(tk.DISABLED if self.single_axis_measurement else tk.NORMAL),
        )
        self.alternating_x_sweep_radio.pack(anchor=tk.W, padx=4)
        self._update_x_sweep_control_state()

        # Create a "Reset Plot" button inside the button frame
        self.reset_plot_button = ttk.Button(self.frame2, text="Plot", command=self.plot_data)
        self.reset_plot_button.pack(side=tk.TOP, fill=tk.X)

        # Create an "Update Plot" button inside the button frame
        self.update_plot_button = ttk.Button(self.frame2, text="Update Plot", command=self.update_plot)
        self.update_plot_button.pack(side=tk.TOP, fill=tk.X)

        # Create a ''Invert Axes'' Button
        self.invert_button = ttk.Button(self.frame2, text="Invert Axes", command=self.toggle_invert)
        self.invert_button.pack(side=tk.BOTTOM)

        # Create Histogram of displayed data
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.horizontal_linecut_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )
        self.vertical_linecut_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.plot_area_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=self.frame2)
        configure_transparent_matplotlib_canvas(
            self.histogram_fig,
            self.histogram_canvas,
        )
        self.histogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.N)
        self.canvas.get_tk_widget().bind(
            '<Configure>',
            self._on_map_canvas_configure,
            add='+',
        )
        self.canvas.mpl_connect('draw_event', self._on_main_canvas_draw)
        self.horizontal_linecut_canvas.mpl_connect(
            'draw_event',
            self._on_horizontal_linecut_draw,
        )
        self.vertical_linecut_canvas.mpl_connect(
            'draw_event',
            self._on_vertical_linecut_draw,
        )
        self.plot_data()
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def _scan_axis_description(self, axis_index):
        """Return an unambiguous label for a logical scan-grid dimension."""
        acquisition_axis_number = self.scan_axis_count - int(axis_index)
        return (
            f'{self.scan_axis_names[axis_index]} '
            f'(scan axis {acquisition_axis_number})'
        )

    def _scan_coordinate_grid(self, axis_index):
        """Reshape one stored scan coordinate into the logical N-D grid."""
        coordinate_array = np.asarray(
            list(reversed(self.data.measure_axis))[axis_index]
        )
        return coordinate_array.swapaxes(0, 1).reshape(self.scan_shape)

    def _selected_additional_index_map(self):
        """Return the current fixed index for every non-displayed scan axis."""
        selected_indices = {}
        for list_index, axis_index in enumerate(
            self.additional_scan_axis_indices
        ):
            combobox = self.parameter_comboboxes[list_index]
            selected_index = combobox.current()
            if selected_index < 0:
                try:
                    selected_value = float(combobox.get())
                    selected_index = self.display_values_list[list_index].index(
                        selected_value
                    )
                except (ValueError, IndexError):
                    selected_index = 0
            selected_indices[axis_index] = selected_index
        return selected_indices

    def _additional_axis_display_values(self, axis_index):
        """Return representative coordinate values for a fixed-axis selector."""
        coordinate_grid = self._scan_coordinate_grid(axis_index)
        display_values = []
        for selected_index in range(self.scan_shape[axis_index]):
            selection = [0] * self.scan_axis_count
            selection[axis_index] = selected_index
            display_values.append(float(coordinate_grid[tuple(selection)]))
        return display_values

    def _rebuild_parameter_selectors(self, preserved_indices=None):
        """Recreate selectors for all axes that are not assigned to X or Y."""
        preserved_indices = dict(preserved_indices or {})
        for selector_frame in self.parameter_selector_frames:
            selector_frame.destroy()

        self.parameter_labels = [
            self.scan_axis_names[axis_index]
            for axis_index in self.additional_scan_axis_indices
        ]
        self.parameter_values = [
            range(self.scan_shape[axis_index])
            for axis_index in self.additional_scan_axis_indices
        ]
        self.parameter_comboboxes = []
        self.parameter_selector_frames = []
        self.display_values_list = []

        for axis_index, label in zip(
            self.additional_scan_axis_indices,
            self.parameter_labels,
        ):
            display_values = self._additional_axis_display_values(axis_index)
            self.display_values_list.append(display_values)
            label_frame = ttk.LabelFrame(self.parameter_frame, text=label)
            label_frame.pack(padx=5, pady=5)
            self.parameter_selector_frames.append(label_frame)
            combobox = ttk.Combobox(
                label_frame,
                values=display_values,
                state='readonly',
                width=10,
            )
            combobox.pack(side=tk.BOTTOM, padx=5, pady=5)
            selected_index = min(
                max(int(preserved_indices.get(axis_index, 0)), 0),
                len(display_values) - 1,
            )
            if display_values:
                combobox.current(selected_index)
            self.parameter_comboboxes.append(combobox)

    def _selected_signal_grid(self):
        """Return the selected stored or derived signal as a logical N-D grid."""
        selected_name = self.data_combobox.get()
        stored_channel_count = len(self.data.measure_data)
        if selected_name in self.name_data[:stored_channel_count]:
            channel_index = self.name_data.index(selected_name)
            return np.asarray(
                self.data.measure_data[channel_index]
            ).swapaxes(0, 1).reshape(self.scan_shape)

        fit_results = getattr(self, 'fit_results_dict', None)
        if fit_results is None:
            traces_fitter = getattr(self, 'traces_fitter', None)
            fit_results = getattr(traces_fitter, 'fit_results_dict', {})
        if selected_name in fit_results:
            return np.asarray(fit_results[selected_name]).reshape(
                self.scan_shape
            )

        if selected_name in ('Tunneling rates in', 'Tunneling rates out'):
            if not self.loaded:
                self.data.set_traces()
                self.data.set_traces_dt()
                self.traces = self.data.traces
                self.times = self.data.get_trace_axis(
                    len(self.traces[0][0])
                )
                self.loaded = True
            if not self.calculated:
                _, self.gamma_up, self.gamma_down = get_t_rates(
                    self.traces,
                    self.times,
                )
                self.calculated = True
            rates = (
                self.gamma_up
                if selected_name == 'Tunneling rates in'
                else self.gamma_down
            )
            return np.asarray(rates).reshape(self.scan_shape)

        raise ValueError(f'No data array is available for {selected_name!r}.')

    def _trace_order_scan_grid(self):
        """Return trace indices arranged like the logical scan dimensions."""
        return reshape_trace_order_to_scan_grid(
            self.data.trace_order,
            self.data.measure_dim,
        )

    def _trace_axis_map_arrays(self):
        """Return trace time, selected scan coordinate, and trace amplitudes."""
        if not self.supports_trace_axis_map:
            raise ValueError(
                'Trace-axis maps are available only in the trace plotter.'
            )
        trace_reference = getattr(self.data, 'trace_reference', None)
        if trace_reference is None:
            raise ValueError('The measurement does not expose trace data.')
        trace_length = int(trace_reference.shape[0])
        trace_x_values = self.data.get_trace_axis(trace_length)
        return build_trace_axis_map(
            trace_reference,
            self._trace_order_scan_grid(),
            trace_x_values,
            self._scan_coordinate_grid(self.y_scan_axis_index),
            self.y_scan_axis_index,
            self._selected_additional_index_map(),
        )

    def _set_trace_axis_data_selector_state(self, enabled):
        """Show that trace amplitudes replace the normal channel selection."""
        if not hasattr(self, 'data_combobox'):
            return
        trace_value_label = getattr(
            self,
            'trace_ylabel',
            'Trace amplitude',
        )
        if enabled:
            if self._data_selection_before_trace_axis_map is None:
                self._data_selection_before_trace_axis_map = (
                    self.data_combobox.get()
                )
            display_values = list(self.name_data)
            if trace_value_label not in display_values:
                display_values.append(trace_value_label)
            self.data_combobox.configure(
                values=display_values,
                state='disabled',
            )
            self.data_combobox.set(trace_value_label)
            return

        previous_selection = self._data_selection_before_trace_axis_map
        self._data_selection_before_trace_axis_map = None
        self.data_combobox.configure(values=self.name_data, state='readonly')
        if previous_selection in self.name_data:
            self.data_combobox.set(previous_selection)
        elif self.name_data:
            self.data_combobox.set(self.name_data[0])

    def _slice_current_scan_grid(self, grid):
        """Slice a full scan grid using the current axis roles and selectors."""
        return slice_scan_grid_for_axes(
            grid,
            self.x_scan_axis_index,
            self.y_scan_axis_index,
            self._selected_additional_index_map(),
        )

    def _default_scan_axis_roles(self):
        return self.scan_axis_count - 1, self.scan_axis_count - 2

    def _scan_axis_rearrangement_available(self):
        """Return whether this plotter has at least one alternate map plane."""
        return (
            self.scan_axis_count > 2
            or (
                self.supports_trace_axis_map
                and self.scan_axis_count >= 2
            )
        )

    def _update_x_sweep_control_state(self):
        """Allow serpentine correction only for the acquisition X/Y plane."""
        if not hasattr(self, 'alternating_x_sweep_radio'):
            return
        default_x_axis, default_y_axis = self._default_scan_axis_roles()
        correction_available = (
            not self.single_axis_measurement
            and not self.trace_axis_map_mode
            and self.x_scan_axis_index == default_x_axis
            and self.y_scan_axis_index == default_y_axis
        )
        if not correction_available:
            self.x_sweep_mode.set('normal')
        self.alternating_x_sweep_radio.configure(
            state=(tk.NORMAL if correction_available else tk.DISABLED)
        )

    def open_rearrange_scan_axes_window(self):
        """Choose the two displayed dimensions of a multidimensional scan."""
        if not self._scan_axis_rearrangement_available():
            return None
        existing_window = getattr(self, 'rearrange_scan_axes_window', None)
        if existing_window is not None and existing_window.winfo_exists():
            existing_window.lift()
            existing_window.focus_force()
            return existing_window

        dialog = ttk.Toplevel(self.root)
        self.rearrange_scan_axes_window = dialog
        dialog.title('Re-arrange Scan Axes')
        dialog.geometry('560x300')
        dialog.resizable(False, False)
        dialog.transient(self.root)
        content = ttk.Frame(dialog, padding=14)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(1, weight=1)

        rearrange_instructions = (
            'Choose two different scan axes for the map. Every remaining '
            'scan axis becomes a fixed-value selector beside the plot.'
        )
        if self.supports_trace_axis_map:
            rearrange_instructions = (
                'Choose the displayed map axes. Every remaining scan axis '
                'becomes a fixed-value selector beside the plot. Select '
                'Trace samples as X to use the stored trace amplitudes as '
                'the map values.'
            )
        ttk.Label(
            content,
            text=rearrange_instructions,
            wraplength=520,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 12))

        axis_descriptions = [
            self._scan_axis_description(axis_index)
            for axis_index in range(self.scan_axis_count)
        ]
        trace_axis_description = (
            f'{getattr(self, "trace_xlabel", "Trace X axis")} '
            '(trace samples)'
        )
        x_axis_descriptions = list(axis_descriptions)
        if self.supports_trace_axis_map:
            x_axis_descriptions.append(trace_axis_description)
        x_axis_variable = tk.StringVar(
            master=dialog,
            value=(
                trace_axis_description
                if self.trace_axis_map_mode
                else axis_descriptions[self.x_scan_axis_index]
            ),
        )
        y_axis_variable = tk.StringVar(
            master=dialog,
            value=axis_descriptions[self.y_scan_axis_index],
        )
        additional_axes_variable = tk.StringVar(master=dialog)

        ttk.Label(content, text='Displayed X axis:').grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=6
        )
        x_axis_combobox = ttk.Combobox(
            content,
            textvariable=x_axis_variable,
            values=x_axis_descriptions,
            state='readonly',
            width=34,
        )
        x_axis_combobox.grid(row=1, column=1, sticky=tk.EW, pady=6)
        ttk.Label(content, text='Displayed Y axis:').grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=6
        )
        y_axis_combobox = ttk.Combobox(
            content,
            textvariable=y_axis_variable,
            values=axis_descriptions,
            state='readonly',
            width=34,
        )
        y_axis_combobox.grid(row=2, column=1, sticky=tk.EW, pady=6)
        ttk.Label(content, text='Additional selectors:').grid(
            row=3, column=0, sticky=tk.NW, padx=(0, 10), pady=6
        )
        ttk.Label(
            content,
            textvariable=additional_axes_variable,
            wraplength=350,
        ).grid(row=3, column=1, sticky=tk.W, pady=6)

        def selected_axis_roles():
            selected_x = x_axis_variable.get()
            x_axis_role = (
                TRACE_SAMPLE_AXIS_ROLE
                if selected_x == trace_axis_description
                else axis_descriptions.index(selected_x)
            )
            return (
                x_axis_role,
                axis_descriptions.index(y_axis_variable.get()),
            )

        def update_additional_axes(_event=None):
            x_axis_role, y_axis_index = selected_axis_roles()
            displayed_scan_axes = {y_axis_index}
            if x_axis_role != TRACE_SAMPLE_AXIS_ROLE:
                displayed_scan_axes.add(x_axis_role)
            additional_axes_variable.set(', '.join(
                axis_descriptions[axis_index]
                for axis_index in range(self.scan_axis_count)
                if axis_index not in displayed_scan_axes
            ) or 'None')

        x_axis_combobox.bind('<<ComboboxSelected>>', update_additional_axes)
        y_axis_combobox.bind('<<ComboboxSelected>>', update_additional_axes)
        update_additional_axes()

        def apply_roles():
            x_axis_role, y_axis_index = selected_axis_roles()
            if x_axis_role == y_axis_index:
                messagebox.showerror(
                    'Invalid Scan Axes',
                    'The displayed X and Y axes must be different.',
                    parent=dialog,
                )
                return
            self.apply_scan_axis_roles(x_axis_role, y_axis_index)
            dialog.destroy()

        def restore_default_roles():
            default_x_axis, default_y_axis = self._default_scan_axis_roles()
            x_axis_variable.set(axis_descriptions[default_x_axis])
            y_axis_variable.set(axis_descriptions[default_y_axis])
            update_additional_axes()

        button_frame = ttk.Frame(content)
        button_frame.grid(
            row=4, column=0, columnspan=2, sticky=tk.EW, pady=(18, 0)
        )
        ttk.Button(
            button_frame,
            text='Restore Default',
            command=restore_default_roles,
            bootstyle='secondary',
        ).pack(side=tk.LEFT)
        ttk.Button(
            button_frame,
            text='Cancel',
            command=dialog.destroy,
            bootstyle='secondary',
        ).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(
            button_frame,
            text='Apply',
            command=apply_roles,
            bootstyle='primary',
        ).pack(side=tk.RIGHT)
        return dialog

    def apply_scan_axis_roles(self, x_axis_index, y_axis_index):
        """Assign scan or trace dimensions and rebuild the selected map."""
        trace_axis_selected = x_axis_index == TRACE_SAMPLE_AXIS_ROLE
        y_axis_index = int(y_axis_index)
        if not self._scan_axis_rearrangement_available():
            raise ValueError(
                'Scan-axis rearrangement is unavailable for this dataset.'
            )
        if not 0 <= y_axis_index < self.scan_axis_count:
            raise ValueError('A selected scan axis is out of range.')
        if trace_axis_selected:
            if not self.supports_trace_axis_map:
                raise ValueError(
                    'Trace samples can be selected only in the trace plotter.'
                )
        else:
            x_axis_index = int(x_axis_index)
            if x_axis_index == y_axis_index:
                raise ValueError(
                    'The displayed X and Y axes must be different.'
                )
            if not 0 <= x_axis_index < self.scan_axis_count:
                raise ValueError('A selected scan axis is out of range.')

        preserved_indices = self._selected_additional_index_map()
        self.trace_axis_map_mode = trace_axis_selected
        self.x_scan_axis_index = (
            None if trace_axis_selected else x_axis_index
        )
        self.y_scan_axis_index = y_axis_index
        self.additional_scan_axis_indices = [
            axis_index
            for axis_index in range(self.scan_axis_count)
            if axis_index != y_axis_index
            and (trace_axis_selected or axis_index != x_axis_index)
        ]
        self.num_dimensions = len(self.additional_scan_axis_indices)
        self._rebuild_parameter_selectors(preserved_indices)
        self._set_trace_axis_data_selector_state(trace_axis_selected)
        self._update_x_sweep_control_state()

        self.x_index = 0
        self.y_index = 0
        self.invert_enabled = False
        if hasattr(self, 'trace_x_index'):
            self.trace_x_index = 0
        if hasattr(self, 'trace_y_index'):
            self.trace_y_index = 0
        self.freeze_linecut = False
        self.linecut_position = None
        self._last_crosshair_indices = None
        self._last_crosshair_canvas_coordinates = None
        self._pending_crosshair_position = None
        self._pending_linecut_request = None
        self._current_linecut_request = None
        self._linecut_axes_signature = None
        self._set_crosshair_overlay_state('hidden')
        if self.roi_mode:
            self.toggle_roi()
        self.roi_corners = []
        self._hide_roi_preview()
        self._remove_current_roi_patch()
        self._cancel_in_progress_line_drawing()
        self.drawn_lines_list = []
        self.linecut_settings_list = []
        if hasattr(self, 'lines_listbox'):
            self.update_lines_listbox()
        if hasattr(self, 'editing_line'):
            self.editing_line = False
            self.editing_line_index = None
        self.plot_data()
        if self.crosshair_enabled:
            self.refresh_crosshair()

    def _alternating_x_sweep_enabled(self):
        """Return whether the displayed map should undo serpentine X scans."""
        default_x_axis, default_y_axis = self._default_scan_axis_roles()
        return (
            not self.single_axis_measurement
            and not self.trace_axis_map_mode
            and self.x_scan_axis_index == default_x_axis
            and self.y_scan_axis_index == default_y_axis
            and self.x_sweep_mode.get() == 'alternating'
        )

    def _current_plot_line_color(self):
        """Return the shared crosshair-linecut and histogram color."""
        return resolve_plot_color(
            self.plot_style['crosshair_histogram_color']
        )

    def _commit_plot_style(self, plot_style):
        """Persist a style through the application or apply it locally."""
        normalized_style = normalize_plot_style(plot_style)
        if self.plot_style_change_callback is None:
            self.apply_plot_style(normalized_style)
            return normalized_style
        try:
            saved_style = self.plot_style_change_callback(normalized_style)
        except (OSError, ValueError) as error:
            self.apply_plot_style(normalized_style)
            messagebox.showwarning(
                'Plot Style Not Saved',
                f'The plot style was applied but could not be remembered:\n{error}',
                parent=self.root,
            )
            return normalized_style
        return normalize_plot_style(saved_style or normalized_style)

    def open_plot_style_settings(self):
        """Edit persistent plot colors from inside the map plotter."""
        return open_plot_style_dialog(
            self.root,
            self.plot_style,
            self._commit_plot_style,
        )

    def _on_preferred_colormap_selected(self, _event=None):
        selected_style = dict(self.plot_style)
        selected_style['preferred_colormap'] = self.colormap_combobox.get()
        self._commit_plot_style(selected_style)

    def _on_extracted_linecut_cycle_selected(self, cycle_name):
        selected_style = dict(self.plot_style)
        selected_style['extracted_linecut_color_cycle'] = cycle_name
        self._commit_plot_style(selected_style)

    def apply_plot_style(self, plot_style):
        """Apply validated plotting colors to all live artists in this window."""
        normalized_style = normalize_plot_style(plot_style)
        previous_colormap = self.plot_style.get('preferred_colormap')
        self.plot_style = normalized_style

        if hasattr(self, 'colormap_combobox'):
            self.colormap_combobox.set(
                normalized_style['preferred_colormap']
            )

        line_color = self._current_plot_line_color()
        for artist_name in (
            'horizontal_linecut_artist',
            'vertical_linecut_artist',
        ):
            artist = getattr(self, artist_name, None)
            if artist is not None:
                artist.set_color(line_color)
        self._apply_crosshair_overlay_color()

        if (
            hasattr(self, 'sliced_data')
            and self.sliced_data is not None
            and hasattr(self, 'histogram_canvas')
        ):
            self.update_histogramm()

        if (
            previous_colormap != normalized_style['preferred_colormap']
            and getattr(self, 'display_sliced_data', None) is not None
        ):
            self.update_pcolormesh(self.vmin, self.vmax)

        if (
            self.crosshair_enabled
            and self._current_linecut_request is not None
        ):
            self._blit_current_linecuts()

        linecut_plotter = getattr(self, 'linecut_plotter', None)
        if linecut_plotter is not None:
            linecut_plotter.set_color_cycle(
                normalized_style['extracted_linecut_color_cycle']
            )
        traces_fitter = getattr(self, 'traces_fitter', None)
        if traces_fitter is not None:
            traces_fitter.set_plot_style(normalized_style)
        self._after_plot_style_applied()
        return normalized_style

    def _after_plot_style_applied(self):
        """Hook for plotters that own additional color-dependent figures."""
        pass

    def _on_x_sweep_mode_changed(self):
        """Reload and redraw the selected channel with the chosen X ordering."""
        self.x_index = 0
        self.y_index = 0
        self.freeze_linecut = False
        self.linecut_position = None
        if hasattr(self, 'trace_x_index'):
            self.trace_x_index = 0
        if hasattr(self, 'trace_y_index'):
            self.trace_y_index = 0
        self.plot_data()

    def _coerce_calculation_arrays_to_float64(self):
        """Keep the authoritative map coordinates and signal at full precision."""
        for attribute_name in ('X', 'Y', 'sliced_data'):
            array = np.asarray(getattr(self, attribute_name))
            if np.iscomplexobj(array):
                if np.any(np.imag(array) != 0):
                    raise ValueError(
                        f'{attribute_name} contains complex values and cannot '
                        'be represented as a real-valued map.'
                    )
                array = np.real(array)
            setattr(
                self,
                attribute_name,
                array.astype(np.float64, copy=False),
            )

    @staticmethod
    def _downsample_indices(point_count, target_count):
        """Return regularly spaced source indices including the final point."""
        point_count = int(point_count)
        target_count = max(2, int(target_count))
        if point_count <= target_count:
            return np.arange(point_count, dtype=int)
        step = max(1, int(np.ceil(point_count / target_count)))
        indices = np.arange(0, point_count, step, dtype=int)
        if indices[-1] != point_count - 1:
            indices = np.append(indices, point_count - 1)
        return indices

    def _map_canvas_pixel_size(self):
        """Return a usable pixel target before and after Tk maps the widget."""
        canvas_widget = self.canvas.get_tk_widget()
        width = int(canvas_widget.winfo_width())
        height = int(canvas_widget.winfo_height())
        if width <= 1:
            width = max(64, int(self.figure.bbox.width))
        if height <= 1:
            height = max(64, int(self.figure.bbox.height))
        return width, height

    def _prepare_display_arrays(self, force=False):
        """Build a canvas-sized float32 view of the float64 calculation grid."""
        self._coerce_calculation_arrays_to_float64()
        canvas_width, canvas_height = self._map_canvas_pixel_size()
        row_indices = self._downsample_indices(
            self.sliced_data.shape[0],
            canvas_height,
        )
        column_indices = self._downsample_indices(
            self.sliced_data.shape[1],
            canvas_width,
        )
        signature = (
            self.sliced_data.shape,
            int(row_indices.size),
            int(column_indices.size),
            int(row_indices[1] - row_indices[0]) if row_indices.size > 1 else 0,
            int(column_indices[1] - column_indices[0])
            if column_indices.size > 1 else 0,
        )
        if not force and signature == self._last_display_sampling_signature:
            return False

        selection = np.ix_(row_indices, column_indices)
        self.display_row_indices = row_indices
        self.display_column_indices = column_indices
        self.display_X = self.X[selection].astype(np.float32, copy=False)
        self.display_Y = self.Y[selection].astype(np.float32, copy=False)
        self.display_sliced_data = self.sliced_data[selection].astype(
            np.float32,
            copy=False,
        )
        center_column = self.Y.shape[1] // 2
        self._crosshair_y_reference = np.array(
            self.Y[:, center_column],
            dtype=np.float64,
            copy=True,
        )
        for row_index in np.flatnonzero(
            ~np.isfinite(self._crosshair_y_reference)
        ):
            finite_row_values = self.Y[row_index][
                np.isfinite(self.Y[row_index])
            ]
            if finite_row_values.size:
                self._crosshair_y_reference[row_index] = finite_row_values[0]
        self._last_display_sampling_signature = signature
        return True

    def _on_map_canvas_configure(self, event):
        """Debounce adaptive display resampling after a canvas resize."""
        if getattr(self, 'sliced_data', None) is None:
            return
        if self._display_resize_after_id is not None:
            self.root.after_cancel(self._display_resize_after_id)
        self._display_resize_after_id = self.root.after(
            180,
            self._refresh_display_after_resize,
        )

    def _on_main_canvas_draw(self, _event):
        """Restore native overlays above the refreshed Matplotlib bitmap."""
        self.root.after_idle(self._restore_map_overlays_after_draw)

    def _raise_map_overlay_items(self):
        """Keep all native interaction previews above the TkAgg photo image."""
        tk_canvas = self.canvas.get_tk_widget()
        for item in (
            getattr(self, 'roi_preview_item', None),
            getattr(self, 'line_preview_item', None),
            getattr(self, 'crosshair_horizontal_shadow_item', None),
            getattr(self, 'crosshair_vertical_shadow_item', None),
            getattr(self, 'crosshair_horizontal_item', None),
            getattr(self, 'crosshair_vertical_item', None),
        ):
            if item is not None:
                tk_canvas.tag_raise(item)

    def _restore_map_overlays_after_draw(self):
        self._raise_map_overlay_items()
        if self.crosshair_enabled:
            self.refresh_crosshair()

    def _refresh_display_after_resize(self):
        self._display_resize_after_id = None
        if getattr(self, 'sliced_data', None) is None:
            return
        if self._prepare_display_arrays(force=False):
            self._draw_main_map(self.vmin, self.vmax)

    def _capture_displayed_data_state(self, operation_name):
        """Return an independent snapshot of the currently displayed data."""
        if any(getattr(self, name, None) is None for name in ('X', 'Y', 'sliced_data')):
            return None

        return {
            'operation': operation_name,
            'X': np.array(self.X, copy=True),
            'Y': np.array(self.Y, copy=True),
            'sliced_data': np.array(self.sliced_data, copy=True),
            'nan_mask': np.array(getattr(self, 'nan_mask', []), copy=True),
            'name_data_x_axis': self.name_data_x_axis,
            'name_data_y_axis': self.name_data_y_axis,
            'name_data_z': self.name_data_z,
            'xlim': tuple(self.xlim) if getattr(self, 'xlim', None) is not None else None,
            'ylim': tuple(self.ylim) if getattr(self, 'ylim', None) is not None else None,
            'vmin': getattr(self, 'vmin', None),
            'vmax': getattr(self, 'vmax', None),
            'auto_scale_factor': getattr(self, 'auto_scale_factor', None),
        }

    @contextmanager
    def data_operation(self, operation_name):
        """Save the pre-operation array state after an operation succeeds."""
        state = self._capture_displayed_data_state(operation_name)
        try:
            yield
        except Exception:
            raise
        else:
            if state is not None:
                self.data_operation_history.append(state)
                self._update_undo_menu_state()

    def _update_undo_menu_state(self):
        """Enable Undo only when a displayed-data snapshot is available."""
        if not hasattr(self, 'file_menu') or not hasattr(self, 'undo_menu_index'):
            return

        if self.data_operation_history:
            operation_name = self.data_operation_history[-1]['operation']
            self.file_menu.entryconfigure(
                self.undo_menu_index,
                label=f"Undo {operation_name}",
                state=tk.NORMAL
            )
        else:
            self.file_menu.entryconfigure(
                self.undo_menu_index,
                label="Undo",
                state=tk.DISABLED
            )

    def undo_last_data_operation(self):
        """Restore and remove the newest displayed-data snapshot."""
        if not self.data_operation_history:
            self._update_undo_menu_state()
            return

        state = self.data_operation_history.pop()
        self.X = np.array(state['X'], copy=True)
        self.Y = np.array(state['Y'], copy=True)
        self.sliced_data = np.array(state['sliced_data'], copy=True)
        self.nan_mask = np.array(state['nan_mask'], copy=True)
        self.name_data_x_axis = state['name_data_x_axis']
        self.name_data_y_axis = state['name_data_y_axis']
        self.name_data_z = state['name_data_z']
        self.xlim = state['xlim']
        self.ylim = state['ylim']
        self.vmin = state['vmin']
        self.vmax = state['vmax']
        self.auto_scale_factor = state['auto_scale_factor']
        self.invert_enabled = False

        row_count, column_count = self.sliced_data.shape[:2]
        self.x_index = min(self.x_index, column_count - 1)
        self.y_index = min(self.y_index, row_count - 1)
        if hasattr(self, 'trace_x_index'):
            self.trace_x_index = min(self.trace_x_index, column_count - 1)
        if hasattr(self, 'trace_y_index'):
            self.trace_y_index = min(self.trace_y_index, row_count - 1)

        self._update_undo_menu_state()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)
        self._after_data_operation_undo()

    def _after_data_operation_undo(self):
        """Hook for plotters with additional views tied to the 2-D map."""
        pass

    def clear_data_operation_history(self):
        """Discard snapshots belonging to a previously displayed source array."""
        self.data_operation_history.clear()
        self._update_undo_menu_state()


    def plot_data(self):
        tick = time.perf_counter()
        self._displayed_acquisition_row_indices = None

        if self.trace_axis_map_mode:
            self.X, self.Y, self.sliced_data = self._trace_axis_map_arrays()
            self.nan_mask = (
                np.isfinite(self.X).all(axis=1)
                & np.isfinite(self.Y).all(axis=1)
            )
            self.X = self.X[self.nan_mask]
            self.Y = self.Y[self.nan_mask]
            self.sliced_data = self.sliced_data[self.nan_mask]
            self.name_data_z = getattr(
                self,
                'trace_ylabel',
                'Trace amplitude',
            )
            self.name_data_x_axis = getattr(
                self,
                'trace_xlabel',
                'Trace X axis',
            )
            self.name_data_y_axis = self.scan_axis_names[
                self.y_scan_axis_index
            ]
            self.ax.clear()
            self.xlim = (np.nanmin(self.X), np.nanmax(self.X))
            self.ylim = (np.nanmin(self.Y), np.nanmax(self.Y))
        elif self.single_axis_measurement:
            x_values = np.ravel(self.data.measure_axis[0])
            original_data = np.ravel(
                self.data.measure_data[
                    self.name_data.index(self.data_combobox.get())
                ]
            )
            point_count = min(x_values.size, original_data.size)
            x_values = x_values[:point_count]
            original_data = original_data[:point_count]
            self.nan_mask = np.isfinite(x_values)
            x_values = x_values[self.nan_mask]

            # Create X as a 2D array with each x-value repeated twice
            # Each row will have identical x-values: [[x1, x1], [x2, x2], ...]
            self.X = np.swapaxes(np.repeat(x_values[:, np.newaxis], 3, axis=1), 0, 1)

            # Create Y with two different values for each X position
            # This creates two rows in the 2D plot
            y_values = np.array([0, 1, 2], dtype=float)
            self.Y = np.swapaxes(np.tile(y_values, (len(x_values), 1)), 0, 1)

            # Handle sliced_data: Repeat each value twice to match X and Y dimensions
            filtered_data = original_data[self.nan_mask]
            # Repeat each value to create a 2D array with identical values in each row
            self.sliced_data = np.swapaxes(np.repeat(filtered_data[:, np.newaxis], 3, axis=1), 0, 1)

            self.ax.clear()
            self.xlim = (np.min(self.X), np.max(self.X))
            self.ylim = (np.min(self.Y), np.max(self.Y))

            self.name_data_z = self.data_combobox.get()
            self.name_data_x_axis = str(self.data.name_axis[0])
            self.name_data_y_axis = 'y-dummy'
        else:
            self.X = self._slice_current_scan_grid(
                self._scan_coordinate_grid(self.x_scan_axis_index)
            )
            self.Y = self._slice_current_scan_grid(
                self._scan_coordinate_grid(self.y_scan_axis_index)
            )
            self.sliced_data = self._slice_current_scan_grid(
                self._selected_signal_grid()
            )
            self.nan_mask = ~np.isnan(self.X).any(axis=1)
            if self._alternating_x_sweep_enabled():
                self._displayed_acquisition_row_indices = np.flatnonzero(
                    self.nan_mask
                )
            self.X = self.X[self.nan_mask]
            self.Y = self.Y[self.nan_mask]
            self.sliced_data = self.sliced_data[self.nan_mask]
            self.name_data_z = self.data_combobox.get()
            self.name_data_x_axis = self.scan_axis_names[
                self.x_scan_axis_index
            ]
            self.name_data_y_axis = self.scan_axis_names[
                self.y_scan_axis_index
            ]

            if self.invert_enabled:
                self.X, self.Y = self.Y.T, self.X.T
                self.sliced_data = self.sliced_data.T
                self.name_data_x_axis, self.name_data_y_axis = (
                    self.name_data_y_axis,
                    self.name_data_x_axis,
                )

            self.ax.clear()
            self.xlim = (np.nanmin(self.X), np.nanmax(self.X))
            self.ylim = (np.nanmin(self.Y), np.nanmax(self.Y))

        self.X, self.Y, self.sliced_data = canonicalize_plot_grid(
            self.X,
            self.Y,
            self.sliced_data,
        )
        if self._alternating_x_sweep_enabled():
            acquisition_rows = self._displayed_acquisition_row_indices
            if (
                acquisition_rows is not None
                and len(acquisition_rows) != self.X.shape[0]
            ):
                acquisition_rows = None
            self.X, self.Y, self.sliced_data = correct_alternating_x_sweep(
                self.X,
                self.Y,
                self.sliced_data,
                acquisition_row_indices=acquisition_rows,
            )
        self._coerce_calculation_arrays_to_float64()
        self._prepare_display_arrays(force=True)

        self.update_histogramm()
        self._draw_main_map(self.vmin, self.vmax)
        self.clear_data_operation_history()
        tock = time.perf_counter()
        print(f'Plotting time: {tock - tick} s')

    def toggle_crosshair(self):
        """
        Enables or disables the crosshair functionality on a matplotlib plot. When enabled, it
        initializes or updates horizontal and vertical crosshair lines and sets up event handlers
        to track cursor movement. When disabled, it removes the crosshair lines and disconnects
        associated event handling.

        :raises NotImplementedError: If any required handler or attribute initialization is
                                      missing (implicit conditions may apply).

        """
        self.crosshair_enabled = not self.crosshair_enabled

        if self.crosshair_enabled:
            self.horizontal_linecut_frame.grid()
            self.vertical_linecut_frame.grid()
            self.ax_vline.set_visible(True)
            self.ax_hline.set_visible(True)
            self._ensure_linecut_axes_configured(force=True)
            self._ensure_crosshair_overlay()
            if self.crosshair_motion_cid is None:
                self.crosshair_motion_cid = self.canvas.mpl_connect(
                    'motion_notify_event',
                    self.on_mouse_move,
                )
            self._apply_crosshair_display_mode()
            self.refresh_crosshair()
        else:
            self.horizontal_linecut_frame.grid_remove()
            self.vertical_linecut_frame.grid_remove()
            self.ax_vline.set_visible(False)
            self.ax_hline.set_visible(False)
            self.canvas.get_tk_widget().configure(cursor='')
            self._set_crosshair_overlay_state('hidden')
            self._pending_crosshair_position = None
            self._pending_linecut_request = None
            self._cancel_scheduled_callback('_crosshair_update_after_id')
            if self._linecut_update_after_id is not None:
                self.root.after_cancel(self._linecut_update_after_id)
                self._linecut_update_after_id = None
            self.horizontal_linecut_canvas.draw_idle()
            self.vertical_linecut_canvas.draw_idle()
            if self.crosshair_motion_cid is not None:
                self.canvas.mpl_disconnect(self.crosshair_motion_cid)
                self.crosshair_motion_cid = None

    def _cancel_scheduled_callback(self, attribute_name):
        """Cancel a stored Tk ``after`` callback and clear its identifier."""
        callback_id = getattr(self, attribute_name, None)
        if callback_id is None:
            return
        try:
            self.root.after_cancel(callback_id)
        except tk.TclError:
            pass
        setattr(self, attribute_name, None)

    def _on_fast_crosshair_changed(self):
        """Apply the selected crosshair renderer without changing linecuts."""
        self._last_crosshair_canvas_coordinates = None
        self._apply_crosshair_display_mode()
        if self.crosshair_enabled and not self.fast_crosshair_var.get():
            self.refresh_crosshair()

    def _apply_crosshair_display_mode(self):
        """Use the native cursor in fast mode and hide expensive guide lines."""
        tk_canvas = self.canvas.get_tk_widget()
        fast_mode = bool(self.fast_crosshair_var.get())
        tk_canvas.configure(
            cursor='crosshair' if self.crosshair_enabled and fast_mode else ''
        )
        if not self.crosshair_enabled or fast_mode:
            self._set_crosshair_overlay_state('hidden')

    def _ensure_crosshair_overlay(self):
        """Create outlined Tk crosshair lines above the static map bitmap."""
        tk_canvas = self.canvas.get_tk_widget()
        line_color = self._current_plot_line_color()
        if self.crosshair_horizontal_shadow_item is None:
            self.crosshair_horizontal_shadow_item = tk_canvas.create_line(
                0,
                0,
                0,
                0,
                fill='black',
                width=3,
                dash=(4, 3),
                state='hidden',
            )
        if self.crosshair_vertical_shadow_item is None:
            self.crosshair_vertical_shadow_item = tk_canvas.create_line(
                0,
                0,
                0,
                0,
                fill='black',
                width=3,
                dash=(4, 3),
                state='hidden',
            )
        if self.crosshair_horizontal_item is None:
            self.crosshair_horizontal_item = tk_canvas.create_line(
                0,
                0,
                0,
                0,
                fill=line_color,
                width=1,
                dash=(4, 3),
                state='hidden',
            )
        if self.crosshair_vertical_item is None:
            self.crosshair_vertical_item = tk_canvas.create_line(
                0,
                0,
                0,
                0,
                fill=line_color,
                width=1,
                dash=(4, 3),
                state='hidden',
            )
        self._raise_map_overlay_items()

    def _apply_crosshair_overlay_color(self):
        """Apply the shared line preference to existing native guides."""
        if not hasattr(self, 'canvas'):
            return
        tk_canvas = self.canvas.get_tk_widget()
        line_color = self._current_plot_line_color()
        for item in (
            getattr(self, 'crosshair_horizontal_item', None),
            getattr(self, 'crosshair_vertical_item', None),
        ):
            if item is not None:
                tk_canvas.itemconfigure(item, fill=line_color)

    def _set_crosshair_overlay_state(self, state):
        visible = state != 'hidden'
        if visible == self._crosshair_overlay_visible:
            return
        tk_canvas = self.canvas.get_tk_widget()
        for item in (
            self.crosshair_horizontal_shadow_item,
            self.crosshair_vertical_shadow_item,
            self.crosshair_horizontal_item,
            self.crosshair_vertical_item,
        ):
            if item is not None:
                tk_canvas.itemconfigure(item, state=state)
        self._crosshair_overlay_visible = visible
        if visible:
            self._raise_map_overlay_items()

    def _position_crosshair_overlay(self, x_value, y_value):
        """Move the Tk overlay without redrawing the Matplotlib figure."""
        if self.fast_crosshair_var.get():
            return
        self._ensure_crosshair_overlay()
        tk_canvas = self.canvas.get_tk_widget()
        canvas_height = tk_canvas.winfo_height()
        x_pixel, y_pixel = self.ax.transData.transform((x_value, y_value))
        axes_bounds = self.ax.bbox
        tk_y = canvas_height - y_pixel
        tk_top = canvas_height - axes_bounds.y1
        tk_bottom = canvas_height - axes_bounds.y0
        canvas_coordinates = (
            int(round(axes_bounds.x0)),
            int(round(tk_y)),
            int(round(axes_bounds.x1)),
            int(round(x_pixel)),
            int(round(tk_top)),
            int(round(tk_bottom)),
        )
        if canvas_coordinates == self._last_crosshair_canvas_coordinates:
            self._set_crosshair_overlay_state('normal')
            self._raise_map_overlay_items()
            return

        for horizontal_item in (
            self.crosshair_horizontal_shadow_item,
            self.crosshair_horizontal_item,
        ):
            tk_canvas.coords(
                horizontal_item,
                canvas_coordinates[0],
                canvas_coordinates[1],
                canvas_coordinates[2],
                canvas_coordinates[1],
            )
        for vertical_item in (
            self.crosshair_vertical_shadow_item,
            self.crosshair_vertical_item,
        ):
            tk_canvas.coords(
                vertical_item,
                canvas_coordinates[3],
                canvas_coordinates[4],
                canvas_coordinates[3],
                canvas_coordinates[5],
            )
        self._last_crosshair_canvas_coordinates = canvas_coordinates
        self._set_crosshair_overlay_state('normal')
        self._raise_map_overlay_items()

    def toggle_invert(self):
        self.invert_enabled = not self.invert_enabled

    def _data_to_tk_canvas_point(self, x_value, y_value):
        """Transform a data coordinate into the map's native Tk coordinates."""
        tk_canvas = self.canvas.get_tk_widget()
        x_pixel, y_pixel = self.ax.transData.transform((x_value, y_value))
        return int(round(x_pixel)), int(round(tk_canvas.winfo_height() - y_pixel))

    def _ensure_roi_preview(self):
        """Create the native ROI outline used only while selecting its size."""
        if self.roi_preview_item is not None:
            return
        tk_canvas = self.canvas.get_tk_widget()
        self.roi_preview_item = tk_canvas.create_rectangle(
            0,
            0,
            0,
            0,
            outline='red',
            width=2,
            state='hidden',
        )
        tk_canvas.tag_raise(self.roi_preview_item)

    def _queue_roi_preview(self, start_point, end_point):
        """Coalesce ROI outline movement so raw mouse events cannot pile up."""
        self._pending_roi_preview = (tuple(start_point), tuple(end_point))
        if self._roi_preview_after_id is None:
            self._roi_preview_after_id = self.root.after(
                self._overlay_update_interval_ms,
                self._flush_roi_preview,
            )

    def _flush_roi_preview(self):
        self._roi_preview_after_id = None
        preview = self._pending_roi_preview
        self._pending_roi_preview = None
        if preview is None or not self.roi_mode:
            return
        self._ensure_roi_preview()
        start_point, end_point = preview
        x1, y1 = self._data_to_tk_canvas_point(*start_point)
        x2, y2 = self._data_to_tk_canvas_point(*end_point)
        tk_canvas = self.canvas.get_tk_widget()
        tk_canvas.coords(
            self.roi_preview_item,
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )
        tk_canvas.itemconfigure(self.roi_preview_item, state='normal')

    def _hide_roi_preview(self):
        self._pending_roi_preview = None
        self._cancel_scheduled_callback('_roi_preview_after_id')
        if self.roi_preview_item is not None:
            self.canvas.get_tk_widget().itemconfigure(
                self.roi_preview_item,
                state='hidden',
            )

    def _remove_current_roi_patch(self):
        if self.current_roi_patch is None:
            return False
        try:
            self.current_roi_patch.remove()
        except ValueError:
            pass
        self.current_roi_patch = None
        return True

    def toggle_roi(self):
        if self.roi_mode:  # If ROI mode is active, deactivate it
            if self.click_cid is not None:
                self.canvas.mpl_disconnect(self.click_cid)
                self.click_cid = None
            if self.move_cid is not None:
                self.canvas.mpl_disconnect(self.move_cid)
                self.move_cid = None
            self.roi_mode = False
            self._hide_roi_preview()
            if self._remove_current_roi_patch():
                self.canvas.draw_idle()
            return

        # Activate ROI mode
        self.roi_mode = True
        self.roi_corners = []  # Initialize the list for ROI corners
        self._remove_current_roi_patch()
        self._hide_roi_preview()

        # Function to handle mouse click
        def on_click(event):
            if event.inaxes != self.ax or event.button != 1:
                return

            if len(self.roi_corners) == 0:
                # First click: Save the initial corner
                self.roi_corners.append((event.xdata, event.ydata))
            elif len(self.roi_corners) == 1:
                # Second click: Save the final corner
                self.roi_corners.append((event.xdata, event.ydata))
                xmin, xmax = sorted([self.roi_corners[0][0], self.roi_corners[1][0]])
                ymin, ymax = sorted([self.roi_corners[0][1], self.roi_corners[1][1]])
                self.roi_corners = [(xmin, ymin), (xmax, ymax)]  # Store sorted coordinates
                self._hide_roi_preview()
                self._remove_current_roi_patch()
                self.current_roi_patch = self.ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        color='red',
                        alpha=0.25,
                    )
                )
                self.canvas.draw_idle()
                print(f"ROI selected: {self.roi_corners}")
            else:
                # Third click: Reset the ROI
                self.roi_corners.clear()
                self._hide_roi_preview()
                if self._remove_current_roi_patch():
                    self.canvas.draw_idle()

        # Function to handle mouse movement
        def on_move(event):
            if (
                len(self.roi_corners) == 1
                and event.inaxes == self.ax
                and event.xdata is not None
                and event.ydata is not None
            ):
                self._queue_roi_preview(
                    self.roi_corners[0],
                    (event.xdata, event.ydata),
                )

        # Connect the click and motion events
        self.click_cid = self.canvas.mpl_connect('button_press_event', on_click)
        self.move_cid = self.canvas.mpl_connect('motion_notify_event', on_move)

    def refresh_crosshair(self):
        if not self.crosshair_enabled:
            return
        row_count, column_count = self.sliced_data.shape
        self.y_index = min(max(int(self.y_index), 0), row_count - 1)
        self.x_index = min(max(int(self.x_index), 0), column_count - 1)
        x_value = float(self.X[self.y_index, self.x_index])
        y_value = float(self.Y[self.y_index, self.x_index])
        self._apply_crosshair_display_mode()
        self._position_crosshair_overlay(x_value, y_value)
        self._last_crosshair_indices = (self.y_index, self.x_index)
        self._ensure_linecut_axes_configured()
        self._queue_crosshair_linecuts(
            self.y_index,
            self.x_index,
            x_value,
            y_value,
        )

    @staticmethod
    def _nearest_finite_index(values, target):
        values = np.asarray(values, dtype=float)
        distances = np.abs(values - target)
        distances[~np.isfinite(distances)] = np.inf
        if not np.any(np.isfinite(distances)):
            return 0
        return int(np.argmin(distances))

    def _nearest_crosshair_indices(self, x_value, y_value):
        """Find a full-resolution cell in O(rows + columns) for grid maps."""
        row_index = self._nearest_finite_index(
            self._crosshair_y_reference,
            y_value,
        )
        column_index = self._nearest_finite_index(
            self.X[row_index, :],
            x_value,
        )
        row_index = self._nearest_finite_index(
            self.Y[:, column_index],
            y_value,
        )
        return row_index, column_index

    def _linecut_signal_limits(self):
        lower = float(self.vmin)
        upper = float(self.vmax)
        if not np.isfinite(lower) or not np.isfinite(upper):
            finite_data = self.sliced_data[np.isfinite(self.sliced_data)]
            if finite_data.size:
                lower = float(np.min(finite_data))
                upper = float(np.max(finite_data))
            else:
                lower, upper = 0.0, 1.0
        lower, upper = sorted((lower, upper))
        if lower == upper:
            padding = max(1.0, abs(lower) * 0.05)
            lower -= padding
            upper += padding
        return lower, upper

    def _ensure_linecut_axes_configured(self, force=False):
        """Create static axes and persistent animated linecut artists."""
        x_limits = tuple(float(value) for value in self.ax.get_xlim())
        y_limits = tuple(float(value) for value in self.ax.get_ylim())
        signal_limits = self._linecut_signal_limits()
        signature = (x_limits, y_limits, signal_limits)
        if not force and signature == self._linecut_axes_signature:
            return

        self.ax_hline.clear()
        self.ax_hline.set_visible(True)
        self.ax_hline.set_xlim(x_limits)
        self.ax_hline.set_ylim(signal_limits)
        self.ax_hline.set_xticklabels([])
        (self.horizontal_linecut_artist,) = self.ax_hline.plot(
            [],
            [],
            color=self._current_plot_line_color(),
            animated=True,
        )
        self.horizontal_linecut_cursor = self.ax_hline.axvline(
            x=x_limits[0],
            color='gray',
            lw=1,
            ls='--',
            animated=True,
        )

        self.ax_vline.clear()
        self.ax_vline.set_visible(True)
        self.ax_vline.set_xlim(signal_limits)
        self.ax_vline.set_ylim(y_limits)
        self.ax_vline.set_yticklabels([])
        for label in self.ax_vline.get_xticklabels():
            label.set_rotation(270)
        (self.vertical_linecut_artist,) = self.ax_vline.plot(
            [],
            [],
            color=self._current_plot_line_color(),
            animated=True,
        )
        self.vertical_linecut_cursor = self.ax_vline.axhline(
            y=y_limits[0],
            color='gray',
            lw=1,
            ls='--',
            animated=True,
        )
        configure_transparent_matplotlib_canvas(
            self.horizontal_linecut_figure,
            self.horizontal_linecut_canvas,
            opaque_for_blitting=True,
        )
        configure_transparent_matplotlib_canvas(
            self.vertical_linecut_figure,
            self.vertical_linecut_canvas,
            opaque_for_blitting=True,
        )

        self._linecut_axes_signature = signature
        self._horizontal_linecut_background = None
        self._vertical_linecut_background = None
        self.horizontal_linecut_canvas.draw_idle()
        self.vertical_linecut_canvas.draw_idle()

    def _on_horizontal_linecut_draw(self, _event):
        if not self.crosshair_enabled or not self.ax_hline.get_visible():
            return
        self._horizontal_linecut_background = (
            self.horizontal_linecut_canvas.copy_from_bbox(self.ax_hline.bbox)
        )
        if self._current_linecut_request is not None:
            self.root.after_idle(self._blit_current_linecuts)

    def _on_vertical_linecut_draw(self, _event):
        if not self.crosshair_enabled or not self.ax_vline.get_visible():
            return
        self._vertical_linecut_background = (
            self.vertical_linecut_canvas.copy_from_bbox(self.ax_vline.bbox)
        )
        if self._current_linecut_request is not None:
            self.root.after_idle(self._blit_current_linecuts)

    def _queue_crosshair_linecuts(
        self,
        row_index,
        column_index,
        x_value,
        y_value,
    ):
        """Retain only the newest linecut request and update at a bounded rate."""
        self._pending_linecut_request = (
            int(row_index),
            int(column_index),
            float(x_value),
            float(y_value),
        )
        if self._linecut_update_after_id is None:
            self._linecut_update_after_id = self.root.after(
                self._linecut_update_interval_ms,
                self._flush_crosshair_linecuts,
            )

    def _update_crosshair_linecuts_now(
        self,
        row_index,
        column_index,
        x_value,
        y_value,
    ):
        """Update linecuts inside an already rate-limited crosshair flush."""
        self._cancel_scheduled_callback('_linecut_update_after_id')
        self._pending_linecut_request = (
            int(row_index),
            int(column_index),
            float(x_value),
            float(y_value),
        )
        self._flush_crosshair_linecuts()

    def _flush_crosshair_linecuts(self):
        self._linecut_update_after_id = None
        request = self._pending_linecut_request
        self._pending_linecut_request = None
        if request is None or not self.crosshair_enabled:
            return
        self._current_linecut_request = request
        row_index, column_index, x_value, y_value = request
        self._ensure_linecut_axes_configured()

        horizontal_indices = self._downsample_indices(
            self.sliced_data.shape[1],
            min(512, self.sliced_data.shape[1]),
        )
        vertical_indices = self._downsample_indices(
            self.sliced_data.shape[0],
            min(512, self.sliced_data.shape[0]),
        )

        self.horizontal_linecut_artist.set_data(
            self.X[row_index, horizontal_indices].astype(np.float32),
            self.sliced_data[row_index, horizontal_indices].astype(np.float32),
        )
        self.horizontal_linecut_cursor.set_xdata([x_value, x_value])
        self.vertical_linecut_artist.set_data(
            self.sliced_data[vertical_indices, column_index].astype(np.float32),
            self.Y[vertical_indices, column_index].astype(np.float32),
        )
        self.vertical_linecut_cursor.set_ydata([y_value, y_value])
        self._blit_current_linecuts()

    def _blit_current_linecuts(self):
        """Blit only animated line artists over cached static linecut axes."""
        if not self.crosshair_enabled:
            return
        if (
            self._horizontal_linecut_background is None
            or self._vertical_linecut_background is None
        ):
            self.horizontal_linecut_canvas.draw_idle()
            self.vertical_linecut_canvas.draw_idle()
            return

        self.horizontal_linecut_canvas.restore_region(
            self._horizontal_linecut_background
        )
        self.ax_hline.draw_artist(self.horizontal_linecut_artist)
        self.ax_hline.draw_artist(self.horizontal_linecut_cursor)
        self.horizontal_linecut_canvas.blit(self.ax_hline.bbox)

        self.vertical_linecut_canvas.restore_region(
            self._vertical_linecut_background
        )
        self.ax_vline.draw_artist(self.vertical_linecut_artist)
        self.ax_vline.draw_artist(self.vertical_linecut_cursor)
        self.vertical_linecut_canvas.blit(self.ax_vline.bbox)

    def init_movable_lines(self):
        # Initial positions for vmin and vmax lines
        if self.auto_scale_var.get() or self.vmin is None or self.vmax is None:
            self.apply_auto_scaling()
            vmin_initial = self.vmin
            vmax_initial = self.vmax
        else:
            vmin_initial = self.vmin
            vmax_initial = self.vmax

        self.vline1 = None
        self.vline2 = None
        # Create vertical lines
        self.vline1 = self.histogram_ax.axvline(vmin_initial, color='red', lw=1, picker=5)
        self.vline2 = self.histogram_ax.axvline(vmax_initial, color='green', lw=1, picker=5)

        # Connect event handlers
        self.histogram_canvas.mpl_connect('pick_event', self.on_pick)
        self.histogram_canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.histogram_canvas.mpl_connect('button_release_event', self.on_release)

        # Set initial vmin and vmax
        self.vmin = vmin_initial
        self.vmax = vmax_initial

    def on_pick(self, event):
        # Called when a line is clicked on
        if isinstance(event.artist, lines.Line2D):
            self.picked_line = event.artist

    def on_drag(self, event):
        # Drag the line
        if event.inaxes == self.histogram_ax and self.picked_line is not None:
            self.picked_line.set_xdata(event.xdata)
            self.histogram_canvas.draw_idle()

            # Update vmin or vmax
            if self.picked_line == self.vline1:
                self.vmin = event.xdata
            elif self.picked_line == self.vline2:
                self.vmax = event.xdata

            # Update the main plot if needed
            self.update_pcolormesh(self.vmin, self.vmax)

    def on_release(self, event):
        # Called when the mouse is released
        self.picked_line = None

    def _queue_crosshair_position(self, x_value, y_value):
        """Retain only the newest pointer position for the next 30 Hz update."""
        self._pending_crosshair_position = (float(x_value), float(y_value))
        if self._crosshair_update_after_id is None:
            self._crosshair_update_after_id = self.root.after(
                self._crosshair_update_interval_ms,
                self._flush_crosshair_position,
            )

    def _flush_crosshair_position(self):
        """Move guides, resolve the data cell, and refresh linecuts together."""
        self._crosshair_update_after_id = None
        position = self._pending_crosshair_position
        self._pending_crosshair_position = None
        if (
            position is None
            or not self.crosshair_enabled
            or self.freeze_linecut
        ):
            return

        x_value, y_value = position
        self._position_crosshair_overlay(x_value, y_value)
        row_index, column_index = self._nearest_crosshair_indices(
            x_value,
            y_value,
        )
        self.y_index, self.x_index = row_index, column_index
        current_indices = (row_index, column_index)
        if current_indices == self._last_crosshair_indices:
            return

        self._last_crosshair_indices = current_indices
        self._update_crosshair_linecuts_now(
            row_index,
            column_index,
            x_value,
            y_value,
        )

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or not self.crosshair_enabled:
            return
        if self.freeze_linecut or event.xdata is None or event.ydata is None:
            return

        self._queue_crosshair_position(
            event.xdata,
            event.ydata,
        )

    def on_key_press(self, event):
        if event.key == 's':
            self.freeze_linecut = not self.freeze_linecut
            self.linecut_position = (self.y_index, self.x_index)
        else:
            pass

    def toggle_interpolation(self):
        self.interpolation_enabled = not self.interpolation_enabled
        if self.interpolation_enabled:
            self.open_interpolation_window()

    def open_interpolation_window(self):
        # Create a new pop-up window for interpolation settings
        self.interpolation_window = ttk.Toplevel(self.root)
        self.interpolation_window.title("Interpolation Settings")
        self.interpolation_window.geometry("400x200")

        # Add a label and entry widget for the first interpolation value
        ttk.Label(self.interpolation_window, text="Enter Interpolation Factor for x-Axis (0.1 - 1.0):").pack()
        self.interpolation_entry_1 = ttk.Entry(self.interpolation_window)
        self.interpolation_entry_1.pack()
        self.interpolation_entry_1.insert(0, "1.0")  # Default value

        # Add a label and entry widget for the second interpolation value
        ttk.Label(self.interpolation_window, text="Enter Interpolation Factor for y-Axis (0.1 - 1.0):").pack()
        self.interpolation_entry_2 = ttk.Entry(self.interpolation_window)
        self.interpolation_entry_2.pack()
        self.interpolation_entry_2.insert(0, "1.0")  # Default value
        submit_button = ttk.Button(self.interpolation_window, text="Apply", command=self.apply_interpolation, bootstyle='primary')
        submit_button.pack()

    def open_gaussian_filter_window(self):
        # Create a new pop-up window for interpolation settings
        self.gaussian_filter_window = ttk.Toplevel(self.root)
        self.gaussian_filter_window.title("Gaussian Filter Settings")
        self.gaussian_filter_window.geometry("400x200")

        # Add a label and entry widget for the first interpolation value
        ttk.Label(self.gaussian_filter_window, text="Pixel for x-Axis:").pack()
        self.filter_pixel_x = ttk.Entry(self.gaussian_filter_window)
        self.filter_pixel_x.pack()
        self.filter_pixel_x.insert(0, "1.0")  # Default value

        # Add a label and entry widget for the second interpolation value
        ttk.Label(self.gaussian_filter_window, text="Pixel for y-Axis:").pack()
        self.filter_pixel_y = ttk.Entry(self.gaussian_filter_window)
        self.filter_pixel_y.pack()
        self.filter_pixel_y.insert(0, "1.0")  # Default value
        submit_button = ttk.Button(self.gaussian_filter_window, text="Apply", command=self.apply_gaussian_filter, bootstyle='primary')
        submit_button.pack()

    def open_background_subtraction_window(self):
        self.background_subtraction_window = ttk.Toplevel(self.root)
        self.background_subtraction_window.title("Background Subtraction")
        self.background_subtraction_window.geometry("400x250")

        self.background_subtraction_combobox = ttk.Combobox(self.background_subtraction_window, values=self.bg_methods, state='readonly', width=10)
        self.background_subtraction_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.background_subtraction_combobox.set('Polynomial')
        self.background_subtraction_combobox.bind('<<ComboboxSelected>>', self.update_bg_subtraction_inputs)

        # Frame to contain method-specific input fields
        self.method_input_frame = ttk.Frame(self.background_subtraction_window)
        self.method_input_frame.pack(fill=tk.BOTH, expand=True)

        submit_button = ttk.Button(self.background_subtraction_window, text="Apply", command=self.apply_background_subtraction, bootstyle='primary')
        submit_button.pack(side=tk.BOTTOM)

        # Initially update inputs for the default selected method
        self.update_bg_subtraction_inputs()

    def open_roi_data_cut_window(self):
        self.roi_data_cut_window = ttk.Toplevel(self.root)
        self.roi_data_cut_window.title("ROI Data")
        self.roi_data_cut_window.geometry("300x250")
        self.roi_cut_entry_list = []
        labels = ["min. X:", "min.  Y:", "max. X:", "max. Y:"]
        coordinates = np.array([self.roi_corners[0][0], self.roi_corners[0][1], self.roi_corners[1][0], self.roi_corners[1][1]])
        for i, label in enumerate(labels):
            ttk.Label(self.roi_data_cut_window, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(self.roi_data_cut_window)
            entry.insert(0, str(coordinates[i]))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.roi_cut_entry_list.append(entry)
        submit_button = ttk.Button(self.roi_data_cut_window, text="Apply", command=self.apply_roi_data_cut, bootstyle='primary')
        submit_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def open_draw_lines_window(self):
        self.draw_lines_window = ttk.Toplevel(self.root)
        self.draw_lines_window.title("Draw Lines")
        self.draw_lines_window.geometry("400x200")
        # Frames

        self.draw_lines_button_frame = ttk.Frame(self.draw_lines_window)
        self.draw_lines_button_frame.pack(side=tk.BOTTOM)

        self.lines_list_frame = ttk.Frame(self.draw_lines_window)
        self.lines_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scrollbar
        self.lines_list_scrollbar = ttk.Scrollbar(self.lines_list_frame)
        self.lines_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox for displaying lines
        self.lines_listbox = ttk.Listbox(self.lines_list_frame, yscrollcommand=self.lines_list_scrollbar.set)
        self.lines_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lines_list_scrollbar.config(command=self.lines_listbox.yview)

        # Buttons
        activate_button = ttk.Button(self.draw_lines_button_frame, text='Activate',
                                    command=self.activate_line_drawing)
        deactivate_button = ttk.Button(self.draw_lines_button_frame, text='Deactivate',
                                      command=self.deactivate_line_drawing)
        reset_lines_button = ttk.Button(self.draw_lines_button_frame, text='Reset',
                                       command=self.reset_lines)
        activate_button.pack(side=tk.LEFT)
        deactivate_button.pack(side=tk.LEFT)
        reset_lines_button.pack(side=tk.LEFT)
        self.setup_line_editing()

    def update_bg_subtraction_inputs(self, event=None):
        # Clear previous inputs
        for widget in self.method_input_frame.winfo_children():
            widget.destroy()

        selected_method = self.background_subtraction_combobox.get()

        # Inputs for Polynomial method
        if selected_method == 'Polynomial':

            self.use_roi_in_bg_subtraction = tk.BooleanVar(value=False)
            ttk.Label(self.method_input_frame, text="X Polynomial Order:").pack()
            self.poly_order_x = ttk.Entry(self.method_input_frame)
            self.poly_order_x.pack()
            self.poly_order_x.insert(0, "1")  # Default value

            ttk.Label(self.method_input_frame, text="Y Polynomial Order:").pack()
            self.poly_order_y = ttk.Entry(self.method_input_frame)
            self.poly_order_y.pack()
            self.poly_order_y.insert(0, "1")  # Default value

            self.use_roi_in_bg_check = ttk.Checkbutton(
                self.method_input_frame,
                text="Use ROI",
                variable=self.use_roi_in_bg_subtraction
            )
            self.use_roi_in_bg_check.pack(side=tk.BOTTOM, pady=2)

        elif selected_method == 'Relation Parameters':
            # Add input fields for the Relation Parameters method
            labels = ["Coefficient before X:", "Power of X:", "Coefficient before Y:", "Power of Y:", "Constant term:"]
            for i, label in enumerate(labels):
                ttk.Label(self.method_input_frame, text=label).grid(row=i, column=0)
                entry = ttk.Entry(self.method_input_frame, width=10)
                entry.insert(0, "0")
                entry.grid(row=i, column=1)
                self.relation_parameter_entry_list.append(entry)

        elif selected_method == 'Subtract Trace Average':
            # Add input fields for the `subtract_trace_average` function
            ttk.Label(self.method_input_frame, text="Number of Traces (n):").pack()
            self.n_traces_entry = ttk.Entry(self.method_input_frame)
            self.n_traces_entry.pack()
            self.n_traces_entry.insert(0, "1")  # Default value

            ttk.Label(self.method_input_frame, text="Axis (0 for rows, 1 for columns):").pack()
            self.axis_entry = ttk.Entry(self.method_input_frame)
            self.axis_entry.pack()
            self.axis_entry.insert(0, "0")  # Default value

            ttk.Label(self.method_input_frame, text="From End (True/False):").pack()
            self.from_end_var = tk.StringVar(value="False")
            self.from_end_checkbox = ttk.Checkbutton(self.method_input_frame, text="From End",
                                                     variable=self.from_end_var, onvalue="True", offvalue="False")
            self.from_end_checkbox.pack()

            ttk.Label(self.method_input_frame, text="Use Gaussian Filter (True/False):").pack()
            self.use_filter_var = tk.StringVar(value="False")
            self.use_filter_checkbox = ttk.Checkbutton(self.method_input_frame, text="Use Filter",
                                                       variable=self.use_filter_var, onvalue="True", offvalue="False")
            self.use_filter_checkbox.pack()

            ttk.Label(self.method_input_frame, text="Polynomial Fit (True/False):").pack()
            self.poly_fit_var = tk.StringVar(value="False")
            self.poly_fit_checkbox = ttk.Checkbutton(self.method_input_frame, text="Polynomial Fit",
                                                     variable=self.poly_fit_var, onvalue="True", offvalue="False")
            self.poly_fit_checkbox.pack()

            ttk.Label(self.method_input_frame, text="Filter Sigma:").pack()
            self.filter_sigma_entry = ttk.Entry(self.method_input_frame)
            self.filter_sigma_entry.pack()
            self.filter_sigma_entry.insert(0, "1.0")  # Default value

            ttk.Label(self.method_input_frame, text="Polynomial Fit Order:").pack()
            self.poly_fit_order_entry = ttk.Entry(self.method_input_frame)
            self.poly_fit_order_entry.pack()
            self.poly_fit_order_entry.insert(0, "1")  # Default value

        elif selected_method == 'Median Difference':
            pass

        elif selected_method == 'Mean of Lines':
            pass

    def apply_background_subtraction(self):
        selected_method = self.background_subtraction_combobox.get()

        # Apply Polynomial background subtraction
        if selected_method == 'Polynomial':
            self.apply_poly_bg()

        elif selected_method == 'Median Difference':
            self.apply_median_difference()

        elif selected_method == 'Relation Parameters':
            self.apply_relation_parameters()

        elif selected_method == 'Mean of Lines':
            self.apply_mean_of_lines()

        elif selected_method == 'Subtract Trace Average':
            self.apply_subtract_trace_average()

    def open_derivative_window(self):

        self.derivative_window = ttk.Toplevel(self.root)
        self.derivative_window.title("Calculate derivative along axis")
        self.derivative_window.geometry("400x200")
        self.axis_selection = ['y', 'x']
        self.derivative_combobox = ttk.Combobox(self.derivative_window, values=self.axis_selection, state='readonly', width=10)
        self.derivative_combobox.pack(side=tk.BOTTOM, padx=5, pady=0)
        self.derivative_combobox.set('x')
        submit_button = ttk.Button(self.derivative_window, text="Apply", command=self.apply_derivative, bootstyle='primary')
        submit_button.pack()

    def open_savitzky_golay_filter_window(self):
        self.savitzky_golay_filter_window = ttk.Toplevel(self.root)
        self.savitzky_golay_filter_window.title("Savitzky-Golay Filter")
        self.savitzky_golay_filter_window.geometry("400x220")

        self.savgol_axis_selection = ['y', 'x']

        ttk.Label(self.savitzky_golay_filter_window, text="Axis:").pack()
        self.savgol_axis_combobox = ttk.Combobox(
            self.savitzky_golay_filter_window,
            values=self.savgol_axis_selection,
            state='readonly',
            width=10
        )
        self.savgol_axis_combobox.pack()
        self.savgol_axis_combobox.set('x')

        ttk.Label(self.savitzky_golay_filter_window, text="Window length:").pack()
        self.savgol_window_entry = ttk.Entry(self.savitzky_golay_filter_window)
        self.savgol_window_entry.pack()
        self.savgol_window_entry.insert(0, "7")

        ttk.Label(self.savitzky_golay_filter_window, text="Polynomial order:").pack()
        self.savgol_poly_entry = ttk.Entry(self.savitzky_golay_filter_window)
        self.savgol_poly_entry.pack()
        self.savgol_poly_entry.insert(0, "2")

        ttk.Label(
            self.savitzky_golay_filter_window,
            text="Derivative order (-1 = coordinate-weighted integral):"
        ).pack()
        self.savgol_deriv_entry = ttk.Entry(self.savitzky_golay_filter_window)
        self.savgol_deriv_entry.pack()
        self.savgol_deriv_entry.insert(0, "0")

        submit_button = ttk.Button(
            self.savitzky_golay_filter_window,
            text="Apply",
            command=self.apply_savitzky_golay_filter
        )
        submit_button.pack(pady=10)

    def apply_savitzky_golay_filter(self):
        axis_name = self.savgol_axis_combobox.get()
        axis = self.savgol_axis_selection.index(axis_name)

        try:
            window_length = int(self.savgol_window_entry.get())
            polyorder = int(self.savgol_poly_entry.get())
            deriv = int(self.savgol_deriv_entry.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Savitzky-Golay settings",
                "Window length, polynomial order, and derivative order must be integers."
            )
            return

        if deriv < -1:
            messagebox.showerror(
                "Invalid Savitzky-Golay settings",
                "Derivative order must be -1 or a non-negative integer."
            )
            return

        if window_length % 2 == 0:
            window_length += 1

        max_window = self.sliced_data.shape[axis]
        if window_length > max_window:
            window_length = max_window if max_window % 2 == 1 else max_window - 1

        if window_length <= polyorder:
            messagebox.showerror(
                "Invalid Savitzky-Golay settings",
                "Window length must be larger than polynomial order."
            )
            return

        filtered_data = savgol_filter(
            self.sliced_data,
            window_length=window_length,
            polyorder=polyorder,
            axis=axis,
            deriv=0 if deriv == -1 else deriv,
            mode='interp'
        )

        operation_name = 'Savitzky-Golay filter'
        if deriv == -1:
            integration_coordinate = self.Y if axis == 0 else self.X
            filtered_data = cumulative_integral(
                filtered_data,
                integration_coordinate,
                axis=axis,
            )
            operation_name = 'Savitzky-Golay cumulative integral'

        with self.data_operation(operation_name):
            self.sliced_data = filtered_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def open_data_axis_transform(self):
        self.data_axis_transform_window = ttk.Toplevel(self.root)
        self.use_trace_wise_min_max_scaling_var = tk.BooleanVar(value=False)
        self.data_axis_transform_window.title("Axis Scaling and Renaming")
        self.data_axis_transform_window.geometry("400x240")

        self.data_axis_transform_naming_frame = ttk.Frame(self.data_axis_transform_window)
        self.data_axis_transform_scaling_frame = ttk.Frame(self.data_axis_transform_window)

        ttk.Label(self.data_axis_transform_naming_frame, text="Axis Names:").pack()
        ttk.Label(self.data_axis_transform_scaling_frame, text="Axis Scaling Factors:").pack()

        self.x_axis_name_input = ttk.Entry(self.data_axis_transform_naming_frame)
        self.x_axis_name_input.pack()
        self.x_axis_name_input.insert(0, self.name_data_x_axis)
        self.x_axis_scale_input = ttk.Entry(self.data_axis_transform_scaling_frame)
        self.x_axis_scale_input.pack()
        self.x_axis_scale_input.insert(0, '1.0')

        self.y_axis_name_input = ttk.Entry(self.data_axis_transform_naming_frame)
        self.y_axis_name_input.pack()
        self.y_axis_name_input.insert(0, self.name_data_y_axis)
        self.y_axis_scale_input = ttk.Entry(self.data_axis_transform_scaling_frame)
        self.y_axis_scale_input.pack()
        self.y_axis_scale_input.insert(0, '1.0')

        self.z_axis_name_input = ttk.Entry(self.data_axis_transform_naming_frame)
        self.z_axis_name_input.pack()
        self.z_axis_name_input.insert(0, self.name_data_z)
        self.z_axis_scale_input = ttk.Entry(self.data_axis_transform_scaling_frame)
        self.z_axis_scale_input.pack()
        self.z_axis_scale_input.insert(0, '1.0')

        ttk.Label(self.data_axis_transform_scaling_frame, text="Auto Scale Factor:").pack()
        self.auto_scale_factor_input = ttk.Entry(self.data_axis_transform_scaling_frame)
        self.auto_scale_factor_input.pack()
        self.auto_scale_factor_input.insert(0, str(self.auto_scale_factor))

        self.use_trace_wise_min_max_scaling_check = ttk.Checkbutton(
            self.data_axis_transform_window,
            text="Trace Wise min-max scaling",
            variable=self.use_trace_wise_min_max_scaling_var
        )
        self.use_trace_wise_min_max_scaling_check.pack(side=tk.BOTTOM, pady=2)

        submit_button = ttk.Button(self.data_axis_transform_window, text="Apply", command=self.apply_data_axis_transform, bootstyle='primary')
        submit_button.pack(side=tk.BOTTOM)
        self.data_axis_transform_naming_frame.pack(side=tk.LEFT)
        self.data_axis_transform_scaling_frame.pack(side=tk.RIGHT)

    def make_params_viewable(self, fit_results_dict):
        """Expose completed batch-fit parameters as selectable map channels."""
        self.fit_results_dict = fit_results_dict
        # Update the global data_combobox with parameter names from fit_results_dict
        current_values = list(self.data_combobox['values'])
        for param in self.fit_results_dict.keys():
            if param not in current_values:
                current_values.append(param)
        self.data_combobox['values'] = current_values

    def fit_traces(self):

        self.traces_fitter = TracesFitter(
            self.data,
            self.root,
            on_fit_all_complete=self.make_params_viewable,
            plot_style=self.plot_style,
        )
        self.traces_fitter.create_widgets()
        self.traces_fitter.update_plot()

    def open_fft_trace_correction_window(self):
        #### Created by Nico Reinders ####
        # Opens the FFT trace correction window.

        # Create a new Toplevel window
        self.fft_plot_window = ttk.Toplevel(self.root)
        self.fft_plot_window.title("FFT Correction")

        # Create a Matplotlib figure and axis
        self.fft_fig, self.fft_ax = plt.subplots(1, 1)

        # Create a canvas to embed the figure into the Tkinter window
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=self.fft_plot_window)
        configure_transparent_matplotlib_canvas(self.fft_fig, self.fft_canvas)
        self.fft_canvas.draw_idle()

        self.fft_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar2Tk(self.fft_canvas, self.fft_plot_window)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        if self.loaded:
            pass
        else:
            print('fetching data...')
            self.data.set_traces()
            self.data.set_traces_dt()
            self.traces = self.data.traces
            self.times = self.data.get_trace_axis(len(self.traces[0][0]))
            self.loaded = True

        self.frequencies_shifted, self.fft_mean, self.fft_signals, self.original_angles = get_fourier(self.traces,
                                                                                                      self.times)
        fft_correction_select(self.frequencies_shifted, self.fft_mean, self.fft_fig, self.fft_ax)

        open_plot_button = ttk.Button(self.fft_plot_window, text="Apply changes", command=self.apply_fft_trace_correction)
        open_plot_button.pack(pady=20)

    def open_2d_fft_filter(self):
        #### Created by Nico Reinders ####
        # Opens the 2D FFT filter tool window.

        # Create a new Toplevel window
        self.fft_filter_window = ttk.Toplevel(self.root)
        self.fft_filter_window.title("2-D FFT Filter")

        left_frame = ttk.Frame(self.fft_filter_window)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Create a Matplotlib figure and axis for the FFT filter
        self.fft_filter_fig, self.fft_filter_ax = plt.subplots(1, 1, figsize=(6, 6))

        self.fft_filter_canvas = FigureCanvasTkAgg(self.fft_filter_fig, master=left_frame)
        configure_transparent_matplotlib_canvas(
            self.fft_filter_fig,
            self.fft_filter_canvas,
        )
        self.fft_filter_canvas.draw_idle()
        self.fft_filter_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fft_filter_toolbar = NavigationToolbar2Tk(self.fft_filter_canvas, left_frame)
        self.fft_filter_toolbar.update()
        self.fft_filter_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        right_frame = ttk.Frame(self.fft_filter_window)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Create radio buttons for selecting the preview mode
        radio_frame = ttk.Frame(right_frame)
        radio_frame.pack(side=tk.TOP, fill=tk.X, pady=25)
        self.fft_filter_mode = tk.StringVar(value='Mask Editor')
        ttk.Radiobutton(radio_frame, text='Mask Editor', variable=self.fft_filter_mode, value='Mask Editor', command=self.update_fft_filter_plot).pack(anchor=tk.W)
        ttk.Radiobutton(radio_frame, text='Filtered Data Preview', variable=self.fft_filter_mode, value='Filtered Data Preview', command=self.update_fft_filter_plot).pack(anchor=tk.W)

        # Create selection shape buttons
        shape_frame = ttk.Frame(right_frame)
        shape_frame.pack(side=tk.TOP, fill=tk.X, pady=25)
        self.selection_shape = tk.StringVar(value='Ellipse')
        self.ellipse_button = ttk.Button(shape_frame, text='Ellipse', command=self.select_ellipse, bootstyle='primary')
        self.ellipse_button.pack(side=tk.LEFT, padx=5)
        self.rectangle_button = ttk.Button(shape_frame, text='Rectangle', command=self.select_rect, bootstyle='secondary outline')
        self.rectangle_button.pack(side=tk.LEFT, padx=5)

        apply_frame = ttk.Frame(right_frame)
        apply_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        ttk.Button(apply_frame, text='Apply Filter', command=self.apply_fft_filter, bootstyle='primary').pack(side=tk.LEFT, padx=5)

        # Add option to toggle the mask outside (negative) checkbox
        mask_frame = ttk.Frame(right_frame)
        mask_frame.pack(side=tk.TOP, fill=tk.X, pady=25)
        self.fft_mask_negative = tk.BooleanVar(value=False)
        self.fft_mask_checkbox = ttk.Checkbutton(mask_frame, text="Mask Outside (Negative)", variable=self.fft_mask_negative, command=self.update_fft_filter_plot)
        self.fft_mask_checkbox.pack(anchor=tk.N)

        self.shapes = np.array([]) # initialize shapes array to store drawn masks

        self.fft_filter_x, self.fft_filter_y, self.fft_filter_sliced_data = two_d_fft_on_data(self.sliced_data, self.X, self.Y, mode='Complex')
        mean = np.mean(np.absolute(self.fft_filter_sliced_data))
        std = np.std(np.absolute(self.fft_filter_sliced_data))
        self.vmin = mean - self.auto_scale_factor * std
        self.vmax = mean + self.auto_scale_factor * std

        # Compute cell centers
        self.xc = (self.fft_filter_x[:-1, :-1] + self.fft_filter_x[1:, 1:]) / 2
        self.yc = (self.fft_filter_y[:-1, :-1] + self.fft_filter_y[1:, 1:]) / 2
        # Flatten for vectorized testing
        self.points = np.vstack((self.xc.ravel(), self.yc.ravel())).T

        # Connect the event handlers
        self.fft_filter_fig.canvas.mpl_connect('button_press_event', self.fft_filter_on_press)
        self.fft_filter_fig.canvas.mpl_connect('button_release_event', self.fft_filter_on_release)
        self.fft_filter_fig.canvas.mpl_connect('motion_notify_event', self.fft_filter_on_motion)

        self.update_fft_filter_plot()

    def apply_fft_mask(self, fft_filter_sliced_data_cut):
        #### Created by Nico Reinders ####
        # Combines all shape masks and applies them to the FFT data.

        combined_mask = np.ones_like(fft_filter_sliced_data_cut[:-1, :-1], dtype=bool)
        for patch in self.shapes:
            patch_path = Path(patch.get_verts())
            mask = patch_path.contains_points(self.fft_filter_ax.transData.transform(self.points))
            mask = mask.reshape(self.xc.shape)
            mask = 1-mask
            combined_mask = np.logical_and(combined_mask, mask)

        if self.fft_mask_negative.get():
            combined_mask = 1 - combined_mask
        fft_filter_sliced_data_cut[:-1, :-1] *= combined_mask

        return fft_filter_sliced_data_cut

    def apply_fft_filter(self):
        #### Created by Nico Reinders ####
        # Finally applies the combined FFT mask and performs the inverse FFT.

        _, _, self.fft_filter_sliced_data = two_d_fft_on_data(self.sliced_data, self.X, self.Y, mode='Complex')
        self.fft_filter_sliced_data_cut = self.apply_fft_mask(self.fft_filter_sliced_data)

        self.fft_filter_sliced_data_preview = two_d_ifft_on_data(self.fft_filter_sliced_data_cut, self.fft_filter_x, self.fft_filter_y, mode='Complex')[2]
        filtered_data = np.abs(self.fft_filter_sliced_data_preview)

        with self.data_operation('2-D FFT filter'):
            self.sliced_data = filtered_data

        # remove all masks
        for i, shape in enumerate(self.shapes):
            shape.remove()
            self.shapes = np.delete(self.shapes, i)


    def select_ellipse(self):
        #### Created by Nico Reinders ####
        # Handles ellipse selection for FFT mask drawing.

        if self.selection_shape.get() == 'Ellipse':
            self.selection_shape.set('Empty')
            self.ellipse_button.configure(bootstyle='secondary outline')
        else:
            self.selection_shape.set('Ellipse')
            self.ellipse_button.configure(bootstyle='primary')
            self.rectangle_button.configure(bootstyle='secondary outline')

    def select_rect(self):
        #### Created by Nico Reinders ####
        # Handles rectangle selection for FFT mask drawing.

        if self.selection_shape.get() == 'Rectangle':
            self.selection_shape.set('Empty')
            self.rectangle_button.configure(bootstyle='secondary outline')
        else:
            self.selection_shape.set('Rectangle')
            self.ellipse_button.configure(bootstyle='secondary outline')
            self.rectangle_button.configure(bootstyle='primary')

    def fft_filter_on_press(self, event):
        #### Created by Nico Reinders ####
        # Handles mouse press event for FFT mask drawing.

        if event.button == 1:  # Left mouse button
            self.start_point = (event.xdata, event.ydata)

    def fft_filter_on_motion(self, event):
        #### Created by Nico Reinders ####
        # Handles mouse motion event for FFT mask drawing.

        # Check for valid coordinates before proceeding
        if (
            self.start_point is None or
            self.start_point[0] is None or
            self.start_point[1] is None or
            event.xdata is None or
            event.ydata is None
        ):
            return  # Ignore the event if coordinates are not valid

        if event.button == 1 and self.start_point and self.fft_filter_mode.get() == 'Mask Editor':
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata
            width =  x1 - x0
            height = y1 - y0
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2

            if hasattr(self, 'preview_ellipse'):
                self.preview_ellipse.remove()

            if hasattr(self, 'preview_rectangle'):
                self.preview_rectangle.remove()

            if self.selection_shape.get() == 'Ellipse':
                # Draw the new preview ellipse
                self.preview_ellipse = matplotlib.patches.Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linestyle='--')
                self.fft_filter_ax.add_patch(self.preview_ellipse)
            elif self.selection_shape.get() == 'Rectangle':
                # Draw the new preview rectangle
                self.preview_rectangle = matplotlib.patches.Rectangle((x0, y0), width, height, edgecolor='white', facecolor='none', linestyle='--')
                self.fft_filter_ax.add_patch(self.preview_rectangle)

            plt.draw()

    def fft_filter_on_release(self, event):
        #### Created by Nico Reinders ####
        # Handles mouse release event for FFT mask drawing.

        # Check for valid coordinates before proceeding
        if self.start_point is None or event.xdata is None or event.ydata is None:
            return  # Ignore the event if coordinates are not valid
        if event.button == 1 and self.fft_filter_mode.get() == 'Mask Editor':  # Left mouse button
            x0, y0 = self.start_point
            x1, y1 = event.xdata, event.ydata
            width =  x1 - x0
            height = y1 - y0
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2

            if self.selection_shape.get() == 'Ellipse':
                # Draw preview ellipse
                ellipse = matplotlib.patches.Ellipse((center_x, center_y), width, height, color='red', fill=True, alpha=0.5)
                mirrored_ellipse = matplotlib.patches.Ellipse((-center_x, -center_y), -width, -height, color='red', fill=True, alpha=0.5)
                self.fft_filter_ax.add_patch(ellipse)
                self.fft_filter_ax.add_patch(mirrored_ellipse)
                self.shapes = np.append(self.shapes, ellipse)
                self.shapes = np.append(self.shapes, mirrored_ellipse)
                if hasattr(self, 'preview_ellipse'):
                    self.preview_ellipse.remove()
                    del self.preview_ellipse

            elif self.selection_shape.get() == 'Rectangle':
                # Draw preview rectangle
                rectangle = matplotlib.patches.Rectangle((x0, y0), width, height, color='red', fill=True, alpha=0.5)
                mirrored_rectangle = matplotlib.patches.Rectangle((-x0, -y0), -width, -height, color='red', fill=True, alpha=0.5)
                self.fft_filter_ax.add_patch(rectangle)
                self.fft_filter_ax.add_patch(mirrored_rectangle)
                self.shapes = np.append(self.shapes, rectangle)
                self.shapes = np.append(self.shapes, mirrored_rectangle)

                if hasattr(self, 'preview_rectangle'):
                    self.preview_rectangle.remove()
                    del self.preview_rectangle

            plt.draw()

        # Right mouse button to remove shapes

        elif event.button == 3 and self.fft_filter_mode.get() == 'Mask Editor':
            x1, y1 = event.xdata, event.ydata
            x1, y1 = self.fft_filter_ax.transData.transform((x1, y1))

            if self.shapes.size > 0:
                for i, shape in enumerate(self.shapes):
                    patch_path = Path(shape.get_verts())

                    if patch_path.contains_point((x1, y1)):
                        shape.remove()
                        self.shapes = np.delete(self.shapes, i)
                        self.update_fft_filter_plot()
                        break

    def update_fft_filter_plot(self):
        #### Created by Nico Reinders ####
        # Updates the FFT filter plot with current mask preview and corrected data preview.

        if self.fft_filter_mode.get() == 'Mask Editor':
            # Mask Editor

            # Show the mask editor control buttons that are hidden in the preview mode
            self.fft_mask_checkbox.pack()
            self.ellipse_button.pack(side=tk.LEFT, padx=5)
            self.rectangle_button.pack(side=tk.LEFT, padx=5)

            # Clear the axes and plot the FFT data with the current masks
            self.fft_filter_ax.clear()
            vmin = np.min(np.abs(self.fft_filter_sliced_data))
            vmax = np.max(np.abs(self.fft_filter_sliced_data))
            self.fft_filter_ax.pcolormesh(self.fft_filter_x, self.fft_filter_y, np.absolute(self.fft_filter_sliced_data), norm=colors.LogNorm(vmin=vmin, vmax=vmax),
                                          cmap=self.colormap_combobox.get(), shading='auto', zorder=1, linewidth=0, rasterized=True)
            for patch in self.shapes:
                self.fft_filter_ax.add_patch(patch)
            self.fft_filter_ax.set_xlim(np.min(self.fft_filter_x), np.max(self.fft_filter_x))
            self.fft_filter_ax.set_ylim(np.min(self.fft_filter_y), np.max(self.fft_filter_y))

            self.fft_filter_ax.set_xlabel('freq. of ' + self.name_data_x_axis)
            self.fft_filter_ax.set_ylabel('freq. of ' + self.name_data_y_axis)
            self.fft_filter_ax.set_title('FFT Amp of ' + self.name_data_z)


            plt.draw()
        else:
            # Filtered Data Preview

            # Hide the mask editor control buttons
            self.fft_mask_checkbox.pack_forget()
            self.rectangle_button.pack_forget()
            self.ellipse_button.pack_forget()

            self.fft_filter_sliced_data_cut = self.apply_fft_mask(self.fft_filter_sliced_data)

            # Show the preview of the filtered data
            self.fft_filter_x_preview, self.fft_filter_y_preview, self.fft_filter_sliced_data_preview = two_d_ifft_on_data(self.fft_filter_sliced_data_cut, self.fft_filter_x, self.fft_filter_y, mode='Complex')
            self.fft_filter_ax.clear()
            vmin = np.min(self.fft_filter_sliced_data_preview)
            vmax = np.max(self.fft_filter_sliced_data_preview)

            # Ensure the preview data is real-valued for plotting
            data_to_plot = self.fft_filter_sliced_data_preview
            if np.iscomplexobj(data_to_plot):
                print("Warning: Complex data in FFT preview. Using np.abs() for plotting.")
                data_to_plot = np.abs(data_to_plot)
            # Compute vmin/vmax from the real-valued data
            vmin = np.min(data_to_plot)
            vmax = np.max(data_to_plot)
            try:
                self.fft_filter_ax.pcolormesh(
                    self.fft_filter_x_preview,
                    self.fft_filter_y_preview,
                    data_to_plot,
                    cmap=self.colormap_combobox.get(), shading='auto', zorder=1, linewidth=0, rasterized=True, vmin=vmin, vmax=vmax)
            except TypeError as e:
                if hasattr(self.fft_filter_sliced_data_preview, 'dtype') and np.issubdtype(self.fft_filter_sliced_data_preview.dtype, np.complexfloating):
                    print("Error: self.fft_filter_sliced_data_preview is complex. Use np.abs() or .real before plotting.")
                else:
                    print(f"Plotting error: {e}")
                raise

            self.fft_filter_sliced_data = two_d_fft_on_data(self.sliced_data, self.X, self.Y, mode='Complex')[2]
            self.fft_filter_ax.set_xlabel(self.name_data_x_axis)
            self.fft_filter_ax.set_ylabel(self.name_data_y_axis)
            self.fft_filter_ax.set_title('Preview of FFT-filtered ' + self.name_data_z)

        self.fft_filter_canvas.draw()

    def apply_data_axis_transform(self):
        name_data_x_axis = str(self.x_axis_name_input.get())
        name_data_y_axis = str(self.y_axis_name_input.get())
        name_data_z = str(self.z_axis_name_input.get())
        auto_scale_factor = np.float64(self.auto_scale_factor_input.get())
        transformed_x = self.X * np.float64(self.x_axis_scale_input.get())
        transformed_y = self.Y * np.float64(self.y_axis_scale_input.get())
        transformed_data = self.sliced_data * np.float64(self.z_axis_scale_input.get())
        if self.use_trace_wise_min_max_scaling_var.get():
            transformed_data = trace_wise_min_max_scaling(transformed_data)

        with self.data_operation('Scale or rename axes and data'):
            self.name_data_x_axis = name_data_x_axis
            self.name_data_y_axis = name_data_y_axis
            self.name_data_z = name_data_z
            self.auto_scale_factor = auto_scale_factor
            self.X = transformed_x
            self.Y = transformed_y
            self.sliced_data = transformed_data
            self.xlim = [np.min(self.X), np.max(self.X)]
            self.ylim = [np.min(self.Y), np.max(self.Y)]
            if self.auto_scale_var.get():
                self.apply_auto_scaling()

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_interpolation(self):
        if self.interpolation_enabled:
            if self.trace_axis_map_mode:
                x, y, sliced_data = self._trace_axis_map_arrays()
            else:
                x = self._slice_current_scan_grid(
                    self._scan_coordinate_grid(self.x_scan_axis_index)
                )
                y = self._slice_current_scan_grid(
                    self._scan_coordinate_grid(self.y_scan_axis_index)
                )
                sliced_data = self._slice_current_scan_grid(
                    self._selected_signal_grid()
                )
            interpolated_x, interpolated_y, interpolated_data = image_down_sampling(
                sliced_data, x, y, (self.interpolation_entry_1.get(), self.interpolation_entry_2.get())
            )
            with self.data_operation('Interpolation'):
                self.X, self.Y, self.sliced_data = interpolated_x, interpolated_y, interpolated_data
                self.xlim = [np.min(self.X), np.max(self.X)]
                self.ylim = [np.min(self.Y), np.max(self.Y)]
            self.update_pcolormesh(self.vmin, self.vmax)

    def apply_poly_bg(self):
        if self.use_roi_in_bg_subtraction.get() and self.roi_mode:
            xmin, xmax = sorted([self.roi_corners[0][0], self.roi_corners[1][0]])
            ymin, ymax = sorted([self.roi_corners[0][1], self.roi_corners[1][1]])
            bg = evaluate_poly_background_2d(self.X, self.Y, self.sliced_data, int(self.poly_order_x.get()),
                                             int(self.poly_order_y.get()), (xmin, xmax), (ymin, ymax))
        else:
            bg = evaluate_poly_background_2d(self.X, self.Y, self.sliced_data, int(self.poly_order_x.get()),
                                             int(self.poly_order_y.get()))

        corrected_data = self.sliced_data - bg
        with self.data_operation('Polynomial background subtraction'):
            self.sliced_data = corrected_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_median_difference(self):
        corrected_data = correct_median_diff(self.sliced_data)
        with self.data_operation('Median-difference correction'):
            self.sliced_data = corrected_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_mean_of_lines(self):
        corrected_data = correct_mean_of_lines(self.sliced_data)
        with self.data_operation('Mean-of-lines correction'):
            self.sliced_data = corrected_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_relation_parameters(self):
        corrected_data = (self.sliced_data + np.float64(self.relation_parameter_entry_list[0].get()) *
                          self.X ** np.float64(self.relation_parameter_entry_list[1].get()) +
                          np.float64(self.relation_parameter_entry_list[2].get()) *
                          self.Y ** np.float64(self.relation_parameter_entry_list[3].get())
                          + np.float64(self.relation_parameter_entry_list[4].get()))

        with self.data_operation('Relation-parameter correction'):
            self.sliced_data = corrected_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_auto_scaling(self):
        self.vmin = np.mean(self.sliced_data) - self.auto_scale_factor * np.std(self.sliced_data)
        self.vmax = np.mean(self.sliced_data) + self.auto_scale_factor * np.std(self.sliced_data)

    def apply_subtract_trace_average(self):
        # Extract parameters from input fields
        n = int(self.n_traces_entry.get())
        axis = int(self.axis_entry.get())
        from_end = self.from_end_var.get() == "True"
        use_filter = self.use_filter_var.get() == "True"
        poly_fit = self.poly_fit_var.get() == "True"
        filter_sigma = float(self.filter_sigma_entry.get())
        poly_fit_order = int(self.poly_fit_order_entry.get())

        # Apply the `subtract_trace_average` function
        corrected_data = subtract_trace_average(
            self.sliced_data,
            n=n,
            axis=axis,
            from_end=from_end,
            use_filter=use_filter,
            poly_fit=poly_fit,
            filter_sigma=filter_sigma,
            poly_fit_order=poly_fit_order
        )
        with self.data_operation('Subtract trace average'):
            self.sliced_data = corrected_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_gaussian_filter(self):
        filtered_data = gaussian_filter(
            self.sliced_data, (float(self.filter_pixel_x.get()), float(self.filter_pixel_y.get()))
        )
        with self.data_operation('Gaussian filter'):
            self.sliced_data = filtered_data
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_derivative(self):
        # Calculate mean spacing between X and Y coordinates
        dx = np.mean((np.diff(self.X, axis=1)).flatten())
        dy = np.mean((np.diff(self.Y, axis=0)).flatten())
        # Calculate gradient
        derivative_data = np.gradient(
            self.sliced_data, dx, dy
        )[self.axis_selection.index(self.derivative_combobox.get())]
        with self.data_operation('Derivative'):
            self.sliced_data = derivative_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_sum_of_gradient(self):
        # Calculate mean spacing between X and Y coordinates
        dx = np.mean((np.diff(self.X, axis=1)).flatten())
        dy = np.mean((np.diff(self.Y, axis=0)).flatten())
        # Calculate gradient
        gradient_data = np.sqrt(np.gradient(self.sliced_data, dx, dy)[0] ** 2
                                + np.gradient(self.sliced_data, dx, dy)[1] ** 2)
        with self.data_operation('Norm of gradient'):
            self.sliced_data = gradient_data
            if self.auto_scale_var.get():
                self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_2d_fft(self):
        fft_x, fft_y, fft_data = two_d_fft_on_data(self.sliced_data, self.X, self.Y, mode='Amplitude')
        with self.data_operation('2-D FFT'):
            self.X, self.Y, self.sliced_data = fft_x, fft_y, fft_data
            self.name_data_z = 'FFT Amp of ' + self.name_data_z
            self.name_data_x_axis = 'freq. of ' + self.name_data_x_axis
            self.name_data_y_axis = 'freq. of ' + self.name_data_y_axis
            self.xlim = [np.min(self.X), np.max(self.X)]
            self.ylim = [np.min(self.Y), np.max(self.Y)]
            self.apply_auto_scaling()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_roi_data_cut(self):
        try:
            # Convert entry values to floats
            x_range = tuple(sorted((
                float(self.roi_cut_entry_list[0].get()),
                float(self.roi_cut_entry_list[2].get()),
            )))
            y_range = tuple(sorted((
                float(self.roi_cut_entry_list[1].get()),
                float(self.roi_cut_entry_list[3].get()),
            )))

            # Apply the cut
            result = cut_data_range(self.X, self.Y, self.sliced_data, x_range, y_range)
            if result is not None:
                # Calculate new min/max if data exists
                if result[2].size > 0:
                    with self.data_operation('ROI cut'):
                        self.X, self.Y, self.sliced_data = result
                        if self.auto_scale_var.get():
                            self.apply_auto_scaling()
                        self.xlim = (np.min(self.X), np.max(self.X))
                        self.ylim = (np.min(self.Y), np.max(self.Y))
                    self.update_plot()
                    self.roi_data_cut_window.destroy()  # Close window on success
                else:
                    messagebox.showwarning(
                        "Empty ROI",
                        "No coordinate points lie inside the selected range."
                    )
            else:
                messagebox.showerror("Error", "Failed to apply ROI cut")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric ROI limits.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def apply_fft_trace_correction(self):
        """
        Apply FFT trace correction to adjust traces based on FFT-derived cuts.

        This method applies Fast Fourier Transform (FFT) correction by obtaining
        cuts from the FFT axis, closing the FFT plot figure, and then applying
        the correction to the traces using these cuts. The function subsequently
        destroys and updates the FFT plot window to complete the operation.

        :raises RuntimeError: If the FFT correction process encounters an error.

        """
        #### Modified by Nico Reinders ####
        self.cuts = get_cuts(self.fft_ax)
        print(self.cuts)
        plt.close(self.fft_fig)

        self.traces = fft_correction_apply(self.traces, self.cuts, self.frequencies_shifted, self.fft_signals,
                                           self.original_angles)
        self.fft_plot_window.destroy()
        self.fft_plot_window.update()

    def _ensure_line_preview(self):
        """Create the native line used only while choosing an endpoint."""
        if self.line_preview_item is not None:
            return
        tk_canvas = self.canvas.get_tk_widget()
        self.line_preview_item = tk_canvas.create_line(
            0,
            0,
            0,
            0,
            fill='red',
            width=2,
            state='hidden',
        )
        tk_canvas.tag_raise(self.line_preview_item)

    def _queue_line_preview(self, start_point, end_point):
        """Retain only the newest temporary line endpoint for rendering."""
        self._pending_line_preview = (tuple(start_point), tuple(end_point))
        if self._line_preview_after_id is None:
            self._line_preview_after_id = self.root.after(
                self._overlay_update_interval_ms,
                self._flush_line_preview,
            )

    def _flush_line_preview(self):
        self._line_preview_after_id = None
        preview = self._pending_line_preview
        self._pending_line_preview = None
        if preview is None or not self.drawing_line:
            return
        self._ensure_line_preview()
        start_point, end_point = preview
        x1, y1 = self._data_to_tk_canvas_point(*start_point)
        x2, y2 = self._data_to_tk_canvas_point(*end_point)
        tk_canvas = self.canvas.get_tk_widget()
        tk_canvas.coords(self.line_preview_item, x1, y1, x2, y2)
        tk_canvas.itemconfigure(self.line_preview_item, state='normal')

    def _hide_line_preview(self):
        self._pending_line_preview = None
        self._cancel_scheduled_callback('_line_preview_after_id')
        if self.line_preview_item is not None:
            self.canvas.get_tk_widget().itemconfigure(
                self.line_preview_item,
                state='hidden',
            )

    def _cancel_in_progress_line_drawing(self):
        """Discard an unfinished native preview without redrawing the map."""
        self.drawing_line = False
        self.current_line = None
        self._hide_line_preview()

    def activate_line_drawing(self):
        if hasattr(self, 'roi_mode') and self.roi_mode:
            self.toggle_roi()  # Turn off ROI before enabling line drawing

        if self.click_cid is not None:
            self.canvas.mpl_disconnect(self.click_cid)
        if self.move_cid is not None:
            self.canvas.mpl_disconnect(self.move_cid)
        self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.move_cid = self.canvas.mpl_connect('motion_notify_event', self.on_canvas_move)

    def deactivate_line_drawing(self):
        if self.click_cid is not None:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None
        if self.move_cid is not None:
            self.canvas.mpl_disconnect(self.move_cid)
            self.move_cid = None
        self._cancel_in_progress_line_drawing()

    def reset_lines(self):
        self._cancel_in_progress_line_drawing()
        self.drawn_lines_list = []
        self.linecut_settings_list = []
        self.update_lines_listbox()
        self.redraw_saved_lines()
        self.canvas.draw_idle()

    def setup_line_editing(self):
        """Set up the line editing functionality with right-click context menu"""
        # Initialize editing state variables
        self.editing_line = False
        self.editing_line_index = None
        self.editing_point = None  # 0 for start point, 1 for end point
        self.edit_markers = []

        # Create a right-click context menu for the lines listbox
        self.lines_context_menu = ttk.Menu(self.lines_listbox, tearoff=0)
        self.lines_context_menu.add_command(label="Edit Line", command=self.start_line_editing)
        self.lines_context_menu.add_command(label="Extract Line", command=self.extract_and_plot_linecut)
        self.lines_context_menu.add_command(label="Open Menu", command=self.open_linecut_settings_window)
        self.lines_context_menu.add_command(label="Delete Line", command=self.delete_selected_line)

        # Bind right-click event to show context menu
        self.lines_listbox.bind("<Button-3>", self.show_lines_context_menu)

        # Also bind double-click as a quick way to edit a line
        self.lines_listbox.bind("<Double-Button-1>", lambda event: self.start_line_editing())

    @staticmethod
    def _default_linecut_settings():
        return {
            'width_pixels': 1,
            'use_average': False,
            'extract_multiple': False,
            'number_of_linecuts': 3,
            'orthogonal': False,
        }

    def _get_linecut_settings(self, line_index):
        while len(self.linecut_settings_list) <= line_index:
            self.linecut_settings_list.append(self._default_linecut_settings())
        return self.linecut_settings_list[line_index]

    def open_linecut_settings_window(self):
        """Open extraction and geometry settings for the selected drawn line."""
        selected_indices = self.lines_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a line to configure.")
            return

        selected_index = selected_indices[0]
        if selected_index >= len(self.drawn_lines_list):
            messagebox.showinfo("Invalid Selection", "The selected line no longer exists.")
            return

        self.linecut_settings_index = selected_index
        settings = self._get_linecut_settings(selected_index)
        if (hasattr(self, 'linecut_settings_window') and
                self.linecut_settings_window.winfo_exists()):
            self.linecut_settings_window.destroy()

        self.linecut_settings_window = ttk.Toplevel(self.root)
        self.linecut_settings_window.title(f"Linecut Settings - Line {selected_index + 1}")
        self.linecut_settings_window.geometry("430x310")
        self.linecut_settings_window.resizable(False, False)
        self.linecut_settings_window.columnconfigure(1, weight=1)

        ttk.Label(
            self.linecut_settings_window,
            text="Linecut width (pixels):"
        ).grid(row=0, column=0, padx=10, pady=8, sticky=tk.W)
        self.linecut_width_var = tk.StringVar(value=str(settings['width_pixels']))
        ttk.Entry(
            self.linecut_settings_window,
            textvariable=self.linecut_width_var,
            width=10
        ).grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)

        self.linecut_average_var = tk.BooleanVar(value=settings['use_average'])
        self._add_linecut_boolean_row(
            row=1,
            label="Use average:",
            variable=self.linecut_average_var,
        )
        self.linecut_multiple_var = tk.BooleanVar(value=settings['extract_multiple'])
        self._add_linecut_boolean_row(
            row=2,
            label="Extract multiple:",
            variable=self.linecut_multiple_var,
        )

        ttk.Label(
            self.linecut_settings_window,
            text="Number of linecuts (N):"
        ).grid(row=3, column=0, padx=10, pady=8, sticky=tk.W)
        self.linecut_count_var = tk.StringVar(value=str(settings['number_of_linecuts']))
        ttk.Entry(
            self.linecut_settings_window,
            textvariable=self.linecut_count_var,
            width=10
        ).grid(row=3, column=1, padx=10, pady=8, sticky=tk.W)

        self.linecut_orthogonal_var = tk.BooleanVar(value=settings['orthogonal'])
        self._add_linecut_boolean_row(
            row=4,
            label="Orthogonal cut:",
            variable=self.linecut_orthogonal_var,
        )

        ttk.Label(
            self.linecut_settings_window,
            text=("Average returns one profile across the selected width.\n"
                  "Extract multiple returns N parallel profiles within that width."),
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=2, padx=10, pady=8, sticky=tk.W)

        ttk.Button(
            self.linecut_settings_window,
            text="Apply",
            command=self.apply_linecut_settings,
            bootstyle='primary',
        ).grid(row=6, column=0, columnspan=2, pady=10)

    def _add_linecut_boolean_row(self, row, label, variable):
        ttk.Label(self.linecut_settings_window, text=label).grid(
            row=row, column=0, padx=10, pady=8, sticky=tk.W
        )
        choice_frame = ttk.Frame(self.linecut_settings_window)
        choice_frame.grid(row=row, column=1, padx=10, pady=8, sticky=tk.W)
        ttk.Radiobutton(
            choice_frame, text="Off", variable=variable, value=False
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            choice_frame, text="On", variable=variable, value=True
        ).pack(side=tk.LEFT)

    def apply_linecut_settings(self):
        """Validate and save settings for the selected drawn line."""
        try:
            width_pixels = int(self.linecut_width_var.get())
            number_of_linecuts = int(self.linecut_count_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Linecut Settings",
                "Width and number of linecuts must be whole numbers."
            )
            return
        if width_pixels < 1 or number_of_linecuts < 1:
            messagebox.showerror(
                "Invalid Linecut Settings",
                "Width and number of linecuts must both be at least 1."
            )
            return

        line_index = self.linecut_settings_index
        settings = self._get_linecut_settings(line_index)
        settings.update({
            'width_pixels': width_pixels,
            'use_average': bool(self.linecut_average_var.get()),
            'extract_multiple': bool(self.linecut_multiple_var.get()),
            'number_of_linecuts': number_of_linecuts,
            'orthogonal': bool(self.linecut_orthogonal_var.get()),
        })
        if settings['orthogonal']:
            start_point, end_point = self.drawn_lines_list[line_index]
            self.drawn_lines_list[line_index] = list(
                self._orthogonalize_line(start_point, end_point)
            )

        self.update_lines_listbox()
        self.redraw_saved_lines()
        self.canvas.draw_idle()
        self.linecut_settings_window.destroy()

    def _orthogonalize_line(self, start_point, end_point):
        """Lock a line to its visually dominant horizontal or vertical axis."""
        start_display = self.ax.transData.transform(start_point)
        end_display = self.ax.transData.transform(end_point)
        display_delta = np.abs(end_display - start_display)
        if display_delta[0] >= display_delta[1]:
            constrained_end = (end_point[0], start_point[1])
        else:
            constrained_end = (start_point[0], end_point[1])
        return tuple(start_point), constrained_end

    @staticmethod
    def _linecut_extraction_offsets(settings):
        """Return average, multiple, and de-duplicated extraction offsets."""
        average_offsets = np.array([], dtype=float)
        multiple_offsets = np.array([], dtype=float)
        if settings['use_average']:
            width = settings['width_pixels']
            average_offsets = np.arange(width, dtype=float) - (width - 1) / 2
        if settings['extract_multiple']:
            count = settings['number_of_linecuts']
            if count == 1:
                multiple_offsets = np.array([0.0])
            else:
                half_width = settings['width_pixels'] / 2
                multiple_offsets = np.linspace(-half_width, half_width, count)
        if not settings['use_average'] and not settings['extract_multiple']:
            multiple_offsets = np.array([0.0])

        unique_offsets = []
        seen_offsets = set()
        for offset in np.concatenate((average_offsets, multiple_offsets)):
            key = round(float(offset), 12)
            if key not in seen_offsets:
                seen_offsets.add(key)
                unique_offsets.append(float(offset))
        return average_offsets, multiple_offsets, unique_offsets

    def extract_and_plot_linecut(self):
        """Extract linecut profiles in a worker and report determinate progress."""
        selected_indices = self.lines_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a line to edit.")
            return

        selected_index = selected_indices[0]
        if selected_index >= len(self.drawn_lines_list):
            messagebox.showinfo("Invalid Selection", "The selected line no longer exists.")
            return
        if getattr(self, 'linecut_extraction_running', False):
            messagebox.showinfo(
                "Linecut Extraction",
                "Another linecut extraction is already running."
            )
            return

        self.editing_line_index = selected_index
        line = self.drawn_lines_list[self.editing_line_index]
        start_point = tuple(line[0])
        end_point = tuple(line[1])
        settings = dict(self._get_linecut_settings(selected_index))
        average_offsets, multiple_offsets, unique_offsets = (
            self._linecut_extraction_offsets(settings)
        )
        total_profiles = len(unique_offsets)

        # Copy map state before entering the worker so it cannot observe a map
        # transformation or undo operation performed during extraction.
        x_grid = np.array(self.X, copy=True)
        y_grid = np.array(self.Y, copy=True)
        data_grid = np.array(self.sliced_data, copy=True)

        progress_window = ttk.Toplevel(self.root)
        progress_window.title("Linecut Extraction")
        progress_window.geometry("460x150")
        progress_window.transient(self.root)
        progress_label_var = tk.StringVar(
            value=f"Preparing {total_profiles} profile(s)…"
        )
        ttk.Label(
            progress_window,
            textvariable=progress_label_var,
        ).pack(padx=20, pady=(15, 6), anchor=tk.W)
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_window,
            variable=progress_var,
            maximum=total_profiles,
            length=400,
        )
        progress_bar.pack(padx=20, pady=8, fill=tk.X)

        cancel_event = threading.Event()

        def cancel_extraction():
            cancel_event.set()
            progress_label_var.set("Cancelling after the current profile…")
            cancel_button.config(state=tk.DISABLED)

        cancel_button = ttk.Button(
            progress_window,
            text="Cancel",
            command=cancel_extraction,
        )
        cancel_button.pack(padx=20, pady=(0, 15), anchor=tk.E)
        progress_window.protocol('WM_DELETE_WINDOW', cancel_extraction)

        message_queue = queue.Queue()
        self.linecut_extraction_running = True

        def worker():
            try:
                profiles_by_offset = {}
                num_points = None
                for completed, offset in enumerate(unique_offsets, start=1):
                    if cancel_event.is_set():
                        message_queue.put(("cancelled", None))
                        return
                    profile = extract_linecut(
                        x_grid,
                        y_grid,
                        data_grid,
                        start_point,
                        end_point,
                        offset_pixels=offset,
                        num_points=num_points,
                    )
                    if num_points is None:
                        num_points = len(profile)
                    profiles_by_offset[round(float(offset), 12)] = profile
                    message_queue.put(("progress", completed))

                if cancel_event.is_set():
                    message_queue.put(("cancelled", None))
                    return

                extracted_profiles = []
                if settings['use_average']:
                    average_profiles = [
                        profiles_by_offset[round(float(offset), 12)]
                        for offset in average_offsets
                    ]
                    average_profile = np.nanmean(
                        np.stack(average_profiles), axis=0
                    )
                    extracted_profiles.append((
                        average_profile,
                        (f"Line {selected_index + 1} average "
                         f"({settings['width_pixels']} px)"),
                    ))

                if settings['extract_multiple']:
                    count = settings['number_of_linecuts']
                    for profile_index, offset in enumerate(
                            multiple_offsets, start=1):
                        extracted_profiles.append((
                            profiles_by_offset[round(float(offset), 12)],
                            (f"Line {selected_index + 1} cut "
                             f"{profile_index}/{count} ({offset:+.2f} px)"),
                        ))

                if not settings['use_average'] and not settings['extract_multiple']:
                    extracted_profiles.append((
                        profiles_by_offset[round(float(unique_offsets[0]), 12)],
                        f"Line {selected_index + 1}",
                    ))

                total_distance = np.linalg.norm(
                    np.subtract(end_point, start_point, dtype=float)
                )
                distance = np.linspace(0, total_distance, num_points)
                message_queue.put((
                    "done", (distance, extracted_profiles)
                ))
            except Exception as error:
                message_queue.put(("error", str(error)))

        def finish_progress_window(delay=0):
            self.linecut_extraction_running = False
            if progress_window.winfo_exists():
                progress_window.after(delay, progress_window.destroy)

        def poll_worker():
            try:
                while True:
                    message_type, payload = message_queue.get_nowait()
                    if message_type == "progress":
                        progress_var.set(payload)
                        progress_label_var.set(
                            f"Extracting profiles… {payload}/{total_profiles}"
                        )
                    elif message_type == "cancelled":
                        progress_label_var.set("Extraction cancelled.")
                        finish_progress_window(delay=300)
                        return
                    elif message_type == "error":
                        progress_label_var.set("Extraction failed.")
                        cancel_button.config(state=tk.DISABLED)
                        messagebox.showerror(
                            "Linecut Extraction Failed",
                            payload,
                            parent=progress_window,
                        )
                        finish_progress_window(delay=300)
                        return
                    elif message_type == "done":
                        distance, extracted_profiles = payload
                        progress_var.set(total_profiles)
                        progress_label_var.set("Extraction complete.")
                        cancel_button.config(state=tk.DISABLED)
                        if (not hasattr(self, 'linecut_plotter') or
                                not hasattr(self.linecut_plotter, 'root') or
                                not self.linecut_plotter.root.winfo_exists()):
                            self.linecut_plotter = UtilityLinePlotter(
                                self.root,
                                color_cycle_name=self.plot_style[
                                    'extracted_linecut_color_cycle'
                                ],
                                color_cycle_change_callback=(
                                    self._on_extracted_linecut_cycle_selected
                                ),
                            )
                            self.linecut_plotter.root.protocol(
                                "WM_DELETE_WINDOW", self.on_linecut_window_close
                            )
                        for profile, label in extracted_profiles:
                            self.linecut_plotter.add_linecut(
                                distance, profile, label=label
                            )
                        finish_progress_window(delay=300)
                        return
            except queue.Empty:
                pass
            if progress_window.winfo_exists():
                progress_window.after(50, poll_worker)

        threading.Thread(target=worker, daemon=True).start()
        poll_worker()

    def show_lines_context_menu(self, event):
        """Show the context menu on right-click in the lines listbox"""
        # Get the line index at the current mouse position
        try:
            index = self.lines_listbox.nearest(event.y)
            # Only show menu if we clicked on an actual line
            if index < len(self.drawn_lines_list):
                # Select the line first
                self.lines_listbox.selection_clear(0, tk.END)
                self.lines_listbox.selection_set(index)
                self.lines_listbox.activate(index)
                # Show the context menu
                self.lines_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            # Make sure to release the menu
            self.lines_context_menu.grab_release()

    def start_line_editing(self):
        """Start editing the selected line"""
        # Get selected line index from listbox
        selected_indices = self.lines_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a line to edit.")
            return

        selected_index = selected_indices[0]
        if selected_index >= len(self.drawn_lines_list):
            messagebox.showinfo("Invalid Selection", "The selected line no longer exists.")
            return

        # Enter editing mode
        self.editing_line = True
        self.editing_line_index = selected_index

        # Disconnect any existing drawing events temporarily
        self.disconnect_drawing_events()

        # Add markers at the endpoints of the selected line
        self.show_endpoint_markers(selected_index)

        # Set up click handler for editing
        self.edit_click_cid = self.canvas.mpl_connect('button_press_event', self.on_edit_click)

        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set(
                "Click and drag a line endpoint (green/blue marker). Right-click or press Esc to finish.")

        # Bind Escape key to finish editing
        self.canvas.get_tk_widget().bind("<Escape>", lambda e: self.finish_line_editing())

        # Also bind right-click to finish editing
        self.edit_right_click_cid = self.canvas.mpl_connect('button_press_event',
                                                            lambda
                                                                event: self.finish_line_editing() if event.button == 3 else None)

    def disconnect_drawing_events(self):
        """Disconnect any active drawing event handlers"""
        if hasattr(self, 'click_cid') and self.click_cid is not None:
            self.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None

        if hasattr(self, 'move_cid') and self.move_cid is not None:
            self.canvas.mpl_disconnect(self.move_cid)
            self.move_cid = None

        # Cancel any in-progress line drawing
        if hasattr(self, 'drawing_line') and self.drawing_line:
            self._cancel_in_progress_line_drawing()

    def show_endpoint_markers(self, line_index):
        """Display markers at the endpoints of the selected line"""
        # Clear any existing markers
        self.clear_edit_markers()

        # Get the line coordinates

        line = self.drawn_lines_list[line_index]
        start_point, end_point = line

        # Create markers for start and end points with different colors
        start_marker = self.ax.plot([start_point[0]], [start_point[1]], 'go',

                                    markersize=10, alpha=0.7, zorder=10)[0]
        end_marker = self.ax.plot([end_point[0]], [end_point[1]], 'bo',

                                  markersize=10, alpha=0.7, zorder=10)[0]

        # Store the markers
        self.edit_markers = [start_marker, end_marker]

        # Draw the updated plot
        self.canvas.draw_idle()

    def clear_edit_markers(self):
        """Remove all endpoint markers"""
        for marker in self.edit_markers:
            if marker in self.ax.lines:
                marker.remove()
        self.edit_markers = []
        self.canvas.draw_idle()

    def on_edit_click(self, event):
        """Handle clicks when in line editing mode"""
        if not event.inaxes or event.inaxes != self.ax or event.button != 1:  # Only handle left clicks
            return

        if not self.editing_line:
            return

        # Get coordinates of the selected line
        line = self.drawn_lines_list[self.editing_line_index]
        start_point, end_point = line

        # Calculate distances to both endpoints
        dist_to_start = np.sqrt((event.xdata - start_point[0]) ** 2 +
                                (event.ydata - start_point[1]) ** 2)
        dist_to_end = np.sqrt((event.xdata - end_point[0]) ** 2 +
                              (event.ydata - end_point[1]) ** 2)

        # Define a threshold for selecting a point (in data units)
        threshold = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.05

        # Check if we clicked near an endpoint
        if dist_to_start < threshold or dist_to_end < threshold:
            # Determine which point is closer
            if dist_to_start < dist_to_end:
                self.editing_point = 0  # Start point
            else:
                self.editing_point = 1  # End point

            # Connect the motion event for dragging
            self.edit_motion_cid = self.canvas.mpl_connect(
                'motion_notify_event', self.on_edit_motion)
            # Connect the release event
            self.edit_release_cid = self.canvas.mpl_connect(
                'button_release_event', self.on_edit_release)

    def on_edit_motion(self, event):
        """Handle mouse motion during endpoint dragging"""
        if not event.inaxes or not self.editing_line or self.editing_point is None:
            return

        # Get the current line
        line = self.drawn_lines_list[self.editing_line_index]
        start_point, end_point = line

        # Update the appropriate endpoint
        if self.editing_point == 0:  # Start point
            new_start = (event.xdata, event.ydata)
            settings = self._get_linecut_settings(self.editing_line_index)
            if settings['orthogonal']:
                _, new_start = self._orthogonalize_line(end_point, new_start)
            self.drawn_lines_list[self.editing_line_index] = [new_start, end_point]
            self.edit_markers[0].set_data([new_start[0]], [new_start[1]])
        else:  # End point
            new_end = (event.xdata, event.ydata)
            settings = self._get_linecut_settings(self.editing_line_index)
            if settings['orthogonal']:
                _, new_end = self._orthogonalize_line(start_point, new_end)
            self.drawn_lines_list[self.editing_line_index] = [start_point, new_end]
            self.edit_markers[1].set_data([new_end[0]], [new_end[1]])

        # Update the line drawing
        self.redraw_saved_lines()
        self.canvas.draw_idle()

    def on_edit_release(self, event):
        """Handle mouse release after dragging endpoint"""
        if not self.editing_line or self.editing_point is None:
            return

        # Disconnect motion and release events
        if hasattr(self, 'edit_motion_cid'):
            self.canvas.mpl_disconnect(self.edit_motion_cid)
            self.edit_motion_cid = None

        if hasattr(self, 'edit_release_cid'):
            self.canvas.mpl_disconnect(self.edit_release_cid)
            self.edit_release_cid = None

        # Reset editing point
        self.editing_point = None

        # Update the listbox with new coordinates
        self.update_lines_listbox()

    def on_linecut_window_close(self):
        """Handle the closing of the linecut plotter window."""
        if hasattr(self, 'linecut_plotter') and hasattr(self.linecut_plotter, 'root'):
            # Explicitly destroy the window
            self.linecut_plotter.root.destroy()
            # Remove our reference to the plotter
            delattr(self, 'linecut_plotter')

    def finish_line_editing(self):
        """Exit line editing mode"""
        if not self.editing_line:
            return

        self.editing_line = False
        self.editing_line_index = None

        # Disconnect edit events
        if hasattr(self, 'edit_click_cid'):
            self.canvas.mpl_disconnect(self.edit_click_cid)
            self.edit_click_cid = None

        if hasattr(self, 'edit_right_click_cid'):
            self.canvas.mpl_disconnect(self.edit_right_click_cid)
            self.edit_right_click_cid = None

        # Clear markers
        self.clear_edit_markers()

        # Unbind escape key
        self.canvas.get_tk_widget().unbind("<Escape>")

        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set("Line editing completed.")

        # Redraw to remove any highlighting
        self.redraw_saved_lines()
        self.canvas.draw_idle()

    def delete_selected_line(self):
        """Delete the selected line"""
        # Get selected line index from listbox
        selected_indices = self.lines_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a line to delete.")
            return

        selected_index = selected_indices[0]
        if selected_index >= len(self.drawn_lines_list):
            messagebox.showinfo("Invalid Selection", "The selected line no longer exists.")
            return

        # Remove the line from our list
        self.drawn_lines_list.pop(selected_index)
        if selected_index < len(self.linecut_settings_list):
            self.linecut_settings_list.pop(selected_index)

        # Update the listbox and redraw
        self.update_lines_listbox()
        self.redraw_saved_lines()
        self.canvas.draw_idle()

        # Update status
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Line {selected_index + 1} deleted.")

    # Update the redraw_saved_lines function to handle highlighting the edited line
    def redraw_saved_lines(self):
        """Redraw all saved lines after clearing the plot."""
        # Remove existing line artists from the plot
        if hasattr(self, 'line_artists'):
            for line_artist in self.line_artists:
                if line_artist in self.ax.lines:
                    line_artist.remove()

        # Create a new list for line artists
        self.line_artists = []

        # Redraw each line from the drawn_lines_list
        for i, line in enumerate(self.drawn_lines_list):
            start_point, end_point = line
            x_values = [start_point[0], end_point[0]]
            y_values = [start_point[1], end_point[1]]
            settings = self._get_linecut_settings(i)

            # Highlight the selected line if in editing mode
            if self.editing_line and i == self.editing_line_index:
                line_color = 'purple'
                line_width = 2
                line_zorder = 6
            else:
                line_color = 'red'
                line_width = 1
                line_zorder = 5

            line_artist, = self.ax.plot(
                x_values,
                y_values,
                color=line_color,
                linewidth=line_width,
                zorder=line_zorder,
            )

            self.line_artists.append(line_artist)

            pixel_normal = get_linecut_pixel_normal(
                self.X, self.Y, start_point, end_point
            )
            half_width_vector = pixel_normal * settings['width_pixels'] / 2
            for endpoint in (np.asarray(start_point), np.asarray(end_point)):
                whisker_start = endpoint - half_width_vector
                whisker_end = endpoint + half_width_vector
                whisker, = self.ax.plot(
                    [whisker_start[0], whisker_end[0]],
                    [whisker_start[1], whisker_end[1]],
                    color=line_color,
                    linewidth=line_width,
                    zorder=line_zorder,
                )
                self.line_artists.append(whisker)

    def update_lines_listbox(self):
        self.lines_listbox.delete(0, tk.END)  # Clear the current contents of the listbox
        for i, line in enumerate(self.drawn_lines_list, start=1):
            # Assuming each line is a tuple of start and end points like ((x1, y1), (x2, y2))
            start_point, end_point = np.round(line, 4)
            width = self._get_linecut_settings(i - 1)['width_pixels']
            line_str = (
                f"Line {i}: Start {start_point} End {end_point} Width {width} px"
            )
            self.lines_listbox.insert(tk.END, line_str)

    def on_canvas_click(self, event):
        if (
            event.inaxes != self.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return  # Ignore clicks outside the axes

        if not self.drawing_line:
            # Start drawing a new line
            self.drawing_line = True
            self.current_line = [(event.xdata, event.ydata), (event.xdata, event.ydata)]
            self._queue_line_preview(
                self.current_line[0],
                self.current_line[1],
            )
        else:
            # Finalize the current line
            self.drawing_line = False
            # Update the final point of the line
            self.current_line[1] = (event.xdata, event.ydata)
            self.drawn_lines_list.append(
                [tuple(self.current_line[0]), tuple(self.current_line[1])]
            )
            self.linecut_settings_list.append(self._default_linecut_settings())
            self._hide_line_preview()
            self.update_lines_listbox()
            self.redraw_saved_lines()
            self.current_line = None  # Reset for the next line
            self.canvas.draw_idle()

    def on_canvas_move(self, event):
        if (
            event.inaxes != self.ax
            or not self.drawing_line
            or event.xdata is None
            or event.ydata is None
        ):
            return  # Ignore if we're not in the process of drawing a line

        # Update the end point of the current line to follow the mouse
        self.current_line[1] = (event.xdata, event.ydata)
        self._queue_line_preview(
            self.current_line[0],
            self.current_line[1],
        )

    def _draw_main_map(self, vmin, vmax):
        """Render only the prepared float32 display grid on the main canvas."""
        self.ax.clear()
        if hasattr(self, 'cbar'):
            self.cbar.remove()
            del self.cbar

        c = self.ax.pcolormesh(
            self.display_X,
            self.display_Y,
            self.display_sliced_data,
            cmap=self.colormap_combobox.get(),
            vmin=vmin,
            vmax=vmax,
            shading='auto',
            zorder=1,
            linewidth=0,
            rasterized=True,
        )
        # NavigationToolbar2 otherwise calls QuadMesh.contains() for every
        # motion event while composing its status message. That cell-by-cell
        # hit test dominates crosshair, ROI, and line interaction on large
        # maps; coordinates and pan/zoom continue to work without it.
        c.mouseover = False
        self.cbar = self.figure.colorbar(c, ax=self.ax, label=self.name_data_z)
        self.ax.set_xlabel(self.name_data_x_axis)
        self.ax.set_ylabel(self.name_data_y_axis)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        # Redraw all the lines in drawn_lines_list
        self.redraw_saved_lines()
        configure_transparent_matplotlib_canvas(self.figure, self.canvas)

        self.canvas.draw_idle()

    def update_pcolormesh(self, vmin, vmax):
        """Refresh the display layer after an explicit data/view operation."""
        self._prepare_display_arrays(force=True)
        self._draw_main_map(vmin, vmax)

    def update_histogramm(self):
        self.histogram_ax.clear()
        self.histogram_ax.hist(
            self.sliced_data.flatten(),
            bins=60,
            color=self._current_plot_line_color(),
            alpha=0.7,
        )
        self.histogram_ax.set_yticklabels([])
        self.histogram_ax.set_xticklabels([])

        self.init_movable_lines()
        configure_transparent_matplotlib_canvas(
            self.histogram_fig,
            self.histogram_canvas,
        )
        self.histogram_canvas.draw_idle()

    def update_plot(self):
        # feature does not work as intended and leads to some weird behaviour which allows to use those bugs as a feature
        if self.invert_enabled:
            with self.data_operation('Invert axes'):
                self.X, self.Y = self.Y.T, self.X.T
                self.sliced_data = self.sliced_data.T
                self.name_data_x_axis, self.name_data_y_axis = self.name_data_y_axis, self.name_data_x_axis
                self.xlim = (np.min(self.X), np.max(self.X))
                self.ylim = (np.min(self.Y), np.max(self.Y))
                self.toggle_invert()
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    ### menu bar functions ###

    def save_data(self):
        pth = filedialog.askdirectory() + '/'
        self.data.set_filename()
        base_name, _ = os.path.splitext(self.data.file_name)
        names_axis = [self.name_data_x_axis, self.name_data_y_axis, self.name_data_z]
        data_and_axis = [self.X, self.Y, self.sliced_data]
        np.save(pth + base_name + 'displayed_tags_.npy', names_axis, allow_pickle=True)
        np.save(pth + 'displayed_data_array_.npy', data_and_axis, allow_pickle=True)

    def save_file(self):
        pth = filedialog.askdirectory() + '/'
        self.data.set_filename()
        base_name, _ = os.path.splitext(self.data.file_name)
        np.save(pth + base_name + 'measurement_axis_tags_.npy', self.data.name_axis, allow_pickle=True)
        np.save(pth + 'measurement_axis_array_.npy', self.data.measure_axis, allow_pickle=True)
        np.save(pth + base_name + 'data_axis_tags_.npy', self.data.name_data, allow_pickle=True)
        np.save(pth + 'data_array_.npy', self.data.measure_data, allow_pickle=True)

    def show_about(self):
        messagebox.showinfo("About", "Interactive Array Plotter\nVersion 0.4.4")

    ### reset functions ###

    def reset(self):
        self.clear_data_operation_history()

        # Clear the plot
        self.ax.clear()
        self.ax_vline.clear()
        self.ax_hline.clear()
        self.histogram_ax.clear()

        # Reset UI elements to their default states
        self.colormap_combobox.set(self.plot_style['preferred_colormap'])
        self.data_combobox.set(self.name_data[0]) if self.name_data else None
        for combobox in self.parameter_comboboxes:
            if combobox['values']:
                combobox.set(combobox['values'][0])

        # Hide additional axes and reset crosshair state
        self.ax_vline.set_visible(False)
        self.ax_hline.set_visible(False)
        self.crosshair_enabled = False
        self.canvas.get_tk_widget().configure(cursor='')
        self._set_crosshair_overlay_state('hidden')
        self._hide_roi_preview()
        self._hide_line_preview()

        # Reset internal data or state as needed
        self.sliced_data = None
        self.X, self.Y = None, None
        self.display_sliced_data = None
        self.display_X, self.display_Y = None, None
        self.xlim, self.ylim = None, None
        self.relation_parameter_entry_list = []
        self.drawn_lines_list = []
        self.linecut_settings_list = []

        # Redraw the canvas to reflect the reset state
        self.canvas.draw_idle()
        self.horizontal_linecut_canvas.draw_idle()
        self.vertical_linecut_canvas.draw_idle()
        self.histogram_canvas.draw_idle()
        for callback_attribute in (
            '_display_resize_after_id',
            '_crosshair_update_after_id',
            '_linecut_update_after_id',
            '_roi_preview_after_id',
            '_line_preview_after_id',
        ):
            self._cancel_scheduled_callback(callback_attribute)
        self.root.destroy()


class InteractiveArrayAndLinePlotter(InteractiveArrayPlotter):
    supports_trace_axis_map = True

    def __init__(
        self,
        root,
        hdf5data,
        plot_style=None,
        plot_style_change_callback=None,
    ):

        self.trace_x_index = 0
        self.trace_y_index = 0

        self.times = 0
        self.trace_xlabel = (
            getattr(hdf5data, 'trace_axis_name', None) or 'Trace X'
        )
        self.trace_ylabel = 'Trace Amplitude (V)'

        self.hist_xlabel = 'Amplitudes (V)'
        self.hist_ylabel = 'Counts'
        self.nbins_traces = 50

        self.trace_selected = 0
        self.enable_hist = False
        # Keep the selected trace on its own canvas so trace changes do not
        # redraw the map canvas.
        self.figure = Figure(figsize=(10, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.trace_figure = Figure(figsize=(8, 2.2), dpi=100)
        self.line_ax = self.trace_figure.add_subplot(111)
        super().__init__(
            root,
            hdf5data,
            self.figure,
            self.ax,
            plot_style=plot_style,
            plot_style_change_callback=plot_style_change_callback,
        )
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
        self.file_menu.add_command(label="Save displayed Trace as NumPy array", command=self.save_trace)

        # Create Trace Menu
        self.trace_menu = ttk.Menu(self.menubar, tearoff=0)
        self.trace_menu.add_command(label="Toggle Histogram", command=self.open_hist_window)
        self.menubar.add_cascade(label="Traces Menu", menu=self.trace_menu)

        self.trace_frame = ttk.Frame(self.plot_area_frame, height=220)
        self.trace_frame.grid(row=2, column=0, sticky=tk.EW)
        self.trace_canvas = FigureCanvasTkAgg(
            self.trace_figure,
            master=self.trace_frame,
        )
        configure_transparent_matplotlib_canvas(
            self.trace_figure,
            self.trace_canvas,
        )
        self.trace_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.initialize_line_plot()
        self.update_line_plot()

    def initialize_line_plot(self):
        # Set up the line plot
        self.line_ax.set_title('Line Plot Title')
        self.line_ax.set_xlabel('X Axis Label')
        self.line_ax.set_ylabel('Y Axis Label')
        self.line_ax.grid(True)

    def plot_data(self):
        super().plot_data()
        if hasattr(self, 'trace_canvas'):
            self.update_line_plot()

    def _after_plot_style_applied(self):
        """Redraw the trace panel with the newly selected shared color."""
        if hasattr(self, 'trace_canvas'):
            self.update_line_plot()

    def update_line_plot(self):
        if self.trace_axis_map_mode:
            self.line_order_indeces = slice_scan_vector_for_axis(
                self._trace_order_scan_grid(),
                self.y_scan_axis_index,
                self._selected_additional_index_map(),
            )
            if (
                np.ndim(self.nan_mask) == 1
                and len(self.nan_mask) == self.line_order_indeces.shape[0]
            ):
                self.line_order_indeces = self.line_order_indeces[
                    self.nan_mask
                ]
            self.trace_y_index = min(
                self.trace_y_index,
                len(self.line_order_indeces) - 1,
            )
            self.trace_x_index = min(
                self.trace_x_index,
                self.X.shape[1] - 1,
            )
            trace_index = int(
                self.line_order_indeces[self.trace_y_index]
            )
        elif self.single_axis_measurement:
            self.line_order_indeces = np.ravel(self.data.trace_order)
            trace_index = int(self.line_order_indeces[self.trace_x_index])
            self.trace_y_index = 0
        else:
            self.line_order_indeces = self._slice_current_scan_grid(
                self._trace_order_scan_grid()
            )
            if self._alternating_x_sweep_enabled():
                self.line_order_indeces = reverse_alternating_rows(
                    self.line_order_indeces
                )
            if (
                np.ndim(self.nan_mask) == 1
                and len(self.nan_mask) == self.line_order_indeces.shape[0]
            ):
                self.line_order_indeces = self.line_order_indeces[
                    self.nan_mask
                ]
            trace_index = int(
                self.line_order_indeces[
                    self.trace_y_index
                ][self.trace_x_index]
            )
        self.trace_selected = self.data.trace_reference[::, 0, trace_index]
        self.times = self.data.get_trace_axis(len(self.trace_selected))
        self.line_ax.clear()

        if not self.enable_hist:
            self.line_ax.set_title(f'Trace at {self.name_data_x_axis}: {self.X[self.trace_y_index][self.trace_x_index]:.3f} ; '
                                   f'{self.name_data_y_axis}: {self.Y[self.trace_y_index][self.trace_x_index]:.3f} ')
            trace_width = max(
                64,
                self.trace_canvas.get_tk_widget().winfo_width(),
            )
            trace_display_indices = self._downsample_indices(
                len(self.trace_selected),
                trace_width,
            )
            self.line_ax.plot(
                self.times[trace_display_indices].astype(np.float32),
                np.asarray(self.trace_selected)[trace_display_indices].astype(
                    np.float32
                ),
                color=self._current_plot_line_color(),
            )
            self.line_ax.set_xlabel(self.trace_xlabel)
            self.line_ax.set_ylabel(self.trace_ylabel)

        elif self.enable_hist:
            self.line_ax.set_title(f'Histogram at {self.name_data_x_axis}: {self.X[self.trace_y_index][self.trace_x_index]:.3f} ; '
                                   f'{self.name_data_y_axis}: {self.Y[self.trace_y_index][self.trace_x_index]:.3f} ')
            self.line_ax.hist(
                self.trace_selected,
                color=self._current_plot_line_color(),
                alpha=0.7,
                edgecolor='black',
                bins=self.nbins_traces,
            )
            self.line_ax.set_xlabel(self.hist_xlabel)
            self.line_ax.set_ylabel(self.hist_ylabel)

        configure_transparent_matplotlib_canvas(
            self.trace_figure,
            self.trace_canvas,
        )
        self.trace_canvas.draw_idle()

    def on_right_click(self, event):
        if event.button == 3:
            self.trace_x_index = np.argmin(np.abs(self.X[0] - event.xdata))
            self.trace_y_index = (
                0
                if self.single_axis_measurement
                else np.argmin(np.abs(self.Y[:, 0] - event.ydata))
            )
            print(f"Right-clicked at coordinates: ({self.trace_x_index}, {self.trace_y_index})")
            self.update_line_plot()

    def toggle_hist(self):
        self.nbins_traces = int(self.nbins_traces_input.get())
        self.enable_hist = not self.enable_hist
        self.update_line_plot()

    def update_plot(self):
        super().update_plot()
        self.update_line_plot()

    def _after_data_operation_undo(self):
        self.update_line_plot()

    def open_hist_window(self):
        self.toggle_hist_window = ttk.Toplevel(self.root)
        self.toggle_hist_window.title("Histogram Settings")
        self.toggle_hist_window.geometry("400x200")

        self.nbins_traces_input = ttk.Entry(self.toggle_hist_window)
        self.nbins_traces_input.pack()
        self.nbins_traces_input.insert(0, self.nbins_traces)
        submit_button = ttk.Button(self.toggle_hist_window, text="Toggle Histogram", command=self.toggle_hist, bootstyle='primary')
        submit_button.pack()


    def apply_data_axis_transform(self):
        super().apply_data_axis_transform()
        self.update_line_plot()


    def save_trace(self):
        pth = filedialog.askdirectory() + '/'
        self.data.set_filename()
        base_name, _ = os.path.splitext(self.data.file_name)
        trace_pos = f'{self.name_data_x_axis}_{self.X[self.trace_y_index][self.trace_x_index]:.3f}_{self.name_data_y_axis}_{self.Y[self.trace_y_index][self.trace_x_index]:.3f}'
        np.save(pth + base_name + 'trace_at_' + trace_pos + '.npy', self.trace_selected, allow_pickle=True)
        np.save(
            pth + base_name + 'trace_axis_for_trace_at_' + trace_pos + '.npy',
            self.times,
            allow_pickle=True,
        )


    def reset(self):
        super().reset()

        self.line_ax.clear()
        self.line_order_indeces = None
        self.times = None
        self.trace_selected = None


class InteractiveTimeTraceMapPlotter(InteractiveArrayPlotter):
    def __init__(
        self,
        root,
        hdf5data,
        plot_style=None,
        plot_style_change_callback=None,
    ):

        super().__init__(
            root,
            hdf5data,
            plot_style=plot_style,
            plot_style_change_callback=plot_style_change_callback,
        )


class FitConfigurationPanel:
    """Shared model-expression and parameter editor for all 1D fitting UIs."""

    def __init__(self, parent, expression=DEFAULT_FIT_EXPRESSION):
        self.frame = ttk.LabelFrame(parent, text='Fit Model')
        self.model_expr_var = tk.StringVar(value=expression)
        self.maxfev_var = tk.IntVar(value=DEFAULT_MAXFEV)
        self.parameter_vars = {}
        self._parameter_value_cache = {}

        ttk.Label(self.frame, text='Model Expression:').grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        ttk.Entry(
            self.frame,
            textvariable=self.model_expr_var,
            width=42,
        ).grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        ttk.Label(
            self.frame,
            text=f'Example: {FIT_EXPRESSION_EXAMPLE}',
            wraplength=440,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5)

        self.parameter_frame = ttk.LabelFrame(
            self.frame,
            text='Initial Parameters',
        )
        self.parameter_frame.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky=tk.NSEW,
            padx=5,
            pady=10,
        )

        self.validation_variable = tk.StringVar()
        ttk.Label(
            self.frame,
            textvariable=self.validation_variable,
            bootstyle='danger',
            wraplength=440,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)

        ttk.Label(
            self.frame,
            text='Max Function Evaluations (maxfev):',
        ).grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(
            self.frame,
            textvariable=self.maxfev_var,
            width=10,
        ).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)

        self.results_text = ttk.Text(
            self.frame,
            height=9,
            width=44,
            wrap=tk.WORD,
        )
        self.results_text.grid(
            row=5,
            column=0,
            columnspan=2,
            sticky=tk.NSEW,
            padx=5,
            pady=5,
        )
        self.results_text.config(state=tk.DISABLED)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(5, weight=1)

        self.model_expr_var.trace_add('write', self._rebuild_parameter_inputs)
        self._rebuild_parameter_inputs()

    def _rebuild_parameter_inputs(self, *_args):
        previous_values = {}
        for name, variable in self.parameter_vars.items():
            try:
                previous_values[name] = variable.get()
            except tk.TclError:
                pass
        self._parameter_value_cache.update(previous_values)

        for widget in self.parameter_frame.winfo_children():
            widget.destroy()
        self.parameter_vars = {}

        try:
            model_definition = FitModelDefinition(self.model_expr_var.get())
        except ValueError as error:
            self.validation_variable.set(str(error))
            return

        self.validation_variable.set('')
        for index, parameter_name in enumerate(model_definition.parameter_names):
            initial_value = previous_values.get(
                parameter_name,
                self._parameter_value_cache.get(
                    parameter_name,
                    default_initial_value(parameter_name),
                ),
            )
            variable = tk.DoubleVar(value=initial_value)
            self.parameter_vars[parameter_name] = variable
            ttk.Label(
                self.parameter_frame,
                text=f'{parameter_name}:',
            ).grid(row=index, column=0, sticky=tk.W, padx=5, pady=3)
            ttk.Entry(
                self.parameter_frame,
                textvariable=variable,
                width=14,
            ).grid(row=index, column=1, sticky=tk.W, padx=5, pady=3)

    def get_fit_setup(self):
        model_definition = FitModelDefinition(self.model_expr_var.get())
        initial_values = {}
        for parameter_name in model_definition.parameter_names:
            variable = self.parameter_vars.get(parameter_name)
            if variable is None:
                raise ValueError(
                    f'No initial-value input exists for {parameter_name!r}.'
                )
            try:
                initial_values[parameter_name] = variable.get()
            except tk.TclError as error:
                raise ValueError(
                    f'Enter a numeric initial value for {parameter_name!r}.'
                ) from error
        try:
            maxfev = self.maxfev_var.get()
        except tk.TclError as error:
            raise ValueError('maxfev must be a positive integer.') from error
        if maxfev <= 0:
            raise ValueError('maxfev must be a positive integer.')
        return model_definition, initial_values, maxfev

    def show_result(self, result, elapsed_seconds=None, additional_lines=None):
        text = format_fit_result(result, elapsed_seconds)
        if additional_lines:
            text += '\n' + '\n'.join(additional_lines)
        self.show_text(text)

    def show_text(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def use_fitted_values_as_initial(self, result):
        for name, value in result.parameters.items():
            variable = self.parameter_vars.get(name)
            if variable is not None:
                variable.set(value)
            self._parameter_value_cache[name] = value


class TracesFitter:
    """
    A GUI application for fitting peak functions/distributions to 1D traces
    """

    def __init__(
        self,
        data,
        master=None,
        on_fit_all_complete=None,
        plot_style=None,
    ):

        # Initialize trace indices
        self.trace_index_x = 0
        self.trace_index_y = 0

        # Set up the main window
        if master is None:
            self.root = ttk.App(theme='bootstrap-light')
            self.root.title("Traces Fitter")
            self.root.geometry("800x600")
        else:
            self.root = ttk.Toplevel(master)
            self.root.title("Traces Fitter")
            # self.root.geometry("800x600")


        self.data = data
        self.on_fit_all_complete = on_fit_all_complete
        self.plot_style = normalize_plot_style(plot_style)
        self.fit_all_running = False

        self.fitted_params = [] # Store fitted parameters
        self.fit_results = [] # Store fit results

        # Create matplotlib figure and axis
        self.fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(
            getattr(self.data, 'trace_axis_name', None) or 'Trace X'
        )
        self.ax.set_ylabel('y')


    def create_widgets(self):
        """Create all GUI widgets for the fitter interface."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        configure_transparent_matplotlib_canvas(self.fig, self.canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        self.model_frame = ttk.Frame(main_frame, width=200)
        self.model_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, expand=True)
        self.fit_panel = FitConfigurationPanel(self.model_frame)
        self.fit_panel.frame.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky=tk.NSEW,
            padx=5,
            pady=5,
        )

        ttk.Label(self.model_frame, text='Trace X Index:').grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.trace_x_index_var = tk.IntVar(value=0)
        trace_x_spinbox = ttk.Spinbox(
            self.model_frame,
            from_=0,
            to=self.data.measure_dim[0] - 1,
            textvariable=self.trace_x_index_var,
            width=10,
        )
        trace_x_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        trace_x_spinbox.bind('<FocusOut>', lambda _event: self.update_plot())
        trace_x_spinbox.bind('<Return>', lambda _event: self.update_plot())

        ttk.Label(self.model_frame, text='Trace Y Index:').grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.trace_y_index_var = tk.IntVar(value=0)
        trace_y_spinbox = ttk.Spinbox(
            self.model_frame,
            from_=0,
            to=self.data.measure_dim[1] - 1,
            textvariable=self.trace_y_index_var,
            width=10,
        )
        trace_y_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        trace_y_spinbox.bind('<FocusOut>', lambda _event: self.update_plot())
        trace_y_spinbox.bind('<Return>', lambda _event: self.update_plot())

        button_frame = ttk.Frame(self.model_frame)
        button_frame.grid(
            row=3,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            padx=5,
            pady=5,
        )
        ttk.Button(
            button_frame,
            text='Fit Preview',
            command=self.fit_preview_trace,
        ).pack(side=tk.LEFT, padx=5)
        self.fit_all_button = ttk.Button(
            button_frame,
            text='Run on all Traces',
            command=self.fit_all_traces,
        )
        self.fit_all_button.pack(side=tk.LEFT, padx=5)

        self.root.update()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

    def get_one_index(self, trace_index_x, trace_index_y): # Convert 2D indices to 1D index
        return self.data.measure_dim[1] * trace_index_x + trace_index_y

    def update_plot(self):
        """
        Update the plot with the selected trace and fitted curve if available.
        """
        # Get the selected trace
        self.trace_index = self.get_one_index(self.trace_x_index_var.get(), self.trace_y_index_var.get())
        self.trace_selected = self.data.trace_reference[::, 0, self.trace_index]
        self.times = self.data.get_trace_axis(len(self.trace_selected))

        self.ax.clear() # Clear previous plot
        self.ax.plot(
            self.times,
            self.trace_selected,
            label='Original Trace',
            color=resolve_plot_color(
                self.plot_style['crosshair_histogram_color']
            ),
        )
        self.ax.set_xlabel(
            getattr(self.data, 'trace_axis_name', None) or 'Trace X'
        )
        if (
            hasattr(self, 'fit_y')
            and getattr(self, 'fit_trace_index', None) == self.trace_index
        ):
            self.ax.plot(self.x_data, self.fit_y, label='Fitted Curve', color='red', linestyle='--')
        self.ax.legend()
        self.canvas.draw()

    def set_plot_style(self, plot_style):
        """Apply the shared trace color without changing fitted-curve colors."""
        self.plot_style = normalize_plot_style(plot_style)
        line_color = resolve_plot_color(
            self.plot_style['crosshair_histogram_color']
        )
        for artist in self.ax.lines:
            if artist.get_label() == 'Original Trace':
                artist.set_color(line_color)
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
        return self.plot_style

    def fit_preview_trace(self):
        """Fit the selected trace through the shared fitting implementation."""
        try:
            model_definition, initial_values, maxfev = (
                self.fit_panel.get_fit_setup()
            )
            start_time = time.perf_counter()
            result = model_definition.fit(
                self.times,
                self.trace_selected,
                initial_values,
                maxfev=maxfev,
            )
            elapsed = time.perf_counter() - start_time
        except Exception as error:
            messagebox.showerror(
                'Fit Error',
                f'Could not fit the preview trace:\n{error}',
                parent=self.root,
            )
            return None

        self.fit_definition = model_definition
        self.fit_preview_result = result
        self.all_param_names = list(result.parameter_names)
        self.fit_params = result.parameters
        self.maxfev = maxfev
        self.x_data = np.asarray(self.times, dtype=float)
        self.y_data = np.asarray(self.trace_selected, dtype=float)
        self.fit_y = result.fitted_y
        self.fit_trace_index = self.trace_index
        self.fit_panel.use_fitted_values_as_initial(result)

        num_traces = int(self.data.trace_reference.shape[2])
        total_estimate = elapsed * num_traces
        self.fit_panel.show_result(
            result,
            elapsed_seconds=elapsed,
            additional_lines=[
                '',
                f'Estimated time for all traces: {total_estimate:.1f} seconds '
                f'({total_estimate / 60:.1f} min)',
            ],
        )
        self.update_plot()
        return result

    def fit_all_traces(self):
        """
        Fit the model to all traces and store the results.
        """
        if self.fit_all_running:
            messagebox.showinfo(
                'Trace Fit in Progress',
                'Please wait for the current batch fit to finish.',
                parent=self.root,
            )
            return None

        preview_result = self.fit_preview_trace()
        if preview_result is None:
            return None

        trace_ref = self.data.trace_reference
        try:
            num_traces = int(trace_ref.shape[2])
        except Exception as error:
            messagebox.showerror(
                'Fit Error',
                'Trace data must have shape (trace_length, 1, trace_count). '
                f'Got {getattr(trace_ref, "shape", None)}.\n{error}',
                parent=self.root,
            )
            return None

        n_params = len(self.all_param_names)
        fit_results = np.full((num_traces, n_params), np.nan, dtype=float)

        # 3) Progress window (UI thread)
        progress_win = ttk.Toplevel(self.root)
        progress_win.title("Fitting Progress")
        progress_win.geometry("460x150")
        progress_win.transient(self.root)

        progress_label_var = tk.StringVar(value="Preparing…")
        ttk.Label(progress_win, textvariable=progress_label_var).pack(padx=20, pady=(15, 6), anchor="w")

        progress_var = tk.DoubleVar(value=0)
        bar = ttk.Progressbar(progress_win, variable=progress_var, maximum=num_traces, length=400)
        bar.pack(padx=20, pady=8, fill=tk.X)

        cancel_flag = {"stop": False}

        def on_cancel():
            cancel_flag["stop"] = True
            progress_label_var.set("Cancelling after current trace…")

        cancel_btn = ttk.Button(progress_win, text="Cancel", command=on_cancel)
        cancel_btn.pack(padx=20, pady=(0, 15), anchor="e")
        progress_win.protocol('WM_DELETE_WINDOW', on_cancel)

        msg_q: "queue.Queue[tuple]" = queue.Queue()

        p0 = np.asarray(preview_result.parameter_values, dtype=float)
        model_definition = self.fit_definition
        maxfev = self.maxfev
        failed_fit_count = 0

        def worker():
            nonlocal p0, failed_fit_count
            try:
                x_data = np.asarray(self.times, dtype=float)

                for i in range(num_traces):
                    if cancel_flag["stop"]:
                        msg_q.put(("done", "Cancelled."))
                        return

                    try:
                        y_data = np.asarray(trace_ref[:, 0, i], dtype=float)
                    except Exception:
                        y_data = np.asarray(trace_ref[i], dtype=float).reshape(-1)

                    try:
                        result = model_definition.fit(
                            x_data,
                            y_data,
                            p0,
                            maxfev=maxfev,
                        )
                        fit_results[i, :] = result.parameter_values
                        p0 = result.parameter_values

                    except Exception as fit_err:
                        failed_fit_count += 1
                        msg_q.put(("warn", f"Fit failed for trace {i + 1}/{num_traces}: {fit_err}"))

                    if (i % 5 == 0) or (i == num_traces - 1):
                        msg_q.put(("progress", i + 1))

                msg_q.put(("done", "Finished."))

            except Exception as e:
                msg_q.put(("error", f"Fatal error: {e}"))

        def poll():
            try:
                while True:
                    kind, payload = msg_q.get_nowait()

                    if kind == "progress":
                        progress_var.set(payload)
                        progress_label_var.set(f"Fitting traces… {payload}/{num_traces}")
                    elif kind == "warn":
                        # Keep GUI responsive; warnings go to console
                        print(payload)
                    elif kind == "error":
                        self.fit_all_running = False
                        self.fit_all_button.config(state=tk.NORMAL)
                        progress_label_var.set('Fit failed.')
                        cancel_btn.config(state=tk.DISABLED)
                        messagebox.showerror(
                            'Trace Fitting Failed',
                            payload,
                            parent=progress_win,
                        )
                        progress_win.after(800, progress_win.destroy)
                        return
                    elif kind == "done":
                        self.fit_all_running = False
                        self.fit_all_button.config(state=tk.NORMAL)
                        progress_label_var.set(payload)
                        cancel_btn.config(state=tk.DISABLED)

                        # Store results on the object
                        self.fit_results = fit_results
                        self.fit_results_dict = {
                            name: self.fit_results[:, idx]
                            for idx, name in enumerate(self.all_param_names)
                        }
                        successful_fit_count = int(np.count_nonzero(
                            np.all(np.isfinite(self.fit_results), axis=1)
                        ))
                        self.fit_panel.show_result(
                            preview_result,
                            additional_lines=[
                                '',
                                f'Batch result: {successful_fit_count}/{num_traces} '
                                'traces fitted successfully.',
                                f'Failed fits: {failed_fit_count}',
                            ],
                        )
                        if self.on_fit_all_complete is not None:
                            self.on_fit_all_complete(self.fit_results_dict)

                        progress_win.after(500, progress_win.destroy)
                        return
            except queue.Empty:
                pass

            progress_win.after(50, poll)

        # 5) Start
        self.fit_all_running = True
        self.fit_all_button.config(state=tk.DISABLED)
        threading.Thread(target=worker, daemon=True).start()
        poll()
        return None



class UtilityLinePlotter:
    def __init__(
        self,
        master=None,
        color_cycle_name=None,
        color_cycle_change_callback=None,
    ):
        """
        Initialize a utility plotter for line cuts.

        """
        if master is None:
            self.root = ttk.App(theme='bootstrap-light')
            self.root.title("Line Cut Plotter")
            self.root.geometry("800x600")
        else:
            self.root = ttk.Toplevel(master)
            self.root.title("Line Cut Plotter")
            self.root.geometry("800x600")

        # Store line cut data as list of tuples (x, y, label)
        self.data = []
        self.selected_line_idx = None
        self.color_cycle_name = (
            color_cycle_name
            if color_cycle_name in COLOR_CYCLE_OPTIONS
            else DEFAULT_PLOT_STYLE['extracted_linecut_color_cycle']
        )
        self.color_cycle_change_callback = color_cycle_change_callback

        # Create the figure and axis for plotting
        self.fig = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('Distance')
        self.ax.set_ylabel('Value')

        # Create widgets
        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        """Create all GUI widgets for the plotter."""
        # Create main frame for layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create frame for plot
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create canvas for matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        configure_transparent_matplotlib_canvas(self.fig, self.canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Create sidebar frame
        sidebar_frame = ttk.Frame(main_frame, width=200)
        sidebar_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # Add line selector frame
        line_select_frame = ttk.LabelFrame(sidebar_frame, text="Line Cuts")
        line_select_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create listbox for line selection
        self.line_listbox = ttk.Listbox(line_select_frame)
        self.line_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.line_listbox.bind('<<ListboxSelect>>', self.on_line_select)

        # Create Context menu
        self.line_context_menu = ttk.Menu(self.line_listbox, tearoff=0)
        self.line_context_menu.add_command(label="Offset and Scale", command=self.open_offset_scale_window)
        self.line_context_menu.add_command(label="Fit Model", command=self.open_fit_custom_model)
        self.line_context_menu.add_command(label="Export Line Trace", command=self.open_export_window)
        self.line_listbox.bind("<Button-3>", self.show_line_context_menu)

        # Add control buttons
        btn_frame = ttk.Frame(sidebar_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.remove_btn = ttk.Button(btn_frame, text="Remove", command=self.remove_line)
        self.remove_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(btn_frame, text="Clear All", command=self.clear_lines)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # Add options frame
        options_frame = ttk.LabelFrame(sidebar_frame, text="Options")
        options_frame.pack(fill=tk.X, pady=5)

        # Legend toggle
        self.show_legend_var = tk.BooleanVar(value=True)
        self.legend_check = ttk.Checkbutton(options_frame, text="Show Legend",
                                            variable=self.show_legend_var,
                                            command=self.update_plot)
        self.legend_check.pack(anchor=tk.W, padx=5, pady=2)

        ttk.Label(options_frame, text='Color cycle:').pack(
            anchor=tk.W,
            padx=5,
            pady=(8, 0),
        )
        self.color_cycle_combobox = ttk.Combobox(
            options_frame,
            values=tuple(COLOR_CYCLE_OPTIONS),
            state='readonly',
            width=24,
        )
        self.color_cycle_combobox.set(self.color_cycle_name)
        self.color_cycle_combobox.pack(
            fill=tk.X,
            padx=5,
            pady=(2, 5),
        )
        self.color_cycle_combobox.bind(
            '<<ComboboxSelected>>',
            self._on_color_cycle_selected,
        )

    def _on_color_cycle_selected(self, _event=None):
        """Apply and persist a cycle chosen in the linecut plotter."""
        cycle_name = self.color_cycle_combobox.get()
        self.set_color_cycle(cycle_name)
        if self.color_cycle_change_callback is not None:
            self.color_cycle_change_callback(cycle_name)

    def set_color_cycle(self, cycle_name):
        """Change the extracted-linecut palette and redraw existing curves."""
        if cycle_name not in COLOR_CYCLE_OPTIONS:
            cycle_name = DEFAULT_PLOT_STYLE[
                'extracted_linecut_color_cycle'
            ]
        self.color_cycle_name = cycle_name
        if hasattr(self, 'color_cycle_combobox'):
            self.color_cycle_combobox.set(cycle_name)
        if hasattr(self, 'canvas'):
            self.update_plot()

    def add_linecut(self, x_data, y_data, label=None):
        """
        Add a new line cut to the plot.

        Parameters:
        -----------
        x_data : array-like
            X values (typically distance along linecut)
        y_data : array-like
            Y values (data values along linecut)
        label : str, optional
            Label for the line cut
        """
        if label is None:
            label = f"Line {len(self.data) + 1}"

        self.data.append((x_data, y_data, label))
        self.line_listbox.insert(tk.END, label)
        self.selected_line_idx = len(self.data) - 1
        self.line_listbox.selection_clear(0, tk.END)
        self.line_listbox.selection_set(self.selected_line_idx)
        self.update_plot()

    def show_line_context_menu(self, event):
        """Show the context menu on right-click in the lines listbox"""
        # Get the line index at the current mouse position
        try:
            index = self.line_listbox.nearest(event.y)
            # Only show menu if we clicked on an actual line
            if index < len(self.data):
                # Select the line first
                self.line_listbox.selection_clear(0, tk.END)
                self.line_listbox.selection_set(index)
                self.line_listbox.activate(index)
                # Show the context menu
                self.line_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            # Make sure to release the menu
            self.line_context_menu.grab_release()

    def on_line_select(self, event):
        """Handle selection of a line from the listbox."""
        selection = self.line_listbox.curselection()
        if selection:
            self.selected_line_idx = selection[0]
            self.update_plot()

    def remove_line(self):
        """Remove the selected line cut."""
        if self.selected_line_idx is not None and 0 <= self.selected_line_idx < len(self.data):
            self.data.pop(self.selected_line_idx)
            self.line_listbox.delete(self.selected_line_idx)

            if len(self.data) > 0:
                self.selected_line_idx = min(self.selected_line_idx, len(self.data) - 1)
                self.line_listbox.selection_set(self.selected_line_idx)
            else:
                self.selected_line_idx = None

            self.update_plot()

    def clear_lines(self):
        """Clear all line cuts from the plot."""
        self.data = []
        self.line_listbox.delete(0, tk.END)
        self.selected_line_idx = None
        self.update_plot()

    def apply_offset_scale(self, x_offset, y_offset, x_scale, y_scale, dialog=None):
        """
        Apply offset and scale to the selected line.

        Parameters:
        -----------
        x_offset : float
            Value to add to x coordinates
        y_offset : float
            Value to add to y coordinates
        x_scale : float
            Factor to multiply x coordinates by
        y_scale : float
            Factor to multiply y coordinates by
        dialog : Toplevel, optional
            Dialog window to close after applying
        """
        if self.selected_line_idx is None or self.selected_line_idx >= len(self.data):
            if dialog:
                dialog.destroy()
            return

        # Get the selected data
        x_data, y_data, label = self.data[self.selected_line_idx]

        # Apply transformations
        new_x = x_data * x_scale + x_offset
        new_y = y_data * y_scale + y_offset

        # Replace the data with the modified version
        self.data[self.selected_line_idx] = (new_x, new_y, label)

        # Update the plot
        self.update_plot()

    def open_offset_scale_window(self):
        """Open a dialog to input offset and scaling parameters for the selected line."""
        if self.selected_line_idx is None or self.selected_line_idx >= len(self.data):
            messagebox.showinfo("No Selection", "Please select a line to modify.")
            return

        # Create a new dialog window
        dialog = ttk.Toplevel(self.root)
        dialog.title("Offset and Scale Line")
        dialog.geometry("300x270")
        #dialog.resizable(False, False)
        #dialog.transient(self.root)  # Set as transient to main window
        #dialog.grab_set()  # Make dialog modal

        # Define the fields needed
        fields = [
            ("X Offset:", "x_offset", 0.0),
            ("Y Offset:", "y_offset", 0.0),
            ("X Scale Factor:", "x_scale", 1.0),
            ("Y Scale Factor:", "y_scale", 1.0)
        ]

        # Dictionary to store the variable references
        vars_dict = {}

        # Create input fields with labels using a loop
        for i, (label_text, var_name, default_value) in enumerate(fields):
            ttk.Label(dialog, text=label_text).grid(row=i, column=0, padx=10, pady=10, sticky=tk.W)
            vars_dict[var_name] = tk.DoubleVar(value=default_value)
            ttk.Entry(dialog, textvariable=vars_dict[var_name], width=15).grid(row=i, column=1, padx=10, pady=10)

        # Create buttons
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=15)

        ttk.Button(button_frame, text="Apply",
                   command=lambda: self.apply_offset_scale(
                       vars_dict["x_offset"].get(),
                       vars_dict["y_offset"].get(),
                       vars_dict["x_scale"].get(),
                       vars_dict["y_scale"].get(),
                       dialog)).pack(side=tk.LEFT, padx=10)

        ttk.Button(button_frame, text="Cancel",
                   command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def open_fit_custom_model(self):
        """Fit the selected line with the same editor used for trace fitting."""
        if self.selected_line_idx is None or self.selected_line_idx >= len(self.data):
            messagebox.showinfo(
                'No Selection',
                'Please select a line to fit.',
                parent=self.root,
            )
            return

        dialog = ttk.Toplevel(self.root)
        dialog.title('Fit Line Cut')
        dialog.geometry('560x650')
        dialog.transient(self.root)

        x_data, y_data, label = self.data[self.selected_line_idx]
        fit_panel = FitConfigurationPanel(dialog)
        fit_panel.frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        def fit_model():
            try:
                model_definition, initial_values, maxfev = (
                    fit_panel.get_fit_setup()
                )
                start_time = time.perf_counter()
                result = model_definition.fit(
                    x_data,
                    y_data,
                    initial_values,
                    maxfev=maxfev,
                )
                elapsed = time.perf_counter() - start_time
            except Exception as error:
                messagebox.showerror(
                    'Fit Error',
                    f'Could not fit the selected line:\n{error}',
                    parent=dialog,
                )
                return

            self.add_linecut(x_data, result.fitted_y, f'Fit: {label}')
            fit_panel.use_fitted_values_as_initial(result)
            fit_panel.show_result(result, elapsed_seconds=elapsed)

        ttk.Button(
            button_frame,
            text='Fit Selected Line',
            command=fit_model,
            bootstyle='success',
        ).pack(side=tk.LEFT, padx=10)
        ttk.Button(
            button_frame,
            text='Close',
            command=dialog.destroy,
            bootstyle='secondary',
        ).pack(side=tk.RIGHT, padx=10)

    def open_export_window(self):
        """Open a dialog to export the selected line trace as text or numpy file."""
        if self.selected_line_idx is None or self.selected_line_idx >= len(self.data):
            messagebox.showinfo("No Selection", "Please select a line trace to export.")
            return

        # Get the selected data
        x_data, y_data, label = self.data[self.selected_line_idx]

        # Create a new dialog window
        export_dialog = ttk.Toplevel(self.root)
        export_dialog.title("Export Line Trace")
        export_dialog.geometry("300x150")
        export_dialog.transient(self.root)
        export_dialog.grab_set()

        # Add info label
        info_label = ttk.Label(export_dialog, text=f"Export data for: {label}")
        info_label.pack(pady=(10, 20))

        # Button frame
        button_frame = ttk.Frame(export_dialog)
        button_frame.pack(fill=tk.X, pady=10)

        def export_as_text():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=f"{label.replace(' ', '_')}.txt"
            )
            if file_path:
                try:
                    with open(file_path, 'w') as f:
                        f.write("# X\tY\n")
                        for x, y in zip(x_data, y_data):
                            f.write(f"{x}\t{y}\n")
                    messagebox.showinfo("Export Successful", f"Data exported to {file_path}")
                    export_dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")

        def export_as_numpy():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".npy",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
                initialfile=f"{label.replace(' ', '_')}.npy"
            )
            if file_path:
                try:
                    data_array = np.column_stack((x_data, y_data))
                    np.save(file_path, data_array)
                    messagebox.showinfo("Export Successful", f"Data exported to {file_path}")
                    export_dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")

        # Export buttons
        txt_button = ttk.Button(button_frame, text="Export as Text", command=export_as_text)
        txt_button.pack(side=tk.LEFT, expand=True, padx=10)

        numpy_button = ttk.Button(button_frame, text="Export as NumPy", command=export_as_numpy)
        numpy_button.pack(side=tk.RIGHT, expand=True, padx=10)

        # Cancel button
        cancel_button = ttk.Button(export_dialog, text="Cancel", command=export_dialog.destroy)
        cancel_button.pack(pady=10)

    def is_alive(self):
        """Check if the toplevel window still exists and is not destroyed."""
        try:
            return self.toplevel.winfo_exists()
        except:
            return False

    def update_plot(self):
        """Update the plot with current data and settings."""
        self.ax.clear()

        line_colors = get_color_cycle(self.color_cycle_name)

        for i, (x, y, label) in enumerate(self.data):
            color = line_colors[i % len(line_colors)]
            if i == self.selected_line_idx:
                # Highlight selected line
                self.ax.plot(x, y, label=label, color=color, linewidth=2.5)
            else:
                self.ax.plot(x, y, label=label, color=color, linewidth=1.5)

        # Add labels and legend if there is data
        if self.data:
            self.ax.set_xlabel('Distance')
            self.ax.set_ylabel('Value')
            if self.show_legend_var.get():
                self.ax.legend()

        # Draw the canvas
        self.canvas.draw()
