import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter import filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import lines
from matplotlib import rc
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scipy.ndimage import gaussian_filter
from Data_analysis_and_transforms import (image_down_sampling, two_d_fft_on_data, evaluate_poly_background_2d,
                                          correct_median_diff, correct_mean_of_lines, gradient_5p_stencil,
                                          subtract_trace_average, cut_data_range, extract_linecut)
from gamma_map import (get_t_rates, get_fourier, fft_correction_select, fft_correction_apply, get_cuts)
from custom_cmap import make_neon_cyclic_colormap, make_bi_colormap
neon_cmap = make_neon_cyclic_colormap()
bi_map = make_bi_colormap() # take out
plt.register_cmap(name='BiMap', cmap=bi_map)
plt.register_cmap(name='NeonPiCy', cmap=neon_cmap)
rc('pdf', fonttype=42)


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
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.row_label = tk.Label(self.root, text="Selected Row:")
        self.row_label.pack()

        # Combobox for row selection
        self.current_row.set(0)
        self.row_combobox = ttk.Combobox(self.root, textvariable=self.current_row, state="readonly")
        self.row_combobox["values"] = list(range(self.data.shape[0]))
        self.row_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        self.row_combobox.pack()

        self.col_label = tk.Label(self.root, text="Selected Column:")
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
    def __init__(self, root, hdf5data, figure=None, ax=None):
        self.root = root
        self.root.title("Interactive Array Plotter")

        # Initialize attributes
        self.data = hdf5data
        #### Created by Nico R.  for trace loading validation
        self.contains_traces = 'Traces' in list(hdf5data.file.keys())
        ####
        self.name_data_z = ''
        self.name_data_y_axis = ''
        self.name_data_x_axis = ''
        self.num_dimensions = len(self.data.measure_dim) - 2
        self.name_data = [str(label) for label in self.data.name_data]
        self.name_data = [label.encode('utf-8').decode('utf-8') for label in self.name_data]
        if self.contains_traces:
            self.name_data.append("Tunneling rates in")
            self.name_data.append("Tunneling rates out")
        self.x_index = 0
        self.y_index = 0
        self.nan_mask = np.array([])
        self.loaded = False # rename to be more discriptiv
        self.calculated = False

        #ROI attributes
        self.roi_mode = False
        self.roi_corners = []  # Store corners of the ROI as [(x1, y1), (x2, y2)]
        self.current_roi_patch = None  # Current ROI rectangle

        # Create Menu Bar
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        # Create File Menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Save whole data as NumPy array", command=self.save_file)
        self.file_menu.add_command(label="Save displayed data as NumPy array", command=self.save_data)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        # Create Data Display Menu
        self.data_menu = tk.Menu(self.menubar, tearoff=0)
        self.data_menu.add_command(label="Interpolation Settings", command=self.open_interpolation_window)
        self.data_menu.add_command(label="Gaussian Filter", command=self.open_gaussian_filter_window)
        self.data_menu.add_command(label="Cut Data to ROI", command=self.open_roi_data_cut_window)
        self.data_menu.add_command(label="Background Subtraction", command=self.open_background_subtraction_window)
        self.data_menu.add_command(label="Rename and Scale Data and Axis", command=self.open_data_axis_transform)
        self.menubar.add_cascade(label="Displayed Data", menu=self.data_menu)

        # Create Tool Menu
        self.tool_menu = tk.Menu(self.menubar, tearoff=0)
        self.tool_menu.add_command(label="Derivative along Axis", command=self.open_derivative_window)
        self.tool_menu.add_command(label="Norm of Gradient", command=self.apply_sum_of_gradient)
        self.tool_menu.add_command(label="2-D FFT on Data", command=self.apply_2d_fft)
        self.tool_menu.add_command(label="Draw Lines", command=self.open_draw_lines_window)
        if self.contains_traces:
            self.tool_menu.add_command(label="Correct Trace Fourier spectrum", command=self.open_fft_trace_correction_window)
        self.menubar.add_cascade(label="Tools", menu=self.tool_menu)

        # Create Help Menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)

        # Create a figure and axis for plotting
        if figure is None or ax is None:
            self.figure, self.ax = plt.subplots()
        else:
            self.figure, self.ax = figure, ax

        self.ax_vline = self.figure.add_axes([0.94, 0.12, 0.05, 0.75])  # Adjusted position and size
        self.ax_hline = self.figure.add_axes([0.12, 0.94, 0.62, 0.05])  # Adjusted position and size

        # Hide the additional axes initially
        self.ax_vline.set_visible(False)
        self.ax_hline.set_visible(False)

        # Create a new figure for the histogram
        self.histogram_fig, self.histogram_ax = plt.subplots(figsize=(3.5, 1.5))
        self.histogram_ax.set_yticklabels([])
        self.histogram_ax.set_xticklabels([])
        self.picked_line = None

        # Define interactive button options
        self.colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'twilight', 'coolwarm', 'Spectral',
                          'gnuplot', 'NeonPiCy', 'BiMap']
        self.bg_methods = ['Polynomial', 'Median Difference', 'Mean of Lines', 'Relation Parameters',
                           'Subtract Trace Average']
        self.relation_parameter_entry_list = []
        self.roi_cut_entry_list = []
        self.drawn_lines_list = []
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.toolbar = NavigationToolbar2Tk(self.canvas, root, pack_toolbar=False)
        self.toolbar.update()
        self.crosshair_button = tk.Button(self.toolbar, text='Crosshair', command=self.toggle_crosshair)
        self.crosshair_button.pack(side=tk.LEFT)
        self.interpol_button = tk.Button(self.toolbar, text='Interpolation', command=self.toggle_interpolation)
        self.interpol_button.pack(side=tk.LEFT)
        self.roi_button = tk.Button(self.toolbar, text='ROI', command=self.toggle_roi)
        self.roi_button.pack(side=tk.LEFT)
        self.horiz_line = None
        self.vert_line = None
        self.auto_scale_var = tk.BooleanVar(value=True)  # Default to True (auto-scaling
        self.crosshair_enabled = False
        self.interpolation_enabled = False
        self.invert_enabled = False
        self.freeze_linecut = False
        self.linecut_position = None
        self.drawing_line = False
        self.current_line = None
        self.click_cid = None
        self.move_cid = None
        self.motion_cid = None
        self.release_cid = None

        # Pre calculate values for selection
        self.parameter_labels = [np.flip(self.data.name_axis)[i] for i in range(self.num_dimensions)]
        self.parameter_values = [list(range(np.flip(self.data.measure_dim)[i])) for i in range(len(np.flip(self.data.measure_dim)))]
        self.parameter_comboboxes = []

        self.display_values_list = []
        for i in range(self.num_dimensions):
            display_values = []
            for index in range(np.flip(self.data.measure_dim)[i]):
                # Construct a selection tuple with the current index
                selection = [0] * self.num_dimensions
                selection[i] = index
                value = np.flip(self.data.measure_axis, axis=0)[i].swapaxes(0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selection)][0][0]
                display_values.append(float(value))
            self.display_values_list.append(display_values)

        # Create a custom style for label frames to make them smaller
        small_label_frame_style = ttk.Style()
        small_label_frame_style.configure('Small.TLabelframe', font=('Arial', 8))  # Adjust font size here
        frame = ttk.Frame(self.root)
        frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.frame2 = ttk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT, padx=5, pady=5)
        for i, label in enumerate(self.parameter_labels):

            # Create a label frame for each parameter
            label_frame = ttk.LabelFrame(frame, text=label)
            label_frame.pack(padx=5, pady=5)

            # Set the width of the combobox and control its placement
            combobox = ttk.Combobox(label_frame, values=self.display_values_list[i], state='readonly', width=6)
            combobox.pack(side=tk.BOTTOM, padx=5, pady=5)

            if self.display_values_list[i]:  # Check if the list is not empty
                combobox.set(self.display_values_list[i][0])

            self.parameter_comboboxes.append(combobox)

        # Create a combobox for colormap selection
        self.colormap_combobox = ttk.Combobox(self.frame2, values=self.colormaps, state='readonly', width=10)
        self.colormap_combobox.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.colormap_combobox.set(self.colormaps[0])  # Set the default colormap

        # Create a combobox for data selection
        self.data_combobox = ttk.Combobox(self.frame2, values=self.name_data, state='readonly', width=20)
        self.data_combobox.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.data_combobox.set(self.name_data[0])  # Set the default colormap

        # Create a Frame for the "Plot" buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.RIGHT, padx=5, pady=5)

        self.auto_scale_check = tk.Checkbutton(
            self.button_frame,
            text="Auto-scale plot bounds",
            variable=self.auto_scale_var
        )
        self.auto_scale_check.pack(side=tk.TOP, pady=2)

        # Create a "Reset Plot" button inside the button frame
        self.reset_plot_button = ttk.Button(self.button_frame, text="Plot", command=self.plot_data)
        self.reset_plot_button.pack(side=tk.TOP, fill=tk.X)

        # Create an "Update Plot" button inside the button frame
        self.update_plot_button = ttk.Button(self.button_frame, text="Update Plot", command=self.update_plot)
        self.update_plot_button.pack(side=tk.TOP, fill=tk.X)

        # Create a ''Invert Axes'' Button
        self.invert_button = ttk.Button(self.frame2, text="Invert Axes", command=self.toggle_invert)
        self.invert_button.pack(side=tk.BOTTOM)

        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.histogram_canvas = FigureCanvasTkAgg(self.histogram_fig, master=self.frame2)
        self.histogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.N)
        self.plot_data()
        self.canvas.mpl_connect('key_press_event', self.on_key_press)

    def plot_data(self):
        tick = time.perf_counter()

        # Get selected parameter values from comboboxes
        selected_display_values = [combobox.get() for combobox in self.parameter_comboboxes]
        selected_colormap = self.colormap_combobox.get()
        selected_indices = []
        for i, display_value in enumerate(selected_display_values):
            # Convert the display value back to an index
            index = self.display_values_list[i].index(float(display_value))
            selected_indices.append(index)

        # Use selected values to slice and plot data
        if not self.invert_enabled:
            self.X = (np.flip(self.data.measure_axis, axis=0)[-1].swapaxes(0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)])
            self.nan_mask = ~np.isnan(self.X).any(axis=1)
            self.X = self.X[self.nan_mask]
            self.Y = (np.flip(self.data.measure_axis, axis=0)[-2].swapaxes(
                0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)])[self.nan_mask]
            try:
                self.sliced_data = ((self.data.measure_data[self.name_data.index(self.data_combobox.get())]).swapaxes(
                    0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)])[self.nan_mask]
            except IndexError:
                if not self.loaded:
                    print('fetching data...')
                    self.data.set_traces()
                    self.data.set_traces_dt()
                    self.traces = self.data.traces
                    self.times = self.data.traces_dt * np.arange(0, len(self.traces[0][0]))
                    self.loaded = True
                else:
                    pass

                if not self.calculated:
                    print('calculating tunneling rates...')
                    _, self.gamma_up, self.gamma_down = get_t_rates(self.traces, self.times)
                    self.calculated = True
                else:
                    pass

                if self.data_combobox.get() == "Tunneling rates in":
                    self.sliced_data = self.gamma_up
                elif self.data_combobox.get() == "Tunneling rates out":
                    self.sliced_data = self.gamma_down

            self.ax.clear()
            self.xlim = (np.min(self.X), np.max(self.X))
            self.ylim = (np.min(self.Y), np.max(self.Y))

            self.name_data_z = self.data_combobox.get()
            self.name_data_x_axis = str(np.flip(self.data.name_axis)[-1])
            self.name_data_y_axis = str(np.flip(self.data.name_axis)[-2])
        #### Modified by Nico R. ####
        if self.invert_enabled:
            self.X = ((np.flip(self.data.measure_axis, axis=0)[-2].swapaxes(0, 1).reshape(
                np.flip(self.data.measure_dim))[tuple(selected_indices)]))
            self.nan_mask = ~np.isnan(self.X).any(axis=1)
            self.X = (self.X[self.nan_mask]).T

            self.Y = ((np.flip(self.data.measure_axis, axis=0)[-1].swapaxes(0, 1).reshape(
                np.flip(self.data.measure_dim))[tuple(selected_indices)])[self.nan_mask]).T
            try:
                self.sliced_data = (((self.data.measure_data[self.name_data.index(self.data_combobox.get())]).swapaxes(
                    0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)])[self.nan_mask]).T
            except IndexError:
                if not self.loaded:
                    print('fetching data...')
                    self.data.set_traces()
                    self.data.set_traces_dt()
                    self.traces = self.data.traces
                    self.times = self.data.traces_dt * np.arange(0, len(self.traces[0][0]))
                    self.loaded = True
                else:
                    pass

                if not self.calculated:
                    print('calculating tunneling rates...')
                    _, self.gamma_up, self.gamma_down = get_t_rates(self.traces, self.times)
                    self.gamma_up.T
                    self.gamma_down.T
                    self.calculated = True
                else:
                    pass

                if self.data_combobox.get() == "Tunneling rates in":
                    self.sliced_data = self.gamma_up.transpose()
                elif self.data_combobox.get() == "Tunneling rates out":
                    self.sliced_data = self.gamma_down.transpose()
            #### end of Modification by Nico R. ####
            self.ax.clear()
            self.xlim = (np.min(self.X), np.max(self.X))
            self.ylim = (np.min(self.Y), np.max(self.Y))

            self.name_data_z = self.data_combobox.get()
            self.name_data_y_axis = str(np.flip(self.data.name_axis)[-1])
            self.name_data_x_axis = str(np.flip(self.data.name_axis)[-2])

        if hasattr(self, 'cbar'):
            self.cbar.remove()
            del self.cbar

        c = self.ax.pcolormesh(self.X, self.Y, self.sliced_data, cmap=selected_colormap, shading='auto', zorder=1,
                               linewidth=0, rasterized=True)

        self.cbar = self.figure.colorbar(c, ax=self.ax, label=self.name_data_z)
        self.ax.set_xlabel(self.name_data_x_axis)
        self.ax.set_ylabel(self.name_data_y_axis)


        # Redraw the canvas to reflect changes
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax_vline.set_ylim(self.ax.get_ylim())
        self.ax_hline.set_xlim(self.ax.get_xlim())
        self.canvas.draw_idle()

        if self.crosshair_enabled:
            self.refresh_crosshair()

        self.update_histogramm()
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

        # Initialize or update crosshair lines
        if self.crosshair_enabled:
            self.ax_vline.set_visible(True)
            self.ax_hline.set_visible(True)
            if self.horiz_line is None:
                self.horiz_line = self.ax.axhline(color='gray', lw=1, ls='--', zorder=10)
            else:
                self.horiz_line.set_visible(True)

            if self.vert_line is None:
                self.vert_line = self.ax.axvline(color='gray', lw=1, ls='--', zorder=10)
            else:
                self.vert_line.set_visible(True)
            self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        else:
            self.ax_vline.set_visible(False)
            self.ax_hline.set_visible(False)

            if self.horiz_line:
                self.horiz_line.set_visible(False)
                self.horiz_line = None
            if self.vert_line:
                self.vert_line.set_visible(False)
                self.vert_line = None
            self.canvas.mpl_disconnect('motion_notify_event')

        # Redraw the entire figure to ensure layout is updated
        self.figure.canvas.draw_idle()

    def toggle_invert(self):
        self.invert_enabled = not self.invert_enabled

    def toggle_roi(self):
        if self.roi_mode:  # If ROI mode is active, deactivate it
            if self.click_cid is not None:
                self.canvas.mpl_disconnect(self.click_cid)
                self.click_cid = None
            if self.move_cid is not None:
                self.canvas.mpl_disconnect(self.move_cid)
                self.move_cid = None
            self.roi_mode = False
            if self.current_roi_patch:
                self.current_roi_patch.remove()  # Remove the rectangle if it exists
                self.current_roi_patch = None
            self.ax.patches.clear()  # Clear any remaining patches
            self.canvas.draw_idle()
            return

        # Activate ROI mode
        self.roi_mode = True
        self.roi_corners = []  # Initialize the list for ROI corners
        self.current_roi_patch = None  # Reset the rectangle

        # Function to handle mouse click
        def on_click(event):
            if not event.inaxes or event.button != 1:  # Only handle left-clicks inside the axes
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
                if self.current_roi_patch:
                    self.current_roi_patch.set_alpha(.25)
                print(f"ROI selected: {self.roi_corners}")
            else:
                # Third click: Reset the ROI
                self.roi_corners.clear()
                if self.current_roi_patch:
                    self.current_roi_patch.remove()
                    self.current_roi_patch = None
                self.ax.patches.clear()
                self.canvas.draw_idle()

        # Function to handle mouse movement
        def on_move(event):
            if len(self.roi_corners) == 1 and event.inaxes:  # If the first point is selected
                x1, y1 = self.roi_corners[0]
                x2, y2 = event.xdata, event.ydata
                if self.current_roi_patch:
                    self.current_roi_patch.remove()  # Remove the previous patch
                self.current_roi_patch = self.ax.add_patch(
                    plt.Rectangle(
                        (min(x1, x2), min(y1, y2)),
                        abs(x2 - x1),
                        abs(y2 - y1),
                        color='red',
                        alpha=0.5
                    )
                )
                self.canvas.draw_idle()

        # Connect the click and motion events
        self.click_cid = self.canvas.mpl_connect('button_press_event', on_click)
        self.move_cid = self.canvas.mpl_connect('motion_notify_event', on_move)

    def refresh_crosshair(self):
        self.toggle_crosshair()
        self.toggle_crosshair()

    def init_movable_lines(self):
        # Initial positions for vmin and vmax lines
        if self.auto_scale_var.get() or self.vmin is None or self.vmax is None:
            vmin_initial = np.nanmin(self.sliced_data)
            vmax_initial = np.nanmax(self.sliced_data)
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

    def on_mouse_move(self, event):

        if not event.inaxes or not self.crosshair_enabled:
            return

        if not self.freeze_linecut:
            # Update the position of the crosshair lines
            self.horiz_line.set_ydata(event.ydata)
            self.vert_line.set_xdata(event.xdata)

            self.x_index = np.argmin(np.abs(self.X[0] - event.xdata))
            self.y_index = np.argmin(np.abs(self.Y[:, 0] - event.ydata))

            # Update the vertical line plot
            self.ax_vline.clear()
            self.ax_vline.plot(self.sliced_data[:, self.x_index], self.Y[:, 0])
            self.ax_vline.set_yticklabels([])
            for label in self.ax_vline.get_xticklabels():
                label.set_rotation(270)
            self.ax_vline.axhline(y=event.ydata, color='gray', lw=1, ls='--')

            # Update the horizontal line plot
            self.ax_hline.clear()
            self.ax_hline.plot(self.X[0], self.sliced_data[self.y_index, :])
            self.ax_hline.set_xticklabels([])
            self.ax_hline.axvline(x=event.xdata, color='gray', lw=1, ls='--')

            # Update the limits of the line plots to match the main plot
            self.ax_vline.set_ylim(self.ax.get_ylim())
            self.ax_hline.set_xlim(self.ax.get_xlim())

            self.figure.canvas.draw_idle()

        elif self.freeze_linecut:
            # it does not work properly, have to invastigate why, works only on the first plot, but why?
            # Update the vertical line plot
            self.ax_vline.clear()
            self.ax_vline.plot(self.sliced_data[:, self.linecut_position[0]], self.Y[:, 0])
            self.ax_vline.set_yticklabels([])
            for label in self.ax_vline.get_xticklabels():
                label.set_rotation(270)
            self.ax_vline.axhline(y=event.ydata, color='gray', lw=1, ls='--')

            # Update the horizontal line plot
            self.ax_hline.clear()
            self.ax_hline.plot(self.X[0], self.sliced_data[self.linecut_position[1], :])
            self.ax_hline.set_xticklabels([])
            self.ax_hline.axvline(x=event.xdata, color='gray', lw=1, ls='--')

            # Update the limits of the line plots to match the main plot
            self.ax_vline.set_ylim(self.ax.get_ylim())
            self.ax_hline.set_xlim(self.ax.get_xlim())

            #self.figure.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 's':
            self.freeze_linecut = not self.freeze_linecut
            self.linecut_position = (self.x_index, self.y_index)
        else:
            pass

    def toggle_interpolation(self):
        self.interpolation_enabled = not self.interpolation_enabled
        if self.interpolation_enabled:
            self.open_interpolation_window()

    def open_interpolation_window(self):
        # Create a new pop-up window for interpolation settings
        self.interpolation_window = tk.Toplevel(self.root)
        self.interpolation_window.title("Interpolation Settings")
        self.interpolation_window.geometry("400x200")

        # Add a label and entry widget for the first interpolation value
        tk.Label(self.interpolation_window, text="Enter Interpolation Factor for x-Axis (0.1 - 1.0):").pack()
        self.interpolation_entry_1 = tk.Entry(self.interpolation_window)
        self.interpolation_entry_1.pack()
        self.interpolation_entry_1.insert(0, "1.0")  # Default value

        # Add a label and entry widget for the second interpolation value
        tk.Label(self.interpolation_window, text="Enter Interpolation Factor for y-Axis (0.1 - 1.0):").pack()
        self.interpolation_entry_2 = tk.Entry(self.interpolation_window)
        self.interpolation_entry_2.pack()
        self.interpolation_entry_2.insert(0, "1.0")  # Default value
        submit_button = tk.Button(self.interpolation_window, text="Apply", command=self.apply_interpolation)
        submit_button.pack()

    def open_gaussian_filter_window(self):
        # Create a new pop-up window for interpolation settings
        self.gaussian_filter_window = tk.Toplevel(self.root)
        self.gaussian_filter_window.title("Gaussian Filter Settings")
        self.gaussian_filter_window.geometry("400x200")

        # Add a label and entry widget for the first interpolation value
        tk.Label(self.gaussian_filter_window, text="Pixel for x-Axis:").pack()
        self.filter_pixel_x = tk.Entry(self.gaussian_filter_window)
        self.filter_pixel_x.pack()
        self.filter_pixel_x.insert(0, "1.0")  # Default value

        # Add a label and entry widget for the second interpolation value
        tk.Label(self.gaussian_filter_window, text="Pixel for y-Axis:").pack()
        self.filter_pixel_y = tk.Entry(self.gaussian_filter_window)
        self.filter_pixel_y.pack()
        self.filter_pixel_y.insert(0, "1.0")  # Default value
        submit_button = tk.Button(self.gaussian_filter_window, text="Apply", command=self.apply_gaussian_filter)
        submit_button.pack()

    def open_background_subtraction_window(self):
        self.background_subtraction_window = tk.Toplevel(self.root)
        self.background_subtraction_window.title("Background Subtraction")
        self.background_subtraction_window.geometry("400x200")

        self.background_subtraction_combobox = ttk.Combobox(self.background_subtraction_window, values=self.bg_methods, state='readonly', width=10)
        self.background_subtraction_combobox.pack(side=tk.TOP, padx=5, pady=5)
        self.background_subtraction_combobox.set('Polynomial')
        self.background_subtraction_combobox.bind('<<ComboboxSelected>>', self.update_bg_subtraction_inputs)

        # Frame to contain method-specific input fields
        self.method_input_frame = tk.Frame(self.background_subtraction_window)
        self.method_input_frame.pack(fill=tk.BOTH, expand=True)

        submit_button = tk.Button(self.background_subtraction_window, text="Apply", command=self.apply_background_subtraction)
        submit_button.pack(side=tk.BOTTOM)

        # Initially update inputs for the default selected method
        self.update_bg_subtraction_inputs()

    def open_roi_data_cut_window(self):
        self.roi_data_cut_window = tk.Toplevel(self.root)
        self.roi_data_cut_window.title("ROI Data")
        self.roi_data_cut_window.geometry("300x250")
        self.roi_cut_entry_list = []
        labels = ["min. X:", "min.  Y:", "max. X:", "max. Y:"]
        coordinates = np.array([self.roi_corners[0][0], self.roi_corners[0][1], self.roi_corners[1][0], self.roi_corners[1][1]])
        for i, label in enumerate(labels):
            tk.Label(self.roi_data_cut_window, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = tk.Entry(self.roi_data_cut_window)
            entry.insert(0, str(coordinates[i]))
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.roi_cut_entry_list.append(entry)
        submit_button = tk.Button(self.roi_data_cut_window, text="Apply", command=self.apply_roi_data_cut)
        submit_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def open_draw_lines_window(self):
        self.draw_lines_window = tk.Toplevel(self.root)
        self.draw_lines_window.title("Draw Lines")
        self.draw_lines_window.geometry("400x200")
        # Frames

        self.draw_lines_button_frame = tk.Frame(self.draw_lines_window)
        self.draw_lines_button_frame.pack(side=tk.BOTTOM)

        self.lines_list_frame = tk.Frame(self.draw_lines_window)
        self.lines_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scrollbar
        self.lines_list_scrollbar = tk.Scrollbar(self.lines_list_frame)
        self.lines_list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox for displaying lines
        self.lines_listbox = tk.Listbox(self.lines_list_frame, yscrollcommand=self.lines_list_scrollbar.set)
        self.lines_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.lines_list_scrollbar.config(command=self.lines_listbox.yview)

        # Buttons
        activate_button = tk.Button(self.draw_lines_button_frame, text='Activate',
                                    command=self.activate_line_drawing)
        deactivate_button = tk.Button(self.draw_lines_button_frame, text='Deactivate',
                                      command=self.deactivate_line_drawing)
        reset_lines_button = tk.Button(self.draw_lines_button_frame, text='Reset',
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
            tk.Label(self.method_input_frame, text="X Polynomial Order:").pack()
            self.poly_order_x = tk.Entry(self.method_input_frame)
            self.poly_order_x.pack()
            self.poly_order_x.insert(0, "1")  # Default value

            tk.Label(self.method_input_frame, text="Y Polynomial Order:").pack()
            self.poly_order_y = tk.Entry(self.method_input_frame)
            self.poly_order_y.pack()
            self.poly_order_y.insert(0, "1")  # Default value

            self.use_roi_in_bg_check = tk.Checkbutton(
                self.method_input_frame,
                text="Use ROI",
                variable=self.use_roi_in_bg_subtraction
            )
            self.use_roi_in_bg_check.pack(side=tk.BOTTOM, pady=2)

        elif selected_method == 'Relation Parameters':
            # Add input fields for the Relation Parameters method
            labels = ["Coefficient before X:", "Power of X:", "Coefficient before Y:", "Power of Y:", "Constant term:"]
            for i, label in enumerate(labels):
                tk.Label(self.method_input_frame, text=label).grid(row=i, column=0)
                entry = tk.Entry(self.method_input_frame, width=10)
                entry.insert(0, "0")
                entry.grid(row=i, column=1)
                self.relation_parameter_entry_list.append(entry)

        elif selected_method == 'Subtract Trace Average':
            # Add input fields for the `subtract_trace_average` function
            tk.Label(self.method_input_frame, text="Number of Traces (n):").pack()
            self.n_traces_entry = tk.Entry(self.method_input_frame)
            self.n_traces_entry.pack()
            self.n_traces_entry.insert(0, "1")  # Default value

            tk.Label(self.method_input_frame, text="Axis (0 for rows, 1 for columns):").pack()
            self.axis_entry = tk.Entry(self.method_input_frame)
            self.axis_entry.pack()
            self.axis_entry.insert(0, "0")  # Default value

            tk.Label(self.method_input_frame, text="From End (True/False):").pack()
            self.from_end_var = tk.StringVar(value="False")
            self.from_end_checkbox = tk.Checkbutton(self.method_input_frame, text="From End",
                                                    variable=self.from_end_var, onvalue="True", offvalue="False")
            self.from_end_checkbox.pack()

            tk.Label(self.method_input_frame, text="Use Gaussian Filter (True/False):").pack()
            self.use_filter_var = tk.StringVar(value="False")
            self.use_filter_checkbox = tk.Checkbutton(self.method_input_frame, text="Use Filter",
                                                      variable=self.use_filter_var, onvalue="True", offvalue="False")
            self.use_filter_checkbox.pack()

            tk.Label(self.method_input_frame, text="Polynomial Fit (True/False):").pack()
            self.poly_fit_var = tk.StringVar(value="False")
            self.poly_fit_checkbox = tk.Checkbutton(self.method_input_frame, text="Polynomial Fit",
                                                    variable=self.poly_fit_var, onvalue="True", offvalue="False")
            self.poly_fit_checkbox.pack()

            tk.Label(self.method_input_frame, text="Filter Sigma:").pack()
            self.filter_sigma_entry = tk.Entry(self.method_input_frame)
            self.filter_sigma_entry.pack()
            self.filter_sigma_entry.insert(0, "1.0")  # Default value

            tk.Label(self.method_input_frame, text="Polynomial Fit Order:").pack()
            self.poly_fit_order_entry = tk.Entry(self.method_input_frame)
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

        self.derivative_window = tk.Toplevel(self.root)
        self.derivative_window.title("Calculate derivative along axis")
        self.derivative_window.geometry("400x200")
        self.axis_selection = ['y', 'x']
        self.derivative_combobox = ttk.Combobox(self.derivative_window, values=self.axis_selection, state='readonly', width=10)
        self.derivative_combobox.pack(side=tk.BOTTOM, padx=5, pady=0)
        self.derivative_combobox.set('x')
        submit_button = tk.Button(self.derivative_window, text="Apply", command=self.apply_derivative)
        submit_button.pack()

    def open_data_axis_transform(self):
        self.data_axis_transform_window = tk.Toplevel(self.root)
        self.data_axis_transform_window.title("Axis Scaling and Renaming")
        self.data_axis_transform_window.geometry("400x200")

        self.data_axis_transform_naming_frame = ttk.Frame(self.data_axis_transform_window)
        self.data_axis_transform_scaling_frame = ttk.Frame(self.data_axis_transform_window)

        tk.Label(self.data_axis_transform_naming_frame, text="Axis Names:").pack()
        tk.Label(self.data_axis_transform_scaling_frame, text="Axis Scaling Factors:").pack()

        self.x_axis_name_input = tk.Entry(self.data_axis_transform_naming_frame)
        self.x_axis_name_input.pack()
        self.x_axis_name_input.insert(0, self.name_data_x_axis)
        self.x_axis_scale_input = tk.Entry(self.data_axis_transform_scaling_frame)
        self.x_axis_scale_input.pack()
        self.x_axis_scale_input.insert(0, '1.0')

        self.y_axis_name_input = tk.Entry(self.data_axis_transform_naming_frame)
        self.y_axis_name_input.pack()
        self.y_axis_name_input.insert(0, self.name_data_y_axis)
        self.y_axis_scale_input = tk.Entry(self.data_axis_transform_scaling_frame)
        self.y_axis_scale_input.pack()
        self.y_axis_scale_input.insert(0, '1.0')

        self.z_axis_name_input = tk.Entry(self.data_axis_transform_naming_frame)
        self.z_axis_name_input.pack()
        self.z_axis_name_input.insert(0, self.name_data_z)
        self.z_axis_scale_input = tk.Entry(self.data_axis_transform_scaling_frame)
        self.z_axis_scale_input.pack()
        self.z_axis_scale_input.insert(0, '1.0')

        submit_button = tk.Button(self.data_axis_transform_window, text="Apply", command=self.apply_data_axis_transform)
        submit_button.pack(side=tk.BOTTOM)
        self.data_axis_transform_naming_frame.pack(side=tk.LEFT)
        self.data_axis_transform_scaling_frame.pack(side=tk.RIGHT)

    def open_fft_trace_correction_window(self):
        #### Created by Nico R. ####
        # Create a new Toplevel window
        self.fft_plot_window = tk.Toplevel(self.root)
        self.fft_plot_window.title("FFT Correction")

        # Create a Matplotlib figure and axis
        self.fft_fig, self.fft_ax = plt.subplots(1, 1)

        # Create a canvas to embed the figure into the Tkinter window
        self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=self.fft_plot_window)
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
            self.times = self.data.traces_dt * np.arange(0, len(self.traces[0][0]))
            self.loaded = True

        self.frequencies_shifted, self.fft_mean, self.fft_signals, self.original_angles = get_fourier(self.traces,
                                                                                                      self.times)
        fft_correction_select(self.frequencies_shifted, self.fft_mean, self.fft_fig, self.fft_ax)

        open_plot_button = ttk.Button(self.fft_plot_window, text="Apply changes", command=self.apply_fft_trace_correction)
        open_plot_button.pack(pady=20)

    def apply_data_axis_transform(self):
        self.name_data_x_axis = str(self.x_axis_name_input.get())
        self.name_data_y_axis = str(self.y_axis_name_input.get())
        self.name_data_z = str(self.z_axis_name_input.get())

        self.X = self.X * np.float64(self.x_axis_scale_input.get())
        self.Y = self.Y * np.float64(self.y_axis_scale_input.get())
        self.sliced_data = self.sliced_data * np.float64(self.z_axis_scale_input.get())

        self.xlim = [np.min(self.X), np.max(self.X)]
        self.ylim = [np.min(self.Y), np.max(self.Y)]
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_interpolation(self):
        if self.interpolation_enabled:
            selected_display_values = [combobox.get() for combobox in self.parameter_comboboxes]
            selected_indices = []
            for i, display_value in enumerate(selected_display_values):
                # Convert the display value back to an index
                index = self.display_values_list[i].index(float(display_value))
                selected_indices.append(index)

            # Use selected values to slice and plot data
            x = np.flip(self.data.measure_axis, axis=0)[-1].swapaxes(0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)]
            y = np.flip(self.data.measure_axis, axis=0)[-2].swapaxes(0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)]
            sliced_data = (self.data.measure_data[self.name_data.index(self.data_combobox.get())]).swapaxes(0, 1).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)]
            self.X, self.Y, self.sliced_data = image_down_sampling(sliced_data, x, y, (self.interpolation_entry_1.get(), self.interpolation_entry_2.get()))
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

        self.sliced_data = self.sliced_data - bg

        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_median_difference(self):
        self.sliced_data = correct_median_diff(self.sliced_data)
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_mean_of_lines(self):
        self.sliced_data = correct_mean_of_lines(self.sliced_data)
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_relation_parameters(self):
        self.sliced_data = (self.sliced_data + np.float64(self.relation_parameter_entry_list[0].get()) *
                            self.X ** np.float64(self.relation_parameter_entry_list[1].get()) +
                            np.float64(self.relation_parameter_entry_list[2].get()) *
                            self.Y ** np.float64(self.relation_parameter_entry_list[3].get())
                            + np.float64(self.relation_parameter_entry_list[4].get()))

        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)

        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

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
        self.sliced_data = subtract_trace_average(
            self.sliced_data,
            n=n,
            axis=axis,
            from_end=from_end,
            use_filter=use_filter,
            poly_fit=poly_fit,
            filter_sigma=filter_sigma,
            poly_fit_order=poly_fit_order
        )
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_gaussian_filter(self):
        self.sliced_data = gaussian_filter(self.sliced_data, (float(self.filter_pixel_x.get()), float(self.filter_pixel_y.get())))
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_derivative(self):
        # Calculate mean spacing between X and Y coordinates
        dx = np.mean((np.diff(self.X, axis=1)).flatten())
        dy = np.mean((np.diff(self.Y, axis=0)).flatten())
        # Calculate gradient
        self.sliced_data = np.gradient(self.sliced_data, dx, dy)[self.axis_selection.index(self.derivative_combobox.get())]
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_sum_of_gradient(self):
        # Calculate mean spacing between X and Y coordinates
        dx = np.mean((np.diff(self.X, axis=1)).flatten())
        dy = np.mean((np.diff(self.Y, axis=0)).flatten())
        # Calculate gradient
        self.sliced_data = np.sqrt(np.gradient(self.sliced_data, dx, dy)[0] ** 2
                            + np.gradient(self.sliced_data, dx, dy)[1] ** 2)
        if self.auto_scale_var.get():
            self.vmin = np.min(self.sliced_data)
            self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_2d_fft(self):
        self.X, self.Y, self.sliced_data = two_d_fft_on_data(self.sliced_data, self.X, self.Y)
        self.name_data_z = 'FFT Amp of ' + self.name_data_z
        self.name_data_x_axis = 'freq. of ' + self.name_data_x_axis
        self.name_data_y_axis = 'freq. of ' + self.name_data_y_axis
        self.xlim = [np.min(self.X), np.max(self.X)]
        self.ylim = [np.min(self.Y), np.max(self.Y)]
        self.vmin = np.min(self.sliced_data)
        self.vmax = np.max(self.sliced_data)
        self.update_histogramm()
        self.update_pcolormesh(self.vmin, self.vmax)

    def apply_roi_data_cut(self):
        try:
            # Convert entry values to floats
            x_range = (float(self.roi_cut_entry_list[0].get()), float(self.roi_cut_entry_list[2].get()))
            y_range = (float(self.roi_cut_entry_list[1].get()), float(self.roi_cut_entry_list[3].get()))

            # Validate ranges
            if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
                messagebox.showerror("Invalid Range", "Min values must be less than max values")
                return

            # Store original data for potential undo
            self.original_data = (self.X.copy(), self.Y.copy(), self.sliced_data.copy())

            # Apply the cut
            result = cut_data_range(self.X, self.Y, self.sliced_data, x_range, y_range)
            if result is not None:
                self.X, self.Y, self.sliced_data = result

                # Calculate new min/max if data exists
                if len(self.sliced_data) > 0:

                    if self.auto_scale_var.get():
                        self.vmin = np.min(self.sliced_data)
                        self.vmax = np.max(self.sliced_data)

                    self.xlim = (np.min(self.X), np.max(self.X))
                    self.ylim = (np.min(self.Y), np.max(self.Y))
                    self.update_plot()
                    self.roi_data_cut_window.destroy()  # Close window on success
                else:
                    messagebox.showwarning("Warning", "No data points in the selected range")
            else:
                messagebox.showerror("Error", "Failed to apply ROI cut")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def apply_fft_trace_correction(self):
        #### Modified by Nico R. ####
        self.cuts = get_cuts(self.fft_ax)
        print(self.cuts)
        plt.close(self.fft_fig)

        self.traces = fft_correction_apply(self.traces, self.cuts, self.frequencies_shifted, self.fft_signals,
                                           self.original_angles)
        self.fft_plot_window.destroy()
        self.fft_plot_window.update()

    def activate_line_drawing(self):
        if hasattr(self, 'roi_mode') and self.roi_mode:
            self.toggle_roi()  # Turn off ROI before enabling line drawing

        self.click_cid = self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.move_cid = self.canvas.mpl_connect('motion_notify_event', self.on_canvas_move)

    def deactivate_line_drawing(self):
        if hasattr(self, 'click_cid'):
            self.canvas.mpl_disconnect(self.click_cid)
            del self.click_cid
        if hasattr(self, 'move_cid'):
            self.canvas.mpl_disconnect(self.move_cid)
            del self.move_cid  # Remove the attribute to clean up

    def reset_lines(self):
        self.drawn_lines_list = []
        self.update_lines_listbox()

    def setup_line_editing(self):
        """Set up the line editing functionality with right-click context menu"""
        # Initialize editing state variables
        self.editing_line = False
        self.editing_line_index = None
        self.editing_point = None  # 0 for start point, 1 for end point
        self.edit_markers = []

        # Create a right-click context menu for the lines listbox
        self.lines_context_menu = tk.Menu(self.lines_listbox, tearoff=0)
        self.lines_context_menu.add_command(label="Edit Line", command=self.start_line_editing)
        self.lines_context_menu.add_command(label="Extract Line", command=self.extract_and_plot_linecut)
        self.lines_context_menu.add_command(label="Delete Line", command=self.delete_selected_line)

        # Bind right-click event to show context menu
        self.lines_listbox.bind("<Button-3>", self.show_lines_context_menu)

        # Also bind double-click as a quick way to edit a line
        self.lines_listbox.bind("<Double-Button-1>", lambda event: self.start_line_editing())

    def extract_and_plot_linecut(self):
        """Extract line cut data and display in a UtilityLinePlotter."""
        selected_indices = self.lines_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("No Selection", "Please select a line to edit.")
            return

        selected_index = selected_indices[0]
        if selected_index >= len(self.drawn_lines_list):
            messagebox.showinfo("Invalid Selection", "The selected line no longer exists.")
            return

        self.editing_line_index = selected_index

        if (not hasattr(self, 'linecut_plotter') or
                not hasattr(self.linecut_plotter, 'root')):
            # Create a new plotter
            self.linecut_plotter = UtilityLinePlotter(self.root)

            # Set up protocol for window close using the root window
            self.linecut_plotter.root.protocol("WM_DELETE_WINDOW", self.on_linecut_window_close)

        line = self.drawn_lines_list[self.editing_line_index]
        start_point, end_point = line

        # Get the current data


        # Extract the line cut using the extract_linecut function
        linecut_result = extract_linecut(self.X, self.Y, self.sliced_data, start_point, end_point)

        # Create distance array (x-axis) for the line cut plot
        x0, y0 = start_point
        x1, y1 = end_point
        total_distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        num_points = len(linecut_result)
        distance = np.linspace(0, total_distance, num_points)

        # Generate a label for the line cut
        line_label = f"Line {len(self.linecut_plotter.data) + 1} )"

        # Add the line cut to the plotter
        self.linecut_plotter.add_linecut(distance, linecut_result, label=line_label)

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
            self.drawing_line = False
            if hasattr(self, 'current_line_artist') and self.current_line_artist is not None:
                self.current_line_artist.remove()
                self.current_line_artist = None
            self.current_line = None
            self.canvas.draw_idle()

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
            self.drawn_lines_list[self.editing_line_index] = [(event.xdata, event.ydata), end_point]
            self.edit_markers[0].set_data([event.xdata], [event.ydata])
        else:  # End point
            self.drawn_lines_list[self.editing_line_index] = [start_point, (event.xdata, event.ydata)]
            self.edit_markers[1].set_data([event.xdata], [event.ydata])

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

            # Highlight the selected line if in editing mode
            if self.editing_line and i == self.editing_line_index:
                line_artist, = self.ax.plot(x_values, y_values, color='purple',
                                            linewidth=2, zorder=6)
            else:
                line_artist, = self.ax.plot(x_values, y_values, color='red',
                                            linewidth=1, zorder=5)

            self.line_artists.append(line_artist)

    def update_lines_listbox(self):
        self.lines_listbox.delete(0, tk.END)  # Clear the current contents of the listbox
        for i, line in enumerate(self.drawn_lines_list, start=1):
            # Assuming each line is a tuple of start and end points like ((x1, y1), (x2, y2))
            start_point, end_point = np.round(line, 4)
            line_str = f"Line {i}: Start {start_point} End {end_point}"
            self.lines_listbox.insert(tk.END, line_str)

    def on_canvas_click(self, event):
        if event.inaxes != self.ax:
            return  # Ignore clicks outside the axes

        if not self.drawing_line:
            # Start drawing a new line
            self.drawing_line = True
            self.current_line = [(event.xdata, event.ydata), (event.xdata, event.ydata)]
            # Draw a line on the plot
            (line,) = self.ax.plot([event.xdata, event.xdata], [event.ydata, event.ydata], color='red')
            self.current_line_artist = line
        else:
            # Finalize the current line
            self.drawing_line = False
            # Update the final point of the line
            self.current_line[1] = (event.xdata, event.ydata)
            self.drawn_lines_list.append(self.current_line)
            self.update_lines_listbox()
            self.current_line = None  # Reset for the next line
            self.canvas.draw_idle()

    def on_canvas_move(self, event):
        if event.inaxes != self.ax or not self.drawing_line:
            return  # Ignore if we're not in the process of drawing a line

        # Update the end point of the current line to follow the mouse
        self.current_line[1] = (event.xdata, event.ydata)
        self.current_line_artist.set_data([self.current_line[0][0], event.xdata],
                                          [self.current_line[0][1], event.ydata])
        self.canvas.draw_idle()

    def update_pcolormesh(self, vmin, vmax):
        # Update the pcolormesh with new vmin and vmax values
        self.ax.clear()
        c = self.ax.pcolormesh(self.X, self.Y, self.sliced_data, cmap=self.colormap_combobox.get(), vmin=vmin,
                                   vmax=vmax,
                                   shading='auto', zorder=1, linewidth=0, rasterized=True)
        if hasattr(self, 'cbar'):
            self.cbar.remove()
            del self.cbar

        self.cbar = self.figure.colorbar(c, ax=self.ax, label=self.name_data_z)
        self.ax.set_xlabel(self.name_data_x_axis)
        self.ax.set_ylabel(self.name_data_y_axis)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax_vline.set_ylim(self.ax.get_ylim())
        self.ax_hline.set_xlim(self.ax.get_xlim())

        # Redraw all the lines in drawn_lines_list
        self.redraw_saved_lines()

        if self.crosshair_enabled:
            self.refresh_crosshair()
        self.canvas.draw_idle()

    def update_histogramm(self):
        self.histogram_ax.clear()
        self.histogram_ax.hist(self.sliced_data.flatten(), bins=60, color='blue', alpha=0.7)
        self.histogram_ax.set_yticklabels([])
        self.histogram_ax.set_xticklabels([])

        self.init_movable_lines()
        self.histogram_canvas.draw_idle()

    def update_plot(self):
        # feature does not work as intended and leads to some weird behaviour which allows to use those bugs as a feature
        if self.invert_enabled:
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
        # Clear the plot
        self.ax.clear()
        self.ax_vline.clear()
        self.ax_hline.clear()
        self.histogram_ax.clear()

        # Reset UI elements to their default states
        self.colormap_combobox.set(self.colormaps[0])
        self.data_combobox.set(self.name_data[0]) if self.name_data else None
        for combobox in self.parameter_comboboxes:
            if combobox['values']:
                combobox.set(combobox['values'][0])

        # Hide additional axes and reset crosshair state
        self.ax_vline.set_visible(False)
        self.ax_hline.set_visible(False)
        self.crosshair_enabled = False

        # Reset internal data or state as needed
        self.sliced_data = None
        self.X, self.Y = None, None
        self.xlim, self.ylim = None, None
        self.relation_parameter_entry_list = []
        self.drawn_lines_list = []

        # Redraw the canvas to reflect the reset state
        self.canvas.draw_idle()
        self.histogram_canvas.draw_idle()
        self.root.destroy()


class InteractiveArrayAndLinePlotter(InteractiveArrayPlotter):
    def __init__(self, root, hdf5data):

        self.trace_x_index = 0
        self.trace_y_index = 0

        self.times = 0
        self.trace_xlabel = 'Time (s)'
        self.trace_ylabel = 'Trace Amplitude (V)'

        self.hist_xlabel = 'Amplitudes (V)'
        self.hist_ylabel = 'Counts'
        self.nbins_traces = 50

        self.trace_selected = 0
        self.enable_hist = False
        # Create a new figure with two subplots
        self.figure, (self.ax, self.line_ax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1]), figsize=(10, 10))
        # Call the constructor of the parent class with the new figure and ax
        super().__init__(root, hdf5data, self.figure, self.ax)
        self.canvas.mpl_connect('button_press_event', self.on_right_click)
        self.file_menu.add_command(label="Save displayed Trace as NumPy array", command=self.save_trace)

        # Create Trace Menu
        self.trace_menu = tk.Menu(self.menubar, tearoff=0)
        self.trace_menu.add_command(label="Toggle Histogram", command=self.open_hist_window)
        self.menubar.add_cascade(label="Traces Menu", menu=self.trace_menu)

        #
        # Additional initialization for the line plot
        self.initialize_line_plot()
        self.figure.subplots_adjust(hspace=0.5)
        self.line_ax.set_position([0.125, 0.1, 0.62, 0.2])
        self.ax_vline.set_position([0.91, 0.42, 0.05, 0.47])  # Adjusted position and size
        self.ax_hline.set_position([0.12, 0.94, 0.62, 0.05])  # Adjusted position and size
        # Redraw the canvas with the new figure
        self.canvas.figure = self.figure
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Redraw the toolbar with the new canvas
        self.toolbar.canvas = self.canvas
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.plot_data()

    def initialize_line_plot(self):
        # Set up the line plot
        self.line_ax.set_title('Line Plot Title')
        self.line_ax.set_xlabel('X Axis Label')
        self.line_ax.set_ylabel('Y Axis Label')
        self.line_ax.grid(True)

    def plot_data(self):
        super().plot_data()
        self.update_line_plot()

    def update_line_plot(self):
        selected_display_values = [combobox.get() for combobox in self.parameter_comboboxes]
        selected_indices = []
        for i, display_value in enumerate(selected_display_values):
            # Convert the display value back to an index
            index = self.display_values_list[i].index(float(display_value))
            selected_indices.append(index)
        self.line_order_indeces = (self.data.trace_order).reshape(np.flip(self.data.measure_dim))[tuple(selected_indices)]
        trace_index = int(self.line_order_indeces[self.trace_y_index][self.trace_x_index])
        self.trace_selected = self.data.trace_reference[::, 0, trace_index]
        self.times = self.data.traces_dt * np.arange(0, len(self.trace_selected))
        self.line_ax.clear()

        if not self.enable_hist:
            self.line_ax.set_title(f'Trace at {self.name_data_x_axis}: {self.X[self.trace_y_index][self.trace_x_index]:.3f} ; '
                                   f'{self.name_data_y_axis}: {self.Y[self.trace_y_index][self.trace_x_index]:.3f} ')
            self.line_ax.plot(self.times, self.trace_selected)
            self.line_ax.set_xlabel(self.trace_xlabel)
            self.line_ax.set_ylabel(self.trace_ylabel)

        elif self.enable_hist:
            self.line_ax.set_title(f'Histogram at {self.name_data_x_axis}: {self.X[self.trace_y_index][self.trace_x_index]:.3f} ; '
                                   f'{self.name_data_y_axis}: {self.Y[self.trace_y_index][self.trace_x_index]:.3f} ')
            self.line_ax.hist(self.trace_selected, color='blue', alpha=0.7, edgecolor='black', bins=self.nbins_traces)
            self.line_ax.set_xlabel(self.hist_xlabel)
            self.line_ax.set_ylabel(self.hist_ylabel)

        self.canvas.draw_idle()

    def on_right_click(self, event):
        if event.button == 3:
            self.trace_x_index = np.argmin(np.abs(self.X[0] - event.xdata))
            self.trace_y_index = np.argmin(np.abs(self.Y[:, 0] - event.ydata))
            print(f"Right-clicked at coordinates: ({self.trace_x_index}, {self.trace_y_index})")
            self.update_line_plot()

    def toggle_hist(self):
        self.nbins_traces = int(self.nbins_traces_input.get())
        self.enable_hist = not self.enable_hist
        self.update_line_plot()

    def update_plot(self):
        super().update_plot()
        self.update_line_plot()

    def open_hist_window(self):
        self.toggle_hist_window = tk.Toplevel(self.root)
        self.toggle_hist_window.title("Histogram Settings")
        self.toggle_hist_window.geometry("400x200")

        self.nbins_traces_input = tk.Entry(self.toggle_hist_window)
        self.nbins_traces_input.pack()
        self.nbins_traces_input.insert(0, self.nbins_traces)
        submit_button = tk.Button(self.toggle_hist_window, text="Toggle Histogram", command=self.toggle_hist)
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
        np.save(pth + base_name + 'times_for_trace_at_' + trace_pos + '.npy', self.times, allow_pickle=True)


    def reset(self):
        super().reset()

        self.line_ax.clear()
        self.line_order_indeces = None
        self.times = None
        self.trace_selected = None


class InteractiveTimeTraceMapPlotter(InteractiveArrayPlotter):
    def __init__(self, root, hdf5data):

        super().__init__(root, hdf5data)


class UtilityLinePlotter:
    def __init__(self, master=None):
        """
        Initialize a utility plotter for line cuts.

        Parameters:
        -----------
        master : Tk window
            Parent window, if None a new Tk window will be created
        """
        if master is None:
            self.root = tk.Tk()
            self.root.title("Line Cut Plotter")
            self.root.geometry("800x600")
        else:
            self.root = tk.Toplevel(master)
            self.root.title("Line Cut Plotter")
            self.root.geometry("800x600")

        # Store line cut data as list of tuples (x, y, label)
        self.data = []
        self.selected_line_idx = None

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
        self.line_listbox = tk.Listbox(line_select_frame)
        self.line_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.line_listbox.bind('<<ListboxSelect>>', self.on_line_select)

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

    def is_alive(self):
        """Check if the toplevel window still exists and is not destroyed."""
        try:
            return self.toplevel.winfo_exists()
        except:
            return False

    def update_plot(self):
        """Update the plot with current data and settings."""
        self.ax.clear()

        colors = plt.cm.tab10.colors

        for i, (x, y, label) in enumerate(self.data):
            color = colors[i % len(colors)]
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


