import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import h5py
import numpy as np
import os
import shutil
from HDF5Data import HDF5Data
from interactive_plotting_tools import InteractiveArrayPlotter
from interactive_plotting_tools import InteractiveArrayAndLinePlotter
from creating_hdf5_files_from_npy_files import CreateHDF5File

import traceback


array_plotters = []
list_name = ['Channels', 'Instrument config', 'Instruments', 'Log list', 'Settings', 'Step config', 'Step list', 'Tags', 'Views']

def data_menu_bar(root, hdf5data):
    menubar = tk.Menu(root)
    # Adding File Menu and commands
    file = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='File', menu=file)
    file.add_command(label='Select File Directory', command=lambda: get_path(hdf5data))
    file.add_command(label='Move File to', command=lambda: move_data(hdf5data))
    file.add_command(label='Save File as', command=lambda: save_data_as(hdf5data))
    file.add_command(label='Create a HDF5 File from Numpy Files', command=lambda : create_hdf5_files_from_npy(root))
    file.add_command(label='Remove Selected Datasets', command=lambda: remove_selected_options_window(root, hdf5data)) #Hannah Vogel: to select datasets to be removed
    file.add_separator()
    file.add_command(label='Add traces from HDF5 File', command=lambda: add_traces_window(hdf5data)) # Nico Reinders: to add traces to current file from another HDF5 file
    file.add_command(label='Generate traces from dataset', command=lambda: transform_traces_window(hdf5data)) # Nico Reinders: create a file with a 'Traces' group that is compatible with the interactive data viewer 
    
    data = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Data', menu=data)
    data.add_command(label='Save Data as Numpy Array', command=lambda: create_data_array(hdf5data))
    data.add_command(label='Save Traces as Numpy Arrays', command=lambda: create_trace_array(hdf5data))
    
    
    data.add_command(label='Calculate Histograms', command=lambda: create_hist_data(hdf5data))
    

    data.add_separator()
    plotting = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Plotting', menu=plotting)
    plotting.add_command(label='Plot Map', command=lambda: plot_array(hdf5data, root))
    plotting.add_command(label='Plot Map with Trace Data', command=lambda: plot_array_with_trace_data(hdf5data, root))
    plotting.add_separator()

    return menubar


def get_path(hdf5Data):
    pth = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")])
    hdf5Data.set_path(pth, 'r')
    hdf5Data.set_filename()
    hdf5Data.vars = []
    return pth

def get_unique_filename(filepath):
    #Returns a unique filename by appending a number if the file already exists
    base, extension = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}({counter}){extension}"
        counter += 1

    return new_filepath


def apply_reshape(selected_dataset, selected_axis_dataset, dimension_index):
    """
    Added by Nico Reinders
    Reshapes the selected dataset to be compatible with the requirements of the data viewer "Plot Map with Trace Data"
    by moving the specified dimension to the first position and reshaping it.
    
    """
    if type(selected_dataset) is not np.ndarray:
        selected_dataset = np.array(selected_dataset[:])
        
    if len(np.shape(selected_dataset)) == 2:
        selected_dataset = np.array([selected_dataset])
        dimension_index += 1
    
    if selected_axis_dataset is None: #if no dataset is selected for the x-axis, use default values
        t0, dt = 0, 1
        print("No axis dataset selected, using default t0=0 and dt=1.")
    else:
        # materialize to a NumPy array (works for h5py.Dataset and ndarrays)
        if type(selected_axis_dataset) is not np.ndarray:
            arr = np.asarray(selected_axis_dataset[()]) if hasattr(selected_axis_dataset, "__getitem__") else np.asarray(selected_axis_dataset)
        arr = np.ravel(arr)  # flatten

        if arr.ndim == 1 and arr.size >= 2: # Check if the axis dataset is 1D and has at least 2 elements
            t0 = arr[0]
            diffs = np.diff(arr)
            dt = np.min(np.abs(diffs))
        else: # if the axis dataset has unusable shape, use default values
            t0, dt = 0, 1
            print("Warning: Selected axis dataset is not 1D or too short, using default t0=0 and dt=1.")

    
    shape_original = selected_dataset.shape
    print(f"Original Shape of spectra: {shape_original}") 
    
    # Keep a copy of the original spectra for mean calculation
    spectra_original = selected_dataset.copy()

    selected_dataset = np.moveaxis(selected_dataset, dimension_index, 0)  # Move the selected axis to the first position
    selected_dataset = np.reshape(selected_dataset, (selected_dataset.shape[0], 1, -1)) # Reshape to required shape
    shape = selected_dataset.shape
    print(f"Shape of spectra: {shape}") 
        
    # Validate shape_original dimensions
    if len(shape_original) < 2:
        raise ValueError("The selected dataset must have at least two dimensions.")

    # Adjust data_data shape to account for dimension_index
    reduced_shape = list(shape_original)
    reduced_shape.pop(dimension_index)  # Remove the selected dimension

    data_data = np.zeros((reduced_shape[0], 3, reduced_shape[1]))

    # Assign values to data_data with explicit broadcasting
    data_data[:, 0, :] = np.broadcast_to(np.arange(reduced_shape[0])[:, None], (reduced_shape[0], reduced_shape[1]))
    data_data[:, 1, :] = np.broadcast_to(np.arange(reduced_shape[1])[None, :], (reduced_shape[0], reduced_shape[1]))
    data_data[:, 2, :] = np.mean(spectra_original, axis=dimension_index).reshape(reduced_shape)  # Reshape mean result to match reduced dimensions

    print(f"Shape of data_data: {data_data.shape}")

    output_path = filedialog.asksaveasfilename(defaultextension=".hdf5", filetypes=[("HDF5 files", "*.hdf5")], title="Save HDF5 file as...")

    if output_path:
        print("Saving traces file to:", output_path)
    else:
        print("User cancelled")

    # Save the reshaped data and traces in the output file
    with h5py.File(output_path, 'w') as out_file:
        traces_grp = out_file.create_group('Traces', track_order=True)
        traces_grp.create_dataset('Data', data=selected_dataset)
        traces_grp.create_dataset('Data_N', data=[shape[0]])
        traces_grp.create_dataset('Alazar Slytherin - Ch1 - Data_t0dt', data=[[t0, dt]])

        data_grp = out_file.create_group('Data')
        data_grp.attrs['Step dimensions'] = [reduced_shape[0], reduced_shape[1]]
        data_grp.attrs['Step index'] = [0, 1]
        data_grp.create_dataset('Data', data=data_data)
        data_grp.create_dataset('Channel names', data=[(b'Axis 1', b''), (b'Axis 2', b''), (b'Channel 1', b'')])
        out_file.create_dataset('Log list', data=[(b'Channel 1', b'')])

        out_file.close()
    
    
def transform_traces_window(hdf5Data):
    '''
    Added by Nico Reinders
    Opens a window to select a dataset to reshape into traces.
    '''
    
    pth = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5"), ("Numpy files", "*.npy")])
    if not pth:
        print("No file selected. Operation cancelled.")
        return
    root, ext = os.path.splitext(pth)

    if ext == 'hdf5':
        reshape_hdf5Data = HDF5Data(wdir=pth)
        reshape_hdf5Data.set_path(pth, 'r')
        # Open file and keep it open for the window lifetime
        reshape_hdf5Data.file = h5py.File(pth, 'r')
    else: 
        arr = np.load(pth)
        print("Select axis numpy file for traces if needed.")
        axis_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if not axis_path:
            axis_arr = None
        else: 
            axis_arr = np.load(axis_path)

    def check_axis_reshape_requirements(selected_item):
        # Check if the selected item is a valid dataset as x-axis for traces
        if not isinstance(selected_item, h5py.Dataset):
            # print("Selected item is not a dataset.")
            return False        
        elif len(selected_item.shape) != 1:
            # print("Selected dataset does not have 1 dimension.")
            return False
        elif selected_item.shape[0] < 2:
            # print("Selected dataset is too short, must have at least 2 elements.")
            return False
        else: 
            return True
    
    def check_dataset_reshape_requirements(selected_item):
        # Check if the selected item is a valid dataset for reshaping to traces
        if selected_item is None:
            print("No item selected.")
            return False
        elif not isinstance(selected_item, h5py.Dataset):
            print("Selected item is not a dataset.")
            return False        
        elif len(selected_item.shape) not in (2, 3):
            print("Selected dataset does not have 2 or 3 dimensions.")
            return False
        else: 
            return True


    def on_var_change(*args): 
        # Update the labels and dimension index based on the selected datasets
        selected_dataset = dataset_map[dataset_selection.get()] if dataset_selection.get() in dataset_map else None
        selected_axis_dataset = axis_map[axis_selection.get()] if axis_selection.get() in axis_map else None
        dataset_label_text.set(f"{selected_dataset if selected_dataset is not None else ''}")
        axis_label_text.set(f"{selected_axis_dataset if selected_axis_dataset is not None else ''}")
        
        if selected_axis_dataset is not None and np.array(selected_axis_dataset[:]).ndim == 1:
            dim = np.shape(selected_dataset).index(len(selected_axis_dataset))
        else:
            dim = 0
        dimension_index.set(dim)

    
            
    transform_options = tk.Toplevel()

    def on_close_transform_options():
        try:
            if reshape_hdf5Data.file:
                reshape_hdf5Data.file.close()
        except Exception:
            pass
        transform_options.destroy()
  
    
    transform_options.protocol("WM_DELETE_WINDOW", on_close_transform_options)

    
    # Frame for dataset labels
    label_frame = tk.Frame(transform_options)
    label_frame.pack(anchor='w', pady=5, padx=5, fill='x')

    # Store the valid datasets and axis datasets in dictionaries
    if ext == 'hdf5':
        dataset_map = {}
        axis_map = {}
        file = reshape_hdf5Data.file
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and check_dataset_reshape_requirements(obj):
                dataset_map[name] = obj  # name is the full HDF5 path
        file.visititems(visitor)

        def visitor_axis(name, obj):
            if isinstance(obj, h5py.Dataset) and check_axis_reshape_requirements(obj):
                axis_map[name] = obj  # name is the full HDF5 path
        file.visititems(visitor_axis)
        
        axis_map['None'] = None  # Add a 'None' option for no axis dataset

        datasets_names = list(dataset_map.keys())
        axis_names = list(axis_map.keys())

        dataset_selection = tk.StringVar(value=datasets_names[0] if datasets_names else "")  # default selection
        database_combo = ttk.Combobox(label_frame, textvariable=dataset_selection, values=datasets_names, state="readonly")
        

        axis_selection = tk.StringVar(value=axis_names[0] if axis_names else "")  # default selection
        axis_combo = ttk.Combobox(label_frame, textvariable=axis_selection, values=axis_names, state="readonly")
        
        dataset_selection.trace_add("write", on_var_change)
        axis_selection.trace_add("write", on_var_change)

        
        dataset_label_text = tk.StringVar(value=f"{dataset_map[dataset_selection.get()] if dataset_selection.get() in dataset_map else ''}")
        axis_label_text = tk.StringVar(value=f"{axis_map[axis_selection.get()] if axis_selection.get() in axis_map else ''}")

        database_combo.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        axis_combo.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        
        tk.Label(label_frame, textvariable=dataset_label_text).grid(row=0, column=2, padx=10, pady=5, sticky='w')    
        tk.Label(label_frame, textvariable=axis_label_text).grid(row=1, column=2, padx=10, pady=5, sticky='w')    
        
        tk.Label(label_frame, text="Selected Dataset:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
        tk.Label(label_frame, text="Selected Axis Dataset:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    else:
        tk.Label(label_frame, text=f"Numpy file shape: {arr.shape}")
    # Frame for spinbox + label
    spin_frame = tk.Frame(transform_options)
    
    spin_frame.pack(anchor='w', pady=5, padx=5, fill='x')

    tk.Label(spin_frame, text="Index of dimension in selected dataset to be used as x axis:").pack(side='left', padx=(0, 5))

    def validate_int(new_value):
        # validate the spinbox input
        if new_value == "":  # allow empty (so user can type)
            return True
        try:
            value = int(new_value)
        except ValueError:
            return False
        return 0 <= value <= 2
    
    vcmd = (transform_options.register(validate_int), '%P')  # %P is the new value of the spinbox    
        
    dimension_index = tk.IntVar(value=0)  # Default to 0
    
    # add a spinbox to select the dimension that will be used as trace length
    tk.Spinbox(spin_frame, from_=0, to=2, increment=1, width=5, textvariable=dimension_index,validate="key", validatecommand=vcmd).pack(side='left')

    if ext == 'hdf5':
        on_var_change()  # Initial call to set labels

    # Buttons frame
    button_frame = tk.Frame(transform_options)
    button_frame.pack(pady=10)

    if ext == 'hdf5':
        confirm_button = tk.Button(
            button_frame,
            text="Confirm Reshape",
            command=lambda: (apply_reshape(dataset_map[dataset_selection.get()], axis_map[axis_selection.get()], int(dimension_index.get())), transform_options.destroy())
        )
    else:
        confirm_button = tk.Button(
            button_frame,
            text="Confirm Reshape",
            command=lambda: (apply_reshape(arr, axis_arr, int(dimension_index.get())), transform_options.destroy())
        )
    confirm_button.pack(side='left', padx=5)

    cancel_button = tk.Button(button_frame, text="Cancel", command=transform_options.destroy)
    cancel_button.pack(side='left', padx=5)
    
    
    
    
def add_traces_window(hdf5Data):
    """
    Added by Nico Reinders
    Opens a window to select a file to copy groups or datasets from
    Then shows the content of the file in a treeview
    Allows the user to select a group or dataset and copy it to the current hdf5 file
    """
    
    pth = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")])
    traces_hdf5Data = HDF5Data(wdir=pth)
    traces_hdf5Data.set_path(pth, 'r')
    
    # open treeview window
    traces_selection_window = tk.Toplevel()
    traces_selection_window.title('Add Traces from HDF5 File')
        
    #add an entry for the group name in the destination file
    group_frame = tk.Frame(traces_selection_window)
    group_frame.pack(pady=5)
    tk.Label(group_frame, text="Destination group name:").pack(side=tk.LEFT)
    group_name_var = tk.StringVar(value='Traces')
    group_name_entry = tk.Entry(group_frame, textvariable=group_name_var, width=30)
    group_name_entry.pack(side=tk.LEFT, padx=5)

    # show treeview of the source file
    traces_tree = display_hdf5_file(traces_selection_window, traces_hdf5Data)

    def copy_selected_dataset():
        """
        Copies the selected dataset or group from the source file traces_hdf5Data to the current hdf5 file.
        """
        
        traces_hdf5Data.set_data()
        selected_items = traces_tree.selection()
        
        if not selected_items:
            print("No Selection", "Please select a dataset or group to copy.")
            return
        selected_item = selected_items[0]
        values_above = get_values_above_clicked_node(selected_item, traces_tree)
        file_dir_sep = '/'
        file_dir = file_dir_sep.join(values_above)
        dest_group = group_name_var.get().strip() or 'Traces'
        # Always use string keys for h5py access
        try:
            h5obj = traces_hdf5Data.file[file_dir]
        except Exception as e:
            print("Error", f"Could not access {file_dir}: {e}")
            return
        
        if isinstance(h5obj, h5py.Group):
            trace_names = [str(name) for name in h5obj.keys()]
            traces = [h5obj[str(trace)] for trace in trace_names]
            # Add datasets to group if it exists, else create group
            with h5py.File(hdf5Data.readpath, 'r+') as dest_file:
                if dest_group in dest_file:
                    group = dest_file[dest_group]
                else:
                    group = dest_file.create_group(dest_group)
                for trace_name, trace in zip(trace_names, traces):
                    if trace_name in group:
                        del group[trace_name]
                    group.create_dataset(trace_name, data=trace[()])
            hdf5Data.set_data()
        elif isinstance(h5obj, h5py.Dataset):
            import time
            trace_name = values_above[-1]
            with h5py.File(hdf5Data.readpath, 'r+') as dest_file:
                t0 = time.time()
                recreate_group = False
                # Always use absolute group path, never nest
                if dest_group in dest_file:
                    old_group = dest_file[dest_group]
                    # Only recreate if track_order is not already True
                    track_order = getattr(old_group, 'track_order', None)
                    if not track_order:
                        recreate_group = True
                if recreate_group:
                    print(f"Recreating group '{dest_group}' with track_order=True to avoid nesting.")
                    new_group = dest_file.create_group('dest_group_tmp', track_order=True)
                    dest_file.copy(old_group, new_group)
                    del dest_file[old_group.name]
                    dest_file.move('dest_group_tmp', dest_group)
                    group = dest_file[dest_group]  # Always re-fetch from root
                elif dest_group in dest_file:
                    group = dest_file[dest_group]
                else:
                    group = dest_file.create_group(dest_group, track_order=True)
                t1 = time.time()
                # Save the dataset directly in the destination group, not as a subgroup
                if trace_name in group:
                    del group[trace_name]
                group.create_dataset(trace_name, data=h5obj[()])
                t2 = time.time()
                print(f"Dataset copy timings: group_prep={t1-t0:.3f}s, create={t2-t1:.3f}s, total={t2-t0:.3f}s")
            hdf5Data.set_data()
        else:
            print("Invalid Selection", "Selected item is neither a group nor a dataset.")
            return

    # Add a button to trigger the copy
    copy_button = tk.Button(traces_selection_window, text="Copy Selected Dataset(s)", command=copy_selected_dataset)
    copy_button.pack(pady=10)
    
        

# Hannah Vogel
# Modified by H.D to additionly handle whole folders instead of single files
def remove_selected_options_window(root, hdf5Data):
    # Removes selected datasets of selected data file
    def confirm_selection():
        # Get the selected groups that will be skipped
        selected_groups = [var.get() for var in vars if var.get()]

        # Create action selection buttons
        action_frame = tk.Frame(newWindow)
        action_frame.pack(fill='x', pady=10)

        tk.Label(action_frame, text="Apply to:").pack(side='left', padx=5)

        # Process single file button
        single_file_button = tk.Button(
            action_frame,
            text="Single File",
            command=lambda: process_single_file(selected_groups)
        )
        single_file_button.pack(side='left', padx=5)

        # Process folder button
        folder_button = tk.Button(
            action_frame,
            text="Folder of Files",
            command=lambda: process_folder(selected_groups)
        )
        folder_button.pack(side='left', padx=5)

    def process_single_file(selected_groups):
        # Original single file processing logic
        dataset = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")])
        if not dataset:  # User canceled
            return

        try:
            src_file = h5py.File(dataset, 'r')  # Open source file in read mode
            unique_dest_filepath = get_unique_filename(dataset.replace('.hdf5', '') + '_reduced.hdf5')
            dest_file = h5py.File(unique_dest_filepath, 'w')  # Open or create destination file in write mode

            # Process status window
            status_window = create_status_window(newWindow)
            status_var = status_window['status_var']

            # Update status
            status_var.set(f"Processing file: {os.path.basename(dataset)}")
            newWindow.update_idletasks()

            # Process the file
            hdf5Data.skip_selected_objects_recursive_in_copying_process(src_file, dest_file, selected_groups)

            src_file.close()
            dest_file.close()

            # Update status and close after delay
            status_var.set(f"Completed! Output: {os.path.basename(unique_dest_filepath)}")
            newWindow.update_idletasks()
            newWindow.after(2000, status_window['window'].destroy)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"An error occurred in process_single_file: {e}")
            traceback.print_exc()

    def process_folder(selected_groups):
        # Select folder containing HDF5 files
        folder_path = filedialog.askdirectory(title="Select Folder with HDF5 Files")
        if not folder_path:  # User canceled
            return

        # Get all HDF5 files in the folder
        hdf5_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(folder_path, file))

        if not hdf5_files:
            messagebox.showinfo("No Files", "No HDF5 files found in the selected folder.")
            return

        # Create output folder
        output_folder = os.path.join(folder_path, "reduced_files")
        os.makedirs(output_folder, exist_ok=True)

        # Process status window with progress bar
        status_window = create_status_window(newWindow, show_progress=True)
        status_var = status_window['status_var']
        progress_var = status_window['progress_var']
        progress_bar = status_window['progress_bar']

        # Update initial status
        status_var.set(f"Processing {len(hdf5_files)} files...")
        progress_var.set(0)
        newWindow.update_idletasks()

        # Process each file
        processed_files = 0
        error_files = 0

        for i, file_path in enumerate(hdf5_files):
            try:
                # Update status for current file
                file_name = os.path.basename(file_path)
                status_var.set(f"Processing file {i + 1}/{len(hdf5_files)}: {file_name}")
                progress_var.set((i / len(hdf5_files)) * 100)
                newWindow.update_idletasks()

                # Create output file path
                output_path = os.path.join(
                    output_folder,
                    file_name.replace('.hdf5', '') + '_reduced.hdf5'
                )

                # Open files
                src_file = h5py.File(file_path, 'r')
                dest_file = h5py.File(output_path, 'w')

                # Process the file
                hdf5Data.skip_selected_objects_recursive_in_copying_process(src_file, dest_file, selected_groups)

                # Close files
                src_file.close()
                dest_file.close()

                processed_files += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                traceback.print_exc()
                error_files += 1

        # Update final status
        progress_var.set(100)
        status_var.set(f"Completed! Processed: {processed_files}, Errors: {error_files}")

        # Add close button
        tk.Button(
            status_window['window'],
            text="Close",
            command=status_window['window'].destroy
        ).pack(pady=5)

    def create_status_window(parent, show_progress=False):
        # Create a status window for showing processing progress
        status_window = tk.Toplevel(parent)
        status_window.title("Processing Status")
        status_window.geometry("400x150")

        # Make it stay on top of the parent window
        status_window.transient(parent)

        # Status label
        status_var = tk.StringVar(value="Processing...")
        status_label = tk.Label(status_window, textvariable=status_var, wraplength=380)
        status_label.pack(pady=10, fill='x')

        # Progress bar (optional)
        progress_var = tk.DoubleVar(value=0)
        progress_bar = None

        if show_progress:
            progress_bar = ttk.Progressbar(
                status_window,
                variable=progress_var,
                maximum=100,
                mode='determinate',
                length=350
            )
            progress_bar.pack(pady=10)

        return {
            'window': status_window,
            'status_var': status_var,
            'progress_var': progress_var,
            'progress_bar': progress_bar
        }


    # Create the selection window
    newWindow = tk.Toplevel(root)
    newWindow.title("Select Datasets to Remove")
    newWindow.geometry("400x500")

    # Instructions label
    tk.Label(
        newWindow,
        text="Select datasets to remove from HDF5 files:",
        wraplength=350
    ).pack(pady=10)

    # Create a frame with scrollbar for checkbuttons
    scroll_frame = tk.Frame(newWindow)
    scroll_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create canvas and scrollbar
    canvas = tk.Canvas(scroll_frame)
    scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)

    # Configure canvas
    checkbutton_frame = tk.Frame(canvas)
    checkbutton_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=checkbutton_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack scrollbar and canvas
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Add checkbuttons to the frame
    vars = []
    for string in list_name:
        var = tk.StringVar()
        checkbutton = tk.Checkbutton(
            checkbutton_frame,
            text=string,
            variable=var,
            onvalue=string,
            offvalue=''
        )
        checkbutton.pack(anchor='w')
        vars.append(var)

    # Add confirm button at the bottom
    confirm_button = tk.Button(
        newWindow,
        text="Confirm Selection",
        command=confirm_selection
    )
    confirm_button.pack(pady=10)


#
def save_data_as(hdf5Data):
    pth = filedialog.askdirectory() + '/'
    hdf5Data.set_path(pth + hdf5Data.file_name, 'w')
    hdf5Data.copy_to(hdf5Data.savepath)


def move_data(hdf5Data):
    pth = filedialog.askdirectory() + '/'
    hdf5Data.set_path(pth + hdf5Data.file_name, 'w')
    hdf5Data.move_and_delete(hdf5Data.savepath)


def create_hist_data(hdf5Data):
    number_str = simpledialog.askstring("Input", "Enter an integer:")
    try:
        # Attempt to convert the entered value to an integer
        nbins = int(number_str)

    except ValueError:
        # Handle the case where the entered value is not a valid integer
        return None
    hdf5Data.set_data()
    hdf5Data.set_traces()
    hdf5Data.calc_hist(nbins)


def create_trace_array(hdf5Data):
    hdf5Data.save_traces_in_wdir()


def create_data_array(hdf5Data):
    pth = filedialog.askdirectory() + '/'
    base_name, _ = os.path.splitext(hdf5Data.file_name)
    new_filename = f'{base_name}.npy'
    hdf5Data.set_data()
    hdf5Data.set_arrays()
    hdf5Data.set_array_tags()
    channel_names = [str(name_i[0]) for name_i in hdf5Data.array_tags]
    np.save(pth + base_name + '_tags_.npy', channel_names, allow_pickle=True)
    np.save(pth + new_filename, hdf5Data.arrays, allow_pickle=True)

def create_hdf5_files_from_npy(root):
    new_window = tk.Toplevel(root)
    new_window.title("Create HDF5 file")
    CreateHDF5File(new_window)

def plot_array(hdf5Data, root):
    new_window = tk.Toplevel(root)
    new_window.title("Array Plotter")
    hdf5Data.set_data()
    hdf5Data.set_measure_dim()
    hdf5Data.set_measure_data_and_axis()
    plotter = InteractiveArrayPlotter(new_window, hdf5Data)
    array_plotters.append(plotter)


def plot_array_with_trace_data(hdf5Data, root):
    new_window = tk.Toplevel(root)
    new_window.title("Array Plotter")
    hdf5Data.set_data()
    hdf5Data.set_measure_dim()
    hdf5Data.set_measure_data_and_axis()
    hdf5Data.trace_loading_with_referance()
    hdf5Data.set_traces_dt()
    plotter = InteractiveArrayAndLinePlotter(new_window, hdf5Data)
    array_plotters.append(plotter)


####
def get_values_above_clicked_node(item, tree):
    values = []
    while item:
        value = tree.item(item, "text")
        values.insert(0, value)
        item = tree.parent(item)
    return values

def display_hdf5_file(root, hdf5Data):

    # Function to open an HDF5 file
    def open_hdf5_file():
        hdf5Data.set_data()
        if hdf5Data:
            with hdf5Data.file as file:
                # Function to display the content of a group recursively
                def display_group(group, parent_tree_node):
                    for name, item in group.items():
                        if isinstance(item, h5py.Group):
                            child_node = tree.insert(parent_tree_node, "end", text=name)
                            display_group(item, child_node)
                        elif isinstance(item, h5py.Dataset):
                            dataset_node = tree.insert(parent_tree_node, "end", text=name, value=(item.shape,))
                        else:
                            tree.insert(parent_tree_node, "end", text=name)

                # Display the content of the root group
                display_group(file, "")

    def close_tree_and_hdf5data(hdf5Data):
        for item in tree.get_children():
            tree.delete(item)
        for ploter in array_plotters:
            ploter.reset()
        if os.path.exists(hdf5Data.wdir) and os.path.isdir(hdf5Data.wdir):
            for filename in os.listdir(hdf5Data.wdir):
                file_path = os.path.join(hdf5Data.wdir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        hdf5Data.reset()
        hdf5Data = None

   

    def open_selection(event):
        item = tree.selection()[0]
        value = tree.item(item, "text")
        position = tree.index(item)
        print(f"Selected item: {value}, Position: {position}")

    def on_single_click(event):
        hdf5Data.set_data()
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        hdf5Data.set_current_h5dir(file_dir)
        print(f"Single-clicked on item: {file_dir}")

    def on_right_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        print(f"right-clicked on item: {values_above}")

    def on_double_right_click(event):
        hdf5Data.set_data()
        parent_item = tree.selection()[0]
        values_above = get_values_above_clicked_node(parent_item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        with hdf5Data.file as file:
            if isinstance(file[file_dir], h5py.Dataset):
                for i, list_values in enumerate(file[file_dir]):
                    tree.insert(parent_item, "end", text=f'{i}', values=(list_values,))

    def on_double_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        print(f"Double-clicked on item: {file_dir}")

    # Frame for buttons
    frame = ttk.Frame(root)
    frame.pack(side=tk.TOP, padx=5, pady=5)
    # Button to close the tree and reset hdf5Data
    close_button = tk.Button(frame, text="Close HDF5 File", command=lambda: close_tree_and_hdf5data(hdf5Data))
    close_button.pack(side=tk.RIGHT, pady=10)
    # Button to open an HDF5 file
    open_button = tk.Button(frame, text="Show HDF5 File", command=open_hdf5_file)
    open_button.pack(side=tk.RIGHT, pady=10)


    # Create a treeview widget to display the HDF5 file structure
    tree = ttk.Treeview(root, columns=("Value"))
    tree.heading("#0", text="HDF5 File Structure", anchor="w")
    tree.heading("Value", text="Value", anchor="w")
    tree.pack(fill="both", expand=True)

    # Create a bindings for treeview widget
    tree.bind('<<TreeviewSelect>>', open_selection)
    tree.bind('<Button-1>', on_single_click)
    tree.bind('<Button-2>', on_right_click)
    tree.bind('<Double-1>', on_double_click)
    tree.bind('<Double-2>', on_double_right_click)
    return tree


def main():
    global wdir

    def on_close():
        if os.path.exists(wdir):
            shutil.rmtree(wdir)
        root.destroy()

    # Set wdir as a sub-folder in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wdir = os.path.join(script_dir, 'wdir')
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    # Create the Tkinter root window
    hdf5Data = HDF5Data(wdir=wdir)
    root = tk.Tk()
    data_bar = data_menu_bar(root, hdf5Data)
    root.config(menu=data_bar)
    root.title('HDF5 File Viewer')
    root.protocol("WM_DELETE_WINDOW", on_close)
    tree = display_hdf5_file(root, hdf5Data)
    root.mainloop()
    # Run the Tkinter main loop


if __name__ == '__main__':
    main()