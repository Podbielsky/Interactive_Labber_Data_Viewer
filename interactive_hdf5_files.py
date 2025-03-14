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

    def get_values_above_clicked_node(item):
        values = []
        while item:
            value = tree.item(item, "text")
            values.insert(0, value)
            item = tree.parent(item)
        return values

    def open_selection(event):
        item = tree.selection()[0]
        value = tree.item(item, "text")
        position = tree.index(item)
        print(f"Selected item: {value}, Position: {position}")

    def on_single_click(event):
        hdf5Data.set_data()
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        hdf5Data.set_current_h5dir(file_dir)
        print(f"Single-clicked on item: {file_dir}")

    def on_right_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item)
        print(f"right-clicked on item: {values_above}")

    def on_double_right_click(event):
        hdf5Data.set_data()
        parent_item = tree.selection()[0]
        values_above = get_values_above_clicked_node(parent_item)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        with hdf5Data.file as file:
            if isinstance(file[file_dir], h5py.Dataset):
                for i, list_values in enumerate(file[file_dir]):
                    tree.insert(parent_item, "end", text=f'{i}', values=(list_values,))

    def on_double_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item)
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