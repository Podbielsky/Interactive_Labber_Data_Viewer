import tkinter as tk
from tkinter import filedialog, messagebox
import os
import tempfile

import ttkbootstrap as ttk
import numpy as np
import h5py


def _as_real_numeric_array(value, field_name):
    """Return a NumPy array suitable for plotting, or raise a useful error."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(
            f"The {field_name!r} field must contain real numeric values; "
            f"its dtype is {array.dtype}."
        )
    return array


def normalize_xyz_arrays(x_data, y_data, z_data):
    """Expand 1D coordinates to grids matching a one- or two-dimensional Z."""
    z_array = _as_real_numeric_array(z_data, 'z')
    if z_array.ndim == 1:
        z_grid = z_array.reshape(1, -1)
    elif z_array.ndim == 2:
        z_grid = z_array
    else:
        raise ValueError(
            f"The z field must be a 1D or 2D array, not {z_array.ndim}D."
        )
    if z_grid.size == 0:
        raise ValueError('The z field must not be empty.')

    rows, columns = z_grid.shape

    def expand_coordinate(value, axis_name):
        coordinate = _as_real_numeric_array(value, axis_name)
        if coordinate.ndim == 2:
            if coordinate.shape != z_grid.shape:
                raise ValueError(
                    f"The 2D {axis_name} grid has shape {coordinate.shape}, but "
                    f"z has shape {z_grid.shape}."
                )
            return coordinate
        if coordinate.ndim != 1:
            raise ValueError(
                f"The {axis_name} field must be a 1D array or 2D grid, not "
                f"{coordinate.ndim}D."
            )

        expected_length = columns if axis_name == 'x' else rows
        if coordinate.size not in (1, expected_length):
            orientation = 'columns' if axis_name == 'x' else 'rows'
            raise ValueError(
                f"The 1D {axis_name} array has length {coordinate.size}. It must "
                f"have length {expected_length} (the number of z {orientation})"
                " or contain one constant value."
            )

        if coordinate.size == 1:
            return np.full(z_grid.shape, coordinate[0], dtype=coordinate.dtype)
        if axis_name == 'x':
            return np.broadcast_to(coordinate.reshape(1, columns), z_grid.shape)
        return np.broadcast_to(coordinate.reshape(rows, 1), z_grid.shape)

    return expand_coordinate(x_data, 'x'), expand_coordinate(y_data, 'y'), z_grid


def write_xyz_hdf5(output_path, x_data, y_data, z_data, channel_names=None,
                   overwrite=False):
    """Write x/y/z data in the Labber-compatible layout used by the viewer."""
    output_path = os.path.abspath(os.path.expanduser(output_path))
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"The output file already exists:\n{output_path}")

    x_grid, y_grid, z_grid = normalize_xyz_arrays(x_data, y_data, z_data)
    names = channel_names or ('X-axis', 'Y-axis', 'Z-axis')
    if len(names) != 3:
        raise ValueError('Exactly three channel names are required.')

    output_directory = os.path.dirname(output_path)
    if not os.path.isdir(output_directory):
        raise FileNotFoundError(
            f"The output directory does not exist:\n{output_directory}"
        )

    dataset = np.stack((x_grid, y_grid, z_grid), axis=1)
    channel_dtype = np.dtype([('Name', 'S256'), ('Info', 'S256')])
    encoded_names = [str(name).encode('utf-8') for name in names]
    channels = np.array(
        [(encoded_name, b'') for encoded_name in encoded_names],
        dtype=channel_dtype,
    )
    log_list = np.array([(encoded_names[2], b'')], dtype=channel_dtype)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f'.{os.path.basename(output_path)}.',
        suffix='.tmp',
        dir=output_directory,
    )
    os.close(descriptor)
    try:
        with h5py.File(temporary_path, 'w') as hdf5_file:
            data_group = hdf5_file.create_group('Data')
            data_group.create_dataset('Data', data=dataset)
            data_group.create_dataset('Channel names', data=channels)
            data_group.attrs['Step dimensions'] = list(z_grid.shape)
            data_group.attrs['Step index'] = [0, 1]
            data_group.attrs['Completed'] = True
            hdf5_file.create_dataset('Log list', data=log_list)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return output_path


def get_npz_output_path(npz_path):
    """Return the sibling .hdf5 path used for a dropped .npz archive."""
    normalized_path = os.path.abspath(os.path.expanduser(npz_path))
    return os.path.splitext(normalized_path)[0] + '.hdf5'


def inspect_npz_fields(npz_path):
    """Validate NPZ fields as safe NumPy arrays and return their descriptions."""
    field_information = []
    invalid_fields = []
    with np.load(npz_path, allow_pickle=False) as archive:
        for field_name in archive.files:
            try:
                value = archive[field_name]
            except ValueError:
                invalid_fields.append(field_name)
                continue
            if not isinstance(value, np.ndarray):
                invalid_fields.append(field_name)
                continue
            field_information.append(
                (field_name, f'shape={value.shape}, dtype={value.dtype}')
            )

    if invalid_fields:
        formatted_names = ', '.join(repr(name) for name in invalid_fields)
        raise ValueError(
            'Every NPZ field must be a NumPy array that can be read without '
            f'pickle. Invalid field(s): {formatted_names}'
        )
    return field_information


class NPZFieldSelectionDialog:
    """Modal dialog for assigning NPZ fields to the viewer's x/y/z roles."""

    def __init__(self, parent, field_information, include_y):
        self.result = None
        self.field_names = [name for name, _detail in field_information]
        self.roles = ('x', 'y', 'z') if include_y else ('x', 'z')
        self.window = ttk.Toplevel(parent)
        self.window.title('Select NPZ Fields')
        self.window.transient(parent)
        self.window.resizable(True, False)
        self.window.protocol('WM_DELETE_WINDOW', self.cancel)

        ttk.Label(
            self.window,
            text=(
                'Assign each data role to a field in the dropped NPZ archive. '
                'Every role must use a different field.'
            ),
            wraplength=620,
            justify='left',
        ).pack(fill='x', padx=16, pady=(16, 8))

        details_frame = ttk.LabelFrame(self.window, text='Available fields')
        details_frame.pack(fill='x', padx=16, pady=8)
        for field_name, detail in field_information:
            ttk.Label(
                details_frame,
                text=f'{field_name}: {detail}',
                anchor='w',
            ).pack(fill='x', padx=10, pady=2)

        selection_frame = ttk.Frame(self.window)
        selection_frame.pack(fill='x', padx=16, pady=8)
        self.variables = {}
        default_fields = self._default_fields()
        for row, role in enumerate(self.roles):
            ttk.Label(selection_frame, text=f'{role.upper()} field:').grid(
                row=row, column=0, sticky='w', padx=(0, 8), pady=4
            )
            variable = tk.StringVar(
                master=self.window,
                value=default_fields[role],
            )
            self.variables[role] = variable
            ttk.Combobox(
                selection_frame,
                textvariable=variable,
                values=self.field_names,
                state='readonly',
                width=42,
            ).grid(row=row, column=1, sticky='ew', pady=4)
        selection_frame.columnconfigure(1, weight=1)

        if not include_y:
            ttk.Label(
                self.window,
                text='Y will be generated as the one-value array [0.0].',
                bootstyle='secondary',
            ).pack(fill='x', padx=16, pady=(0, 8))

        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill='x', padx=16, pady=(8, 16))
        ttk.Button(
            button_frame,
            text='Cancel',
            command=self.cancel,
            bootstyle='secondary',
        ).pack(side='right')
        ttk.Button(
            button_frame,
            text='Create HDF5',
            command=self.confirm,
            bootstyle='success',
        ).pack(side='right', padx=(0, 8))

        self.window.bind('<Escape>', lambda _event: self.cancel())
        self.window.grab_set()

    def _default_fields(self):
        lower_case_names = {
            name.casefold(): name for name in self.field_names
        }
        defaults = {}
        unused_names = list(self.field_names)
        for role in self.roles:
            preferred_name = lower_case_names.get(role)
            if preferred_name in unused_names:
                selected_name = preferred_name
            else:
                selected_name = unused_names[0]
            defaults[role] = selected_name
            unused_names.remove(selected_name)
        return defaults

    def confirm(self):
        selected = {role: variable.get() for role, variable in self.variables.items()}
        if len(set(selected.values())) != len(selected):
            messagebox.showerror(
                'Duplicate NPZ Field',
                'Choose a different NPZ field for each role.',
                parent=self.window,
            )
            return
        if 'y' not in selected:
            selected['y'] = None
        self.result = selected
        self.window.destroy()

    def cancel(self):
        self.window.destroy()

    def show(self):
        self.window.wait_window()
        return self.result


def select_npz_field_mapping(parent, npz_path):
    """Ask the user to assign NPZ fields to x/y/z roles."""
    field_information = inspect_npz_fields(npz_path)
    field_names = [name for name, _detail in field_information]
    if len(field_names) < 2:
        raise ValueError('An NPZ file must contain at least two fields.')

    dialog = NPZFieldSelectionDialog(
        parent,
        field_information,
        include_y=len(field_names) != 2,
    )
    return dialog.show()


def convert_npz_to_hdf5(npz_path, field_mapping, overwrite=False):
    """Convert selected fields from an NPZ archive into a sibling HDF5 file."""
    normalized_path = os.path.abspath(os.path.expanduser(npz_path))
    if os.path.splitext(normalized_path)[1].lower() != '.npz':
        raise ValueError('Only .npz archives can be converted by this function.')
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"The dropped file does not exist:\n{normalized_path}")

    required_roles = ('x', 'z')
    if not all(field_mapping.get(role) for role in required_roles):
        raise ValueError('The NPZ field mapping must define x and z fields.')

    with np.load(normalized_path, allow_pickle=False) as archive:
        missing_fields = [
            field_name
            for field_name in field_mapping.values()
            if field_name is not None and field_name not in archive.files
        ]
        if missing_fields:
            raise ValueError(
                'The selected NPZ field was not found: ' + ', '.join(missing_fields)
            )
        x_data = np.array(archive[field_mapping['x']], copy=True)
        z_data = np.array(archive[field_mapping['z']], copy=True)
        if field_mapping.get('y') is None:
            y_data = np.array([0.0])
            y_name = 'Y (dummy)'
        else:
            y_data = np.array(archive[field_mapping['y']], copy=True)
            y_name = field_mapping['y']

    return write_xyz_hdf5(
        get_npz_output_path(normalized_path),
        x_data,
        y_data,
        z_data,
        channel_names=(field_mapping['x'], y_name, field_mapping['z']),
        overwrite=overwrite,
    )


class CreateHDF5File:
    def __init__(self, root):
        self.root = root
        self.root.title("File Selector")
        self.root.geometry("800x400")  # Set the window size

        self.file_paths = {'x': None, 'y': None, 'z': None, 'single': None}
        self.data_names = {'x': None, 'y': None, 'z': None}
        self.text_boxes = {}

        self.selection_var = tk.StringVar(value="single")
        self.dim_type = tk.StringVar(value="None")  # Global dimension type

        self.dataset = None  # Internal storage for the final 3D array
        self.step_dimensions = None  # To store step dimensions before transposing

        self.create_selection_area()
        self.multiple_frame = self.create_multiple_area()
        self.single_frame = self.create_single_area()

        self.create_save_button()
        self.update_visibility()

    def create_selection_area(self):
        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(frame, text="Select Input Type:").pack(side='left', padx=5)
        ttk.Radiobutton(frame, text="Multiple Files", variable=self.selection_var, value="multiple",
                        command=self.update_visibility).pack(side='left', padx=5)
        ttk.Radiobutton(frame, text="Single 3D File", variable=self.selection_var, value="single",
                        command=self.update_visibility).pack(side='left', padx=5)

    def create_multiple_area(self):
        frame = ttk.Frame(self.root)

        # Global dimensionality selection
        dim_frame = ttk.Frame(frame)
        dim_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(dim_frame, text="Global Dimension Type:").pack(side='left', padx=5)
        ttk.Radiobutton(dim_frame, text="None", variable=self.dim_type, value="None").pack(side='left')
        ttk.Radiobutton(dim_frame, text="1D (array)", variable=self.dim_type, value="1D").pack(side='left')
        ttk.Radiobutton(dim_frame, text="2D (grid)", variable=self.dim_type, value="2D").pack(side='left')

        # File selection areas
        self.create_area(frame, 'x')
        self.create_area(frame, 'y')
        self.create_area(frame, 'z')
        return frame

    def create_single_area(self):
        frame = ttk.Frame(self.root)

        file_frame = ttk.Frame(frame)
        file_frame.pack(padx=10, pady=10, fill='x')

        button = ttk.Button(file_frame, text="Select single .npy file [x, y, z]", command=self.browse_single_file)
        button.pack(side='left')

        text_box = ttk.Entry(file_frame, width=50)
        text_box.pack(side='left', padx=5)
        self.text_boxes['single'] = text_box

        for name in ['x', 'y', 'z']:
            field_frame = ttk.Frame(frame)
            field_frame.pack(padx=10, pady=5, fill='x')

            ttk.Label(field_frame, text=f"Data name for {name}:").pack(side='left', padx=5)
            name_entry = ttk.Entry(field_frame, width=20)
            name_entry.pack(side='left', padx=5)
            name_entry.bind("<KeyRelease>", lambda event, n=name: self.update_data_name(event, n))

        return frame

    def create_area(self, parent, name):
        frame = ttk.Frame(parent)
        frame.pack(padx=10, pady=10, fill='x')

        button = ttk.Button(frame, text=f"Select .npy file for {name}", command=lambda: self.browse_files(name))
        button.pack(side='left')

        text_box = ttk.Entry(frame, width=50)
        text_box.pack(side='left', padx=5)
        self.text_boxes[name] = text_box

        ttk.Label(frame, text="Data name:").pack(side='left', padx=5)
        name_entry = ttk.Entry(frame, width=20)
        name_entry.pack(side='left', padx=5)
        name_entry.bind("<KeyRelease>", lambda event, n=name: self.update_data_name(event, n))

    def create_save_button(self):
        button = ttk.Button(self.root, text="Save HDF5 File", command=self.save_hdf5_file, bootstyle='success')
        button.pack(pady=20)

    def browse_files(self, name):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            if file_path.lower().endswith('.npy'):
                self.file_paths[name] = file_path
                self.text_boxes[name].delete(0, tk.END)
                self.text_boxes[name].insert(0, file_path)
                print(f"File selected for {name}: {file_path}")
            else:
                messagebox.showerror("Invalid file", "Please select a valid .npy file")

    def process_multiple_files(self):
        try:
            z_data = np.load(self.file_paths['z'], allow_pickle=True)
            if self.dim_type.get() == "None":
                x_data = np.arange(z_data.shape[0])
                y_data = np.arange(z_data.shape[1])
                x_data, y_data = np.meshgrid(x_data, y_data)
            elif self.dim_type.get() == "1D":
                x_data = np.load(self.file_paths['x'], allow_pickle=True)
                y_data = np.load(self.file_paths['y'], allow_pickle=True)
                x_data, y_data = np.meshgrid(x_data, y_data)
            elif self.dim_type.get() == "2D":
                x_data = np.load(self.file_paths['x'], allow_pickle=True)
                y_data = np.load(self.file_paths['y'], allow_pickle=True)

            if x_data is None or y_data is None or z_data is None:
                raise ValueError("Missing or invalid data for one or more axes.")

            self.step_dimensions = np.shape(z_data)
            self.dataset = np.swapaxes(np.stack([y_data, x_data, z_data], axis=0), 0, 1)

            print(f"Processed dataset shape: {self.dataset.shape}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not process multiple files: {e}")

    def browse_single_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
        if file_path:
            if file_path.lower().endswith('.npy'):
                try:
                    data = np.load(file_path, allow_pickle=True)
                    if data.ndim == 3:
                        self.file_paths['single'] = file_path
                        self.text_boxes['single'].delete(0, tk.END)
                        self.text_boxes['single'].insert(0, file_path)
                        x_data = data[0].T
                        y_data = data[1].T
                        z_data = data[2].T
                        self.step_dimensions = x_data.shape  # Capture dimensions before transpose
                        self.dataset = np.swapaxes(np.stack([x_data, y_data, z_data], axis=0), 1, 0)  # Store with swapped axes
                        print(f"Single file selected with shape {data.shape}: {file_path}")
                    else:
                        raise ValueError(f"Expected 3D data, but got {data.ndim}D data.")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not load file: {e}")
            else:
                messagebox.showerror("Invalid file", "Please select a valid .npy file")

    def save_hdf5_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".hdf5", filetypes=[("HDF5 files", "*.hdf5")])
        if not file_path:  # Check if the user canceled the dialog
            messagebox.showwarning("Save Canceled", "No file selected. The save operation was canceled.")
            return

        try:

            if self.selection_var.get() == "multiple":
                self.process_multiple_files()
            # Prepare metadata and data for saving
            x_name = self.data_names['x'] or "X-axis"
            y_name = self.data_names['y'] or "Y-axis"
            z_name = self.data_names['z'] or "Z-axis"

            # Create a compound data type for channel names
            dt = np.dtype([('Name', 'S20'), ('Info', 'S20')])
            channel_names = np.array(
                [(x_name.encode('utf-8'), b""),
                 (y_name.encode('utf-8'), b""),
                 (z_name.encode('utf-8'), b"")],
                dtype=dt
            )

            x_data = np.arange(self.step_dimensions[0])  # Generate X-axis indices
            y_data = np.arange(self.step_dimensions[1])  # Generate Y-axis indices

            metadata = {
                "Step dimensions": list(self.step_dimensions),
                "Step index": [0, 1],
                "Completed": True
            }

            log_list = [(z_name.encode('utf-8'), b"")]

            with h5py.File(file_path, "w") as hdf:
                # Create the Data group
                data_group = hdf.create_group("Data")
                data_group.create_dataset("Data", data=self.dataset)
                data_group.create_dataset("Channel names", data=channel_names)
                for key, value in metadata.items():
                    data_group.attrs[key] = value

                # Add the Log list
                hdf.create_dataset("Log list", data=np.array(log_list, dtype='S'))

            messagebox.showinfo("Success", f"HDF5 file saved successfully at {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save HDF5 file: {e}")

    def update_data_name(self, event, name):
        self.data_names[name] = event.widget.get()
        print(f"Data name for {name} set to: {self.data_names[name]}")

    def update_visibility(self):
        if self.selection_var.get() == "multiple":
            self.single_frame.pack_forget()
            self.multiple_frame.pack(padx=10, pady=10, fill='x')
        else:
            self.multiple_frame.pack_forget()
            self.single_frame.pack(padx=10, pady=10, fill='x')

if __name__ == "__main__":
    root = ttk.App(theme='bootstrap-light')
    app = CreateHDF5File(root)
    root.mainloop()
