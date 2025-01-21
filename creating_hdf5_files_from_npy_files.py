import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import h5py

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
        frame = tk.Frame(self.root)
        frame.pack(padx=10, pady=10, fill='x')

        tk.Label(frame, text="Select Input Type:").pack(side='left', padx=5)
        tk.Radiobutton(frame, text="Multiple Files", variable=self.selection_var, value="multiple",
                       command=self.update_visibility).pack(side='left', padx=5)
        tk.Radiobutton(frame, text="Single 3D File", variable=self.selection_var, value="single",
                       command=self.update_visibility).pack(side='left', padx=5)

    def create_multiple_area(self):
        frame = tk.Frame(self.root)

        # Global dimensionality selection
        dim_frame = tk.Frame(frame)
        dim_frame.pack(padx=10, pady=5, fill='x')
        tk.Label(dim_frame, text="Global Dimension Type:").pack(side='left', padx=5)
        tk.Radiobutton(dim_frame, text="None", variable=self.dim_type, value="None").pack(side='left')
        tk.Radiobutton(dim_frame, text="1D (array)", variable=self.dim_type, value="1D").pack(side='left')
        tk.Radiobutton(dim_frame, text="2D (grid)", variable=self.dim_type, value="2D").pack(side='left')

        # File selection areas
        self.create_area(frame, 'x')
        self.create_area(frame, 'y')
        self.create_area(frame, 'z')
        return frame

    def create_single_area(self):
        frame = tk.Frame(self.root)

        file_frame = tk.Frame(frame)
        file_frame.pack(padx=10, pady=10, fill='x')

        button = tk.Button(file_frame, text="Select single .npy file [x, y, z]", command=self.browse_single_file)
        button.pack(side='left')

        text_box = tk.Entry(file_frame, width=50)
        text_box.pack(side='left', padx=5)
        self.text_boxes['single'] = text_box

        for name in ['x', 'y', 'z']:
            field_frame = tk.Frame(frame)
            field_frame.pack(padx=10, pady=5, fill='x')

            tk.Label(field_frame, text=f"Data name for {name}:").pack(side='left', padx=5)
            name_entry = tk.Entry(field_frame, width=20)
            name_entry.pack(side='left', padx=5)
            name_entry.bind("<KeyRelease>", lambda event, n=name: self.update_data_name(event, n))

        return frame

    def create_area(self, parent, name):
        frame = tk.Frame(parent)
        frame.pack(padx=10, pady=10, fill='x')

        button = tk.Button(frame, text=f"Select .npy file for {name}", command=lambda: self.browse_files(name))
        button.pack(side='left')

        text_box = tk.Entry(frame, width=50)
        text_box.pack(side='left', padx=5)
        self.text_boxes[name] = text_box

        tk.Label(frame, text="Data name:").pack(side='left', padx=5)
        name_entry = tk.Entry(frame, width=20)
        name_entry.pack(side='left', padx=5)
        name_entry.bind("<KeyRelease>", lambda event, n=name: self.update_data_name(event, n))

    def create_save_button(self):
        button = tk.Button(self.root, text="Save HDF5 File", command=self.save_hdf5_file)
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
    root = tk.Tk()
    app = CreateHDF5File(root)
    root.mainloop()

