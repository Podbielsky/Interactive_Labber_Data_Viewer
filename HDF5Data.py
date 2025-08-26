import h5py
import numpy as np
import os


class HDF5Data:
    '''
    A class for handling Labber HDF5 files including reading, writing, and manipulating the files and their data.

    Attributes:

        readpath (str):
        Path to the HDF5 file to read.

        shape_data (tuple):
        Shape of the data array.

        shape_trace (tuple):
        Shape of the trace data.

        file (h5py.File object):
        HDF5 file object.


        file_name (str):
        Name of the HDF5 file.

        arrays (numpy.array):
        Data arrays stored in the HDF5 file.

        array_tags (numpy.array):
        Tags or labels for the data arrays.

        measure_axis (numpy.array):
        Measurement axis data.

        name_axis (list):
        Names of the measurement axes.

        name_data (list):
        Names of the data arrays.

        measure_data (numpy.array):
        Measurement data arrays.

        measure_dim (list):
        Dimensions of the measurement data.

        completed_measurement (bool):
        Flag indicating if the measurement is complete.

        current_h5dir (str):
        Current HDF5 directory.

        savepath (str):
        Path to save the HDF5 file.

        traces (numpy.array):
        Trace data arrays.

        traces_time (numpy.array):
        Time data for traces.

        trace_order (numpy.array):
        Order of traces

        traces_dt (float):
        Time interval between traces.

        trace_reference (numpy.array):
        Reference for trace data.

        saved_traces (bool):
        Flag indicating if traces are saved.

        hist (numpy.array):
        Histogram data.

        bins (numpy.array):
        Bins for the histogram.

        wdir (str):
        Working directory.

    Methods:
        set_path(path_read_output, intention='r'):
            Sets the read or save path for the HDF5 file based on the intention ('r' for read, 'w' for write).

        set_data():
            Opens the HDF5 file for reading and sets it to the file attribute.

        #fehlt copy_objects_recursive?

        copy_to(destination_dir):
            Copies the HDF5 file to a specified destination directory.

        move_and_delete(destination_dir):
            Moves the HDF5 file to a new location and deletes the original file.

        set_filename():
            Sets the filename attribute based on the read path of the HDF5 file.

        set_current_h5dir(current_dir):
            Sets the current HDF5 directory for operations within the file.

        set_data_shape():
            Determines and sets the shape of the primary data within the HDF5 file.

        set_trace_shape():
            Determines and sets the shape of the trace data within the HDF5 file.

        set_array_tags():
            Reads and sets the tags or labels for the data arrays stored in the HDF5 file.

        set_arrays():
            Loads and sets the data arrays from the HDF5 file into memory.

        set_measure_dim():
            Reads and sets the measurement dimensions from the HDF5 file.

        complete_status():
            Checks and sets the completion status of the measurement data within the HDF5 file.

        set_measure_data_and_axis():
            Organizes and sets the measurement data and corresponding axes based on the HDF5 file structure.

        set_traces():
            Loads and sets the trace data from the HDF5 file into memory.

        set_traces_dt():
            Sets the time interval between traces based on the HDF5 file metadata.

        save_traces_in_wdir():
            Saves trace data into the working directory specified by the wdir attribute.

        trace_loading_with_reference():
            Loads trace data along with a reference index or key.

        calc_hist(nbins):
            Calculates and stores the histogram of trace data based on a specified number of bins.

        replace_trace_with_hists(nbins, tracedir):
            Replaces the raw trace data in the HDF5 file with histogram data.

        delete_data_set(dataset_name):
            Deletes a specified dataset from the HDF5 file.

        delete_datasets_in_group(group_name, datasets_to_delete):
            Deletes specific datasets within a given group in the HDF5 file.

        add_group_and_datasets(group_name, dataset_names, datasets):
            Adds a new group to the HDF5 file and populates it with datasets.

        reset():
            Resets the attributes of the HDF5Data object to their default states, essentially reinitializing the object.

    '''

    def __init__(self, wdir=None, readpath=None, file=None, file_name=None, arrays=None, array_tags=None,
                 measure_axis=None, name_axis=None, measure_data=None, name_data=None, measure_dim=None,
                 shape_data=None, current_h5dir=None, savepath=None, traces=None, shape_trace=None, trace_time=None,
                 trace_order=None, traces_dt=None, trace_reference=None, hist=None, bins=None):

        self.readpath = readpath
        self.shape_data = shape_data
        self.shape_trace = shape_trace
        self.file = file
        self.file_name = file_name
        self.arrays = arrays
        self.array_tags = array_tags
        self.measure_axis = measure_axis
        self.name_axis = name_axis
        self.name_data = name_data
        self.measure_data = measure_data
        self.channels = None
        self.measure_dim = measure_dim
        self.completed_measurement = True
        self.current_h5dir = current_h5dir
        self.savepath = savepath
        self.traces = traces
        self.traces_time = trace_time
        self.traces_dt = traces_dt
        self.trace_order = trace_order
        self.trace_reference = trace_reference
        self.saved_traces = False
        self.hist = hist
        self.bins = bins
        self.wdir = wdir

    def set_path(self, path_read_inout, intention='r'):
        if intention == 'r':
            self.readpath = path_read_inout
        elif intention == 'w':
            self.savepath = path_read_inout

    def set_data(self):
        try:
            self.file = h5py.File(self.readpath, "r+")
        except Exception as e:
            print(f"Error setting HDF5 data: {e}")

    #  Hannah Vogel
    def skip_selected_objects_recursive_in_copying_process(self, src, dest, selected_options):
        # to skip datasets selected in checkbutton window in remove_selected_options_window (interactive_hdf5_files)
        for name, item in src.items():
            if name in selected_options:
                print(f"Skipping {name}")
                continue
            try:
                if isinstance(item, h5py.Group):
                    new_group = dest.create_group(name)
                    for key, value in item.attrs.items():
                        new_group.attrs[key] = value
                    self.skip_selected_objects_recursive_in_copying_process(item, new_group, selected_options)
                elif isinstance(item, h5py.Dataset):
                    # Copy datasets
                    new_dataset = dest.create_dataset(
                        name,
                        data=item[()],
                        compression=item.compression,
                        compression_opts=item.compression_opts
                    )
                    # Copy attributes
                    for key, value in item.attrs.items():
                        new_dataset.attrs[key] = value
                else:
                    print(f"Unsupported item type: {name}")
            except Exception as e:
                print(f"Error processing {name}: {e}")

    #

    def _copy_objects_recursive(self, src, dest):
        for name, item in src.items():
            if isinstance(item, h5py.Group):
                # Recursively copy groups
                new_group = dest.create_group(name)
                # Copy attributes
                for key, value in item.attrs.items():
                    new_group.attrs[key] = value
                self._copy_objects_recursive(item, new_group)
            elif isinstance(item, h5py.Dataset):
                # Copy datasets
                new_dataset = dest.create_dataset(
                    name,
                    data=item[()],
                    compression=item.compression,
                    compression_opts=item.compression_opts
                )
                # Copy attributes
                for key, value in item.attrs.items():
                    new_dataset.attrs[key] = value
            else:
                print(f"Unsupported item: {name} ({type(item)})")

    def copy_to(self, destination_dir):
        try:
            # Ensure the file is opened
            if self.file is None:
                self.set_data()

            # Create the destination file path
            dest_path = os.path.join(destination_dir, os.path.basename(self.readpath))

            # Create a new HDF5 file
            with h5py.File(dest_path, 'w') as dest_file:
                # Copy attributes of the root group
                for key, value in self.file.attrs.items():
                    dest_file.attrs[key] = value
                # Recursively copy groups and datasets
                self._copy_objects_recursive(self.file, dest_file)

            print(f"HDF5 data copied successfully to {dest_path}")
            return dest_path

        except Exception as e:
            print(f"Error copying HDF5 data: {e}")
            return None

    def move_and_delete(self, destination_dir):
        try:
            copied_file_path = self.copy_to(destination_dir)

            if copied_file_path:
                # Close the original HDF5 file
                if self.file:
                    self.file.close()

                # Delete the original data
                os.remove(self.readpath)
                print(f"Original HDF5 data deleted successfully.")
        except Exception as e:
            print(f"Error moving and deleting HDF5 data: {e}")

    def set_filename(self):
        self.file_name = os.path.basename(self.readpath)

    def set_current_h5dir(self, current_dir):
        self.current_h5dir = current_dir

    def set_data_shape(self):
        try:
            self.set_data()
            self.shape_data = np.shape(self.file['Data/Data'])
        except Exception as e:
            print(f"Error getting shape of data: {e}")

    def set_trace_shape(self):
        try:
            self.set_data()
            trace_keys = list(self.file['Traces'].keys())
            self.shape_trace = np.shape(self.file[f'Traces/{trace_keys[0]}'])
        except Exception as e:
            print(f"Error getting shape of data: {e}")

    def set_array_tags(self):
        try:
            self.set_data()
            self.array_tags = np.array(self.file['Data/Channel names'])
        except Exception as e:
            print(f"Error setting Channel names: {e}")

    def set_arrays(self):
        try:
            self.set_data()
            self.arrays = np.array(self.file['Data/Data']).swapaxes(0, 1)
        except Exception as e:
            print(f"Error creating data-array: {e}")

    def set_measure_dim(self):
        try:
            if self.file is None:
                self.set_data()
            attrs = self.file['Data'].attrs.items()
            step_dim = 0
            for attr_2 in attrs:
                for attr in attrs:
                    if attr[0] == 'Step dimensions' and attr_2[0] == 'Step index':
                        step_dim = [attr[1][i] for i in attr_2[1]]
            self.measure_dim = step_dim
        except Exception as e:
            print(f"Error getting the measurement dimensions : {e}")

    def complete_status(self):
        try:
            if self.file is None:
                self.set_data()
            attrs = self.file['Data'].attrs.items()
            for attr in attrs:
                if attr[0] == 'Completed':
                    self.completed_measurement = attr[1]
        except Exception as e:
            print(f"Error getting data : {e}")

    def set_measure_data_and_axis(self):
        try:
            # Set up arrays, tags, and dimensions
            if self.arrays is None:
                self.set_arrays()
            if self.array_tags is None:
                self.set_array_tags()
            if self.measure_dim is None:
                self.set_measure_dim()
            if self.channels is None:
                self.channels = self.file['Channels']
            self.complete_status()
            # Calculate the expected shape of the arrays
            should_array_shape = (int(self.measure_dim[0]), int(np.prod(np.array(self.measure_dim)[1:])))

            log_list = self.file['Log list'][:]
            array_tags_names = [tag[0] for tag in self.array_tags]
            log_list_names = [tag[0] for tag in log_list]
            measurement_data = []
            name_data = []
            measurement_axis = []
            name_axis = []

            # Create a dictionary to store channel parameters for all channels
            channel_params = {}
            for channel in self.channels:
                channel_name = channel['name']
                channel_params[channel_name] = {
                    'gain': channel['gain'],
                    'offset': channel['offset'],
                    'amp': channel['amp']
                }

            for name in array_tags_names:
                index = array_tags_names.index(name)
                is_shape = self.arrays[index].shape
                target_array = self.arrays[index]

                # Check if the measurement is complete
                if self.complete_status and is_shape != should_array_shape:
                    # Pad the array if the shape is not as expected
                    padded_array = np.full(should_array_shape, np.nan)
                    slices = tuple(slice(0, min(dim, size)) for size, dim in zip(is_shape, should_array_shape))
                    padded_array[slices] = target_array
                    target_array = padded_array

                # Process the array with the formula if channel parameters exist
                processed_array = target_array
                if name in channel_params:
                    params = channel_params[name]
                    gain = params['gain']
                    offset = params['offset']
                    amp = params['amp']

                    # Avoid division by zero
                    if amp != 0 and gain != 0:
                        processed_array = (target_array / amp - offset) / gain
                    else:
                        # If gain or amp is zero, just use the original array
                        print(f"Warning: gain or amp is zero for channel {name}. Using original data.")

                # Append data to the corresponding lists
                if name in log_list_names:
                    name_data.append(name)
                    measurement_data.append(processed_array)
                else:
                    name_axis.append(name)
                    measurement_axis.append(processed_array)

            # Set the class attributes
            self.measure_data = measurement_data
            self.name_data = name_data
            self.measure_axis = measurement_axis
            self.name_axis = name_axis

            # Clear temporarily used variables and attributes
            log_list = None
            array_tags_names = None
            log_list_names = None
            measurement_data = None
            name_data = None
            measurement_axis = None
            name_axis = None
            self.arrays = None
            self.array_tags = None
        except Exception as e:
            print(f"Error dividing measurement into data and axis : {e}")

    def set_traces(self):
        try:
            self.set_data()
            self.set_data_shape()
            trace_keys = list(self.file['Traces'].keys())
            traces_i = (
                np.array(self.file[f'Traces/{trace_keys[0]}'], dtype=np.float32).swapaxes(0, 2).flatten()).reshape(
                int(self.shape_data[0] * self.shape_data[-1]),
                np.array(self.file[f'Traces/{trace_keys[1]}'])[0])
            self.traces = traces_i.reshape(self.shape_data[-1], self.shape_data[0],
                                           np.array(self.file[f'Traces/{trace_keys[1]}'])[0])
            traces_i = None
        except Exception as e:
            print(f"Error creating traces as array: {e}")

    def set_traces_dt(self):
        if self.file is None:
            self.set_data()
        self.traces_dt = self.file['Traces']['Alazar Slytherin - Ch1 - Data_t0dt'][0][1]

    def save_traces_in_wdir(self):
        if not self.saved_traces:
            if self.file is None:
                self.set_data()
            if self.measure_dim is None:
                self.set_measure_dim()
            trace_order_matrix = []
            should_array_shape = (int(self.measure_dim[0]), int(np.prod(np.array(self.measure_dim)[1:])))
            trace_keys = list(self.file['Traces'].keys())
            traces_i = (np.array(self.file[f'Traces/{trace_keys[0]}'], dtype=np.float32).swapaxes(0, 2).reshape(
                int(np.prod(should_array_shape)), np.array(self.file[f'Traces/{trace_keys[1]}'])[0]))
            save_path = self.wdir + '/traces'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i, trace_single in enumerate(traces_i):
                trace_order_matrix.append(i)
                np.save(save_path + '/' + 'trace' + str(i) + '.npy', trace_single, allow_pickle=True)
            self.trace_order = np.reshape(trace_order_matrix, should_array_shape)
            print(self.trace_order)
            traces_i = None
            self.saved_traces = True

    def trace_loading_with_referance(self):
        if self.file is None:
            self.set_data()
        if self.measure_dim is None:
            self.set_measure_dim()
        should_array_shape = (int(self.measure_dim[0]), int(np.prod(np.array(self.measure_dim)[1:])))
        trace_keys = list(self.file['Traces'].keys())
        self.trace_reference = self.file[f'Traces/{trace_keys[0]}']
        self.trace_order = np.reshape([i for i in range(self.trace_reference.shape[-1])], should_array_shape)

    def calc_hist(self, nbins):
        self.nbins = nbins
        counts = [np.histogram(subarray, bins=self.nbins, density=True)[0] for row in self.traces for subarray in row]
        bins = [np.histogram(subarray, bins=self.nbins, density=True)[1] for row in self.traces for subarray in row]
        print(np.shape(counts))
        print(np.shape(bins))
        counts = np.array(counts).reshape(np.shape(self.traces)[0], np.shape(self.traces)[1], self.nbins)
        bins = np.array(bins).reshape(np.shape(self.traces)[0], np.shape(self.traces)[1], self.nbins + 1)
        self.bins = bins
        self.hist = counts

    def replace_trace_with_hists(self, nbins, tracedir):
        if tracedir == 'Traces':
            self.set_data()
            self.set_data_shape()
            dataset_names = [name for name in self.file[tracedir]]
            for name in dataset_names:
                if name == 'Time stamp':
                    time_stamp_data = self.file['/'.join((tracedir, name))]
            self.set_traces()
            self.set_trace_shape()
            self.calc_hist(nbins)
            min_max_bins = [[0, (np.max(self.bins) - np.min(self.bins)) / nbins]]
            hist_data_shape = (nbins, 1, int(self.shape_data[0] * self.shape_data[2]))
            datasets = [np.reshape(self.hist.flatten(), hist_data_shape), np.int32(nbins), min_max_bins,
                        time_stamp_data]
            attrs_trace_data = self.file['/'.join((tracedir, dataset_names[0]))].attrs
            self.copy_to('/Users/hubert.D/Documents/Triton3_cd_data/copy_folder')
            self.delete_data_set('Traces')
            self.add_group_and_datasets('Traces', dataset_names, datasets)
            for name, value in attrs_trace_data.items():
                self.file['/'.join((tracedir, dataset_names[0]))].attrs[name] = value
            self.file.close()
            self.set_data()
        else:
            return 'Wrong directory inside .hdf5 file! Select the Group where the traces are stored and try again ...'

    def delete_data_set(self, dataset_name):
        temp_file_path = 'temp_file.hdf5'
        with h5py.File(self.readpath, 'r') as old_file, h5py.File(temp_file_path, 'w') as new_file:
            for key, value in old_file.attrs.items():
                new_file.attrs[key] = value
            for name, dataset in old_file.items():
                if name != dataset_name:
                    old_file.copy(name, new_file)

        os.replace(temp_file_path, self.readpath)
        print(f"Dataset '{dataset_name}' deleted, and file size reduced.")

    def delete_datasets_in_group(self, group_name, datasets_to_delete):
        temp_file_path = 'temp_file.hdf5'
        with h5py.File(self.readpath, 'a') as old_file, h5py.File(temp_file_path, 'w') as new_file:
            # Iterate over items in the old file
            for key, value in old_file.attrs.items():
                new_file.attrs[key] = value
            for name, item in old_file.items():
                if isinstance(item, h5py.Group) and name == group_name:
                    # If the item is a group and matches the specified group_name
                    # Create a new group in the new file
                    new_group = new_file.create_group(name)

                    # Iterate over datasets within the group and exclude the ones you want to delete
                    for dataset_name, dataset in item.items():
                        if dataset_name not in datasets_to_delete:
                            # Copy non-deleted datasets to the new group
                            item.copy(dataset_name, new_group.create_dataset(dataset_name, data=dataset))
                else:
                    # Copy non-group items as is
                    old_file.copy(name, new_file)

        # Replace the original file with the new file
        os.replace(temp_file_path, self.readpath)
        print(f"Datasets in '{group_name}' deleted, and file size reduced.")

    def add_group_and_datasets(self, group_name, dataset_names, datasets):
        try:
            with h5py.File(self.readpath, 'r+') as file:
                group = file.create_group(group_name)

                for i, name in enumerate(dataset_names):
                    data = datasets[i]
                    group.create_dataset(name, data=data)

                print(f"Group '{group_name}' and datasets added successfully.")
                self.set_data()
        except Exception as e:
            print(f"Error: {e}")

    def reset(self):
        self.readpath = None
        self.shape_data = None
        self.shape_trace = None
        self.file = None
        self.file_name = None
        self.arrays = None
        self.array_tags = None
        self.measure_axis = None
        self.name_axis = None
        self.name_data = None
        self.measure_data = None
        self.measure_dim = None
        self.completed_measurement = True
        self.current_h5dir = None
        self.savepath = None
        self.traces = None
        self.traces_time = None
        self.trace_order = None
        self.saved_traces = False
        self.trace_reference = None
        self.hist = None
        self.bins = None
        self.wdir = None
