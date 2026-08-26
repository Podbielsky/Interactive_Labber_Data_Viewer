import h5py
import math
import numpy as np
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class HDF5MapPreview:
    """A lightweight, read-only map representation for database previews."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    x_label: str
    y_label: str
    z_label: str
    is_linecut: bool = False


@dataclass(frozen=True)
class HDF5TraceChannel:
    """One selectable trace-amplitude dataset inside an HDF5 measurement."""

    identifier: str
    label: str
    dataset_path: str
    sample_count: int
    trace_count: int


def _decode_hdf5_text(value):
    """Decode the first textual field used by Labber metadata records."""
    if isinstance(value, np.void) and value.dtype.names:
        field_name = 'Name' if 'Name' in value.dtype.names else value.dtype.names[0]
        value = value[field_name]
    elif isinstance(value, (tuple, list, np.ndarray)) and np.ndim(value) > 0:
        value = value[0]

    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace').rstrip('\x00')
    return str(value)


def _read_hdf5_names(dataset):
    """Return channel names from a Labber metadata dataset."""
    values = np.asarray(dataset[()])
    if values.ndim == 0:
        values = values.reshape(1)
    return [_decode_hdf5_text(value) for value in values]


def _trace_data_candidates(group):
    """Return trace datasets using the samples × 1 × traces layout."""
    return [
        dataset
        for dataset in group.values()
        if (
            isinstance(dataset, h5py.Dataset)
            and dataset.ndim == 3
            and dataset.shape[1] == 1
            and dataset.shape[0] > 0
            and dataset.shape[2] > 0
        )
    ]


def find_trace_groups(hdf5_file, expected_trace_count=None):
    """Find groups containing trace amplitudes, including below /Traces."""
    trace_containers = []
    for name, item in hdf5_file.items():
        if (
            isinstance(item, h5py.Group)
            and name.casefold() in {'trace', 'traces'}
        ):
            trace_containers.append(item)

    def compatible(group):
        candidates = _trace_data_candidates(group)
        if expected_trace_count is not None:
            candidates = [
                dataset
                for dataset in candidates
                if dataset.shape[2] == int(expected_trace_count)
            ]
        return bool(candidates)

    def compatible_groups_below(container):
        groups = [container] if compatible(container) else []

        def visitor(_name, item):
            if isinstance(item, h5py.Group) and compatible(item):
                groups.append(item)

        container.visititems(visitor)
        return groups

    grouped_channels = []
    for container in trace_containers:
        grouped_channels.extend(compatible_groups_below(container))
    if grouped_channels:
        return grouped_channels

    # Older/generated files may use an arbitrarily named trace group rather
    # than a /Traces container. Preserve that single-group compatibility.
    discovered_groups = []

    def fallback_visitor(_name, item):
        if isinstance(item, h5py.Group) and compatible(item):
            discovered_groups.append(item)

    hdf5_file.visititems(fallback_visitor)
    return discovered_groups


def find_trace_group(hdf5_file, expected_trace_count=None):
    """Return the first compatible trace group in a measurement."""
    groups = find_trace_groups(
        hdf5_file,
        expected_trace_count=expected_trace_count,
    )
    return groups[0] if groups else None


def find_trace_data_dataset(trace_group, expected_trace_count=None):
    """Select the trace-value dataset by shape rather than HDF5 key order."""
    candidates = _trace_data_candidates(trace_group)
    if not candidates:
        raise ValueError(
            'No trace dataset with shape (trace_length, 1, trace_count) '
            'was found in the selected trace group.'
        )
    if expected_trace_count is not None:
        matching_candidates = [
            dataset
            for dataset in candidates
            if dataset.shape[2] == int(expected_trace_count)
        ]
        if matching_candidates:
            candidates = matching_candidates
    return max(
        candidates,
        key=lambda dataset: (
            dataset.name.rsplit('/', 1)[-1].casefold() == 'data',
            int(np.prod(dataset.shape)),
        ),
    )


def _numeric_values(dataset):
    """Return a numeric dataset as an array, or None for text/object data."""
    if dataset.dtype.kind not in 'biufc':
        return None
    return np.asarray(dataset[()])


def _trace_name_match_score(dataset_name, channel_name):
    """Score how likely a metadata dataset belongs to an amplitude channel."""
    dataset_name = dataset_name.rsplit('/', 1)[-1].casefold()
    channel_name = str(channel_name).rsplit('/', 1)[-1].casefold()
    score = 0
    if channel_name and dataset_name.startswith(channel_name):
        score += 1000
    channel_prefix = channel_name
    for suffix in ('_data', ' - data', ' data'):
        if channel_prefix.endswith(suffix):
            channel_prefix = channel_prefix[:-len(suffix)]
            break
    if channel_prefix and dataset_name.startswith(channel_prefix):
        score += 500
    score += len(os.path.commonprefix([dataset_name, channel_name]))
    return score


def _trace_channel_label(trace_group, trace_data_dataset, channel_count):
    """Return the amplitude name represented by one trace dataset."""
    for owner in (trace_group, trace_data_dataset):
        for attribute_name in (
            'label',
            'channel_name',
            'name',
            'long_name',
        ):
            if attribute_name in owner.attrs:
                label = _decode_hdf5_text(owner.attrs[attribute_name]).strip()
                if label:
                    return label

    dataset_name = trace_data_dataset.name.rsplit('/', 1)[-1].strip()
    if channel_count > 1:
        return dataset_name

    group_name = trace_group.name.rsplit('/', 1)[-1].strip()
    if group_name.casefold() not in {'trace', 'traces'}:
        return group_name

    # Flat legacy files have no amplitude subgroup. In those files the
    # instrument/channel name is commonly encoded by the *_t0dt dataset.
    for dataset in trace_group.values():
        if (
            isinstance(dataset, h5py.Dataset)
            and dataset.name.casefold().endswith('_t0dt')
        ):
            basename = dataset.name.rsplit('/', 1)[-1]
            return basename[:-len('_t0dt')]

    dataset_label = _trace_axis_label(trace_data_dataset, 'Trace amplitude')
    return (
        'Trace amplitude'
        if dataset_label.casefold() == 'data'
        else dataset_label
    )


def _trace_sample_count(trace_group, trace_data_dataset):
    """Read the valid sample count from metadata local to one trace group."""
    maximum_count = int(trace_data_dataset.shape[0])
    basename = trace_data_dataset.name.rsplit('/', 1)[-1]
    candidates = [
        dataset
        for dataset in trace_group.values()
        if (
            isinstance(dataset, h5py.Dataset)
            and dataset.dtype.kind in 'biuf'
            and dataset.name.rsplit('/', 1)[-1].casefold().endswith('_n')
            and dataset.size >= 1
        )
    ]
    candidates.sort(
        key=lambda dataset: _trace_name_match_score(dataset.name, basename),
        reverse=True,
    )
    if candidates:
        counts = np.asarray(candidates[0][()]).reshape(-1)
        try:
            count = int(counts[0])
        except (TypeError, ValueError, OverflowError):
            count = maximum_count
        if 0 < count <= maximum_count:
            return count
    return maximum_count


def list_trace_channels(trace_group, expected_trace_count=None):
    """Describe every trace-amplitude dataset local to one trace group."""
    datasets = _trace_data_candidates(trace_group)
    if expected_trace_count is not None:
        datasets = [
            dataset
            for dataset in datasets
            if dataset.shape[2] == int(expected_trace_count)
        ]
    channel_count = len(datasets)
    return [
        HDF5TraceChannel(
            identifier=dataset.name,
            label=_trace_channel_label(
                trace_group,
                dataset,
                channel_count,
            ),
            dataset_path=dataset.name,
            sample_count=_trace_sample_count(trace_group, dataset),
            trace_count=int(dataset.shape[2]),
        )
        for dataset in datasets
    ]


def _trace_axis_label(dataset, fallback):
    """Return a readable axis label from optional HDF5 metadata."""
    for attribute_name in ('label', 'axis_name', 'name', 'long_name'):
        if attribute_name in dataset.attrs:
            label = _decode_hdf5_text(dataset.attrs[attribute_name]).strip()
            if label:
                return label
    dataset_name = dataset.name.rsplit('/', 1)[-1].strip()
    return dataset_name or fallback


def _trace_data_axis_label(trace_data_dataset, fallback):
    """Read Labber's X-axis name/unit attributes from a trace dataset."""
    axis_name = ''
    axis_unit = ''
    for attribute_name in ('x, name', 'x_name', 'x name', 'x-axis name'):
        if attribute_name in trace_data_dataset.attrs:
            axis_name = _decode_hdf5_text(
                trace_data_dataset.attrs[attribute_name]
            ).strip()
            if axis_name:
                break
    for attribute_name in ('x, unit', 'x_unit', 'x unit', 'x-axis unit'):
        if attribute_name in trace_data_dataset.attrs:
            axis_unit = _decode_hdf5_text(
                trace_data_dataset.attrs[attribute_name]
            ).strip()
            if axis_unit:
                break
    if axis_name and axis_unit:
        return f'{axis_name} ({axis_unit})'
    return axis_name or fallback


def read_trace_axis(
    trace_group,
    trace_data_dataset,
    sample_count=None,
    channel_name='',
):
    """Read an arbitrary explicit trace axis or reconstruct a legacy one."""
    trace_length = int(sample_count or trace_data_dataset.shape[0])
    explicit_axis_candidates = []
    legacy_axis_candidates = []
    for dataset in trace_group.values():
        if not isinstance(dataset, h5py.Dataset) or dataset == trace_data_dataset:
            continue
        dataset_name = dataset.name.casefold()
        values = _numeric_values(dataset)
        if values is None:
            continue
        if 't0dt' not in dataset_name and not dataset_name.endswith('_n'):
            axis_values = None
            if dataset.ndim == 1 and dataset.size == trace_length:
                axis_values = values
            elif dataset.ndim == 2:
                if dataset.size == trace_length and 1 in dataset.shape:
                    axis_values = values.reshape(-1)
            if axis_values is not None:
                explicit_axis_candidates.append((dataset, axis_values))
        if 't0dt' in dataset_name or (
            dataset.size == 2 and not dataset_name.endswith('_n')
        ):
            metadata_values = None
            if dataset.size == 2:
                metadata_values = values.reshape(-1)[:2]
            if metadata_values is not None:
                legacy_axis_candidates.append((dataset, metadata_values))

    def select_channel_metadata(candidates):
        scores = [
            max(
                _trace_name_match_score(candidate[0].name, channel_name),
                _trace_name_match_score(
                    candidate[0].name,
                    trace_data_dataset.name,
                ),
            )
            for candidate in candidates
        ]
        return candidates[int(np.argmax(scores))]

    if explicit_axis_candidates:
        axis_dataset, axis_values = select_channel_metadata(
            explicit_axis_candidates
        )
        return (
            np.asarray(axis_values).reshape(-1).astype(np.float64, copy=False),
            _trace_axis_label(axis_dataset, 'Trace X'),
        )

    if legacy_axis_candidates:
        metadata_dataset, metadata_values = select_channel_metadata(
            legacy_axis_candidates
        )
        start, spacing = np.asarray(metadata_values).reshape(-1)[:2]
        axis_values = float(start) + float(spacing) * np.arange(trace_length)
        label = _trace_axis_label(metadata_dataset, 'Trace X')
        if 't0dt' in label.casefold():
            label = _trace_data_axis_label(trace_data_dataset, 'Trace X')
        return axis_values, label

    return (
        np.arange(trace_length, dtype=np.float64),
        _trace_data_axis_label(trace_data_dataset, 'Trace sample'),
    )


def inspect_viewer_hdf5(path):
    """Validate a Labber/viewer-compatible HDF5 file and return its metadata."""
    normalized_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(normalized_path):
        raise ValueError(f'The HDF5 file does not exist:\n{normalized_path}')
    if not h5py.is_hdf5(normalized_path):
        raise ValueError(f'The file is not a valid HDF5 container:\n{normalized_path}')

    try:
        with h5py.File(normalized_path, 'r') as hdf5_file:
            required_paths = ('Data', 'Data/Data', 'Data/Channel names', 'Log list')
            missing_paths = [name for name in required_paths if name not in hdf5_file]
            if missing_paths:
                raise ValueError(
                    'The file is not in the supported Labber/viewer format. '
                    f"Missing: {', '.join(missing_paths)}"
                )

            data_group = hdf5_file['Data']
            data_dataset = hdf5_file['Data/Data']
            if data_dataset.ndim != 3:
                raise ValueError(
                    'Data/Data must be a three-dimensional dataset; '
                    f'found shape {data_dataset.shape}.'
                )

            channel_names = _read_hdf5_names(hdf5_file['Data/Channel names'])
            if len(channel_names) != data_dataset.shape[1]:
                raise ValueError(
                    'The number of channel names does not match the channel '
                    'dimension of Data/Data.'
                )

            log_names = _read_hdf5_names(hdf5_file['Log list'])
            logged_channels = [name for name in log_names if name in channel_names]
            if not logged_channels:
                raise ValueError('Log list does not reference a channel in Data/Data.')

            if 'Step dimensions' not in data_group.attrs:
                raise ValueError('Data is missing the Step dimensions attribute.')
            step_dimensions = tuple(
                int(value) for value in np.ravel(data_group.attrs['Step dimensions'])
            )
            if not step_dimensions or any(value <= 0 for value in step_dimensions):
                raise ValueError('Step dimensions must contain positive integers.')

            stored_points = int(data_dataset.shape[0] * data_dataset.shape[2])
            expected_points = math.prod(step_dimensions)
            if stored_points > expected_points:
                raise ValueError(
                    'Data/Data contains more points than declared by Step dimensions.'
                )

            axis_names = [name for name in channel_names if name not in log_names]
            if not axis_names:
                raise ValueError('The file does not contain a sweep-axis channel.')

            return {
                'path': normalized_path,
                'channel_names': channel_names,
                'axis_names': axis_names,
                'log_names': logged_channels,
                'step_dimensions': step_dimensions,
                'data_shape': tuple(data_dataset.shape),
                'has_traces': find_trace_group(
                    hdf5_file,
                    expected_trace_count=expected_points,
                ) is not None,
            }
    except OSError as error:
        raise ValueError(f'Could not read the HDF5 file: {error}') from error


def load_hdf5_map_preview(path, maximum_points_per_axis=None):
    """Read only the final two-dimensional map needed for a lightweight preview."""
    metadata = inspect_viewer_hdf5(path)
    if maximum_points_per_axis is not None:
        maximum_points_per_axis = int(maximum_points_per_axis)
        if maximum_points_per_axis <= 0:
            raise ValueError('The preview size limit must be positive.')

    with h5py.File(metadata['path'], 'r') as hdf5_file:
        channel_names = metadata['channel_names']
        step_dimensions = metadata['step_dimensions']
        data_dataset = hdf5_file['Data/Data']

        map_column_count = step_dimensions[0]
        map_row_count = step_dimensions[1] if len(step_dimensions) > 1 else 1
        column_step = 1
        row_step = 1
        if maximum_points_per_axis is not None:
            column_step = max(
                1, math.ceil(map_column_count / maximum_points_per_axis)
            )
            row_step = max(
                1, math.ceil(map_row_count / maximum_points_per_axis)
            )

        # Data/Data stores a channel as (map columns, flattened remaining
        # dimensions). The flattened order used by the full plotter puts the
        # displayed row dimension last. Select the last available map directly
        # in that storage layout instead of constructing an N-dimensional array.
        expected_flat_count = (
            math.prod(step_dimensions[1:])
            if len(step_dimensions) > 1
            else 1
        )
        available_flat_count = min(data_dataset.shape[2], expected_flat_count)
        if available_flat_count <= 0 or data_dataset.shape[0] <= 0:
            raise ValueError('The selected preview map does not contain data.')
        final_map_index = max(0, (available_flat_count - 1) // map_row_count)
        flat_start = final_map_index * map_row_count
        flat_stop = min(
            flat_start + map_row_count,
            available_flat_count,
        )

        sampled_column_count = len(
            range(0, map_column_count, column_step)
        )
        sampled_row_count = len(range(0, map_row_count, row_step))

        def read_preview_channel(channel_index):
            """Read one strided map channel and pad an incomplete measurement."""
            stored_column_stop = min(map_column_count, data_dataset.shape[0])
            channel_values = np.asarray(
                data_dataset[
                    0:stored_column_stop:column_step,
                    channel_index,
                    flat_start:flat_stop:row_step,
                ],
                dtype=float,
            ).T
            preview_values = np.full(
                (sampled_row_count, sampled_column_count),
                np.nan,
                dtype=float,
            )
            preview_values[
                :channel_values.shape[0], :channel_values.shape[1]
            ] = channel_values
            return preview_values

        def finite_median(values, fallback):
            """Return a finite median for coordinate boundaries."""
            values = np.asarray(values, dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return float(fallback)
            return float(np.median(finite_values))

        def read_coordinate_grid(channel_index, axis):
            """Read only coordinate edges and construct a raster display grid."""
            stored_column_stop = min(map_column_count, data_dataset.shape[0])
            if axis == 'x':
                start_values = data_dataset[
                    0,
                    channel_index,
                    flat_start:flat_stop:row_step,
                ]
                stop_values = data_dataset[
                    stored_column_stop - 1,
                    channel_index,
                    flat_start:flat_stop:row_step,
                ]
                start = finite_median(start_values, 0)
                stop = finite_median(stop_values, sampled_column_count - 1)
                coordinate_line = np.linspace(
                    start, stop, sampled_column_count, dtype=float
                )
                return np.tile(coordinate_line, (sampled_row_count, 1))

            start_values = data_dataset[
                0:stored_column_stop:column_step,
                channel_index,
                flat_start,
            ]
            stop_values = data_dataset[
                0:stored_column_stop:column_step,
                channel_index,
                flat_stop - 1,
            ]
            start = finite_median(start_values, 0)
            stop = finite_median(stop_values, sampled_row_count - 1)
            coordinate_line = np.linspace(
                start, stop, sampled_row_count, dtype=float
            )
            return np.tile(
                coordinate_line.reshape(-1, 1),
                (1, sampled_column_count),
            )

        z_label = metadata['log_names'][-1]
        z_index = channel_names.index(z_label)
        z_grid = read_preview_channel(z_index)

        axis_names = metadata['axis_names']
        is_linecut = len(axis_names) == 1
        x_label = axis_names[0]
        x_index = channel_names.index(x_label)
        if maximum_points_per_axis is None or is_linecut:
            x_grid = read_preview_channel(x_index)
        else:
            x_grid = read_coordinate_grid(x_index, 'x')

        if is_linecut:
            line_x = np.ravel(x_grid)
            line_z = np.ravel(z_grid)
            valid_points = np.isfinite(line_x)
            line_x = line_x[valid_points]
            line_z = line_z[valid_points]
            if line_x.size == 0:
                raise ValueError('The one-axis measurement does not contain data.')
            x_grid = np.tile(line_x, (3, 1))
            y_grid = np.tile(
                np.arange(3, dtype=float).reshape(3, 1),
                (1, x_grid.shape[1]),
            )
            z_grid = np.tile(line_z, (3, 1))
            y_label = 'y-dummy'
        else:
            y_label = axis_names[1]
            y_index = channel_names.index(y_label)
            if maximum_points_per_axis is None:
                y_grid = read_preview_channel(y_index)
            else:
                y_grid = read_coordinate_grid(y_index, 'y')

        if x_grid.shape != y_grid.shape or x_grid.shape != z_grid.shape:
            raise ValueError('The preview axes and data do not have matching shapes.')
        if z_grid.size == 0:
            raise ValueError('The selected preview map is empty.')

        return HDF5MapPreview(
            x=np.array(x_grid, copy=True),
            y=np.array(y_grid, copy=True),
            z=np.array(z_grid, copy=True),
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            is_linecut=is_linecut,
        )


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
            X-axis coordinate for traces (time, frequency, voltage, etc.).

        trace_axis_name (str):
            Display label for the trace X-axis.

        trace_channel_name (str):
            Display label for the selected trace-amplitude channel.

        trace_sample_count (int):
            Number of valid samples for the selected amplitude channel.

        trace_order (numpy.array):
        Order of traces

        traces_dt (float):
            Representative spacing between trace X-axis samples.

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
            Loads the trace X-axis and its representative sample spacing from
            the HDF5 file metadata.

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
                 trace_order=None, traces_dt=None, trace_reference=None, hist=None, bins=None,
                 trace_axis_name=None, trace_group_path=None,
                 trace_dataset_path=None, trace_channel_name=None,
                 trace_sample_count=None):

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
        self.trace_axis_name = trace_axis_name
        self.trace_group_path = trace_group_path
        self.trace_dataset_path = trace_dataset_path
        self.trace_channel_name = trace_channel_name
        self.trace_sample_count = trace_sample_count
        self.traces_dt = traces_dt
        self.trace_order = trace_order
        self.trace_reference = trace_reference
        self.saved_traces = False
        self.hist = hist
        self.bins = bins
        self.wdir = wdir

    def set_path(self, path_read_inout, intention='r'):
        if intention == 'r':
            path_changed = self.readpath != path_read_inout
            self.readpath = path_read_inout
            if path_changed:
                self.trace_dataset_path = None
                self.trace_group_path = None
                self.trace_channel_name = None
                self.trace_sample_count = None
                self.trace_reference = None
                self.traces = None
                self.traces_time = None
                self.trace_axis_name = None
                self.traces_dt = None
        elif intention == 'w':
            self.savepath = path_read_inout

    def set_data(self):
        try:
            self.file = h5py.File(self.readpath, "r+")
        except Exception as e:
            print(f"Error setting HDF5 data: {e}")

    def get_trace_group(self):
        """Return the compatible trace group in the currently open file."""
        if self.file is None:
            self.set_data()
        if self.trace_dataset_path and self.trace_dataset_path in self.file:
            trace_group = self.file[self.trace_dataset_path].parent
        else:
            trace_group = find_trace_group(
                self.file,
                expected_trace_count=self._expected_trace_count(),
            )
        if trace_group is None:
            raise ValueError('The HDF5 file does not contain compatible traces.')
        self.trace_group_path = trace_group.name
        return trace_group

    def _expected_trace_count(self):
        """Return the number of scan points when the measurement declares it."""
        if self.measure_dim is not None:
            return int(np.prod(self.measure_dim))
        if self.file is not None and 'Data' in self.file:
            dimensions = self.file['Data'].attrs.get('Step dimensions')
            if dimensions is not None:
                return int(np.prod(np.asarray(dimensions, dtype=int)))
        return None

    def get_trace_channels(self):
        """Return every selectable trace-amplitude channel in the file."""
        if self.file is None:
            self.set_data()
        expected_trace_count = self._expected_trace_count()
        channels = []
        for trace_group in find_trace_groups(
            self.file,
            expected_trace_count=expected_trace_count,
        ):
            channels.extend(list_trace_channels(
                trace_group,
                expected_trace_count=expected_trace_count,
            ))

        label_counts = {}
        for channel in channels:
            label_counts[channel.label] = label_counts.get(channel.label, 0) + 1
        if any(count > 1 for count in label_counts.values()):
            channels = [
                HDF5TraceChannel(
                    identifier=channel.identifier,
                    label=(
                        f'{channel.label} '
                        f'[{self.file[channel.dataset_path].parent.name}]'
                        if label_counts[channel.label] > 1
                        else channel.label
                    ),
                    dataset_path=channel.dataset_path,
                    sample_count=channel.sample_count,
                    trace_count=channel.trace_count,
                )
                for channel in channels
            ]
        return channels

    def get_selected_trace_channel(self):
        """Return the selected amplitude channel, choosing the first by default."""
        if (
            self.file is not None
            and self.trace_dataset_path
            and self.trace_dataset_path in self.file
            and self.trace_channel_name
            and self.trace_sample_count is not None
        ):
            dataset = self.file[self.trace_dataset_path]
            if (
                isinstance(dataset, h5py.Dataset)
                and dataset.ndim == 3
                and dataset.shape[1] == 1
            ):
                return HDF5TraceChannel(
                    identifier=dataset.name,
                    label=self.trace_channel_name,
                    dataset_path=self.trace_dataset_path,
                    sample_count=int(self.trace_sample_count),
                    trace_count=int(dataset.shape[2]),
                )
        channels = self.get_trace_channels()
        if not channels:
            raise ValueError('The HDF5 file does not contain compatible traces.')
        selected = next(
            (
                channel
                for channel in channels
                if (
                    channel.dataset_path == self.trace_dataset_path
                )
            ),
            channels[0],
        )
        self.trace_dataset_path = selected.dataset_path
        self.trace_group_path = self.file[selected.dataset_path].parent.name
        self.trace_channel_name = selected.label
        self.trace_sample_count = selected.sample_count
        return selected

    def select_trace_channel(self, channel_identifier):
        """Select one trace amplitude by identifier, label, or channel object."""
        if isinstance(channel_identifier, HDF5TraceChannel):
            requested_identifier = channel_identifier.identifier
        else:
            requested_identifier = str(channel_identifier)
        selected = next(
            (
                channel
                for channel in self.get_trace_channels()
                if requested_identifier in {
                    channel.identifier,
                    channel.label,
                }
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f'Unknown trace amplitude channel: {requested_identifier!r}.'
            )
        self.trace_dataset_path = selected.dataset_path
        self.trace_group_path = self.file[selected.dataset_path].parent.name
        self.trace_channel_name = selected.label
        self.trace_sample_count = selected.sample_count
        self.trace_reference = self.file[selected.dataset_path]
        self.traces = None
        self.traces_time = None
        self.trace_axis_name = None
        self.traces_dt = None
        self.saved_traces = False
        self.hist = None
        self.bins = None
        return selected

    def has_traces(self):
        """Return whether the current file contains a compatible trace group."""
        try:
            return bool(self.get_trace_channels())
        except (OSError, ValueError):
            return False

    def get_trace_data_dataset(self):
        """Return the dataset containing the selected amplitude channel."""
        if (
            self.file is not None
            and self.trace_dataset_path
            and self.trace_dataset_path in self.file
        ):
            return self.file[self.trace_dataset_path]
        channel = self.get_selected_trace_channel()
        return self.file[channel.dataset_path]

    def get_trace_sample_count(self):
        """Return the valid sample count of the selected amplitude channel."""
        if self.trace_sample_count is not None:
            return int(self.trace_sample_count)
        return int(self.get_selected_trace_channel().sample_count)

    def get_trace_values(self, trace_index):
        """Read one selected amplitude trace without loading other channels."""
        dataset = self.get_trace_data_dataset()
        return np.asarray(dataset[
            :self.get_trace_sample_count(),
            0,
            int(trace_index),
        ])

    def get_trace_matrix(self, trace_indices):
        """Read selected traces from the active amplitude channel."""
        trace_indices = np.asarray(trace_indices, dtype=int).ravel()
        if trace_indices.size == 0:
            raise ValueError('At least one trace must be selected.')
        dataset = self.get_trace_data_dataset()
        trace_count = int(dataset.shape[2])
        if np.any(trace_indices < 0) or np.any(trace_indices >= trace_count):
            raise IndexError('A selected trace index is outside the trace dataset.')
        unique_indices, inverse_indices = np.unique(
            trace_indices,
            return_inverse=True,
        )
        selected_traces = np.asarray(dataset[
            :self.get_trace_sample_count(),
            0,
            unique_indices,
        ])
        if selected_traces.ndim == 1:
            selected_traces = selected_traces[:, np.newaxis]
        return selected_traces[:, inverse_indices].T

    def get_trace_axis(self, trace_length=None):
        """Return the stored or reconstructed trace X coordinate."""
        if self.traces_time is None:
            self.set_traces_dt()
        axis_values = np.asarray(self.traces_time, dtype=np.float64)
        if trace_length is not None and axis_values.size != int(trace_length):
            raise ValueError(
                'The trace X-axis length does not match the trace data length.'
            )
        return axis_values

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
            trace_dataset = self.get_trace_data_dataset()
            self.shape_trace = (
                self.get_trace_sample_count(),
                1,
                int(trace_dataset.shape[2]),
            )
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
            data_attributes = self.file['Data'].attrs
            step_dimensions = np.atleast_1d(
                data_attributes['Step dimensions']
            ).astype(int).ravel()
            step_indices = np.atleast_1d(
                data_attributes.get(
                    'Step index', np.arange(step_dimensions.size)
                )
            ).astype(int).ravel()
            if step_indices.size == 0:
                step_indices = np.arange(step_dimensions.size)
            if np.any(step_indices < 0) or np.any(
                step_indices >= step_dimensions.size
            ):
                raise ValueError('Step index references a missing step dimension.')
            self.measure_dim = [
                int(step_dimensions[index]) for index in step_indices
            ]
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

            # Check if 'Channels' exists in the file before accessing it
            if self.channels is None:
                if 'Channels' in self.file:
                    self.channels = self.file['Channels']
                else:
                    # Initialize as empty if 'Channels' doesn't exist
                    self.channels = []

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
            # Only process channels if they exist
            if len(self.channels) > 0:
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
            if self.measure_dim is None:
                self.set_measure_dim()
            trace_dataset = self.get_trace_data_dataset()
            trace_length = self.get_trace_sample_count()
            trace_count = int(trace_dataset.shape[2])
            expected_trace_count = int(np.prod(self.measure_dim))
            if trace_count != expected_trace_count:
                raise ValueError(
                    f'Trace count {trace_count} does not match the '
                    f'{expected_trace_count} scan points.'
                )
            traces_i = np.asarray(
                trace_dataset[
                    :trace_length,
                    0,
                    :,
                ],
                dtype=np.float32,
            ).T.reshape(trace_count, trace_length)
            self.traces = traces_i.reshape(
                int(np.prod(self.measure_dim[1:])),
                int(self.measure_dim[0]),
                trace_length,
            )
            traces_i = None
        except Exception as e:
            print(f"Error creating traces as array: {e}")

    def set_traces_dt(self):
        if self.file is None:
            self.set_data()
        trace_group = self.get_trace_group()
        trace_dataset = self.get_trace_data_dataset()
        axis_values, axis_name = read_trace_axis(
            trace_group,
            trace_dataset,
            sample_count=self.get_trace_sample_count(),
            channel_name=self.trace_channel_name or '',
        )
        self.traces_time = axis_values
        self.trace_axis_name = axis_name
        if axis_values.size >= 2:
            finite_steps = np.diff(axis_values)
            finite_steps = finite_steps[
                np.isfinite(finite_steps) & (finite_steps != 0)
            ]
            self.traces_dt = (
                float(np.median(finite_steps))
                if finite_steps.size
                else 1.0
            )
        else:
            self.traces_dt = 1.0

    def save_traces_in_wdir(self):
        if not self.saved_traces:
            if self.file is None:
                self.set_data()
            if self.measure_dim is None:
                self.set_measure_dim()
            trace_order_matrix = []
            should_array_shape = (int(self.measure_dim[0]), int(np.prod(np.array(self.measure_dim)[1:])))
            trace_dataset = self.get_trace_data_dataset()
            trace_length = self.get_trace_sample_count()
            traces_i = (
                np.asarray(
                    trace_dataset[
                        :trace_length,
                        0,
                        :,
                    ],
                    dtype=np.float32,
                )
                .T.reshape(int(np.prod(should_array_shape)), trace_length)
            )
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

    def trace_loading_with_referance(self, trace_channel=None):
        if self.file is None:
            self.set_data()
        if self.measure_dim is None:
            self.set_measure_dim()
        if trace_channel is not None:
            self.select_trace_channel(trace_channel)
        else:
            self.get_selected_trace_channel()
        should_array_shape = (int(self.measure_dim[0]), int(np.prod(np.array(self.measure_dim)[1:])))
        self.trace_reference = self.get_trace_data_dataset()
        trace_count = int(self.trace_reference.shape[-1])
        if trace_count != int(np.prod(should_array_shape)):
            raise ValueError(
                f'Trace count {trace_count} does not match the scan shape '
                f'{tuple(self.measure_dim)}.'
            )
        self.trace_order = np.arange(trace_count).reshape(should_array_shape)

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
        self.trace_axis_name = None
        self.trace_group_path = None
        self.trace_dataset_path = None
        self.trace_channel_name = None
        self.trace_sample_count = None
        self.trace_order = None
        self.saved_traces = False
        self.trace_reference = None
        self.hist = None
        self.bins = None
        self.wdir = None
