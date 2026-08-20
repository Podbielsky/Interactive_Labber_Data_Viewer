# Trace File Manipulation User Guide

This guide describes the following commands in the HDF5 File Viewer:

- `File` > `Add Traces from HDF5 File`
- `File` > `Generate Traces from Dataset`

These commands are not equivalent:

- **Add Traces** copies existing HDF5 objects. It does not reshape, convert, or
  validate spectroscopy data.
- **Generate Traces** converts one numeric 2D or 3D dataset into the special
  `/Traces` representation expected by `Plot Map with Trace Data`.

## Current implementation status

`Add Traces from HDF5 File` does **not currently work reliably for creating a
new plot-compatible `/Traces` group**. Tests against the included example files
showed the following problems:

1. Copying the complete source `/Traces` group creates the destination group
   without HDF5 creation-order tracking. The loader then sees the alphabetically
   first `t0dt` dataset instead of `Data` and can fail with an error such as
   `cannot reshape array of size 2`.
2. Although the tree permits Ctrl/Shift multi-selection, the copy command uses
   only the first selected item.
3. Copying the required datasets one at a time can create nested paths such as
   `/Traces/Traces/Data`, which the trace loader does not recognize.
4. Copy errors are often printed only to the terminal from which the viewer was
   started.

Copying into an already valid `/Traces` group may work if its creation order is
preserved and every shape matches. Back up the target file before trying it.
For a reliable copy into a file without traces, use the Python workaround in
[Reliable copy workaround](#reliable-copy-workaround).

## Required HDF5 format

For a two-dimensional map, define:

- `D0`: number of points along the displayed x index
- `D1`: number of points along the displayed y index
- `L`: number of samples in every spectrum or trace
- `P = D0 * D1`: total number of map points and therefore total number of traces

For measurements with more map dimensions, `P` is the product of all values in
the `/Data` attribute `Step dimensions`.

The target file must already be a valid map file and must contain the following
objects:

```text
/
├── Data/
│   ├── attribute: Step dimensions = [D0, D1]
│   ├── attribute: Step index      = [0, 1]
│   ├── Data
│   └── Channel names
├── Log list
└── Traces/                         # must track link creation order
    ├── Data                        # must be the first link
    ├── Data_N                      # must be the second link
    └── Alazar Slytherin - Ch1 - Data_t0dt
```

The trace datasets have these requirements:

| HDF5 path | Required shape | Required contents |
| --- | --- | --- |
| `/Traces/Data` | `(L, 1, P)` | Numeric trace samples. Trace `k` is `Data[:, 0, k]`. |
| `/Traces/Data_N` | `(1,)` | One integer equal to `L`. |
| `/Traces/Alazar Slytherin - Ch1 - Data_t0dt` | `(1, 2)` | `[[t0, dt]]`, where `dt` is the sample spacing. |

Important constraints:

- Group and dataset names are case-sensitive. The root group must be named
  exactly `Traces`, with an uppercase `T`.
- `/Traces/Data.shape[2]` must equal `prod(/Data.attrs['Step dimensions'])`.
  One spectrum is required for every map point; there is no broadcasting of one
  spectrum to multiple points.
- The middle dimension of `/Traces/Data` must be exactly `1`, because the viewer
  reads traces as `Data[:, 0, trace_index]`.
- All traces must have the same length `L`. Ragged arrays and NumPy `object`
  arrays are not supported.
- Use a homogeneous numeric dtype such as `float32`, `float64`, or an integer
  dtype. Finite floating-point values are recommended; NaN/Inf values can break
  fitting and analysis tools.
- The `/Traces` group must be created with `track_order=True`, and its three
  datasets must be inserted in the order shown above. This is a limitation of
  the current loader, which uses the first and second group keys instead of
  always addressing `Data` and `Data_N` by name.

### Sample-axis limitations

The application does not store or read a complete spectroscopy axis. It stores
only `t0` and `dt`, and the current plotter constructs its horizontal trace axis
as:

```python
sample_axis = dt * np.arange(L)
```

Consequently:

- `t0` is currently ignored by the trace plot.
- The axis must be uniformly spaced for the displayed coordinates to be valid.
- Descending and nonlinear energy/frequency axes cannot be represented exactly.
- The plot is currently labelled `Time (s)` even when the samples represent
  energy, frequency, wavelength, or another spectroscopic coordinate.

The trace amplitudes can still contain spectroscopic values, but the horizontal
coordinate is treated as a uniformly sampled, time-like axis.

## Per-point trace ordering

For a two-dimensional target with step dimensions `[D0, D1]`, the viewer uses:

```text
trace_index = y_index * D0 + x_index
```

Therefore a NumPy spectroscopy array with shape `(D0, D1, L)` and indexing
`spectra[x_index, y_index, sample_index]` must be converted as follows:

```python
import numpy as np

spectra = np.asarray(spectra)
D0, D1, L = spectra.shape

# (D0, D1, L) -> (L, D0, D1) -> (L, D1, D0) -> (L, 1, D0*D1)
trace_data = np.moveaxis(spectra, -1, 0)
trace_data = trace_data.swapaxes(1, 2).reshape(L, 1, D0 * D1)

# Verification for one point:
x_index, y_index = 2, 3
k = y_index * D0 + x_index
np.testing.assert_array_equal(trace_data[:, 0, k], spectra[x_index, y_index, :])
```

If the trace/sample dimension is not the final input dimension, replace `-1`
in `np.moveaxis` with its dimension index. After moving it, the two remaining
dimensions must be ordered as `(D0, D1)` before the `swapaxes` call.

## Add Traces from HDF5 File

Use this command only when the source file already contains the exact compatible
datasets described above.

1. Back up the target HDF5 file. The command modifies it in place.
2. Open the **target** map with `File` > `Select File Directory`.
3. Select `File` > `Add Traces from HDF5 File` and choose the **source** HDF5
   file.
4. Keep `Destination group name` set to `Traces`.
5. The intended operation is to select the source `/Traces` group and press
   `Copy Selected Dataset(s)`.
6. Close and reopen the target file before selecting
   `Plotting` > `Plot Map with Trace Data`.

Do not select an arbitrary 3D spectroscopy dataset and copy it into `Traces`.
That only creates `/Traces/<original_dataset_name>`; it does not create the
required `Data`, `Data_N`, or `t0dt` structure and does not reshape point data.
Use `Generate Traces from Dataset` or construct the group explicitly.

### Reliable copy workaround

Until the GUI copy defects are fixed, the following script copies an already
compatible source `/Traces` group while preserving the required key order,
attributes, datatype, and compression. Close the files in the viewer first and
replace the two paths before running it.

```python
import h5py
import numpy as np

source_path = r"source_with_traces.hdf5"
target_path = r"target_map_without_traces.hdf5"

ordered_names = (
    "Data",
    "Data_N",
    "Alazar Slytherin - Ch1 - Data_t0dt",
)

with h5py.File(source_path, "r") as source, h5py.File(target_path, "r+") as target:
    source_traces = source["Traces"]
    step_dimensions = np.asarray(
        target["Data"].attrs["Step dimensions"], dtype=int
    )
    trace_shape = source_traces["Data"].shape

    if len(trace_shape) != 3 or trace_shape[1] != 1:
        raise ValueError(f"Expected /Traces/Data shape (L, 1, P), got {trace_shape}")
    if trace_shape[2] != int(np.prod(step_dimensions)):
        raise ValueError(
            "Trace count does not match the number of target map points: "
            f"{trace_shape[2]} != {int(np.prod(step_dimensions))}"
        )
    if int(source_traces["Data_N"][0]) != trace_shape[0]:
        raise ValueError("/Traces/Data_N does not equal the trace length")

    # This replaces any existing trace group; make a backup of the file first.
    if "Traces" in target:
        del target["Traces"]
    target_traces = target.create_group("Traces", track_order=True)

    for name in ordered_names:
        if name not in source_traces:
            raise KeyError(f"Missing required source dataset: /Traces/{name}")
        source_traces.copy(name, target_traces, name=name)
```

## Generate Traces from Dataset

This command creates a new HDF5 file containing a `/Traces` group and a minimal
`/Data` map whose displayed value is the mean of every trace.

### Accepted source data

- HDF5 input: a numeric dataset with exactly 2 or 3 dimensions. The file picker
  currently offers `.hdf5` files.
- NumPy input: a numeric `.npy` array with 2 or 3 dimensions.
- Every point must have the same trace length.
- A selected axis dataset must be one-dimensional, contain at least two values,
  and have the same length as the selected trace dimension.
- If no axis is selected, the output uses `t0 = 0` and `dt = 1`.

For 3D data, exactly one dimension is the trace/sample dimension. The two other
dimensions become the map dimensions `D0` and `D1`. For example:

```text
input shape:          (101, 181, 1001)
trace dimension:      2
map dimensions:       D0=101, D1=181
trace length:         L=1001
number of traces:     P=101*181=18281
output trace shape:   (1001, 1, 18281)
```

For 2D input, the converter inserts a map dimension of length one. For example,
`(500, 2048)` with trace dimension `1` produces 500 traces of length 2048 on a
`1 x 500` map.

### Procedure

1. Select `File` > `Generate Traces from Dataset`.
2. Choose the source `.hdf5` or `.npy` file.
3. For HDF5 input, select the numeric 2D/3D dataset containing the spectra.
4. Optionally select its one-dimensional sample-axis dataset.
5. Set `Index of dimension ... to be used as x axis` to the **trace/sample
   dimension index**, not a map dimension. For `(101, 181, 1001)`, enter `2`.
6. Press `Confirm Reshape`, choose a nonempty output filename, and wait for the
   write operation to finish.
7. Validate the output using the checklist below before using
   `Plot Map with Trace Data`.

### Spatial-order correction for generated 3D files

The current generator flattens the remaining map dimensions in `(D0, D1)`
order, while the viewer looks up traces in `(D1, D0)` order. Without correction,
a trace can be displayed at the wrong map point. This is visible even for square
maps and is especially confusing for nonsquare maps.

Run this correction once on a newly generated file. It changes the data in
place, so make a backup first:

```python
import h5py
import numpy as np

generated_path = r"generated_traces.hdf5"

with h5py.File(generated_path, "r+") as file:
    D0, D1 = map(int, file["Data"].attrs["Step dimensions"])
    dataset = file["Traces/Data"]
    L = dataset.shape[0]

    corrected = (
        dataset[...]
        .reshape(L, D0, D1)
        .transpose(0, 2, 1)
        .reshape(L, 1, D0 * D1)
    )
    dataset[...] = corrected
```

This transpose is harmless for the promoted `1 x D` map created from 2D input.

## Validation checklist

The following read-only script checks the structural requirements:

```python
import h5py
import numpy as np

path = r"file_to_check.hdf5"
required_order = [
    "Data",
    "Data_N",
    "Alazar Slytherin - Ch1 - Data_t0dt",
]

with h5py.File(path, "r") as file:
    dimensions = np.asarray(file["Data"].attrs["Step dimensions"], dtype=int)
    group = file["Traces"]
    trace_data = group["Data"]

    print("Step dimensions:", dimensions)
    print("Traces keys:", list(group.keys()))
    print("Trace data shape:", trace_data.shape)
    print("Data_N:", group["Data_N"][...])
    print("t0, dt:", group["Alazar Slytherin - Ch1 - Data_t0dt"][...])

    assert list(group.keys()) == required_order
    assert trace_data.ndim == 3 and trace_data.shape[1] == 1
    assert trace_data.shape[2] == int(np.prod(dimensions))
    assert int(group["Data_N"][0]) == trace_data.shape[0]
    assert group["Alazar Slytherin - Ch1 - Data_t0dt"].shape == (1, 2)

print("The trace structure is compatible with the current loader.")
```

Structural compatibility does not prove that each trace is associated with the
correct map coordinate. For known test data, also compare selected
`/Traces/Data[:, 0, k]` values against their original spectra using
`k = y_index * D0 + x_index`.

## Troubleshooting

- **`cannot reshape array of size 2 into shape (...)`**: the loader selected the
  `(1, 2)` `t0dt` dataset as the first trace-group item. Recreate `/Traces` with
  `track_order=True` and the required insertion order.
- **`/Traces/Traces/Data` exists**: datasets were copied individually and became
  nested. Recreate one flat `/Traces` group using the workaround above.
- **Trace count mismatch**: the source and target maps have different point
  counts. A target with dimensions `[D0, D1]` requires exactly `D0*D1` traces.
- **The trace shape is correct but the wrong spectrum appears**: the point order
  is wrong. Apply or reproduce the `(L, D1, D0)` flattening described above.
- **The spectroscopy x axis is shifted or nonlinear**: the current viewer uses
  only `dt` and ignores `t0` and the full axis array.
- **Nothing appears after pressing Copy**: inspect the terminal output. The
  current copy window does not show all errors in a message box.
