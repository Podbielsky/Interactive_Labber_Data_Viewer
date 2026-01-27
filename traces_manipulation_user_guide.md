# Trace File Manipulation User Guide

This is a User Guide for the menu options `Add traces from HDF5 file` and `Generate traces from dataset` under `File`
in the HDF5 File Viewer. 

## Add Traces from HDF5 File

This feature allows the user to take a group or dataset from one HDF5 file and copy it to another.
It is meant to copy traces from one measurement to a map of another measurement, but works for any dataset of group.

1. To use it you must first open your target HDF5 file. Click `File` > `Select File Directory` and select your target HDF5 file in the file dialogue. 

2. Then you may click `File` > `Add traces from HDF5 file` and select your source file. 

3. Click `Show HDF5 File` and use (ctrl + or shift +) left mouse click to select the group or dataset(s) you wish to copy to your destination file.
The entry in the text box at the top determines the group name in the target file. If you select a group the name of that group will change to this value in the target file. If you select datasets, they will all be saved under this group name (if the group already exists, the datasets will simply be added, if the group does not exist yet, it will be created). 

4. To apply the changes, simply press the `Copy Selected Datasets ` button at the bottom of the window.


## Generate Traces from Datset

This feature is meant to generate a new HDF5 file with a 'Traces' group that uses the necessary format for `Plot Map with Trace Data`. It can be applied to any 2D or 3D HDF5 dataset or .npy array. Aside from the traces group, a Data group is also generated. It contains a dataset with the mean values of the traces and is generated because it is required by the plotter.  

1. Open the feature by clicking `File` > `Generate Traces from Dataset`

2. Select your source .hdf5 or .npy file. 

**Then in the case of a .hdf5 file:**

3. Select your target dataset that contains the traces from the list of suitable datasets. If the list is empty, your hdf5 contains no suitable datasets for conversion to traces. 

4. Optionally select an axis dataset. This contains the x-axis values (e.g. times or energies)

5. Select the dimension to be used as x-axis. (e.g. if your dataset has shape (80, 1152, 1600) and your x-axis has shape (1600) you should select 2 as your dimension index)

6. Click the `Confirm Reshape` button to apply. And select your target directory. (This may take a while, depending on dataset size.)

**In the case of a .npy file:**

3. After selecting your source file, another file dialogue will pop up where you can select your axis dataset file. Click `cancel` if you do not wish to use one.

Then continue with step 5 from above. 

