@echo off

REM Define the name of the virtual environment
set "ENV_NAME=%USERPROFILE%\AppData\Local\Programs\Python\Labber_View_GUI"

REM Create a new virtual environment using Python 3.9
"%USERPROFILE%\AppData\Local\Programs\Python\Python39\python.exe" -m venv %ENV_NAME%

REM Check if the environment was created successfully
IF EXIST "%ENV_NAME%\Scripts\activate.bat" (
    echo Virtual environment '%ENV_NAME%' created successfully.

    REM Activate the virtual environment
    CALL "%ENV_NAME%\Scripts\activate.bat"

    echo Virtual environment '%ENV_NAME%' is now active.

    REM Install required packages
    echo Installing required packages...
    pip install numpy==1.22.4 scipy==1.7.1 matplotlib==3.4.3 numba==0.58.1 h5py

    echo Required packages installed successfully.

    REM Copy Python scripts to the Scripts directory of the virtual environment
    copy "%~dp0interactive_hdf5_files.py" "%ENV_NAME%\Scripts"
    copy "%~dp0HDF5Data.py" "%ENV_NAME%\Scripts"
    copy "%~dp0interactive_plotting_tools.py" "%ENV_NAME%\Scripts"
    copy "%~dp0Data_analysis_and_transforms.py" "%ENV_NAME%\Scripts"
    copy "%~dp0custom_cmap.py" "%ENV_NAME%\Scripts"
    copy "%~dp0gamma_map.py" "%ENV_NAME%\Scripts"

    echo Python script copied successfully.

    REM You can add additional commands here to run in the virtual environment
) ELSE (
    echo Failed to create virtual environment.
)

pause