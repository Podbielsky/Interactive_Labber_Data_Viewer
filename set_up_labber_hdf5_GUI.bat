@echo off

REM Python 3.10.11 is the newest Python 3.10 release with an official Windows installer.
set "PYTHON_VERSION=3.10.11"
set "PYTHON_DIR=%USERPROFILE%\AppData\Local\Programs\Python\Python310"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PYTHON_INSTALLER=%TEMP%\python-%PYTHON_VERSION%-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe"

REM Install Python 3.10 for the current user if it is not already installed.
IF NOT EXIST "%PYTHON_EXE%" (
    echo Python %PYTHON_VERSION% was not found in the user folder.
    echo Downloading Python %PYTHON_VERSION%...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri $env:PYTHON_URL -OutFile $env:PYTHON_INSTALLER"

    IF ERRORLEVEL 1 (
        echo Failed to download Python %PYTHON_VERSION%.
        GOTO :END
    )

    echo Installing Python %PYTHON_VERSION% for the current user...
    "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 Include_test=0 PrependPath=0 Shortcuts=0 TargetDir="%PYTHON_DIR%"

    IF ERRORLEVEL 1 (
        echo Failed to install Python %PYTHON_VERSION%.
        GOTO :END
    )

    DEL /Q "%PYTHON_INSTALLER%" >NUL 2>&1
)

IF NOT EXIST "%PYTHON_EXE%" (
    echo Python was not found at "%PYTHON_EXE%".
    GOTO :END
)

REM Define the name of the virtual environment
set "ENV_NAME=%USERPROFILE%\AppData\Local\Programs\Python\Labber_View_GUI"
set "LABBER_SHORTCUT_TARGET=%ENV_NAME%\Scripts\labber_hdf5_viewer.bat"
set "LABBER_ICON_DIR=%ENV_NAME%\icons"
set "LABBER_SHORTCUT_ICON=%LABBER_ICON_DIR%\labber_viewer_ICON.ico"
set "LABBER_SHORTCUT_WORKING_DIR=%ENV_NAME%\Scripts"
set "LABBER_SOURCE_ROOT=%~dp0"
set "LABBER_VERSION_FILE=%ENV_NAME%\Scripts\labber_hdf5_viewer_version.json"
set "LABBER_GITHUB_REPOSITORY=Podbielsky/Interactive_Labber_Data_Viewer"

REM Create a new virtual environment using Python 3.10
"%PYTHON_EXE%" -m venv "%ENV_NAME%"

IF ERRORLEVEL 1 (
    echo Failed to create the Python 3.10 virtual environment.
    GOTO :END
)

REM Check if the environment was created successfully
IF EXIST "%ENV_NAME%\Scripts\activate.bat" (
    echo Virtual environment '%ENV_NAME%' created successfully.

    REM Activate the virtual environment
    CALL "%ENV_NAME%\Scripts\activate.bat"

    echo Virtual environment '%ENV_NAME%' is now active.

    REM Install required packages
    echo Installing required packages...
    "%ENV_NAME%\Scripts\python.exe" -m pip install numpy==1.22.4 scipy==1.7.3 matplotlib==3.5.0 numba==0.58.1 h5py tkinterdnd2==0.6.2 ttkbootstrap==2.2.2

    IF ERRORLEVEL 1 (
        echo Failed to install one or more required packages.
        GOTO :END
    )

    echo Required packages installed successfully.

    REM Copy Python scripts to the Scripts directory of the virtual environment
    copy "%~dp0source\interactive_hdf5_files.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\creating_hdf5_files_from_npy_files.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\HDF5Data.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\interactive_plotting_tools.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\fitting_tools.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\Data_analysis_and_transforms.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\custom_cmap.py" "%ENV_NAME%\Scripts"
    copy "%~dp0source\gamma_map.py" "%ENV_NAME%\Scripts"
    copy /Y "%~dp0labber_hdf5_viewer.bat" "%LABBER_SHORTCUT_TARGET%"

    IF ERRORLEVEL 1 (
        echo Failed to copy labber_hdf5_viewer.bat.
        GOTO :END
    )

    REM Record the installed commit for the in-application update checker.
    echo Recording the installed Git commit...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $commit = ''; $versionSource = 'unknown'; $gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue; $gitMetadata = Join-Path $env:LABBER_SOURCE_ROOT '.git'; if ($gitCommand -and (Test-Path -LiteralPath $gitMetadata)) { try { $candidate = & $gitCommand.Source -C $env:LABBER_SOURCE_ROOT rev-parse HEAD 2>$null | Select-Object -First 1; if ($candidate) { $candidate = $candidate.Trim() }; if ($candidate -match '^[0-9a-fA-F]{40}$') { $commit = $candidate.ToLower(); $versionSource = 'git-checkout' } } catch {} }; if ($commit -notmatch '^[0-9a-f]{40}$') { try { $headers = @{ Accept = 'application/vnd.github+json'; 'User-Agent' = 'Labber-HDF5-Viewer-Installer'; 'X-GitHub-Api-Version' = '2022-11-28' }; $response = Invoke-RestMethod -Uri ('https://api.github.com/repos/' + $env:LABBER_GITHUB_REPOSITORY + '/commits/main') -Headers $headers -TimeoutSec 10; if ($response.sha -match '^[0-9a-fA-F]{40}$') { $commit = $response.sha.ToLower(); $versionSource = 'github-main-at-install' } } catch {} }; $metadata = [ordered]@{ repository = $env:LABBER_GITHUB_REPOSITORY; branch = 'main'; commit = $commit; source = $versionSource; installed_at = (Get-Date).ToUniversalTime().ToString('o') }; $metadata | ConvertTo-Json | Set-Content -LiteralPath $env:LABBER_VERSION_FILE -Encoding UTF8; if ($commit -notmatch '^[0-9a-f]{40}$') { Write-Warning 'The installed Git commit could not be determined.' }"

    IF ERRORLEVEL 1 (
        echo Warning: version metadata could not be written.
    )

    IF NOT EXIST "%LABBER_ICON_DIR%" mkdir "%LABBER_ICON_DIR%"

    IF ERRORLEVEL 1 (
        echo Failed to create the icon directory.
        GOTO :END
    )

    copy /Y "%~dp0icons\labber_viewer_ICON.ico" "%LABBER_SHORTCUT_ICON%"

    IF ERRORLEVEL 1 (
        echo Failed to copy icons\labber_viewer_ICON.ico.
        GOTO :END
    )

    echo Application files copied successfully.

    REM Create or replace the current user's Desktop shortcut.
    echo Creating Desktop shortcut...
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $shell = New-Object -ComObject WScript.Shell; $desktop = [Environment]::GetFolderPath('Desktop'); $shortcut = $shell.CreateShortcut((Join-Path $desktop 'Labber HDF5 Viewer.lnk')); $shortcut.TargetPath = $env:LABBER_SHORTCUT_TARGET; $shortcut.WorkingDirectory = $env:LABBER_SHORTCUT_WORKING_DIR; $shortcut.IconLocation = $env:LABBER_SHORTCUT_ICON + ',0'; $shortcut.Description = 'Labber HDF5 Viewer'; $shortcut.Save()"

    IF ERRORLEVEL 1 (
        echo Failed to create the Desktop shortcut.
        GOTO :END
    )

    echo Desktop shortcut created successfully.
) ELSE (
    echo Failed to create virtual environment.
)

:END
pause
