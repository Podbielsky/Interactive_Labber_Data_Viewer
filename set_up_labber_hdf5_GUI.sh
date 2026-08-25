#!/usr/bin/env bash

set -Eeuo pipefail

labber_fail() {
    printf '\nError: %s\n' "$1" >&2
    exit 1
}

if [[ -z "${HOME:-}" ]]; then
    labber_fail "The HOME directory is not available."
fi

LABBER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LABBER_SOURCE_DIR="$LABBER_SCRIPT_DIR/source"
LABBER_OS="$(uname -s)"

case "$LABBER_OS" in
    Darwin | Linux)
        ;;
    *)
        labber_fail "This installer supports macOS and Linux only."
        ;;
esac

LABBER_USER_DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"
LABBER_INSTALL_ROOT="$LABBER_USER_DATA_DIR/Labber_View_GUI"
LABBER_VENV_DIR="$LABBER_INSTALL_ROOT/venv"
LABBER_APP_DIR="$LABBER_INSTALL_ROOT/app"
LABBER_ICON_DIR="$LABBER_INSTALL_ROOT/icons"
LABBER_VERSION_FILE="$LABBER_APP_DIR/labber_hdf5_viewer_version.json"
LABBER_GITHUB_REPOSITORY="Podbielsky/Interactive_Labber_Data_Viewer"
LABBER_UV_INSTALL_DIR="$LABBER_INSTALL_ROOT/tools"
LABBER_UV_BIN="$LABBER_UV_INSTALL_DIR/uv"
LABBER_TEMP_DIR=""

labber_cleanup() {
    if [[ -n "$LABBER_TEMP_DIR" && -d "$LABBER_TEMP_DIR" ]]; then
        rm -rf -- "$LABBER_TEMP_DIR"
    fi
}

trap labber_cleanup EXIT

LABBER_APPLICATION_FILES=(
    "interactive_hdf5_files.py"
    "creating_hdf5_files_from_npy_files.py"
    "HDF5Data.py"
    "database_manager.py"
    "database_browser.py"
    "interactive_plotting_tools.py"
    "plot_style.py"
    "fitting_tools.py"
    "Data_analysis_and_transforms.py"
    "custom_cmap.py"
    "gamma_map.py"
)

LABBER_ICON_FILES=(
    "labber_viewer_ICON.icns"
    "labber_viewer_ICON.png"
)

for LABBER_REQUIRED_FILE in "${LABBER_APPLICATION_FILES[@]}"; do
    if [[ ! -f "$LABBER_SOURCE_DIR/$LABBER_REQUIRED_FILE" ]]; then
        labber_fail "Required file is missing: source/$LABBER_REQUIRED_FILE"
    fi
done

for LABBER_ICON_FILE in "${LABBER_ICON_FILES[@]}"; do
    if [[ ! -f "$LABBER_SCRIPT_DIR/icons/$LABBER_ICON_FILE" ]]; then
        labber_fail "Required icon is missing: icons/$LABBER_ICON_FILE"
    fi
done

mkdir -p \
    "$LABBER_INSTALL_ROOT" \
    "$LABBER_APP_DIR" \
    "$LABBER_ICON_DIR" \
    "$LABBER_UV_INSTALL_DIR"

printf 'Installing or updating the user-local Python bootstrapper...\n'
LABBER_TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/labber-hdf5-viewer.XXXXXX")"
LABBER_UV_INSTALLER="$LABBER_TEMP_DIR/install-uv.sh"

if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error \
        https://astral.sh/uv/install.sh \
        --output "$LABBER_UV_INSTALLER"
elif command -v wget >/dev/null 2>&1; then
    wget --quiet \
        https://astral.sh/uv/install.sh \
        --output-document="$LABBER_UV_INSTALLER"
else
    labber_fail "curl or wget is required to download Python."
fi

env UV_INSTALL_DIR="$LABBER_UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 \
    sh "$LABBER_UV_INSTALLER"

if [[ ! -x "$LABBER_UV_BIN" ]]; then
    labber_fail "The Python bootstrapper was not installed correctly."
fi

printf 'Installing the newest available Python 3.10 release for this user...\n'
"$LABBER_UV_BIN" python install --no-bin 3.10
LABBER_PYTHON_EXE="$("$LABBER_UV_BIN" python find --managed-python 3.10)"

if [[ ! -x "$LABBER_PYTHON_EXE" ]]; then
    labber_fail "The managed Python 3.10 executable was not found."
fi

LABBER_MANAGED_PYTHON_VERSION="$(
    "$LABBER_PYTHON_EXE" -c 'import platform; print(platform.python_version())'
)"
LABBER_CREATE_VENV=1

if [[ -x "$LABBER_VENV_DIR/bin/python" ]]; then
    LABBER_VENV_PYTHON_VERSION="$(
        "$LABBER_VENV_DIR/bin/python" -c 'import platform; print(platform.python_version())' \
            2>/dev/null || true
    )"

    if [[ "$LABBER_VENV_PYTHON_VERSION" == "$LABBER_MANAGED_PYTHON_VERSION" ]]; then
        LABBER_CREATE_VENV=0
        printf 'Reusing the existing Python %s virtual environment.\n' \
            "$LABBER_VENV_PYTHON_VERSION"
    fi
fi

if [[ "$LABBER_CREATE_VENV" -eq 1 ]]; then
    if [[ -e "$LABBER_VENV_DIR" ]]; then
        LABBER_VENV_BACKUP="${LABBER_VENV_DIR}.backup-$(date +%Y%m%d-%H%M%S)"
        mv "$LABBER_VENV_DIR" "$LABBER_VENV_BACKUP"
        printf 'Preserved the previous environment at %s\n' "$LABBER_VENV_BACKUP"
    fi

    "$LABBER_UV_BIN" venv --no-config \
        --python "$LABBER_PYTHON_EXE" \
        "$LABBER_VENV_DIR"
fi

printf 'Installing required packages...\n'
"$LABBER_UV_BIN" pip install --no-config \
    --python "$LABBER_VENV_DIR/bin/python" \
    "numpy==1.22.4" \
    "scipy==1.7.3" \
    "matplotlib==3.5.0" \
    "numba==0.58.1" \
    h5py \
    "tkinterdnd2==0.6.2" \
    "ttkbootstrap==2.2.2"

"$LABBER_VENV_DIR/bin/python" -c \
    'import h5py, matplotlib, numba, numpy, scipy, tkinter, tkinterdnd2, ttkbootstrap'

printf 'Copying application files...\n'
for LABBER_APPLICATION_FILE in "${LABBER_APPLICATION_FILES[@]}"; do
    install -m 0644 \
        "$LABBER_SOURCE_DIR/$LABBER_APPLICATION_FILE" \
        "$LABBER_APP_DIR/$LABBER_APPLICATION_FILE"
done

printf 'Recording the installed Git commit...\n'
LABBER_INSTALLED_COMMIT=""
LABBER_VERSION_SOURCE=""

if [[ -e "$LABBER_SCRIPT_DIR/.git" ]] && command -v git >/dev/null 2>&1; then
    LABBER_INSTALLED_COMMIT="$(
        git -C "$LABBER_SCRIPT_DIR" rev-parse HEAD 2>/dev/null || true
    )"
    if [[ "$LABBER_INSTALLED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
        LABBER_VERSION_SOURCE="git-checkout"
    fi
fi

if [[ ! "$LABBER_INSTALLED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
    LABBER_INSTALLED_COMMIT="$(
        "$LABBER_VENV_DIR/bin/python" -c \
            'import json, urllib.request; request = urllib.request.Request("https://api.github.com/repos/Podbielsky/Interactive_Labber_Data_Viewer/commits/main", headers={"Accept": "application/vnd.github+json", "User-Agent": "Labber-HDF5-Viewer-Installer", "X-GitHub-Api-Version": "2022-11-28"}); print(json.load(urllib.request.urlopen(request, timeout=10))["sha"])' \
            2>/dev/null || true
    )"
    if [[ "$LABBER_INSTALLED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
        LABBER_VERSION_SOURCE="github-main-at-install"
    else
        LABBER_INSTALLED_COMMIT=""
        LABBER_VERSION_SOURCE="unknown"
        printf 'Warning: the installed Git commit could not be determined.\n' >&2
    fi
fi

env \
    LABBER_VERSION_COMMIT="$LABBER_INSTALLED_COMMIT" \
    LABBER_VERSION_SOURCE="$LABBER_VERSION_SOURCE" \
    LABBER_VERSION_FILE="$LABBER_VERSION_FILE" \
    LABBER_VERSION_REPOSITORY="$LABBER_GITHUB_REPOSITORY" \
    "$LABBER_VENV_DIR/bin/python" -c \
    'import datetime, json, os; metadata = {"repository": os.environ["LABBER_VERSION_REPOSITORY"], "branch": "main", "commit": os.environ["LABBER_VERSION_COMMIT"], "source": os.environ["LABBER_VERSION_SOURCE"], "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}; file = open(os.environ["LABBER_VERSION_FILE"], "w", encoding="utf-8"); json.dump(metadata, file, indent=2); file.write("\n"); file.close()'

install -m 0644 "$LABBER_SCRIPT_DIR/icons/labber_viewer_ICON.icns" \
    "$LABBER_ICON_DIR/labber_viewer_ICON.icns"
install -m 0644 "$LABBER_SCRIPT_DIR/icons/labber_viewer_ICON.png" \
    "$LABBER_ICON_DIR/labber_viewer_ICON.png"

LABBER_LAUNCHER="$LABBER_INSTALL_ROOT/labber_hdf5_viewer.sh"
printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\nexec %q %q "$@"\n' \
    "$LABBER_APP_DIR" \
    "$LABBER_VENV_DIR/bin/python" \
    "$LABBER_APP_DIR/interactive_hdf5_files.py" \
    > "$LABBER_LAUNCHER"
chmod 0755 "$LABBER_LAUNCHER"

if [[ "$LABBER_OS" == "Darwin" ]]; then
    LABBER_MAC_APP="$HOME/Desktop/Labber HDF5 Viewer.app"
    LABBER_MAC_CONTENTS="$LABBER_MAC_APP/Contents"
    mkdir -p "$LABBER_MAC_CONTENTS/MacOS" "$LABBER_MAC_CONTENTS/Resources"

    install -m 0644 "$LABBER_ICON_DIR/labber_viewer_ICON.icns" \
        "$LABBER_MAC_CONTENTS/Resources/labber_viewer_ICON.icns"

    cat > "$LABBER_MAC_CONTENTS/Info.plist" <<'LABBER_PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>Labber HDF5 Viewer</string>
    <key>CFBundleExecutable</key>
    <string>labber-hdf5-viewer</string>
    <key>CFBundleIconFile</key>
    <string>labber_viewer_ICON.icns</string>
    <key>CFBundleIdentifier</key>
    <string>local.labber.hdf5viewer</string>
    <key>CFBundleName</key>
    <string>Labber HDF5 Viewer</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
LABBER_PLIST

    printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$LABBER_LAUNCHER" \
        > "$LABBER_MAC_CONTENTS/MacOS/labber-hdf5-viewer"
    chmod 0755 "$LABBER_MAC_CONTENTS/MacOS/labber-hdf5-viewer"
    touch "$LABBER_MAC_APP"
    LABBER_SHORTCUT_PATH="$LABBER_MAC_APP"
else
    LABBER_APPLICATIONS_DIR="$LABBER_USER_DATA_DIR/applications"
    LABBER_LINUX_ENTRY="$LABBER_APPLICATIONS_DIR/labber-hdf5-viewer.desktop"
    mkdir -p "$LABBER_APPLICATIONS_DIR"

    {
        printf '[Desktop Entry]\n'
        printf 'Type=Application\n'
        printf 'Name=Labber HDF5 Viewer\n'
        printf 'Comment=View and analyze Labber HDF5 data\n'
        printf 'Exec="%s"\n' "$LABBER_LAUNCHER"
        printf 'Icon=%s\n' "$LABBER_ICON_DIR/labber_viewer_ICON.png"
        printf 'Terminal=false\n'
        printf 'Categories=Science;Utility;\n'
        printf 'StartupNotify=true\n'
    } > "$LABBER_LINUX_ENTRY"
    chmod 0755 "$LABBER_LINUX_ENTRY"

    if command -v desktop-file-validate >/dev/null 2>&1; then
        desktop-file-validate "$LABBER_LINUX_ENTRY"
    fi

    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$LABBER_APPLICATIONS_DIR" >/dev/null 2>&1 || true
    fi

    LABBER_DESKTOP_DIR=""
    if command -v xdg-user-dir >/dev/null 2>&1; then
        LABBER_DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
    fi
    if [[ -z "$LABBER_DESKTOP_DIR" ]]; then
        LABBER_DESKTOP_DIR="$HOME/Desktop"
    fi

    if [[ -d "$LABBER_DESKTOP_DIR" ]]; then
        LABBER_DESKTOP_ENTRY="$LABBER_DESKTOP_DIR/Labber HDF5 Viewer.desktop"
        install -m 0755 "$LABBER_LINUX_ENTRY" "$LABBER_DESKTOP_ENTRY"
        if command -v gio >/dev/null 2>&1; then
            gio set "$LABBER_DESKTOP_ENTRY" metadata::trusted true \
                >/dev/null 2>&1 || true
        fi
        LABBER_SHORTCUT_PATH="$LABBER_DESKTOP_ENTRY"
    else
        LABBER_SHORTCUT_PATH="$LABBER_LINUX_ENTRY"
        printf 'No Desktop directory was found; an application-menu entry was created instead.\n'
    fi
fi

printf '\nInstallation completed successfully.\n'
printf 'Python: %s\n' "$LABBER_MANAGED_PYTHON_VERSION"
printf 'Application: %s\n' "$LABBER_INSTALL_ROOT"
printf 'Shortcut: %s\n' "$LABBER_SHORTCUT_PATH"
