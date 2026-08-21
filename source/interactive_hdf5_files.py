# Version 0.4.7
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import datetime
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import ttkbootstrap as ttk
from tkinterdnd2 import COPY, DND_FILES, REFUSE_DROP, TkinterDnD
import h5py
import numpy as np
import os
import shutil
from HDF5Data import HDF5Data
from interactive_plotting_tools import InteractiveArrayPlotter
from interactive_plotting_tools import InteractiveArrayAndLinePlotter
from creating_hdf5_files_from_npy_files import (
    CreateHDF5File,
    convert_npz_to_hdf5,
    get_npz_output_path,
    select_npz_field_mapping,
)
from database_browser import DatabaseBrowser
from database_manager import MeasurementDatabase

import traceback


array_plotters = []
list_name = ['Channels', 'Instrument config', 'Instruments', 'Log list', 'Settings', 'Step config', 'Step list', 'Tags', 'Views']
DEFAULT_THEME = 'bootstrap-light'
HDF5_FILE_EXTENSIONS = {'.h5', '.hdf5'}
NPZ_FILE_EXTENSION = '.npz'
GITHUB_REPOSITORY = 'Podbielsky/Interactive_Labber_Data_Viewer'
GITHUB_MAIN_BRANCH = 'main'
GITHUB_API_URL = f'https://api.github.com/repos/{GITHUB_REPOSITORY}'
GITHUB_REPOSITORY_URL = f'https://github.com/{GITHUB_REPOSITORY}'
VERSION_METADATA_FILE = 'labber_hdf5_viewer_version.json'
UPDATE_CHECK_TIMEOUT_SECONDS = 10
UPDATE_DOWNLOAD_TIMEOUT_SECONDS = 30
UPDATE_SOURCE_DIRECTORY = 'source'
UPDATE_ICON_DIRECTORY = 'icons'
UPDATE_ICON_EXTENSIONS = {'.icns', '.ico', '.png'}
PREFERENCES_FILE_NAME = 'preferences.json'
PREFERENCES_DIRECTORY_NAME = 'Labber HDF5 Viewer'
DEFAULT_APPLICATION_PREFERENCES = {
    'theme': DEFAULT_THEME,
    'database_folder': None,
}


def set_application_icon(window):
    """Set the application icon in both source and installed layouts."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_name = 'labber_viewer_ICON.png'
    icon_paths = (
        os.path.join(script_dir, 'icons', icon_name),
        os.path.join(os.path.dirname(script_dir), 'icons', icon_name),
    )

    for icon_path in icon_paths:
        if os.path.isfile(icon_path):
            try:
                icon_image = tk.PhotoImage(file=icon_path)
                window.iconphoto(True, icon_image)
                window._labber_icon_image = icon_image
            except tk.TclError:
                pass
            return


def get_preferences_path():
    """Return the user-local preferences path for the current platform."""
    user_home = os.path.expanduser('~')

    if os.name == 'nt':
        config_directory = os.environ.get('APPDATA')
        if not config_directory:
            config_directory = os.path.join(user_home, 'AppData', 'Roaming')
    elif sys.platform == 'darwin':
        config_directory = os.path.join(user_home, 'Library', 'Application Support')
    else:
        config_directory = os.environ.get('XDG_CONFIG_HOME')
        if not config_directory or not os.path.isabs(config_directory):
            config_directory = os.path.join(user_home, '.config')

    return os.path.join(
        config_directory,
        PREFERENCES_DIRECTORY_NAME,
        PREFERENCES_FILE_NAME,
    )


def load_application_preferences():
    """Load preferences, returning defaults for missing or invalid files."""
    try:
        with open(get_preferences_path(), 'r', encoding='utf-8-sig') as preferences_file:
            preferences = json.load(preferences_file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return dict(DEFAULT_APPLICATION_PREFERENCES)

    if not isinstance(preferences, dict):
        return dict(DEFAULT_APPLICATION_PREFERENCES)

    merged_preferences = dict(DEFAULT_APPLICATION_PREFERENCES)
    merged_preferences.update(preferences)
    if not isinstance(merged_preferences.get('database_folder'), (str, type(None))):
        merged_preferences['database_folder'] = None
    return merged_preferences


def save_application_preferences(updates):
    """Atomically merge application preference updates into the user JSON."""
    if not isinstance(updates, dict):
        raise ValueError('Preference updates must be provided as a dictionary.')

    preferences_path = get_preferences_path()
    preferences = load_application_preferences()
    preferences.update(updates)
    os.makedirs(os.path.dirname(preferences_path), exist_ok=True)

    temporary_path = f'{preferences_path}.{os.getpid()}.tmp'
    try:
        with open(temporary_path, 'w', encoding='utf-8') as preferences_file:
            json.dump(preferences, preferences_file, indent=2)
            preferences_file.write('\n')
        os.replace(temporary_path, preferences_path)
    except OSError:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise


def save_application_style(theme_name):
    """Persist the selected theme in the current user's preferences."""
    save_application_preferences({'theme': theme_name})


def ensure_application_preference_defaults():
    """Write newly introduced defaults once without replacing unknown keys."""
    preferences_path = get_preferences_path()
    try:
        with open(preferences_path, 'r', encoding='utf-8-sig') as preferences_file:
            stored_preferences = json.load(preferences_file)
    except FileNotFoundError:
        stored_preferences = {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return

    if not isinstance(stored_preferences, dict):
        return
    if all(key in stored_preferences for key in DEFAULT_APPLICATION_PREFERENCES):
        return
    save_application_preferences({})


def apply_saved_application_style(root):
    """Apply the saved theme when it is still available."""
    theme_name = load_application_preferences().get('theme')
    if isinstance(theme_name, str) and theme_name in root.theme_names():
        root.theme_use(theme_name)


def change_application_style(root, theme_variable, theme_name):
    """Apply a ttkbootstrap theme to the running application."""
    try:
        root.theme_use(theme_name)
        theme_variable.set(theme_name)
    except tk.TclError as error:
        messagebox.showerror(
            'Style Error',
            f'Could not apply the {theme_name!r} style:\n{error}',
            parent=root,
        )
        return

    try:
        save_application_style(theme_name)
    except OSError as error:
        messagebox.showwarning(
            'Style Not Saved',
            f'The style was applied but could not be remembered:\n{error}',
            parent=root,
        )


def add_style_menu(root, menubar):
    """Add runtime-selectable light and dark themes to the menu bar."""
    style_menu = ttk.Menu(menubar, tearoff=0)
    light_menu = ttk.Menu(style_menu, tearoff=0)
    dark_menu = ttk.Menu(style_menu, tearoff=0)
    theme_variable = tk.StringVar(master=root, value=root.theme_use())

    for theme_name in sorted(root.theme_names()):
        target_menu = dark_menu if theme_name.endswith('-dark') else light_menu
        target_menu.add_radiobutton(
            label=theme_name,
            variable=theme_variable,
            value=theme_name,
            command=lambda selected_theme=theme_name: change_application_style(
                root, theme_variable, selected_theme
            ),
        )

    style_menu.add_cascade(label='Light', menu=light_menu)
    style_menu.add_cascade(label='Dark', menu=dark_menu)
    menubar.add_cascade(label='Style', menu=style_menu)

    # Keep the Tk variable alive for as long as the application window exists.
    root._labber_theme_variable = theme_variable


def is_commit_sha(value):
    """Return whether value is a complete Git commit SHA."""
    return isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{40}', value) is not None


def get_installed_commit():
    """Read the installed commit metadata or fall back to a source checkout."""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    metadata_paths = (
        os.path.join(script_directory, VERSION_METADATA_FILE),
        os.path.join(os.path.dirname(script_directory), VERSION_METADATA_FILE),
    )

    for metadata_path in metadata_paths:
        if not os.path.isfile(metadata_path):
            continue
        try:
            with open(metadata_path, 'r', encoding='utf-8-sig') as metadata_file:
                metadata = json.load(metadata_file)
            commit = metadata.get('commit')
            if is_commit_sha(commit):
                return commit.lower()
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

    repository_directory = os.path.dirname(script_directory)
    if not os.path.exists(os.path.join(repository_directory, '.git')):
        return None

    try:
        result = subprocess.run(
            ['git', '-C', repository_directory, 'rev-parse', 'HEAD'],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    commit = result.stdout.strip()
    return commit.lower() if is_commit_sha(commit) else None


def request_github_json(url):
    """Request JSON from the public GitHub API."""
    request = urllib.request.Request(
        url,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'Labber-HDF5-Viewer',
            'X-GitHub-Api-Version': '2022-11-28',
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=UPDATE_CHECK_TIMEOUT_SECONDS,
    ) as response:
        return json.load(response)


def request_github_bytes(url):
    """Download a repository file from GitHub."""
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'Labber-HDF5-Viewer-Updater'},
    )
    with urllib.request.urlopen(
        request,
        timeout=UPDATE_DOWNLOAD_TIMEOUT_SECONDS,
    ) as response:
        return response.read()


def fetch_update_information(installed_commit):
    """Compare an installed commit with the repository's main branch."""
    if not is_commit_sha(installed_commit):
        raise ValueError('The installed version does not contain a valid commit SHA.')

    branch_commit = request_github_json(
        f'{GITHUB_API_URL}/commits/{GITHUB_MAIN_BRANCH}'
    )
    if not isinstance(branch_commit, dict):
        raise ValueError('GitHub returned an unexpected main-branch response.')

    latest_commit = branch_commit.get('sha')
    if not is_commit_sha(latest_commit):
        raise ValueError('GitHub did not return a valid main-branch commit SHA.')
    latest_commit = latest_commit.lower()
    installed_commit = installed_commit.lower()

    if installed_commit == latest_commit:
        return {
            'status': 'identical',
            'ahead_by': 0,
            'behind_by': 0,
            'total_commits': 0,
            'installed_commit': installed_commit,
            'latest_commit': latest_commit,
            'commits': [],
            'changed_files': [],
            'compare_url': f'{GITHUB_REPOSITORY_URL}/commits/{GITHUB_MAIN_BRANCH}',
        }

    compare_api_url = (
        f'{GITHUB_API_URL}/compare/{installed_commit}...{latest_commit}'
    )
    comparison = request_github_json(compare_api_url)
    if not isinstance(comparison, dict):
        raise ValueError('GitHub returned an unexpected update response.')

    commits = []
    for commit_entry in comparison.get('commits') or []:
        commit_details = commit_entry.get('commit') or {}
        author_details = commit_details.get('author') or {}
        commits.append({
            'sha': commit_entry.get('sha', ''),
            'message': commit_details.get('message', '').strip(),
            'author': author_details.get('name', 'Unknown author'),
            'date': author_details.get('date', ''),
            'url': commit_entry.get('html_url', ''),
        })

    changed_files = []
    for file_entry in comparison.get('files') or []:
        changed_files.append({
            'filename': file_entry.get('filename', ''),
            'previous_filename': file_entry.get('previous_filename', ''),
            'status': file_entry.get('status', 'modified'),
            'additions': file_entry.get('additions', 0),
            'deletions': file_entry.get('deletions', 0),
        })

    return {
        'status': comparison.get('status', 'unknown'),
        'ahead_by': int(comparison.get('ahead_by') or 0),
        'behind_by': int(comparison.get('behind_by') or 0),
        'total_commits': int(comparison.get('total_commits') or 0),
        'installed_commit': installed_commit,
        'latest_commit': latest_commit,
        'commits': commits,
        'changed_files': changed_files,
        'compare_url': comparison.get(
            'html_url',
            f'{GITHUB_REPOSITORY_URL}/compare/{installed_commit}...{GITHUB_MAIN_BRANCH}',
        ),
    }


def get_update_installation_layout(script_directory=None):
    """Resolve source, icon, and launcher locations for this installation."""
    if script_directory is None:
        script_directory = os.path.dirname(os.path.abspath(__file__))
    script_directory = os.path.abspath(script_directory)
    parent_directory = os.path.dirname(script_directory)
    uses_source_directory = os.path.basename(script_directory) == UPDATE_SOURCE_DIRECTORY

    if uses_source_directory:
        icon_directory = os.path.join(parent_directory, UPDATE_ICON_DIRECTORY)
        launcher_path = os.path.join(parent_directory, 'labber_hdf5_viewer.bat')
    else:
        icon_directory = os.path.join(parent_directory, UPDATE_ICON_DIRECTORY)
        local_icon_directory = os.path.join(script_directory, UPDATE_ICON_DIRECTORY)
        if os.path.isdir(local_icon_directory):
            icon_directory = local_icon_directory
        launcher_path = os.path.join(script_directory, 'labber_hdf5_viewer.bat')

    return {
        'application_directory': script_directory,
        'icon_directory': icon_directory,
        'launcher_path': launcher_path,
        'source_checkout': (
            uses_source_directory
            and os.path.exists(os.path.join(parent_directory, '.git'))
        ),
    }


def normalize_repository_path(repository_path):
    """Return a safe, normalized repository-relative path."""
    if not isinstance(repository_path, str):
        return None
    normalized_path = repository_path.replace('\\', '/').strip('/')
    path_parts = normalized_path.split('/')
    if not normalized_path or any(part in ('', '.', '..') for part in path_parts):
        return None
    return normalized_path


def get_update_target(repository_path, installation_layout):
    """Map an updateable repository file to its installed destination."""
    repository_path = normalize_repository_path(repository_path)
    if repository_path is None:
        return None

    path_parts = repository_path.split('/')
    if (
        len(path_parts) == 2
        and path_parts[0] == UPDATE_SOURCE_DIRECTORY
        and path_parts[1].lower().endswith('.py')
    ):
        return os.path.join(
            installation_layout['application_directory'], path_parts[1]
        )

    if (
        len(path_parts) == 2
        and path_parts[0] == UPDATE_ICON_DIRECTORY
        and os.path.splitext(path_parts[1])[1].lower() in UPDATE_ICON_EXTENSIONS
    ):
        return os.path.join(installation_layout['icon_directory'], path_parts[1])

    if repository_path == 'labber_hdf5_viewer.bat' and os.name == 'nt':
        return installation_layout['launcher_path']

    return None


def update_changes_require_setup(update_information):
    """Return whether the update changes the Python environment definition."""
    return any(
        normalize_repository_path(file_entry.get('filename')) == 'requirements.txt'
        for file_entry in update_information.get('changed_files', [])
    )


def get_update_installation_block_reason(update_information, script_directory=None):
    """Explain why this update cannot be installed automatically, if applicable."""
    if update_information.get('status') != 'ahead':
        return (
            'Automatic installation is only available when the installed '
            'version is directly behind the main branch.'
        )
    if update_changes_require_setup(update_information):
        return (
            'This update changes requirements.txt. Run the newest setup script '
            'so that Python packages and program files are updated together.'
        )
    if get_update_installation_layout(script_directory)['source_checkout']:
        return (
            'Automatic installation is disabled for Git source checkouts because '
            'it would overwrite the working tree. Update the checkout with Git, '
            'then run the setup script.'
        )
    return None


def build_update_operations(update_information, installation_layout):
    """Build deduplicated replace/delete operations for installed program files."""
    operations_by_target = {}

    def add_operation(repository_path, action):
        normalized_path = normalize_repository_path(repository_path)
        target_path = get_update_target(normalized_path, installation_layout)
        if normalized_path is None or target_path is None:
            return
        operations_by_target[os.path.normcase(target_path)] = {
            'repository_path': normalized_path,
            'target_path': target_path,
            'action': action,
        }

    for file_entry in update_information.get('changed_files', []):
        status = file_entry.get('status', 'modified')
        repository_path = file_entry.get('filename')

        if status == 'renamed':
            add_operation(file_entry.get('previous_filename'), 'delete')

        add_operation(
            repository_path,
            'delete' if status == 'removed' else 'replace',
        )

    return list(operations_by_target.values())


def get_raw_repository_file_url(commit_sha, repository_path):
    """Return the raw GitHub URL for one file at an exact commit."""
    quoted_repository = urllib.parse.quote(GITHUB_REPOSITORY, safe='/')
    quoted_commit = urllib.parse.quote(commit_sha, safe='')
    quoted_path = urllib.parse.quote(repository_path, safe='/')
    return (
        f'https://raw.githubusercontent.com/{quoted_repository}/'
        f'{quoted_commit}/{quoted_path}'
    )


def write_update_metadata(metadata_path, commit_sha, staging_directory):
    """Stage metadata identifying the exact commit being installed."""
    metadata = {
        'repository': GITHUB_REPOSITORY,
        'branch': GITHUB_MAIN_BRANCH,
        'commit': commit_sha,
        'source': 'in-application-update',
        'installed_at': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }
    staged_path = os.path.join(staging_directory, 'version-metadata.new')
    with open(staged_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
        metadata_file.write('\n')
    return {
        'repository_path': VERSION_METADATA_FILE,
        'target_path': metadata_path,
        'action': 'replace',
        'staged_path': staged_path,
    }


def apply_staged_update_operations(operations, staging_directory):
    """Apply staged files atomically and roll back the transaction on failure."""
    backup_directory = os.path.join(staging_directory, 'backups')
    os.makedirs(backup_directory, exist_ok=True)
    applied_operations = []

    try:
        for index, operation in enumerate(operations):
            target_path = operation['target_path']
            target_directory = os.path.dirname(target_path)
            os.makedirs(target_directory, exist_ok=True)

            target_existed = os.path.lexists(target_path)
            backup_path = None
            if target_existed:
                if not os.path.isfile(target_path):
                    raise OSError(f'Update target is not a regular file: {target_path}')
                backup_path = os.path.join(backup_directory, f'{index:04d}.backup')
                shutil.copy2(target_path, backup_path)

            if operation['action'] == 'delete':
                if target_existed:
                    os.remove(target_path)
            else:
                os.replace(operation['staged_path'], target_path)

            applied_operations.append((target_path, backup_path))
    except Exception:
        for target_path, backup_path in reversed(applied_operations):
            try:
                if backup_path is None:
                    if os.path.isfile(target_path):
                        os.remove(target_path)
                else:
                    os.replace(backup_path, target_path)
            except OSError:
                pass
        raise


def install_program_update(update_information, script_directory=None):
    """Download and transactionally install program files from one commit."""
    latest_commit = update_information.get('latest_commit')
    if not is_commit_sha(latest_commit):
        raise ValueError('The update does not contain a valid commit SHA.')
    latest_commit = latest_commit.lower()

    installation_layout = get_update_installation_layout(script_directory)
    block_reason = get_update_installation_block_reason(
        update_information, script_directory
    )
    if block_reason is not None:
        raise RuntimeError(block_reason)

    operations = build_update_operations(update_information, installation_layout)
    entry_script_path = os.path.join(
        installation_layout['application_directory'],
        os.path.basename(os.path.abspath(__file__)),
    )
    if any(
        operation['target_path'] == entry_script_path
        and operation['action'] == 'delete'
        for operation in operations
    ):
        raise RuntimeError('The update removes the application entry point.')

    staging_directory = tempfile.mkdtemp(
        prefix='.labber-update-',
        dir=installation_layout['application_directory'],
    )
    try:
        for index, operation in enumerate(operations):
            if operation['action'] == 'delete':
                continue

            repository_path = operation['repository_path']
            file_data = request_github_bytes(
                get_raw_repository_file_url(latest_commit, repository_path)
            )
            if repository_path.lower().endswith('.py'):
                compile(file_data, repository_path, 'exec')

            staged_path = os.path.join(staging_directory, f'{index:04d}.new')
            with open(staged_path, 'wb') as staged_file:
                staged_file.write(file_data)
            operation['staged_path'] = staged_path

        metadata_path = os.path.join(
            installation_layout['application_directory'], VERSION_METADATA_FILE
        )
        operations.append(
            write_update_metadata(
                metadata_path, latest_commit, staging_directory
            )
        )
        apply_staged_update_operations(operations, staging_directory)
    finally:
        shutil.rmtree(staging_directory, ignore_errors=True)

    return [
        operation['repository_path']
        for operation in operations
        if operation['repository_path'] != VERSION_METADATA_FILE
    ]


def format_update_report(update_information):
    """Format commit messages and changed files for the update dialog."""
    installed_commit = update_information['installed_commit']
    latest_commit = update_information['latest_commit']
    ahead_by = update_information['ahead_by']
    commits = update_information['commits']
    changed_files = update_information['changed_files']

    lines = [
        f'{ahead_by} newer commit(s) are available on {GITHUB_MAIN_BRANCH}.',
        '',
        f'Installed commit: {installed_commit}',
        f'Latest main commit: {latest_commit}',
        '',
        'Commits:',
    ]

    for index, commit in enumerate(commits, start=1):
        commit_sha = commit['sha'][:7] if commit['sha'] else 'unknown'
        commit_date = commit['date'][:10] if commit['date'] else 'unknown date'
        message_lines = commit['message'].splitlines() or ['No commit message']
        lines.append(
            f'{index}. {commit_sha} — {message_lines[0]} '
            f'({commit["author"]}, {commit_date})'
        )
        for message_line in message_lines[1:]:
            if message_line.strip():
                lines.append(f'   {message_line}')

    omitted_commits = max(0, ahead_by - len(commits))
    if omitted_commits:
        lines.extend([
            '',
            f'{omitted_commits} additional commit(s) are available on GitHub.',
        ])

    if changed_files:
        lines.extend(['', 'Changed files:'])
        for file_entry in changed_files:
            lines.append(
                f'- {file_entry["filename"]} [{file_entry["status"]}; '
                f'+{file_entry["additions"]}/-{file_entry["deletions"]}]'
            )

    return '\n'.join(lines)


def show_update_dialog(root, update_information):
    """Display available commits and files in a scrollable dialog."""
    dialog = ttk.Toplevel(root)
    dialog.title('Labber HDF5 Viewer Update Available')
    dialog.geometry('780x610')
    dialog.minsize(620, 420)
    dialog.transient(root)
    set_application_icon(dialog)

    ttk.Label(
        dialog,
        text='A newer version is available',
        font=('TkDefaultFont', 14, 'bold'),
        bootstyle='primary',
    ).pack(anchor='w', padx=16, pady=(16, 8))

    report_frame = ttk.Frame(dialog)
    report_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
    report_frame.rowconfigure(0, weight=1)
    report_frame.columnconfigure(0, weight=1)

    report_text = ttk.Text(report_frame, wrap=tk.WORD)
    report_scrollbar = ttk.Scrollbar(
        report_frame,
        orient=tk.VERTICAL,
        command=report_text.yview,
    )
    report_text.configure(yscrollcommand=report_scrollbar.set)
    report_text.grid(row=0, column=0, sticky='nsew')
    report_scrollbar.grid(row=0, column=1, sticky='ns')
    report_text.insert(tk.END, format_update_report(update_information))
    report_text.configure(state=tk.DISABLED)

    installation_block_reason = get_update_installation_block_reason(
        update_information
    )
    status_variable = tk.StringVar(
        master=dialog,
        value=(
            installation_block_reason
            or 'The update can be installed directly into this application.'
        ),
    )
    ttk.Label(
        dialog,
        textvariable=status_variable,
        bootstyle='warning' if installation_block_reason else 'secondary',
        wraplength=730,
    ).pack(fill=tk.X, padx=16, pady=(4, 2))

    progress_bar = ttk.Progressbar(dialog, mode='indeterminate')
    progress_bar.pack(fill=tk.X, padx=16, pady=(0, 8))

    button_frame = ttk.Frame(dialog)
    button_frame.pack(fill=tk.X, padx=16, pady=(0, 16))
    ttk.Button(
        button_frame,
        text='Open comparison on GitHub',
        command=lambda: webbrowser.open(update_information['compare_url']),
        bootstyle='primary',
    ).pack(side=tk.LEFT)
    install_button = ttk.Button(
        button_frame,
        text='Install update and restart',
        bootstyle='success',
    )
    install_button.pack(side=tk.LEFT, padx=(8, 0))
    close_button = ttk.Button(
        button_frame,
        text='Close',
        command=dialog.destroy,
        bootstyle='secondary',
    )
    close_button.pack(side=tk.RIGHT)
    install_button.configure(
        command=lambda: install_update_from_dialog(
            root,
            dialog,
            update_information,
            install_button,
            close_button,
            progress_bar,
            status_variable,
        )
    )
    if installation_block_reason:
        install_button.configure(state=tk.DISABLED)


def restart_application(root):
    """Close this process and launch the updated entry script."""
    script_directory = os.path.dirname(os.path.abspath(__file__))
    restart_command = [
        sys.executable,
        os.path.abspath(__file__),
        *sys.argv[1:],
    ]
    helper_code = (
        'import json, subprocess, sys, time\n'
        'time.sleep(1.0)\n'
        'subprocess.Popen(json.loads(sys.argv[1]), '
        'cwd=sys.argv[2], close_fds=True)\n'
    )
    helper_command = [
        sys.executable,
        '-c',
        helper_code,
        json.dumps(restart_command),
        script_directory,
    ]
    popen_options = {
        'cwd': script_directory,
        'close_fds': True,
    }
    if os.name == 'nt':
        popen_options['creationflags'] = (
            getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
            | getattr(subprocess, 'DETACHED_PROCESS', 0)
        )
    else:
        popen_options['start_new_session'] = True

    try:
        subprocess.Popen(helper_command, **popen_options)
    except OSError as error:
        messagebox.showerror(
            'Restart Failed',
            f'The update was installed, but the viewer could not restart:\n{error}\n\n'
            'Please start the viewer again manually.',
            parent=root,
        )
        return

    close_application = getattr(root, '_labber_close_application', root.destroy)
    close_application()


def install_update_from_dialog(
    root,
    dialog,
    update_information,
    install_button,
    close_button,
    progress_bar,
    status_variable,
):
    """Install an available update in a worker and restart after success."""
    if getattr(root, '_labber_update_install_running', False):
        return

    if not messagebox.askyesno(
        'Install Update',
        'Download the changed program files, install them, and restart the viewer?',
        parent=dialog,
    ):
        return

    root._labber_update_install_running = True
    root.configure(cursor='watch')
    install_button.configure(state=tk.DISABLED)
    close_button.configure(state=tk.DISABLED)
    dialog.protocol('WM_DELETE_WINDOW', lambda: None)
    status_variable.set('Downloading and validating the update...')
    progress_bar.start(12)
    result_queue = queue.Queue(maxsize=1)

    def update_worker():
        try:
            updated_paths = install_program_update(update_information)
            result_queue.put(('success', updated_paths))
        except Exception as error:
            result_queue.put(('error', error))

    def finish_update_installation():
        try:
            result_type, result = result_queue.get_nowait()
        except queue.Empty:
            try:
                root.after(100, finish_update_installation)
            except tk.TclError:
                pass
            return

        root._labber_update_install_running = False
        root.configure(cursor='')
        progress_bar.stop()

        if result_type == 'error':
            install_button.configure(state=tk.NORMAL)
            close_button.configure(state=tk.NORMAL)
            dialog.protocol('WM_DELETE_WINDOW', dialog.destroy)
            status_variable.set('The update was not installed.')
            messagebox.showerror(
                'Update Installation Failed',
                f'Could not install the update:\n{result}',
                parent=dialog,
            )
            return

        file_count = len(result)
        if file_count:
            status_variable.set(
                f'Installed {file_count} changed program file(s). Restarting...'
            )
        else:
            status_variable.set(
                'No runtime files changed; the installed version was updated. '
                'Restarting...'
            )
        root.after(500, lambda: restart_application(root))

    threading.Thread(target=update_worker, daemon=True).start()
    root.after(100, finish_update_installation)


def check_for_updates(root, silent_if_current=False):
    """Check GitHub in the background and report main-branch updates."""
    if getattr(root, '_labber_update_check_running', False):
        if not silent_if_current:
            messagebox.showinfo(
                'Update Check',
                'An update check is already running.',
                parent=root,
            )
        return

    installed_commit = get_installed_commit()
    if installed_commit is None:
        if not silent_if_current:
            messagebox.showwarning(
                'Installed Version Unknown',
                'The installed commit could not be determined. Run the current '
                'setup script once to create version metadata.',
                parent=root,
            )
        return

    result_queue = queue.Queue(maxsize=1)
    root._labber_update_check_running = True
    if not silent_if_current:
        root.configure(cursor='watch')

    def update_worker():
        try:
            result_queue.put(('success', fetch_update_information(installed_commit)))
        except Exception as error:
            result_queue.put(('error', error))

    def finish_update_check():
        try:
            result_type, result = result_queue.get_nowait()
        except queue.Empty:
            root.after(100, finish_update_check)
            return

        root._labber_update_check_running = False
        if not silent_if_current:
            root.configure(cursor='')

        if result_type == 'error':
            if not silent_if_current:
                if isinstance(result, urllib.error.HTTPError) and result.code == 403:
                    error_message = (
                        'GitHub temporarily rejected the request. The public API '
                        'rate limit may have been reached.'
                    )
                else:
                    error_message = f'Could not check for updates:\n{result}'
                messagebox.showerror('Update Check Failed', error_message, parent=root)
            return

        if result['ahead_by'] > 0:
            show_update_dialog(root, result)
        elif not silent_if_current:
            if result['status'] == 'behind':
                status_message = (
                    'This installation contains a commit newer than or outside '
                    'the current main branch.'
                )
            else:
                status_message = 'You are using the newest version from the main branch.'
            messagebox.showinfo('No Updates Available', status_message, parent=root)

    threading.Thread(target=update_worker, daemon=True).start()
    root.after(100, finish_update_check)


def load_configured_database(root):
    """Attach the configured database when its portable index still exists."""
    root._labber_database = None
    root._labber_database_browsers = []
    database_folder = load_application_preferences().get('database_folder')
    if not database_folder:
        return None

    normalized_folder = os.path.abspath(os.path.expanduser(database_folder))
    if MeasurementDatabase.is_initialized(normalized_folder):
        root._labber_database = MeasurementDatabase(normalized_folder)
    return root._labber_database


def update_database_menu_state(root):
    """Enable database browsing only when an index is configured."""
    file_menu = getattr(root, '_labber_file_menu', None)
    menu_index = getattr(root, '_labber_browse_database_menu_index', None)
    if file_menu is None or menu_index is None:
        return
    state = tk.NORMAL if getattr(root, '_labber_database', None) else tk.DISABLED
    file_menu.entryconfigure(menu_index, state=state)


def refresh_open_database_browsers(root):
    """Refresh browser trees after a managed drag-and-drop import."""
    browsers = getattr(root, '_labber_database_browsers', [])
    active_browsers = []
    for browser in browsers:
        if not browser.closed and browser.window.winfo_exists():
            browser.refresh_records()
            active_browsers.append(browser)
    root._labber_database_browsers = active_browsers


def open_database_browser(root):
    """Open the optional database overview window."""
    database = getattr(root, '_labber_database', None)
    if database is None:
        messagebox.showinfo(
            'No Database Configured',
            'Use File → Set Database Folder… before browsing measurements.',
            parent=root,
        )
        return None
    open_file_callback = getattr(root, '_labber_open_hdf5_path', None)
    if open_file_callback is None:
        messagebox.showerror(
            'File Viewer Not Ready',
            'The main HDF5 file view is not ready yet.',
            parent=root,
        )
        return None

    def unregister_browser(browser):
        browsers = getattr(root, '_labber_database_browsers', [])
        root._labber_database_browsers = [
            candidate for candidate in browsers if candidate is not browser
        ]

    browser = DatabaseBrowser(
        root,
        database,
        open_file_callback,
        icon_callback=set_application_icon,
        on_close=unregister_browser,
    )
    root._labber_database_browsers.append(browser)
    return browser


def set_database_folder(root):
    """Create/open a managed database folder and index it in a worker."""
    if getattr(root, '_labber_database_scan_running', False):
        messagebox.showinfo(
            'Database Scan in Progress',
            'Please wait for the current database scan to finish.',
            parent=root,
        )
        return

    selected_folder = filedialog.askdirectory(
        title='Select or Create a Measurement Database Folder',
        parent=root,
    )
    if not selected_folder:
        return

    database = MeasurementDatabase(selected_folder)
    try:
        database.initialize()
        save_application_preferences(
            {'database_folder': database.root_directory}
        )
    except (OSError, ValueError) as error:
        messagebox.showerror(
            'Could Not Create Database', str(error), parent=root
        )
        return

    for browser in list(getattr(root, '_labber_database_browsers', [])):
        if not browser.closed:
            browser.close()
    root._labber_database = database
    update_database_menu_state(root)

    progress_window = ttk.Toplevel(root)
    progress_window.title('Index Measurement Database')
    progress_window.geometry('520x165')
    progress_window.resizable(False, False)
    progress_window.transient(root)
    set_application_icon(progress_window)
    status_variable = tk.StringVar(value='Searching for HDF5 measurements…')
    ttk.Label(
        progress_window,
        textvariable=status_variable,
        wraplength=480,
    ).pack(fill=tk.X, padx=20, pady=(18, 8))
    progress_variable = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(
        progress_window,
        variable=progress_variable,
        maximum=1,
        mode='determinate',
    )
    progress_bar.pack(fill=tk.X, padx=20, pady=6)

    cancel_event = threading.Event()
    cancel_button = ttk.Button(
        progress_window,
        text='Cancel',
        bootstyle='secondary',
    )
    cancel_button.pack(pady=(5, 12))
    result_queue = queue.Queue()
    root._labber_database_scan_running = True

    def cancel_scan():
        cancel_event.set()
        cancel_button.configure(state=tk.DISABLED)
        status_variable.set('Cancelling after the current file…')

    cancel_button.configure(command=cancel_scan)
    progress_window.protocol('WM_DELETE_WINDOW', cancel_scan)

    def report_progress(processed, total, path):
        result_queue.put(('progress', processed, total, path))

    def scan_worker():
        try:
            result_queue.put(
                (
                    'success',
                    database.scan(
                        progress_callback=report_progress,
                        cancel_event=cancel_event,
                    ),
                )
            )
        except Exception as error:
            result_queue.put(('error', error))

    def finish_scan():
        try:
            while True:
                result = result_queue.get_nowait()
                if result[0] == 'progress':
                    _, processed, total, path = result
                    progress_bar.configure(maximum=max(1, total))
                    progress_variable.set(processed)
                    status_variable.set(
                        f'Indexing {processed}/{total}: '
                        f'{os.path.basename(path) if path else ""}'
                    )
                    continue

                root._labber_database_scan_running = False
                progress_window.destroy()
                if result[0] == 'error':
                    messagebox.showerror(
                        'Database Scan Failed', str(result[1]), parent=root
                    )
                    return

                scan_result = result[1]
                refresh_open_database_browsers(root)
                if scan_result.cancelled:
                    messagebox.showinfo(
                        'Database Scan Cancelled',
                        'The database folder remains configured. Run Browse '
                        'Database → Rescan Database to index it later.',
                        parent=root,
                    )
                    return

                summary = (
                    f'Indexed {scan_result.indexed} valid measurement(s) from '
                    f'{scan_result.discovered} HDF5 file(s).'
                )
                if scan_result.invalid:
                    summary += (
                        f'\n\nSkipped {len(scan_result.invalid)} incompatible '
                        'or unreadable file(s).'
                    )
                messagebox.showinfo('Database Ready', summary, parent=root)
                return
        except queue.Empty:
            pass

        try:
            root.after(100, finish_scan)
        except tk.TclError:
            pass

    threading.Thread(target=scan_worker, daemon=True).start()
    root.after(100, finish_scan)


def data_menu_bar(root, hdf5data):
    menubar = ttk.Menu(root)
    # Adding File Menu and commands
    file = ttk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='File', menu=file)
    file.add_command(label='Select File Directory', command=lambda: get_path(hdf5data))
    file.add_command(label='Move File to', command=lambda: move_data(hdf5data))
    file.add_command(label='Save File as', command=lambda: save_data_as(hdf5data))
    file.add_command(label='Create a HDF5 File from Numpy Files', command=lambda : create_hdf5_files_from_npy(root))
    file.add_separator()
    file.add_command(
        label='Set Database Folder…',
        command=lambda: set_database_folder(root),
    )
    file.add_command(
        label='Browse Database…',
        command=lambda: open_database_browser(root),
    )
    root._labber_file_menu = file
    root._labber_browse_database_menu_index = file.index('end')
    update_database_menu_state(root)
    file.add_separator()
    file.add_command(label='Remove Selected Datasets', command=lambda: remove_selected_options_window(root, hdf5data)) #Hannah Vogel: to select datasets to be removed
    file.add_separator()
    file.add_command(label='Add Traces from HDF5 File', command=lambda: add_traces_window(hdf5data)) # Nico Reinders: to add traces to current file from another HDF5 file
    file.add_command(label='Generate Traces from Dataset', command=lambda: transform_traces_window(hdf5data)) # Nico Reinders: create a file with a 'Traces' group that is compatible with the interactive data viewer 
    
    data = ttk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Data', menu=data)
    data.add_command(label='Save Data as Numpy Array', command=lambda: create_data_array(hdf5data))
    data.add_command(label='Save Traces as Numpy Arrays', command=lambda: create_trace_array(hdf5data))
    
    
    data.add_command(label='Calculate Histograms', command=lambda: create_hist_data(hdf5data))
    

    data.add_separator()
    plotting = ttk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label='Plotting', menu=plotting)
    plotting.add_command(label='Plot Map', command=lambda: plot_array(hdf5data, root))
    plotting.add_command(label='Plot Map with Trace Data', command=lambda: plot_array_with_trace_data(hdf5data, root))
    plotting.add_separator()

    add_style_menu(root, menubar)

    help_menu = ttk.Menu(menubar, tearoff=0)
    help_menu.add_command(
        label='Check for Updates...',
        command=lambda: check_for_updates(root),
    )
    menubar.add_cascade(label='Help', menu=help_menu)

    return menubar


def set_hdf5_path(hdf5Data, path):
    """Validate an HDF5 file and make it the viewer's current file."""
    normalized_path = os.path.abspath(os.path.expanduser(path))
    extension = os.path.splitext(normalized_path)[1].lower()

    if extension not in HDF5_FILE_EXTENSIONS:
        raise ValueError('Only .hdf5 and .h5 files can be opened.')
    if not os.path.isfile(normalized_path):
        raise ValueError(f'The dropped file does not exist:\n{normalized_path}')
    if not h5py.is_hdf5(normalized_path):
        raise ValueError(f'The selected file is not a valid HDF5 file:\n{normalized_path}')

    hdf5Data.set_path(normalized_path, 'r')
    hdf5Data.set_filename()
    hdf5Data.vars = []
    return normalized_path


def get_path(hdf5Data):
    """Select and validate an HDF5 file using the system file dialog."""
    path = filedialog.askopenfilename(
        filetypes=[('HDF5 files', ('*.hdf5', '*.h5'))]
    )
    if not path:
        return None

    try:
        return set_hdf5_path(hdf5Data, path)
    except (OSError, ValueError) as error:
        messagebox.showerror('Invalid HDF5 File', str(error))
        return None

def get_unique_filename(filepath):
    #Returns a unique filename by appending a number if the file already exists
    base, extension = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath

    while os.path.exists(new_filepath):
        new_filepath = f"{base}({counter}){extension}"
        counter += 1

    return new_filepath


def apply_reshape(selected_dataset, selected_axis_dataset, dimension_index):
    """
    Added by Nico Reinders
    Reshapes the selected dataset to be compatible with the requirements of the data viewer "Plot Map with Trace Data"
    by moving the specified dimension to the first position and reshaping it.
    
    """
    if type(selected_dataset) is not np.ndarray:
        selected_dataset = np.array(selected_dataset[:])
        
    if len(np.shape(selected_dataset)) == 2:
        selected_dataset = np.array([selected_dataset])
        dimension_index += 1

    if selected_axis_dataset is None:  # if no dataset is selected for the x-axis, use default values
        t0, dt = 0, 1
        print("No axis dataset selected, using default t0=0 and dt=1.")
    else:
        # Always materialize axis data into a NumPy array first
        if isinstance(selected_axis_dataset, h5py.Dataset):
            arr = np.asarray(selected_axis_dataset[()])
        else:
            arr = np.asarray(selected_axis_dataset)

        arr = np.ravel(arr)  # flatten

        if arr.ndim == 1 and arr.size >= 2:  # Check if the axis dataset is 1D and has at least 2 elements
            t0 = arr[0]
            diffs = np.diff(arr)
            dt = np.min(np.abs(diffs))
        else:  # if the axis dataset has unusable shape, use default values
            t0, dt = 0, 1
            print("Warning: Selected axis dataset is not 1D or too short, using default t0=0 and dt=1.")

    shape_original = selected_dataset.shape
    print(f"Original Shape of spectra: {shape_original}")

    # Keep a copy of the original spectra for mean calculation
    spectra_original = selected_dataset.copy()

    selected_dataset = np.moveaxis(selected_dataset, dimension_index, 0)  # Move the selected axis to the first position
    selected_dataset = np.reshape(selected_dataset, (selected_dataset.shape[0], 1, -1))  # Reshape to required shape
    shape = selected_dataset.shape
    print(f"Shape of spectra: {shape}")
        
    # Validate shape_original dimensions
    if len(shape_original) < 2:
        raise ValueError("The selected dataset must have at least two dimensions.")

    # Adjust data_data shape to account for dimension_index
    reduced_shape = list(shape_original)
    reduced_shape.pop(dimension_index)  # Remove the selected dimension

    data_data = np.zeros((reduced_shape[0], 3, reduced_shape[1]))

    # Assign values to data_data with explicit broadcasting
    data_data[:, 0, :] = np.broadcast_to(np.arange(reduced_shape[0])[:, None], (reduced_shape[0], reduced_shape[1]))
    data_data[:, 1, :] = np.broadcast_to(np.arange(reduced_shape[1])[None, :], (reduced_shape[0], reduced_shape[1]))
    data_data[:, 2, :] = np.mean(spectra_original, axis=dimension_index).reshape(reduced_shape)  # Reshape mean result to match reduced dimensions

    print(f"Shape of data_data: {data_data.shape}")

    output_path = filedialog.asksaveasfilename(defaultextension=".hdf5", filetypes=[("HDF5 files", "*.hdf5")], title="Save HDF5 file as...")

    if output_path:
        print("Saving traces file to:", output_path)
    else:
        print("User cancelled")

    # Save the reshaped data and traces in the output file
    with h5py.File(output_path, 'w') as out_file:
        traces_grp = out_file.create_group('Traces', track_order=True)
        traces_grp.create_dataset('Data', data=selected_dataset)
        traces_grp.create_dataset('Data_N', data=[shape[0]])
        traces_grp.create_dataset('Alazar Slytherin - Ch1 - Data_t0dt', data=[[t0, dt]])

        data_grp = out_file.create_group('Data')
        data_grp.attrs['Step dimensions'] = [reduced_shape[0], reduced_shape[1]]
        data_grp.attrs['Step index'] = [0, 1]
        data_grp.create_dataset('Data', data=data_data)
        data_grp.create_dataset('Channel names', data=[(b'Axis 1', b''), (b'Axis 2', b''), (b'Channel 1', b'')])
        out_file.create_dataset('Log list', data=[(b'Channel 1', b'')])

        out_file.close()
    
    
def transform_traces_window(hdf5Data):
    '''
    Added by Nico Reinders
    Opens a window to select a dataset to reshape into traces.
    '''
    
    pth = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5"), ("Numpy files", "*.npy")])
    if not pth:
        print("No file selected. Operation cancelled.")
        return
    root, ext = os.path.splitext(pth)
    if ext == '.hdf5':
        print('hdf5')
        reshape_hdf5Data = HDF5Data(wdir=pth)
        reshape_hdf5Data.set_path(pth, 'r')
        # Open file and keep it open for the window lifetime
        reshape_hdf5Data.file = h5py.File(pth, 'r')
    elif ext == '.npy':
        print('npy') 
        arr = np.load(pth, allow_pickle=True)
        print("Select axis numpy file for traces if needed.")
        axis_path = filedialog.askopenfilename(filetypes=[("Numpy files", "*.npy")])
        if not axis_path:
            axis_arr = None
        else: 
            axis_arr = np.load(axis_path, allow_pickle=True)
    else:
        messagebox.showerror("Invalid File", "Please select a valid HDF5 or Numpy (.npy) file.")
        return
    
    def check_axis_reshape_requirements(selected_item):
        # Check if the selected item is a valid dataset as x-axis for traces
        if not isinstance(selected_item, h5py.Dataset):
            # print("Selected item is not a dataset.")
            return False        
        elif len(selected_item.shape) != 1:
            # print("Selected dataset does not have 1 dimension.")
            return False
        elif selected_item.shape[0] < 2:
            # print("Selected dataset is too short, must have at least 2 elements.")
            return False
        else: 
            return True
    
    def check_dataset_reshape_requirements(selected_item):
        # Check if the selected item is a valid dataset for reshaping to traces
        if selected_item is None:
            print("No item selected.")
            return False
        elif not isinstance(selected_item, h5py.Dataset):
            print("Selected item is not a dataset.")
            return False        
        elif len(selected_item.shape) not in (2, 3):
            print("Selected dataset does not have 2 or 3 dimensions.")
            return False
        else: 
            return True


    def on_var_change(*args): 
        # Update the labels and dimension index based on the selected datasets
        selected_dataset = dataset_map[dataset_selection.get()] if dataset_selection.get() in dataset_map else None
        selected_axis_dataset = axis_map[axis_selection.get()] if axis_selection.get() in axis_map else None
        dataset_label_text.set(f"{selected_dataset if selected_dataset is not None else ''}")
        axis_label_text.set(f"{selected_axis_dataset if selected_axis_dataset is not None else ''}")
        
        if selected_axis_dataset is not None and np.array(selected_axis_dataset[:]).ndim == 1:
            dim = np.shape(selected_dataset).index(len(selected_axis_dataset))
        else:
            dim = 0
        dimension_index.set(dim)

    
            
    transform_options = ttk.Toplevel()

    def on_close_transform_options():
        try:
            if reshape_hdf5Data.file:
                reshape_hdf5Data.file.close()
        except Exception:
            pass
        transform_options.destroy()
  
    
    transform_options.protocol("WM_DELETE_WINDOW", on_close_transform_options)

    
    # Frame for dataset labels
    label_frame = ttk.Frame(transform_options)
    label_frame.pack(anchor='w', pady=5, padx=5, fill='x')

    # Store the valid datasets and axis datasets in dictionaries
    if ext == '.hdf5':
        dataset_map = {}
        axis_map = {}
        file = reshape_hdf5Data.file
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and check_dataset_reshape_requirements(obj):
                dataset_map[name] = obj  # name is the full HDF5 path
        file.visititems(visitor)

        def visitor_axis(name, obj):
            if isinstance(obj, h5py.Dataset) and check_axis_reshape_requirements(obj):
                axis_map[name] = obj  # name is the full HDF5 path
        file.visititems(visitor_axis)
        
        axis_map['None'] = None  # Add a 'None' option for no axis dataset

        datasets_names = list(dataset_map.keys())
        axis_names = list(axis_map.keys())

        dataset_selection = tk.StringVar(value=datasets_names[0] if datasets_names else "")  # default selection
        database_combo = ttk.Combobox(label_frame, textvariable=dataset_selection, values=datasets_names, state="readonly")
        

        axis_selection = tk.StringVar(value=axis_names[0] if axis_names else "")  # default selection
        axis_combo = ttk.Combobox(label_frame, textvariable=axis_selection, values=axis_names, state="readonly")
        
        dataset_selection.trace_add("write", on_var_change)
        axis_selection.trace_add("write", on_var_change)

        
        dataset_label_text = tk.StringVar(value=f"{dataset_map[dataset_selection.get()] if dataset_selection.get() in dataset_map else ''}")
        axis_label_text = tk.StringVar(value=f"{axis_map[axis_selection.get()] if axis_selection.get() in axis_map else ''}")

        database_combo.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        axis_combo.grid(row=1, column=1, padx=10, pady=10, sticky='w')
        
        ttk.Label(label_frame, textvariable=dataset_label_text).grid(row=0, column=2, padx=10, pady=5, sticky='w')
        ttk.Label(label_frame, textvariable=axis_label_text).grid(row=1, column=2, padx=10, pady=5, sticky='w')
        
        ttk.Label(label_frame, text="Selected Dataset:").grid(row=0, column=0, padx=10, pady=5, sticky='e')
        ttk.Label(label_frame, text="Selected Axis Dataset:").grid(row=1, column=0, padx=10, pady=5, sticky='e')
    else:
        ttk.Label(label_frame, text=f"Numpy file shape: {arr.shape}").pack(anchor='w')
    # Frame for spinbox + label
    spin_frame = ttk.Frame(transform_options)
    
    spin_frame.pack(anchor='w', pady=5, padx=5, fill='x')

    ttk.Label(spin_frame, text="Index of dimension in selected dataset to be used as x axis:").pack(side='left', padx=(0, 5))

    def validate_int(new_value):
        # validate the spinbox input
        if new_value == "":  # allow empty (so user can type)
            return True
        try:
            value = int(new_value)
        except ValueError:
            return False
        return 0 <= value <= 2
    
    vcmd = (transform_options.register(validate_int), '%P')  # %P is the new value of the spinbox    
        
    dimension_index = tk.IntVar(value=0)  # Default to 0
    
    # add a spinbox to select the dimension that will be used as trace length
    ttk.Spinbox(spin_frame, from_=0, to=2, increment=1, width=5, textvariable=dimension_index, validate="key", validatecommand=vcmd).pack(side='left')

    if ext == '.hdf5':
        on_var_change()  # Initial call to set labels

    # Buttons frame
    button_frame = ttk.Frame(transform_options)
    button_frame.pack(pady=10)

    if ext == '.hdf5':
        confirm_button = ttk.Button(
            button_frame,
            text="Confirm Reshape",
            command=lambda: (apply_reshape(dataset_map[dataset_selection.get()], axis_map[axis_selection.get()], int(dimension_index.get())), transform_options.destroy())
        )
    else:
        confirm_button = ttk.Button(
            button_frame,
            text="Confirm Reshape",
            command=lambda: (apply_reshape(arr, axis_arr, int(dimension_index.get())), transform_options.destroy())
        )
    confirm_button.pack(side='left', padx=5)

    cancel_button = ttk.Button(button_frame, text="Cancel", command=transform_options.destroy, bootstyle='secondary')
    cancel_button.pack(side='left', padx=5)
    
    
    
    
def add_traces_window(hdf5Data):
    """
    Added by Nico Reinders
    Opens a window to select a file to copy groups or datasets from
    Then shows the content of the file in a treeview
    Allows the user to select a group or dataset and copy it to the current hdf5 file
    """
    
    pth = filedialog.askopenfilename(filetypes=[("HDF5 files", "*.hdf5")])
    traces_hdf5Data = HDF5Data(wdir=pth)
    traces_hdf5Data.set_path(pth, 'r')
    
    # open treeview window
    traces_selection_window = ttk.Toplevel()
    traces_selection_window.title('Add Traces from HDF5 File')
        
    #add an entry for the group name in the destination file
    group_frame = ttk.Frame(traces_selection_window)
    group_frame.pack(pady=5)
    ttk.Label(group_frame, text="Destination group name:").pack(side=tk.LEFT)
    group_name_var = tk.StringVar(value='Traces')
    group_name_entry = ttk.Entry(group_frame, textvariable=group_name_var, width=30)
    group_name_entry.pack(side=tk.LEFT, padx=5)

    # show treeview of the source file
    traces_tree = display_hdf5_file(traces_selection_window, traces_hdf5Data)

    def copy_selected_dataset():
        """
        Copies the selected dataset or group from the source file traces_hdf5Data to the current hdf5 file.
        """
        
        traces_hdf5Data.set_data()
        selected_items = traces_tree.selection()
        
        if not selected_items:
            print("No Selection", "Please select a dataset or group to copy.")
            return
        selected_item = selected_items[0]
        values_above = get_values_above_clicked_node(selected_item, traces_tree)
        file_dir_sep = '/'
        file_dir = file_dir_sep.join(values_above)
        dest_group = group_name_var.get().strip() or 'Traces'
        # Always use string keys for h5py access
        try:
            h5obj = traces_hdf5Data.file[file_dir]
        except Exception as e:
            print("Error", f"Could not access {file_dir}: {e}")
            return
        
        if isinstance(h5obj, h5py.Group):
            trace_names = [str(name) for name in h5obj.keys()]
            traces = [h5obj[str(trace)] for trace in trace_names]
            # Add datasets to group if it exists, else create group
            with h5py.File(hdf5Data.readpath, 'r+') as dest_file:
                if dest_group in dest_file:
                    group = dest_file[dest_group]
                else:
                    group = dest_file.create_group(dest_group)
                for trace_name, trace in zip(trace_names, traces):
                    if trace_name in group:
                        del group[trace_name]
                    group.create_dataset(trace_name, data=trace[()])
            hdf5Data.set_data()
        elif isinstance(h5obj, h5py.Dataset):
            import time
            trace_name = values_above[-1]
            with h5py.File(hdf5Data.readpath, 'r+') as dest_file:
                t0 = time.time()
                recreate_group = False
                # Always use absolute group path, never nest
                if dest_group in dest_file:
                    old_group = dest_file[dest_group]
                    # Only recreate if track_order is not already True
                    track_order = getattr(old_group, 'track_order', None)
                    if not track_order:
                        recreate_group = True
                if recreate_group:
                    print(f"Recreating group '{dest_group}' with track_order=True to avoid nesting.")
                    new_group = dest_file.create_group('dest_group_tmp', track_order=True)
                    dest_file.copy(old_group, new_group)
                    del dest_file[old_group.name]
                    dest_file.move('dest_group_tmp', dest_group)
                    group = dest_file[dest_group]  # Always re-fetch from root
                elif dest_group in dest_file:
                    group = dest_file[dest_group]
                else:
                    group = dest_file.create_group(dest_group, track_order=True)
                t1 = time.time()
                # Save the dataset directly in the destination group, not as a subgroup
                if trace_name in group:
                    del group[trace_name]
                group.create_dataset(trace_name, data=h5obj[()])
                t2 = time.time()
                print(f"Dataset copy timings: group_prep={t1-t0:.3f}s, create={t2-t1:.3f}s, total={t2-t0:.3f}s")
            hdf5Data.set_data()
        else:
            print("Invalid Selection", "Selected item is neither a group nor a dataset.")
            return

    # Add a button to trigger the copy
    copy_button = ttk.Button(traces_selection_window, text="Copy Selected Dataset(s)", command=copy_selected_dataset)
    copy_button.pack(pady=10)
    
        

# Hannah Vogel
# Modified by H.D to additionly handle whole folders instead of single files
def remove_selected_options_window(root, hdf5Data):
    # Removes selected datasets of selected data file
    def confirm_selection():
        # Get the selected groups that will be skipped
        selected_groups = [var.get() for var in vars if var.get()]

        # Create action selection buttons
        action_frame = ttk.Frame(newWindow)
        action_frame.pack(fill='x', pady=10)

        ttk.Label(action_frame, text="Apply to:").pack(side='left', padx=5)

        # Process single file button
        single_file_button = ttk.Button(
            action_frame,
            text="Single File",
            command=lambda: process_single_file(selected_groups)
        )
        single_file_button.pack(side='left', padx=5)

        # Process folder button
        folder_button = ttk.Button(
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
        ttk.Button(
            status_window['window'],
            text="Close",
            command=status_window['window'].destroy
        ).pack(pady=5)

    def create_status_window(parent, show_progress=False):
        # Create a status window for showing processing progress
        status_window = ttk.Toplevel(parent)
        status_window.title("Processing Status")
        status_window.geometry("400x150")

        # Make it stay on top of the parent window
        status_window.transient(parent)

        # Status label
        status_var = tk.StringVar(value="Processing...")
        status_label = ttk.Label(status_window, textvariable=status_var, wraplength=380)
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
    newWindow = ttk.Toplevel(root)
    newWindow.title("Select Datasets to Remove")
    newWindow.geometry("400x500")

    # Instructions label
    ttk.Label(
        newWindow,
        text="Select datasets to remove from HDF5 files:",
        wraplength=350
    ).pack(pady=10)

    # Create a frame with scrollbar for checkbuttons
    scroll_frame = ttk.Frame(newWindow)
    scroll_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Create canvas and scrollbar
    canvas = ttk.Canvas(scroll_frame)
    scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)

    # Configure canvas
    checkbutton_frame = ttk.Frame(canvas)
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
        checkbutton = ttk.Checkbutton(
            checkbutton_frame,
            text=string,
            variable=var,
            onvalue=string,
            offvalue=''
        )
        checkbutton.pack(anchor='w')
        vars.append(var)

    # Add confirm button at the bottom
    confirm_button = ttk.Button(
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
    new_window = ttk.Toplevel(root)
    new_window.title("Create HDF5 file")
    CreateHDF5File(new_window)

def plot_array(hdf5Data, root):
    new_window = ttk.Toplevel(root)
    new_window.title("Array Plotter")
    hdf5Data.set_data()
    hdf5Data.set_measure_dim()
    hdf5Data.set_measure_data_and_axis()
    plotter = InteractiveArrayPlotter(new_window, hdf5Data)
    array_plotters.append(plotter)


def plot_array_with_trace_data(hdf5Data, root):
    new_window = ttk.Toplevel(root)
    new_window.title("Array Plotter")
    hdf5Data.set_data()
    hdf5Data.set_measure_dim()
    hdf5Data.set_measure_data_and_axis()
    hdf5Data.trace_loading_with_referance()
    hdf5Data.set_traces_dt()
    plotter = InteractiveArrayAndLinePlotter(new_window, hdf5Data)
    array_plotters.append(plotter)


####
def get_values_above_clicked_node(item, tree):
    values = []
    while item:
        value = tree.item(item, "text")
        values.insert(0, value)
        item = tree.parent(item)
    return values

def display_hdf5_file(root, hdf5Data):

    drop_instruction = (
        'Drop one .hdf5, .h5, or .npz file onto the tree to open it'
    )

    # Function to open an HDF5 file
    def open_hdf5_file():
        if not hdf5Data.readpath:
            messagebox.showinfo(
                'No HDF5 File Selected',
                'Select an HDF5 file or drop one onto the file tree first.',
                parent=root,
            )
            return False

        def display_group(group, parent_tree_node):
            for name, item in group.items():
                if isinstance(item, h5py.Group):
                    child_node = tree.insert(parent_tree_node, 'end', text=name)
                    display_group(item, child_node)
                elif isinstance(item, h5py.Dataset):
                    tree.insert(
                        parent_tree_node,
                        'end',
                        text=name,
                        value=(item.shape,),
                    )
                else:
                    tree.insert(parent_tree_node, 'end', text=name)

        try:
            with h5py.File(hdf5Data.readpath, 'r') as hdf5_file:
                for item in tree.get_children():
                    tree.delete(item)
                display_group(hdf5_file, '')
        except (OSError, ValueError) as error:
            messagebox.showerror(
                'Could Not Open HDF5 File',
                str(error),
                parent=root,
            )
            return False

        root.title(f'HDF5 File Viewer — {hdf5Data.file_name}')
        return True

    def open_hdf5_path(path):
        """Load an explicit path and immediately refresh the main file tree."""
        try:
            set_hdf5_path(hdf5Data, path)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                'Could Not Open HDF5 File', str(error), parent=root
            )
            return False
        return open_hdf5_file()

    # DatabaseBrowser uses the same validated path-opening route as drag/drop.
    root._labber_open_hdf5_path = open_hdf5_path

    def begin_database_import(source_path):
        """Copy a dropped measurement into the active database in a worker."""
        database = getattr(root, '_labber_database', None)
        if database is None:
            return open_hdf5_path(source_path)
        if getattr(root, '_labber_database_import_running', False):
            messagebox.showinfo(
                'Database Import in Progress',
                'Please wait for the current measurement import to finish.',
                parent=root,
            )
            return False

        root._labber_database_import_running = True
        root.configure(cursor='watch')
        drop_label.configure(
            text=f'Adding {os.path.basename(source_path)} to the database…',
            bootstyle='info',
        )
        import_results = queue.Queue(maxsize=1)

        def import_worker():
            try:
                import_results.put(
                    ('success', database.import_dropped_file(source_path))
                )
            except Exception as error:
                import_results.put(('error', error))

        def finish_database_import():
            try:
                result_type, result = import_results.get_nowait()
            except queue.Empty:
                try:
                    root.after(100, finish_database_import)
                except tk.TclError:
                    pass
                return

            root._labber_database_import_running = False
            root.configure(cursor='')
            drop_label.configure(text=drop_instruction, bootstyle='secondary')
            if result_type == 'error':
                messagebox.showwarning(
                    'Database Import Failed',
                    f'The file could not be added to the configured database:\n'
                    f'{result}\n\nThe original file will be opened without '
                    'adding it to the database.',
                    parent=root,
                )
                return open_hdf5_path(source_path)

            managed_path = database.get_absolute_path(result.record)
            opened = open_hdf5_path(managed_path)
            if opened:
                refresh_open_database_browsers(root)
            return opened

        threading.Thread(target=import_worker, daemon=True).start()
        root.after(100, finish_database_import)
        return True

    def open_dropped_hdf5(path):
        """Open directly or import into the enabled managed database first."""
        if getattr(root, '_labber_database', None) is not None:
            return begin_database_import(path)
        return open_hdf5_path(path)

    def begin_npz_conversion(dropped_path):
        """Collect field choices, then convert an NPZ without blocking Tk."""
        if getattr(root, '_labber_npz_conversion_running', False):
            messagebox.showinfo(
                'NPZ Conversion in Progress',
                'Please wait for the current NPZ conversion to finish.',
                parent=root,
            )
            return

        try:
            field_mapping = select_npz_field_mapping(root, dropped_path)
            if field_mapping is None:
                return

            hdf5_path = get_npz_output_path(dropped_path)
            overwrite = False
            if os.path.exists(hdf5_path):
                overwrite = messagebox.askyesno(
                    'Replace Existing HDF5 File',
                    'The target HDF5 file already exists:\n'
                    f'{hdf5_path}\n\nReplace it with data from the NPZ file?',
                    parent=root,
                )
                if not overwrite:
                    return
        except Exception as error:
            messagebox.showerror(
                'Could Not Read NPZ File',
                str(error),
                parent=root,
            )
            return

        root._labber_npz_conversion_running = True
        root.configure(cursor='watch')
        drop_label.configure(
            text=f'Converting {os.path.basename(dropped_path)} to HDF5...',
            bootstyle='info',
        )
        result_queue = queue.Queue(maxsize=1)

        def conversion_worker():
            try:
                converted_path = convert_npz_to_hdf5(
                    dropped_path,
                    field_mapping,
                    overwrite=overwrite,
                )
                result_queue.put(('success', converted_path))
            except Exception as error:
                result_queue.put(('error', error))

        def finish_npz_conversion():
            try:
                result_type, result = result_queue.get_nowait()
            except queue.Empty:
                try:
                    root.after(100, finish_npz_conversion)
                except tk.TclError:
                    pass
                return

            root._labber_npz_conversion_running = False
            root.configure(cursor='')
            drop_label.configure(text=drop_instruction, bootstyle='secondary')

            if result_type == 'error':
                messagebox.showerror(
                    'NPZ Conversion Failed',
                    str(result),
                    parent=root,
                )
                return

            open_dropped_hdf5(result)

        threading.Thread(target=conversion_worker, daemon=True).start()
        root.after(100, finish_npz_conversion)

    def on_data_file_drop(event):
        """Open HDF5 directly or schedule conversion of a dropped NPZ file."""
        dropped_paths = root.tk.splitlist(event.data)
        if len(dropped_paths) != 1:
            messagebox.showerror(
                'Drop One Data File',
                'Please drop exactly one .hdf5, .h5, or .npz file at a time.',
                parent=root,
            )
            return REFUSE_DROP

        dropped_path = os.path.abspath(os.path.expanduser(dropped_paths[0]))
        extension = os.path.splitext(dropped_path)[1].lower()
        if extension == NPZ_FILE_EXTENSION:
            # Returning from the TkDND callback before opening a modal dialog
            # avoids a nested-event-loop deadlock on some Tk/macOS versions.
            root.after_idle(
                lambda selected_path=dropped_path: begin_npz_conversion(
                    selected_path
                )
            )
            return COPY

        if extension not in HDF5_FILE_EXTENSIONS:
            messagebox.showerror(
                'Could Not Open Data File',
                'Only .hdf5, .h5, and .npz files can be dropped here.',
                parent=root,
            )
            return REFUSE_DROP

        if getattr(root, '_labber_database', None) is not None:
            root.after_idle(
                lambda selected_path=dropped_path: open_dropped_hdf5(selected_path)
            )
            return COPY
        return COPY if open_dropped_hdf5(dropped_path) else REFUSE_DROP

    def close_tree_and_hdf5data(hdf5Data):
        working_directory = hdf5Data.wdir
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
        hdf5Data.wdir = working_directory
        root.title('HDF5 File Viewer')

   

    def open_selection(event):
        item = tree.selection()[0]
        value = tree.item(item, "text")
        position = tree.index(item)
        print(f"Selected item: {value}, Position: {position}")

    def on_single_click(event):
        hdf5Data.set_data()
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        hdf5Data.set_current_h5dir(file_dir)
        print(f"Single-clicked on item: {file_dir}")

    def on_right_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        print(f"right-clicked on item: {values_above}")

    def on_double_right_click(event):
        hdf5Data.set_data()
        parent_item = tree.selection()[0]
        values_above = get_values_above_clicked_node(parent_item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        with hdf5Data.file as file:
            if isinstance(file[file_dir], h5py.Dataset):
                for i, list_values in enumerate(file[file_dir]):
                    tree.insert(parent_item, "end", text=f'{i}', values=(list_values,))

    def on_double_click(event):
        item = tree.selection()[0]
        values_above = get_values_above_clicked_node(item, tree)
        file_dir_sep ='/'
        file_dir = file_dir_sep.join(values_above)
        print(f"Double-clicked on item: {file_dir}")

    # Frame for buttons
    frame = ttk.Frame(root)
    frame.pack(side=tk.TOP, padx=5, pady=5)
    # Button to close the tree and reset hdf5Data
    close_button = ttk.Button(frame, text="Close HDF5 File", command=lambda: close_tree_and_hdf5data(hdf5Data), bootstyle='secondary')
    close_button.pack(side=tk.RIGHT, pady=10)
    # Button to open an HDF5 file
    open_button = ttk.Button(frame, text="Show HDF5 File", command=open_hdf5_file, bootstyle='primary')
    open_button.pack(side=tk.RIGHT, pady=10)


    drop_label = ttk.Label(
        root,
        text=drop_instruction,
        bootstyle='secondary',
    )
    drop_label.pack(padx=10, pady=(0, 5), anchor='w')

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

    try:
        TkinterDnD.require(root)
        tree.drop_target_register(DND_FILES)
        tree.dnd_bind('<<Drop>>', on_data_file_drop)
    except (RuntimeError, tk.TclError) as error:
        drop_label.configure(
            text=(
                'Drag and drop is unavailable; HDF5 files can still be opened '
                'from File → Select File Directory.'
            ),
            bootstyle='warning',
        )
        print(f'Drag and drop could not be enabled: {error}')

    return tree


def main():
    global wdir

    def on_close():
        if getattr(root, '_labber_update_install_running', False):
            messagebox.showinfo(
                'Update In Progress',
                'Please wait until the update installation has finished.',
                parent=root,
            )
            return
        if getattr(root, '_labber_npz_conversion_running', False):
            messagebox.showinfo(
                'NPZ Conversion in Progress',
                'Please wait until the NPZ conversion has finished.',
                parent=root,
            )
            return
        if getattr(root, '_labber_database_import_running', False):
            messagebox.showinfo(
                'Database Import in Progress',
                'Please wait until the database import has finished.',
                parent=root,
            )
            return
        if getattr(root, '_labber_database_scan_running', False):
            messagebox.showinfo(
                'Database Scan in Progress',
                'Cancel or finish the database scan before closing the viewer.',
                parent=root,
            )
            return
        try:
            if os.path.exists(wdir):
                shutil.rmtree(wdir)
        except OSError as error:
            print(f'Could not remove the working directory during shutdown: {error}')
        finally:
            root.destroy()

    # Set wdir as a sub-folder in the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wdir = os.path.join(script_dir, 'wdir')
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    # Create the themed application root window.
    hdf5Data = HDF5Data(wdir=wdir)
    root = ttk.App(theme=DEFAULT_THEME)
    try:
        ensure_application_preference_defaults()
    except OSError as error:
        print(f'Could not add preference defaults: {error}')
    apply_saved_application_style(root)
    set_application_icon(root)
    load_configured_database(root)
    data_bar = data_menu_bar(root, hdf5Data)
    root.config(menu=data_bar)
    root.title('HDF5 File Viewer')
    root.protocol("WM_DELETE_WINDOW", on_close)
    root._labber_close_application = on_close
    tree = display_hdf5_file(root, hdf5Data)
    root.after(2000, lambda: check_for_updates(root, silent_if_current=True))
    root.mainloop()
    # Run the Tkinter main loop


if __name__ == '__main__':
    main()
