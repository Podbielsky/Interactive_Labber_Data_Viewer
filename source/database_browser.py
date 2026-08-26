"""Tk database browser for managed Labber HDF5 measurements."""

import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox

import numpy as np
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from plot_style import (
    configure_transparent_matplotlib_canvas,
    normalize_plot_style,
    resolve_plot_color,
)

class DatabaseBrowser:
    """Display the database directory tree, previews, stars, and comments."""

    PREVIEW_MAX_POINTS_PER_AXIS = 256

    def __init__(
        self,
        parent,
        database,
        open_file_callback,
        icon_callback=None,
        on_close=None,
        plot_style=None,
    ):
        self.parent = parent
        self.database = database
        self.open_file_callback = open_file_callback
        self.on_close_callback = on_close
        self.plot_style = normalize_plot_style(plot_style)
        self.window = ttk.Toplevel(parent)
        self.window.title('Browse Measurement Database')
        self.window.geometry('1100x720')
        self.window.minsize(850, 560)
        if icon_callback is not None:
            icon_callback(self.window)

        self.records = {}
        self.tree_items_by_measurement = {}
        self.current_measurement_id = None
        self.previewed_measurement_id = None
        self.current_preview = None
        self.preview_request_number = 0
        self.keyboard_preview_after_id = None
        self.preview_loading = False
        self.scan_running = False
        self.closed = False
        self.worker_results = queue.Queue()
        self.preview_requests = queue.Queue(maxsize=1)
        self.preview_worker_stop = threading.Event()

        self._build_widgets()
        self.refresh_records()
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.after(100, self._poll_worker_results)
        self.preview_worker_thread = threading.Thread(
            target=self._preview_worker_loop,
            daemon=True,
        )
        self.preview_worker_thread.start()

    def _build_widgets(self):
        """Create the two-pane database browser layout."""
        main_pane = ttk.Panedwindow(self.window, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=2)
        main_pane.add(right_frame, weight=3)

        database_name = os.path.basename(self.database.root_directory) or str(
            self.database.root_directory
        )
        ttk.Label(
            left_frame,
            text=f'Database: {database_name}',
            font=('', 11, 'bold'),
        ).pack(anchor='w', padx=4, pady=(2, 5))

        tree_frame = ttk.Frame(left_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=('star', 'channel', 'dimensions'),
            selectmode='browse',
        )
        self.tree.heading('#0', text='Directory / Measurement', anchor='w')
        self.tree.heading('star', text='★', anchor='center')
        self.tree.heading('channel', text='Data channel', anchor='w')
        self.tree.heading('dimensions', text='Shape', anchor='w')
        self.tree.column('#0', width=280, minwidth=180)
        self.tree.column('star', width=38, minwidth=38, stretch=False, anchor='center')
        self.tree.column('channel', width=120, minwidth=80)
        self.tree.column('dimensions', width=90, minwidth=60)
        tree_scrollbar = ttk.Scrollbar(
            tree_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scrollbar.set)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.tree.bind('<ButtonRelease-1>', self._on_tree_click)
        self.tree.bind('<Up>', lambda event: self._move_keyboard_selection(-1))
        self.tree.bind('<Down>', lambda event: self._move_keyboard_selection(1))
        self.tree.bind('<Return>', lambda event: self.open_selected())

        left_buttons = ttk.Frame(left_frame)
        left_buttons.pack(fill=tk.X, pady=(6, 0))
        self.rescan_button = ttk.Button(
            left_buttons,
            text='Rescan Database',
            command=self.start_rescan,
            bootstyle='secondary',
        )
        self.rescan_button.pack(side=tk.LEFT)
        self.open_button = ttk.Button(
            left_buttons,
            text='Open Selected',
            command=self.open_selected,
            state=tk.DISABLED,
            bootstyle='primary',
        )
        self.open_button.pack(side=tk.RIGHT)

        self.status_variable = tk.StringVar(value='Select a measurement to preview it.')
        ttk.Label(
            left_frame,
            textvariable=self.status_variable,
            wraplength=420,
            bootstyle='secondary',
        ).pack(fill=tk.X, padx=4, pady=(6, 0))

        self.measurement_title = tk.StringVar(value='No measurement selected')
        ttk.Label(
            right_frame,
            textvariable=self.measurement_title,
            font=('', 12, 'bold'),
        ).pack(anchor='w', padx=5, pady=(2, 4))

        self.measurement_details = tk.StringVar(value='')
        ttk.Label(
            right_frame,
            textvariable=self.measurement_details,
            wraplength=580,
            bootstyle='secondary',
        ).pack(anchor='w', padx=5, pady=(0, 4))

        self.preview_figure = Figure(figsize=(6, 4), dpi=100, tight_layout=True)
        self.preview_axis = self.preview_figure.add_subplot(111)
        self.preview_axis.set_title('Measurement preview')
        self.preview_axis.set_axis_off()
        self.preview_canvas = FigureCanvasTkAgg(
            self.preview_figure, master=right_frame
        )
        self.refresh_theme_style(redraw=False)
        self.preview_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        metadata_frame = ttk.LabelFrame(right_frame, text='Measurement notes')
        metadata_frame.pack(fill=tk.X, padx=4, pady=(6, 2))
        self.starred_variable = tk.BooleanVar(value=False)
        self.starred_button = ttk.Checkbutton(
            metadata_frame,
            text='Mark as special measurement ★',
            variable=self.starred_variable,
            command=self._save_starred,
            state=tk.DISABLED,
            bootstyle='warning-round-toggle',
        )
        self.starred_button.pack(anchor='w', padx=6, pady=(5, 3))

        comment_frame = ttk.Frame(metadata_frame)
        comment_frame.pack(fill=tk.X, padx=6, pady=(2, 6))
        self.comment_text = tk.Text(comment_frame, height=4, wrap=tk.WORD)
        comment_scrollbar = ttk.Scrollbar(
            comment_frame, orient=tk.VERTICAL, command=self.comment_text.yview
        )
        self.comment_text.configure(yscrollcommand=comment_scrollbar.set)
        comment_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comment_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.comment_text.configure(state=tk.DISABLED)

        self.save_comment_button = ttk.Button(
            metadata_frame,
            text='Save Comment',
            command=self._save_comment,
            state=tk.DISABLED,
            bootstyle='success',
        )
        self.save_comment_button.pack(anchor='e', padx=6, pady=(0, 6))

    def refresh_theme_style(self, redraw=True):
        """Blend the preview figure into the active ttkbootstrap theme."""
        configure_transparent_matplotlib_canvas(
            self.preview_figure,
            self.preview_canvas,
        )
        if redraw:
            self.preview_canvas.draw_idle()

    def refresh_records(self):
        """Reload the Treeview from SQLite without rescanning the filesystem."""
        try:
            records = self.database.list_measurements()
        except Exception as error:
            messagebox.showerror(
                'Could Not Read Database', str(error), parent=self.window
            )
            return

        selected_id = self.current_measurement_id
        self.records = {record.measurement_id: record for record in records}
        self.tree_items_by_measurement.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)

        folder_items = {'': ''}
        for record in records:
            path_parts = record.relative_path.split('/')
            parent_item = ''
            accumulated_parts = []
            for directory_name in path_parts[:-1]:
                accumulated_parts.append(directory_name)
                directory_path = '/'.join(accumulated_parts)
                if directory_path not in folder_items:
                    folder_items[directory_path] = self.tree.insert(
                        parent_item,
                        'end',
                        text=directory_name,
                        open=True,
                    )
                parent_item = folder_items[directory_path]

            tree_item = self.tree.insert(
                parent_item,
                'end',
                text=record.file_name,
                values=(
                    '★' if record.starred else '',
                    record.data_channel,
                    ' × '.join(str(value) for value in record.step_dimensions),
                ),
            )
            self.tree_items_by_measurement[record.measurement_id] = tree_item

        if selected_id in self.records:
            tree_item = self.tree_items_by_measurement[selected_id]
            self.tree.selection_set(tree_item)
            self.tree.see(tree_item)
            self._show_record_metadata(self.records[selected_id])
        elif not records:
            self._clear_selection('The database does not contain valid measurements.')
        else:
            self._clear_selection(
                f'{len(records)} measurement(s) indexed. Select one to preview it.'
            )

    def _measurement_id_for_item(self, tree_item):
        """Resolve a Treeview leaf to its measurement ID."""
        for measurement_id, candidate_item in self.tree_items_by_measurement.items():
            if candidate_item == tree_item:
                return measurement_id
        return None

    def _on_tree_click(self, event):
        """Preview on the first click and open on a repeated click."""
        tree_item = self.tree.identify_row(event.y)
        measurement_id = self._measurement_id_for_item(tree_item)
        if measurement_id is None:
            return

        self.tree.focus_set()
        self.tree.focus(tree_item)
        self._cancel_keyboard_preview()
        if (
            measurement_id == self.previewed_measurement_id
            and not self.preview_loading
        ):
            self.tree.selection_set(tree_item)
            self.current_measurement_id = measurement_id
            self.open_selected()
            return

        self.tree.selection_set(tree_item)
        self.current_measurement_id = measurement_id
        self._begin_preview(self.records[measurement_id])

    def _cancel_keyboard_preview(self):
        """Cancel a preview that was waiting for rapid key navigation to stop."""
        if self.keyboard_preview_after_id is None:
            return
        try:
            self.window.after_cancel(self.keyboard_preview_after_id)
        except tk.TclError:
            pass
        self.keyboard_preview_after_id = None

    def _move_keyboard_selection(self, direction):
        """Move Up/Down between measurement leaves and refresh their preview."""
        measurement_ids = list(self.tree_items_by_measurement)
        if not measurement_ids:
            return 'break'

        if self.current_measurement_id in measurement_ids:
            current_index = measurement_ids.index(self.current_measurement_id)
            target_index = max(
                0, min(len(measurement_ids) - 1, current_index + direction)
            )
        else:
            target_index = 0 if direction > 0 else len(measurement_ids) - 1

        measurement_id = measurement_ids[target_index]
        if measurement_id == self.current_measurement_id:
            return 'break'

        tree_item = self.tree_items_by_measurement[measurement_id]
        self.tree.selection_set(tree_item)
        self.tree.focus(tree_item)
        self.tree.see(tree_item)
        self.current_measurement_id = measurement_id
        record = self.records[measurement_id]
        self._show_record_metadata(record)

        # Invalidate an older worker immediately, then wait briefly so holding
        # an arrow key does not start one HDF5 read for every repeated keypress.
        self.preview_request_number += 1
        self.previewed_measurement_id = None
        self.current_preview = None
        self.preview_loading = False
        self._cancel_keyboard_preview()
        self.status_variable.set(f'Preparing preview for {record.file_name}…')

        def load_keyboard_preview():
            self.keyboard_preview_after_id = None
            if self.current_measurement_id == measurement_id and not self.closed:
                self._begin_preview(record)

        self.keyboard_preview_after_id = self.window.after(
            150, load_keyboard_preview
        )
        return 'break'

    def _show_record_metadata(self, record):
        """Update the controls below the preview for the selected record."""
        self.measurement_title.set(record.file_name)
        details = record.relative_path
        if record.has_traces:
            details += '  •  contains traces'
        self.measurement_details.set(details)
        self.starred_variable.set(record.starred)
        self.starred_button.configure(state=tk.NORMAL)
        self.comment_text.configure(state=tk.NORMAL)
        self.comment_text.delete('1.0', tk.END)
        self.comment_text.insert('1.0', record.comment)
        self.save_comment_button.configure(state=tk.NORMAL)
        self.open_button.configure(state=tk.NORMAL)

    def _clear_selection(self, status_message):
        """Reset controls when no measurement is currently available."""
        self.current_measurement_id = None
        self.previewed_measurement_id = None
        self.current_preview = None
        self.measurement_title.set('No measurement selected')
        self.measurement_details.set('')
        self.starred_variable.set(False)
        self.starred_button.configure(state=tk.DISABLED)
        self.comment_text.configure(state=tk.NORMAL)
        self.comment_text.delete('1.0', tk.END)
        self.comment_text.configure(state=tk.DISABLED)
        self.save_comment_button.configure(state=tk.DISABLED)
        self.open_button.configure(state=tk.DISABLED)
        self.status_variable.set(status_message)

    def _begin_preview(self, record):
        """Queue the selected measurement for the single preview worker."""
        self._cancel_keyboard_preview()
        self._show_record_metadata(record)
        self.preview_request_number += 1
        request_number = self.preview_request_number
        self.preview_loading = True
        self.previewed_measurement_id = None
        self.current_preview = None
        self.status_variable.set(f'Loading preview for {record.file_name}…')
        self.preview_figure.clear()
        preview_axis = self.preview_figure.add_subplot(111)
        preview_axis.text(
            0.5,
            0.5,
            'Loading preview…',
            ha='center',
            va='center',
            transform=preview_axis.transAxes,
        )
        preview_axis.set_axis_off()
        self.refresh_theme_style(redraw=False)
        self.preview_canvas.draw_idle()

        preview_request = (request_number, record)
        try:
            while True:
                self.preview_requests.get_nowait()
        except queue.Empty:
            pass
        self.preview_requests.put_nowait(preview_request)

    @classmethod
    def _preview_resolution(cls, record):
        """Use a smaller raster for unusually large displayed map planes."""
        if len(record.step_dimensions) < 2:
            return cls.PREVIEW_MAX_POINTS_PER_AXIS
        map_point_count = record.step_dimensions[0] * record.step_dimensions[1]
        if map_point_count > 16_000_000:
            return 64
        if map_point_count > 4_000_000:
            return 128
        return cls.PREVIEW_MAX_POINTS_PER_AXIS

    def _preview_worker_loop(self):
        """Process at most one preview read and one newest pending request."""
        while not self.preview_worker_stop.is_set():
            try:
                request_number, record = self.preview_requests.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                preview = self.database.load_map_preview(
                    record,
                    maximum_points_per_axis=self._preview_resolution(record),
                )
                self.worker_results.put(
                    ('preview_success', request_number, record.measurement_id, preview)
                )
            except Exception as error:
                self.worker_results.put(
                    ('preview_error', request_number, record.measurement_id, error)
                )

    @staticmethod
    def _preview_extent(preview):
        """Approximate coordinate bounds for fast raster preview rendering."""
        with np.errstate(invalid='ignore'):
            x_start = float(np.nanmedian(preview.x[:, 0]))
            x_stop = float(np.nanmedian(preview.x[:, -1]))
            y_start = float(np.nanmedian(preview.y[0, :]))
            y_stop = float(np.nanmedian(preview.y[-1, :]))

        if not np.isfinite(x_start) or not np.isfinite(x_stop):
            x_start, x_stop = 0.0, float(preview.z.shape[1])
        if not np.isfinite(y_start) or not np.isfinite(y_stop):
            y_start, y_stop = 0.0, float(preview.z.shape[0])
        if x_start == x_stop:
            x_start -= 0.5
            x_stop += 0.5
        if y_start == y_stop:
            y_start -= 0.5
            y_stop += 0.5
        return x_start, x_stop, y_start, y_stop

    def _display_preview(self, measurement_id, preview):
        """Render a completed preview on the Tk event thread."""
        self.preview_figure.clear()
        self.preview_axis = self.preview_figure.add_subplot(111)
        if preview.is_linecut:
            line_x = np.ravel(preview.x[0])
            line_z = np.ravel(preview.z[0])
            finite_points = np.isfinite(line_x) & np.isfinite(line_z)
            self.preview_axis.plot(
                line_x[finite_points],
                line_z[finite_points],
                color=resolve_plot_color(
                    self.plot_style['crosshair_histogram_color']
                ),
                linewidth=1.2,
            )
            self.preview_axis.set_xlabel(preview.x_label)
            self.preview_axis.set_ylabel(preview.z_label)
            self.preview_axis.set_title(preview.z_label)
            self.preview_axis.grid(True, alpha=0.25)
        else:
            finite_z = np.ma.masked_invalid(preview.z)
            preview_image = self.preview_axis.imshow(
                finite_z,
                origin='lower',
                aspect='auto',
                interpolation='nearest',
                extent=self._preview_extent(preview),
                cmap=self.plot_style['preferred_colormap'],
            )
            self.preview_axis.set_xlabel(preview.x_label)
            self.preview_axis.set_ylabel(preview.y_label)
            self.preview_axis.set_title(preview.z_label)
            self.preview_figure.colorbar(
                preview_image, ax=self.preview_axis, label=preview.z_label
            )
        self.refresh_theme_style(redraw=False)
        self.preview_canvas.draw_idle()
        self.previewed_measurement_id = measurement_id
        self.current_preview = preview
        self.preview_loading = False
        record = self.records.get(measurement_id)
        if record is not None:
            self.status_variable.set(
                'Preview loaded. Click the measurement again to open it.'
            )

    def set_plot_style(self, plot_style):
        """Apply preferred map and line colors to the current preview."""
        normalized_style = normalize_plot_style(plot_style)
        colormap_changed = (
            normalized_style['preferred_colormap']
            != self.plot_style['preferred_colormap']
        )
        line_color_changed = (
            resolve_plot_color(
                normalized_style['crosshair_histogram_color']
            )
            != resolve_plot_color(
                self.plot_style['crosshair_histogram_color']
            )
        )
        self.plot_style = normalized_style
        if (
            self.current_preview is not None
            and self.previewed_measurement_id is not None
            and (
                (self.current_preview.is_linecut and line_color_changed)
                or (
                    not self.current_preview.is_linecut
                    and colormap_changed
                )
            )
        ):
            self._display_preview(
                self.previewed_measurement_id,
                self.current_preview,
            )
        return normalized_style

    def _display_preview_error(self, measurement_id, error):
        """Show preview errors without closing the database browser."""
        self.preview_figure.clear()
        self.preview_axis = self.preview_figure.add_subplot(111)
        self.preview_axis.text(
            0.5,
            0.5,
            f'Could not load preview:\n{error}',
            ha='center',
            va='center',
            wrap=True,
            transform=self.preview_axis.transAxes,
        )
        self.preview_axis.set_axis_off()
        self.refresh_theme_style(redraw=False)
        self.preview_canvas.draw_idle()
        self.previewed_measurement_id = measurement_id
        self.current_preview = None
        self.preview_loading = False
        self.status_variable.set('The selected measurement could not be previewed.')

    def open_selected(self):
        """Load the selected managed path into the main HDF5 file viewer."""
        record = self.records.get(self.current_measurement_id)
        if record is None:
            messagebox.showinfo(
                'No Measurement Selected',
                'Select a measurement first.',
                parent=self.window,
            )
            return
        try:
            opened = self.open_file_callback(self.database.get_absolute_path(record))
        except Exception as error:
            messagebox.showerror(
                'Could Not Open Measurement', str(error), parent=self.window
            )
            return
        if opened is not False:
            self.status_variable.set(
                f'{record.file_name} is loaded in the main file viewer.'
            )

    def _save_starred(self):
        """Save and immediately reflect the selected starred state."""
        measurement_id = self.current_measurement_id
        if measurement_id is None:
            return
        try:
            self.database.set_starred(
                measurement_id, self.starred_variable.get()
            )
            record = self.database.get_measurement(measurement_id)
        except Exception as error:
            messagebox.showerror(
                'Could Not Save Star', str(error), parent=self.window
            )
            return
        if record is not None:
            self.records[measurement_id] = record
            tree_item = self.tree_items_by_measurement.get(measurement_id)
            if tree_item is not None:
                current_values = list(self.tree.item(tree_item, 'values'))
                current_values[0] = '★' if record.starred else ''
                self.tree.item(tree_item, values=current_values)

    def _save_comment(self):
        """Save the comment displayed below the current preview."""
        measurement_id = self.current_measurement_id
        if measurement_id is None:
            return
        comment = self.comment_text.get('1.0', 'end-1c')
        try:
            self.database.set_comment(measurement_id, comment)
            record = self.database.get_measurement(measurement_id)
        except Exception as error:
            messagebox.showerror(
                'Could Not Save Comment', str(error), parent=self.window
            )
            return
        if record is not None:
            self.records[measurement_id] = record
        self.status_variable.set('Comment saved.')

    def start_rescan(self):
        """Reconcile files and SQLite records without blocking Tk."""
        if self.scan_running:
            return
        self.scan_running = True
        self.rescan_button.configure(state=tk.DISABLED)
        self.status_variable.set('Scanning the database…')

        def report_progress(processed, total, path):
            self.worker_results.put(('scan_progress', processed, total, path))

        def scan_worker():
            try:
                result = self.database.scan(progress_callback=report_progress)
                self.worker_results.put(('scan_success', result))
            except Exception as error:
                self.worker_results.put(('scan_error', error))

        threading.Thread(target=scan_worker, daemon=True).start()

    def _poll_worker_results(self):
        """Apply worker results serially on the Tk event thread."""
        if self.closed:
            return
        try:
            while True:
                result = self.worker_results.get_nowait()
                result_type = result[0]
                if result_type in ('preview_success', 'preview_error'):
                    _, request_number, measurement_id, payload = result
                    if request_number != self.preview_request_number:
                        continue
                    if result_type == 'preview_success':
                        self._display_preview(measurement_id, payload)
                    else:
                        self._display_preview_error(measurement_id, payload)
                elif result_type == 'scan_progress':
                    _, processed, total, path = result
                    file_name = os.path.basename(path) if path else ''
                    self.status_variable.set(
                        f'Scanning {processed}/{total}: {file_name}'
                    )
                elif result_type == 'scan_success':
                    scan_result = result[1]
                    self.scan_running = False
                    self.rescan_button.configure(state=tk.NORMAL)
                    self.refresh_records()
                    self.status_variable.set(
                        f'Indexed {scan_result.indexed} of '
                        f'{scan_result.discovered} HDF5 file(s).'
                    )
                    if scan_result.invalid:
                        invalid_lines = [
                            f'• {path}: {error}'
                            for path, error in scan_result.invalid[:10]
                        ]
                        if len(scan_result.invalid) > 10:
                            invalid_lines.append(
                                f'…and {len(scan_result.invalid) - 10} more.'
                            )
                        messagebox.showwarning(
                            'Some Files Were Skipped',
                            'The following files are not compatible with the '
                            'map viewer:\n\n' + '\n'.join(invalid_lines),
                            parent=self.window,
                        )
                elif result_type == 'scan_error':
                    self.scan_running = False
                    self.rescan_button.configure(state=tk.NORMAL)
                    self.status_variable.set('Database scan failed.')
                    messagebox.showerror(
                        'Database Scan Failed', str(result[1]), parent=self.window
                    )
        except queue.Empty:
            pass
        self.window.after(100, self._poll_worker_results)

    def close(self):
        """Close the browser and unregister it from the main application."""
        self.closed = True
        self._cancel_keyboard_preview()
        self.preview_worker_stop.set()
        try:
            while True:
                self.preview_requests.get_nowait()
        except queue.Empty:
            pass
        if self.on_close_callback is not None:
            self.on_close_callback(self)
        self.window.destroy()
