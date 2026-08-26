"""Shared plotting-style preferences and selection dialog."""

import tkinter as tk
from tkinter import messagebox

import matplotlib
from matplotlib.colors import is_color_like
import ttkbootstrap as ttk
from ttkbootstrap.themes.standard import STANDARD_THEMES


def configure_transparent_matplotlib_canvas(
    figure,
    canvas,
    opaque_for_blitting=False,
):
    """Blend an embedded Matplotlib figure into the active ttk theme."""
    style = ttk.Style()
    background = (
        style.lookup('TFrame', 'background')
        or style.lookup('.', 'background')
        or 'white'
    )
    foreground = (
        style.lookup('TLabel', 'foreground')
        or style.lookup('.', 'foreground')
        or 'black'
    )

    # Keep newly created axes and axes reset by ``clear()`` transparent and
    # readable in the current light or dark ttkbootstrap theme.
    matplotlib.rcParams['figure.facecolor'] = 'none'
    matplotlib.rcParams['axes.facecolor'] = 'none'
    matplotlib.rcParams['axes.edgecolor'] = foreground
    matplotlib.rcParams['axes.labelcolor'] = foreground
    matplotlib.rcParams['text.color'] = foreground
    matplotlib.rcParams['xtick.color'] = foreground
    matplotlib.rcParams['ytick.color'] = foreground

    figure.patch.set_facecolor(background if opaque_for_blitting else 'none')
    figure.patch.set_alpha(1.0 if opaque_for_blitting else 0.0)
    for axis in figure.axes:
        axis.set_facecolor(background if opaque_for_blitting else 'none')
        axis.patch.set_alpha(1.0 if opaque_for_blitting else 0.0)
        axis.tick_params(axis='both', colors=foreground)
        axis.xaxis.label.set_color(foreground)
        axis.yaxis.label.set_color(foreground)
        axis.title.set_color(foreground)
        for spine in axis.spines.values():
            spine.set_color(foreground)

    # Tk canvases cannot inherit a truly transparent widget background. Match
    # it to the ttk frame beneath the alpha-enabled Agg image instead.
    canvas.get_tk_widget().configure(
        background=background,
        highlightthickness=0,
        borderwidth=0,
    )


AVAILABLE_COLORMAPS = (
    'viridis',
    'plasma',
    'inferno',
    'magma',
    'cividis',
    'twilight',
    'twilight_shifted',
    'BlueMap',
    'RedMap',
    'coolwarm',
    'Spectral',
    'gnuplot',
    'NeonPiCy',
    'BiMap',
)


# Preferences store these readable names instead of backend-specific RGBA
# tuples. Unknown but valid Matplotlib colors are also accepted when a user
# edits preferences.json directly.
PLOT_COLOR_OPTIONS = {
    'Matplotlib blue': 'tab:blue',
    'Matplotlib orange': 'tab:orange',
    'Matplotlib green': 'tab:green',
    'Matplotlib red': 'tab:red',
    'Matplotlib purple': 'tab:purple',
    'Matplotlib brown': 'tab:brown',
    'Matplotlib pink': 'tab:pink',
    'Matplotlib gray': 'tab:gray',
    'Matplotlib olive': 'tab:olive',
    'Matplotlib cyan': 'tab:cyan',
    'Black': 'black',
    'White': 'white',
    'Viridis purple': '#440154',
    'Viridis blue': '#3b528b',
    'Viridis teal': '#21918c',
    'Viridis green': '#5ec962',
    'Viridis yellow': '#fde725',
    'Plasma violet': '#5c01a6',
    'Plasma magenta': '#cc4778',
    'Plasma orange': '#f89540',
    'BiMap pink': '#d60270',
    'BiMap purple': '#9b4f96',
    'BiMap blue': '#0038a8',
    'NeonPiCy pink': '#ff1493',
    'NeonPiCy blue-green': '#0cc6ba',
    'RedMap coral': '#d95f5f',
    'BlueMap azure': '#4169a1',
}


COLOR_CYCLE_OPTIONS = {
    'Matplotlib tab10': (
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ),
    'Matplotlib tab20': (
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
    ),
    'Matplotlib Set1': (
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999',
    ),
    'Matplotlib Dark2': (
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a',
        '#66a61e', '#e6ab02', '#a6761d', '#666666',
    ),
    'Viridis inspired': (
        '#440154', '#3b528b', '#21918c', '#5ec962', '#fde725',
    ),
    'Plasma inspired': (
        '#0d0887', '#5c01a6', '#9c179e', '#cc4778', '#ed7953',
        '#fdb42f', '#f0f921',
    ),
    'Inferno inspired': (
        '#000004', '#420a68', '#932667', '#dd513a', '#fca50a',
        '#fcffa4',
    ),
    'Cividis inspired': (
        '#00224e', '#31446b', '#666970', '#958f78', '#c8b866',
        '#fee838',
    ),
    'BiMap inspired': (
        '#d60270', '#e84a9a', '#9b4f96', '#5357a6', '#0038a8',
    ),
    'NeonPiCy inspired': (
        '#ff1493', '#a40775', '#000000', '#076d72', '#0cc6ba',
        '#ffffff',
    ),
    'RedMap / BlueMap': (
        '#762a83', '#d95f5f', '#f7b6b2', '#4169a1', '#92c5de',
        '#0571b0',
    ),
}


TTKBOOTSTRAP_CYCLE_COLOR_ORDER = (
    'primary',
    'success',
    'info',
    'warning',
    'danger',
    'secondary',
)


def _ttkbootstrap_color_cycles():
    """Build line cycles from the installed ttkbootstrap theme palettes."""
    theme_cycles = {}
    for theme_name, theme_definition in STANDARD_THEMES.items():
        theme_colors = theme_definition.get('colors', {})
        cycle = tuple(
            theme_colors.get(color_name)
            for color_name in TTKBOOTSTRAP_CYCLE_COLOR_ORDER
        )
        if all(isinstance(color, str) and is_color_like(color) for color in cycle):
            theme_cycles[f'ttkbootstrap: {theme_name.title()}'] = cycle
    return theme_cycles


# Generate these entries from ttkbootstrap itself so the choices match the
# exact theme version installed by the setup scripts.
COLOR_CYCLE_OPTIONS.update(_ttkbootstrap_color_cycles())


DEFAULT_PLOT_STYLE = {
    'preferred_colormap': 'viridis',
    'crosshair_histogram_color': 'Matplotlib blue',
    'extracted_linecut_color_cycle': 'Matplotlib tab10',
}


def resolve_plot_color(color_name):
    """Return a Matplotlib-compatible color for a stored preference value."""
    color = PLOT_COLOR_OPTIONS.get(color_name, color_name)
    if not isinstance(color, str) or not is_color_like(color):
        return PLOT_COLOR_OPTIONS[
            DEFAULT_PLOT_STYLE['crosshair_histogram_color']
        ]
    return color


def get_color_cycle(cycle_name):
    """Return the selected extracted-linecut color cycle."""
    return tuple(COLOR_CYCLE_OPTIONS.get(
        cycle_name,
        COLOR_CYCLE_OPTIONS[
            DEFAULT_PLOT_STYLE['extracted_linecut_color_cycle']
        ],
    ))


def normalize_plot_style(plot_style):
    """Merge and validate a possibly partial plot-style preference."""
    normalized = dict(DEFAULT_PLOT_STYLE)
    if isinstance(plot_style, dict):
        normalized.update(plot_style)

    if normalized['preferred_colormap'] not in AVAILABLE_COLORMAPS:
        normalized['preferred_colormap'] = DEFAULT_PLOT_STYLE[
            'preferred_colormap'
        ]

    color_name = normalized['crosshair_histogram_color']
    if (
        not isinstance(color_name, str)
        or (
            color_name not in PLOT_COLOR_OPTIONS
            and not is_color_like(color_name)
        )
    ):
        normalized['crosshair_histogram_color'] = DEFAULT_PLOT_STYLE[
            'crosshair_histogram_color'
        ]

    if (
        normalized['extracted_linecut_color_cycle']
        not in COLOR_CYCLE_OPTIONS
    ):
        normalized['extracted_linecut_color_cycle'] = DEFAULT_PLOT_STYLE[
            'extracted_linecut_color_cycle'
        ]
    return normalized


def open_plot_style_dialog(parent, current_style, apply_callback):
    """Open a reusable editor for persistent plotting-style preferences."""
    dialog = ttk.Toplevel(parent)
    dialog.title('Plot Style Preferences')
    dialog.geometry('520x350')
    dialog.resizable(False, False)
    dialog.transient(parent)

    style = normalize_plot_style(current_style)
    colormap_variable = tk.StringVar(
        master=dialog,
        value=style['preferred_colormap'],
    )
    line_color_variable = tk.StringVar(
        master=dialog,
        value=style['crosshair_histogram_color'],
    )
    cycle_variable = tk.StringVar(
        master=dialog,
        value=style['extracted_linecut_color_cycle'],
    )

    content = ttk.Frame(dialog, padding=14)
    content.pack(fill=tk.BOTH, expand=True)
    content.columnconfigure(1, weight=1)

    ttk.Label(content, text='Preferred map colormap:').grid(
        row=0, column=0, sticky=tk.W, padx=(0, 12), pady=6
    )
    colormap_combobox = ttk.Combobox(
        content,
        textvariable=colormap_variable,
        values=AVAILABLE_COLORMAPS,
        state='readonly',
        width=27,
    )
    colormap_combobox.grid(row=0, column=1, sticky=tk.EW, pady=6)

    ttk.Label(content, text='Crosshair and histogram:').grid(
        row=1, column=0, sticky=tk.W, padx=(0, 12), pady=6
    )
    color_combobox = ttk.Combobox(
        content,
        textvariable=line_color_variable,
        values=tuple(PLOT_COLOR_OPTIONS),
        state='readonly',
        width=27,
    )
    color_combobox.grid(row=1, column=1, sticky=tk.EW, pady=6)

    ttk.Label(content, text='Extracted-linecut cycle:').grid(
        row=2, column=0, sticky=tk.W, padx=(0, 12), pady=6
    )
    cycle_combobox = ttk.Combobox(
        content,
        textvariable=cycle_variable,
        values=tuple(COLOR_CYCLE_OPTIONS),
        state='readonly',
        width=27,
    )
    cycle_combobox.grid(row=2, column=1, sticky=tk.EW, pady=6)

    preview_frame = ttk.LabelFrame(content, text='Color preview')
    preview_frame.grid(
        row=3,
        column=0,
        columnspan=2,
        sticky=tk.EW,
        pady=(14, 8),
    )
    preview_canvas = tk.Canvas(
        preview_frame,
        height=60,
        highlightthickness=0,
        borderwidth=0,
    )
    preview_canvas.pack(fill=tk.X, padx=8, pady=8)

    def update_preview(_event=None):
        preview_canvas.delete('all')
        width = max(300, preview_canvas.winfo_width())
        background = (
            ttk.Style().lookup('TFrame', 'background') or 'white'
        )
        preview_canvas.configure(background=background)
        line_color = resolve_plot_color(line_color_variable.get())
        preview_canvas.create_line(
            8, 14, width - 8, 14, fill=line_color, width=4
        )
        cycle = get_color_cycle(cycle_variable.get())
        swatch_width = max(1, (width - 16) / len(cycle))
        for index, color in enumerate(cycle):
            x_start = 8 + index * swatch_width
            preview_canvas.create_rectangle(
                x_start,
                34,
                x_start + swatch_width,
                54,
                fill=color,
                outline=color,
            )

    for combobox in (color_combobox, cycle_combobox):
        combobox.bind('<<ComboboxSelected>>', update_preview)
    preview_canvas.bind('<Configure>', update_preview)

    def restore_defaults():
        colormap_variable.set(DEFAULT_PLOT_STYLE['preferred_colormap'])
        line_color_variable.set(
            DEFAULT_PLOT_STYLE['crosshair_histogram_color']
        )
        cycle_variable.set(
            DEFAULT_PLOT_STYLE['extracted_linecut_color_cycle']
        )
        update_preview()

    def apply_style():
        selected_style = normalize_plot_style({
            'preferred_colormap': colormap_variable.get(),
            'crosshair_histogram_color': line_color_variable.get(),
            'extracted_linecut_color_cycle': cycle_variable.get(),
        })
        try:
            apply_callback(selected_style)
        except (OSError, ValueError) as error:
            messagebox.showerror(
                'Plot Style Not Saved',
                f'Could not save the plot style:\n{error}',
                parent=dialog,
            )
            return
        dialog.destroy()

    button_frame = ttk.Frame(content)
    button_frame.grid(
        row=4, column=0, columnspan=2, sticky=tk.EW, pady=(14, 0)
    )
    ttk.Button(
        button_frame,
        text='Restore Defaults',
        command=restore_defaults,
        bootstyle='secondary',
    ).pack(side=tk.LEFT)
    ttk.Button(
        button_frame,
        text='Cancel',
        command=dialog.destroy,
        bootstyle='secondary',
    ).pack(side=tk.RIGHT, padx=(8, 0))
    ttk.Button(
        button_frame,
        text='Save',
        command=apply_style,
        bootstyle='primary',
    ).pack(side=tk.RIGHT)

    dialog.after_idle(update_preview)
    return dialog
