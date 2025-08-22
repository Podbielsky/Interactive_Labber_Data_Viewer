# Developer Guidelines for Interactive Labber Data Viewer 


## Code Structure & Organization
- Keep code modular: add new features as separate functions or classes.
    - create an ``open_feature_name`` method to open a new interactive window 
    - Store persistent state (e.g., current ROI, selected lines, mask shapes) as instance attributes.
    - add changes to parent class if possible (e.g. to ``InteractiveArrayPlotter`` instead of ``InteractiveArrayAndLinePlotter``)
- use common naming convention to indicate the type of function (``get_variable_name``, ``set_variable_name``, ``update_function_name``, etc.)
- do not add to deprecated classes 
- if you add to the ``__init__`` method of a class, label what feature your code belongs to 
- Place new code in the most appropriate file/module.
    - ``interactive_plotting_tools.py`` for adding GUI elements  
    - ``Data_analysis_and_transforms.py`` for generic mathematical or data analysis methods 
    - ``HDF5Data.py`` for hdf5-file handling
- Avoid code duplication; reuse existing functions where possible.
- ==Make clear what code belongs to you==
- Do not change the code of another author unless necessary 
    - Communicate with them about the changes

## Coding Style
- Use clear variable names.
    - use snake case and make sure that your variable name is unique within the file.   
- ==Add comments== to explain non-obvious logic or important steps.
    - add a comment to every major method that you add to briefly describe what it does and where it belongs to.  
    - It is better to add to many comments than to few 

## Error Handling
- Implement ``try ... except`` error handling to ensure comprehensible error messages.
- Validate user input before using it in calculations.

## User Interface
- Keep the GUI intuitive and consistent with existing design.
    - Always label new UI elements clearly
- ==Label all visible elements clearly==
    - this includes axis labels, plot titles, button labels, colorbars, etc.
    - do not forget to update these when the underlying data is changed 
- Document keyboard/mouse controls in user manual if needed.  
- Use Tkinter’s Toplevel for new dialogs/windows, following the style of existing ones.
- Use Matplotlib's ``FigureCanvasTkAgg`` for embedding plots in Tkinter windows.
  
## Documentation & Version Control
- Update ``TODO.txt`` with new features or known issues.
- Update user manual if you add or change user-facing features.
- Create a new fork for adding a signigicant feature.
- Use clear commit messages describing what was changed and why.
    - include what feature you were working on


## Testing
- Test new features with various data files and edge cases.
- Ensure existing functionality is not broken after changes before pushing.
- Print timing or debug information (e.g., plotting time) only when useful for development; remove or comment out before pushing unless necessary  