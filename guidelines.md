# Developer Guidelines for Interactive Labber Data Viewer 

## Code Structure & Organization
- Keep code modular: add new features as separate functions or classes.
    - create an open_*feature_name* method to open a new interactive window 
- Place new code in the most appropriate file/module.
    - interactive_plotting_tools.py for adding GUI elements  
    - Data_analysis_and_transforms.py for generic mathematical or data analysis methods 
    - HDF5Data.py for hdf5-file handling
- Avoid code duplication; reuse existing functions where possible.
- Add your name to the code snippets added/modified by you 
- Do not change the code of another author unless necessary 
    - Communicate with them about the changes

## Coding Style
- Use clear variable names.
    - use snake case and make sure that your variable name is unique within the file.   
- Add comments to explain non-obvious logic or important steps.
    - add a comment to every major method that you add to briefly describe what it does and where it belongs to.  

## Error Handling
- Implement "try ... except" error handling to ensure comprehensible error messages.

## User Interface
- Keep the GUI intuitive and consistent with existing design.
- Label all buttons and controls clearly.
- Document keyboard/mouse controls in user manual if needed.  
- Use Tkinter’s Toplevel for new dialogs/windows, following the style of existing ones.


## Documentation & Version Control
- Update TODO.txt with new features or known issues.
- Update user manual if you add or change user-facing features.
- Create a new fork for adding a signigicant feature.
- Use clear commit messages describing what was changed and why.
    - include what feature you were working on


## Testing
- Test new features with various data files and edge cases.
- Ensure existing functionality is not broken after changes before pushing.