import os

# Define the directory and file structure
structure = [
    "well_analysis/requirements.txt",
    "well_analysis/setup.py",
    "well_analysis/config/__init__.py",
    "well_analysis/config/hpc_config.json",
    "well_analysis/config/params_config.yaml",
    "well_analysis/src/__init__.py",
    "well_analysis/src/core/__init__.py",
    "well_analysis/src/core/constants.py",
    "well_analysis/src/core/data_structures.py",
    "well_analysis/src/core/data_loader.py",
    "well_analysis/src/core/data_generator.py",
    "well_analysis/src/analysis/__init__.py",
    "well_analysis/src/analysis/monte_carlo.py",
    "well_analysis/src/analysis/decline_analysis.py",
    "well_analysis/src/analysis/economic_analysis.py",
    "well_analysis/src/analysis/environmental_analysis.py",
    "well_analysis/src/analysis/maintenance_analysis.py",
    "well_analysis/src/compute/__init__.py",
    "well_analysis/src/compute/hpc_manager.py",
    "well_analysis/src/compute/local_compute.py",
    "well_analysis/src/visualization/__init__.py",
    "well_analysis/src/visualization/visualizer.py",
    "well_analysis/src/visualization/dashboard.py",
    "well_analysis/tests/__init__.py",
    "well_analysis/tests/test_data/production_data.xlsx",
    "well_analysis/tests/test_data/maintenance_data.xlsx",
    "well_analysis/tests/test_data/environmental_data.xlsx",
    "well_analysis/tests/test_data/financial_data.xlsx"
]

# Create each directory and file in the structure
for path in structure:
    # Extract the directory part from the path
    directory = os.path.dirname(path)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Create an empty file at the path
    open(path, 'a').close()

print("Directory structure created successfully.")
