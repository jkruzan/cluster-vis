# Cluster Vis

A Django web app to aid in the visual analysis of clustering.

The homepage currently lists all features and choosing features brings you to a `/chart` page where a parallel coordinates plot with those features is shown.

## Setup
Dependencies
- `python` (I used version 3.9.13)
- `conda`

Create a virtual environment for this project with `conda env create -f env.yml`.
Activate the virtual environment with `conda activate cvis`.

Start the server by navigating to the `clustervis/` directory containing `manage.py` and run `python manage.py runserver`.

Navigate to http://localhost:8000

# Development
To update the conda environment yaml, run `conda env export > env.yml`.