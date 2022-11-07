# Cluster Vis

A Django web app to aid in the visual analysis of clustering.

Currently shows the list of features on the homepage and a
parallel coordinates plot at `/chart`.

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