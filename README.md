# Cluster Vis

A plotly Dash web app to aid in the visual analysis of clustering.

## Setup
Dependencies
- `python` (I used version 3.9.13)
- `conda`

Create a virtual environment for this project with `conda env create -f env.yml`.
Activate the virtual environment with `conda activate cvis`.

You will need to download `BinaryImages.mat` and `data_normalized.mat` from Google Drive and place them
in the data directory.

Save the images individually by creating a directory `dash/data/binary_images/`, change directories to `/dash/` and
run `binary_images.py`

Start the server by navigating to the `dash/` directory and run `python app.py`.

Navigate to http://localhost:8050 to view the server

# Development
To update the conda environment yaml, run `conda env export > env.yml`.