# Cluster Vis

A plotly Dash web app to aid in the visual analysis of clustering.

## Setup
Dependencies
- `python` (I used version 3.9.13)
- `conda`


### Data Files
You will need to download `BinaryImages.mat` and `data_normalized.mat` from Google Drive and place them
in the data directory.

### Virtual Environment
Create a virtual environment for this project with `conda env create -f env.yml`.
Activate the virtual environment with `conda activate cvis`.

### Loading Images
If you save all the images in the git repo, VS Code breaks cause its trying to list 100k image names so create a directory outside of `cluster-vis`: from this directory, run `mkdir ../binary_images`. Then from the `dash/` directory, run `python binary_images.py`.

Start the server by navigating to the `dash/` directory and run `python app.py`.

Navigate to http://localhost:8050 to view the app.

# Development
To update the conda environment yaml, run `conda env export > env.yml`.