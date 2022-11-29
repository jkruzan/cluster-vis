import numpy as np
import pandas as pd
from PIL import Image

def get_image(unique_id):
    path = '../../binary_images/binary_image_' + str(unique_id) + '.bmp'
    return Image.open(path)

def get_feature_names(df, path=None):
    if df is None:
        print("LOADING CSV FOR FEATURE NAMES")
        df = get_csv_df(path)
    return list(df.columns)

def get_csv_df(path=None):
    print("LOAD THE BIG CSV")
    if path is None:
        path = './data/data.csv'
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['Unique ID'] = df['Unique ID'].astype('int32')
    df['Cluster'] = df['Cluster'].astype('int32')
    df.set_index('Unique ID', inplace=True)
    # LOOK INTO NAN
    return df