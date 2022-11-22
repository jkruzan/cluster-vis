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

def get_cluster_names_pretty(df):
    if df is None:
        print("LOADING CSV TO GET CLUSTER NAMES")
        labels = get_csv_df()['Cluster'].values
    else:
        labels = df['Cluster'].values
    min_label, max_label = labels.min(), labels.max()
    return ["Cluster "+ str(i) for i in range(min_label, max_label+1)]

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