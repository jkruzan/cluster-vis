import numpy as np
import pandas as pd
from scipy.io import loadmat

def get_raw_data():
    path = './data/data_normalized.mat'
    raw_data = loadmat(path)
    raw_data = raw_data['DataFileNorm']
    return(raw_data)

def get_feature_names():
    path = './data/feature_names.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    return np.array(names)

def get_features_and_conditions():
    raw_data = get_raw_data()
    cell_cond = raw_data[:,3]
    feat_arr = raw_data[:,4:]
    return feat_arr, cell_cond

def get_data_frame():
    raw_data = get_raw_data()
    feature_names = get_feature_names()
    df = pd.DataFrame(raw_data, columns=feature_names).set_index('Unique ID')
    # df = df[df['Maximum Curvature'] < 10]
    # df = df[df['Circularity'] < 4]
    # df = df[df['Circular Diameter'] < 1]
    return df