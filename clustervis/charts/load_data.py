import numpy as np
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
    return names

def get_features_and_conditions():
    raw_data = get_raw_data()
    cell_cond = raw_data[:,3]
    feat_arr = raw_data[:,4:]
    return feat_arr, cell_cond
