import numpy as np
import pandas as pd
from scipy.io import loadmat
import mat73
def get_binary_images(path=None):
    if path is None:
        path = './data/BinaryImages.mat'
    raw_data = mat73.loadmat(path)
    raw_data = raw_data['DataFileNorm']
    print("HKJHKJFGKJHEGFLKJEGRLKJGELKJRGWERLJHWEGRKWEJHRGWEKRJHWGRWEKJ")
    print(raw_data.shape)

def get_raw_cell_data(path=None):
    if path is None:
        path = './data/data_normalized.mat'
    raw_data = loadmat(path)
    raw_data = raw_data['DataFileNorm']
    return(raw_data)

def get_feature_names(path=None):
    if path is None:
        path = './data/feature_names.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    # Include additional cluster label
    names.append("Cluster Label")
    return np.array(names)

def get_short_feature_names(path=None):
    if path is None:
        path = './data/short_feature_names.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    names.append("Cluster")
    return np.array(names)

def get_cluster_labels(path=None):
    if path is None:
        path = './data/cluster_labels.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    return np.array(names)

def get_data_frame():
    raw_data = get_raw_cell_data()
    feature_names = get_feature_names()
    df = pd.DataFrame(raw_data, columns=feature_names[:-1]).set_index('Unique ID')
    cluster_labels = get_cluster_labels()
    df['Cluster Label'] = cluster_labels
    # df = df[df['Maximum Curvature'] < 10]
    # df = df[df['Circularity'] < 4]
    # df = df[df['Circular Diameter'] < 1]
    return df