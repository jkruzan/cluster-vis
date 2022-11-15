import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image

def get_embedded_clusters(path=None):
    if path is None:
        path = './data/labels_and_embedding.txt'
    df = pd.read_csv(path, sep=',', header=None, names=['Unique ID', 'Cluster Label', 'Embed1', 'Embed2']).set_index('Unique ID')
    return df

def get_image(unique_id):
    path = '../../binary_images/binary_image_' + str(unique_id) + '.bmp'
    return Image.open(path)

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

def get_cluster_names_pretty(path=None):
    labels = get_cluster_labels_ints()
    min_label, max_label = labels.min(), labels.max()
    return ["Cluster "+ str(i) for i in range(min_label, max_label+1)]

def get_short_feature_names(path=None):
    if path is None:
        path = './data/short_feature_names.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    names.append("Cluster")
    return np.array(names)

def get_cluster_labels_ints(path=None):
    if path is None:
        path = './data/cluster_labels.txt'
    file = open(path, "r")
    names = file.readlines()
    names = [name.strip('\n') for name in names]
    return np.array(names, dtype=int)

def get_data_frame():
    raw_data = get_raw_cell_data()
    feature_names = get_feature_names()
    df = pd.DataFrame(raw_data, columns=feature_names[:-1]).set_index('Unique ID')
    cluster_labels = get_cluster_labels_ints()
    df['Cluster Label'] = cluster_labels
    # df = df[df['Maximum Curvature'] < 10]
    # df = df[df['Circularity'] < 4]
    # df = df[df['Circular Diameter'] < 1]
    return df