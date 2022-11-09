from plotly.offline import plot
from .load_data import get_data_frame, get_feature_names, get_short_feature_names
import plotly.express as px
import numpy as np

FEATURE_NAMES = get_feature_names()
SHORT_FEATURE_NAMES = get_short_feature_names()

def get_plots(feature_indexes):
    features = np.take(FEATURE_NAMES, feature_indexes)
    par_plot = get_plot(features, px.parallel_coordinates)
    labels = {FEATURE_NAMES[i]:SHORT_FEATURE_NAMES[i] for i in feature_indexes} 
    scatter_mat = get_plot(features, px.scatter_matrix, labels)
    return par_plot, scatter_mat

def get_plot(features, plot_type, labels=None):
    df = get_data_frame()
    colored_feat = 'Experimental Condition'
    df[colored_feat] = df[colored_feat].astype(int)
    df = df.sample(frac=0.001)
    if labels is None:
        labels = features 
    fig = plot_type(df, color=colored_feat,
                        dimensions=features,
                        labels=labels)
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)')
    return plot(fig,output_type='div',show_link=False,link_text="")