from plotly.offline import plot
from .load_data import get_data_frame, get_feature_names
import plotly.express as px
import numpy as np

FEATURE_NAMES = get_feature_names()

def get_plots(features):
    features = np.take(FEATURE_NAMES, features)
    # return get_parallel_coordinates(features), get_scatter_matrix(features)
    return get_plot(features, px.parallel_coordinates), get_plot(features, px.scatter_matrix)

def get_plot(features, plot_type):
    df = get_data_frame()
    colored_feat = 'Experimental Condition'
    df[colored_feat] = df[colored_feat].astype(int)
    df = df.sample(frac=0.001)
    fig = plot_type(df, color=colored_feat,
                                dimensions=features)
    return plot(fig,output_type='div',show_link=False,link_text="")