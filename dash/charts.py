from plotly.offline import plot
from load_data import get_data_frame, get_feature_names, get_short_feature_names
import plotly.express as px
import numpy as np

FEATURE_NAMES = get_feature_names()
SHORT_FEATURE_NAMES = get_short_feature_names()

def get_plots(features):
    par_plot = get_plot(features, px.parallel_coordinates)
    scatter_mat = get_plot(features, scatter_matrix, True)
    return par_plot, scatter_mat

def get_plot(features, plot_type, use_labels=False):
    if use_labels:
        labels = {feature: SHORT_FEATURE_NAMES[np.where(FEATURE_NAMES==feature)[0][0]] for feature in features} 
    else:
        labels = features
    df = get_data_frame()
    colored_feat = 'Cluster Label'
    df[colored_feat] = df[colored_feat].astype(int)
    df = df.sample(frac=0.001)
    fig = plot_type(df, color=colored_feat,
                        dimensions=features,
                        labels=labels)
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)')
    return fig

def scatter_matrix(df, color, dimensions, labels):
    fig = px.scatter_matrix(df, color=color, dimensions=dimensions, labels=labels, hover_data={df.index.name: df.index})
    fig.update_traces(
        hoverinfo='none',
        hovertemplate=None,
    )
    fig.update_layout(hovermode='closest')
    return fig

# def bar_chart():
#     df = get_data_frame()
#     df = df.sample(frac=0.001)
#     fig = px.bar(df, x='Experimental Condition', y='Cluster Label', color='Cluster Label')
#     return fig
