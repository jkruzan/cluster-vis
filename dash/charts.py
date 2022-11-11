from plotly.offline import plot
from load_data import get_data_frame, get_feature_names, get_short_feature_names, get_embedded_clusters
import plotly.express as px
import numpy as np


def parallel_coordinates(features):
    # Get data and sample 1/1000 of it
    df = get_data_frame().sample(frac=0.001)
    # Cast cluster labels to int (required for parallel coordinates color)
    colored_feat = 'Cluster Label'
    df[colored_feat] = df[colored_feat].astype(int)
    
    # Get figure
    fig = px.parallel_coordinates(df, color=colored_feat,
                        dimensions=features)
    fig.update_layout(paper_bgcolor='rgb(0,0,0,0)')
    return fig

def scatter_matrix(features):
    # Get data and sample 1/1000 of it
    df = get_data_frame().sample(frac=0.001)

    # Get shortened feature names for scatter plots
    all_feature_names = get_feature_names()
    short_feature_names = get_short_feature_names()
    labels = {feature: short_feature_names[np.where(all_feature_names==feature)[0][0]] for feature in features} 
    
    # Get figure
    fig = px.scatter_matrix(df, color='Cluster Label', 
                            dimensions=features, 
                            labels=labels, 
                            hover_data={df.index.name: df.index},
                            category_orders=get_cluster_order(df)
                            )
    # Update traces (required for showing image on hover)
    fig.update_traces(hoverinfo='none',hovertemplate=None,)
    fig.update_layout(legend=dict(bgcolor='white'), paper_bgcolor='rgb(0,0,0,0)')
    return fig

def embed_scatter():
    '''
    Scatter plot for with embedded dimensions and color showing cluster label
    '''
    # Get data (currently not sampling data)
    df = get_embedded_clusters()

    ##  Manually order the clusters for the legend
    # Find max and min of cluster labels
    cluster_min = df['Cluster Label'].min()
    cluster_max = df['Cluster Label'].max()
    # Convert integer labels to strings (required for discrete colors)
    df['Cluster Label'] = 'Cluster ' + df['Cluster Label'].astype(str)
    cluster_order = {'Cluster Label':["Cluster "+ str(i) for i in range(cluster_min, cluster_max+1)]}

    # Get figure
    fig = px.scatter(df, x='Embed1', y='Embed2', 
                    color='Cluster Label', 
                    hover_data={df.index.name: df.index},
                    category_orders=cluster_order
                    )

    fig.update_traces(
        hoverinfo='none',
        hovertemplate=None,
    )
    fig.update_layout(autosize=False, margin=dict(t=10))
    return fig


def get_cluster_order(df):
    """
    Helper function to get the cluster labels in order from (min, max). 
    Otherwise plotly would order by whatever appears first in the data 
    """
    ##  Manually order the clusters for the legend
    # Find max and min of cluster labels
    cluster_min = df['Cluster Label'].astype(int).min()
    cluster_max = df['Cluster Label'].astype(int).max()
    # Convert integer labels to strings (required for discrete colors)
    df['Cluster Label'] = 'Cluster ' + df['Cluster Label'].astype(str)
    return {'Cluster Label':[f'Cluster {i}' for i in range(cluster_min, cluster_max+1)]}
