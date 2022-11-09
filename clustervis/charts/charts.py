from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
from .load_data import get_data_frame, get_feature_names
import plotly.express as px

# Test function to get a chart with toy data
def get_basic_parallel_coords_chart():
    df = px.data.iris()
    colored_feat = "species_id"
    features = ['sepal_width', 'sepal_length', 'petal_width',
                                            'petal_length']
    return get_parallel_coordinates_chart(df, colored_feat, features)

######################### Parallel Coordinates #########################

def get_cell_parallel_coordinates_chart(features):
    df = get_data_frame()
    colored_feat = 'Experimental Condition'
    return get_parallel_coordinates_chart(df.sample(frac=0.001), colored_feat, features)

def get_parallel_coordinates_chart(df, colored_feat, features):
    fig = px.parallel_coordinates(df, color=colored_feat,
                                dimensions=features,
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=df[colored_feat].max()/2)
    return plot(fig,output_type='div',show_link=False,link_text="")

######################### Scatter-plot Matrix #########################

def get_cell_scatter_matrix_chart(features):
    df = get_data_frame()
    colored_feat = 'Experimental Condition'
    return get_scatter_matrix_chart(df, colored_feat, features)

def get_scatter_matrix_chart(df, colored_feat, features):
    fig = px.scatter_matrix(df, color=colored_feat,
                                dimensions=features)
    return plot(fig,output_type='div',show_link=False,link_text="")