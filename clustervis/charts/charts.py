from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter
from .load_data import get_data_frame, get_feature_names
import plotly.express as px

def get_basic_chart():
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div',
               show_link=False,
               link_text="")
    return plot_div

def get_basic_parallel_coords_chart():

    df = px.data.iris()
    colored_feat = "species_id"
    features = ['sepal_width', 'sepal_length', 'petal_width',
                                            'petal_length']
    return get_parallel_coordinates_chart(df, colored_feat, features)

def get_cell_parallel_coordinates_chart(features=None):
    df = get_data_frame()
    colored_feat = 'Experimental Condition'
    if features is None:
        features = ['Experimental Condition', 'Velocity', 'Elongation']
    return get_parallel_coordinates_chart(df, colored_feat, features)

def get_parallel_coordinates_chart(df, colored_feat, features):
    fig = px.parallel_coordinates(df.sample(frac=0.001), color=colored_feat,
                                dimensions=features,
                                color_continuous_scale=px.colors.diverging.Tealrose,
                                color_continuous_midpoint=df[colored_feat].median())
    return plot(fig,output_type='div',show_link=False,link_text="")