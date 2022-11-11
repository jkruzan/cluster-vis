from load_data import get_binary_images as gbi
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from load_data import get_feature_names

from load_data import get_data_frame
from charts import get_plots, get_plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    # Header
    html.H1("Cluster Vis"),
    # Feature Selection
    html.H3("Features:"),
    dcc.Dropdown(get_feature_names(), ['Experimental Condition', 'Area', 'Cluster Label'], multi=True, id='selected-features'),
    # Parallel Coordinates
    html.H3("Parallel Coordinates Plot"),
    dcc.Graph(id='parallel-coords'),
    # Scatter Matrix
    html.Hr(),
    html.H3("Scatter Matrix"),
    dcc.Graph(id='scatter-matrix'),
])

@app.callback(
    Output('parallel-coords', 'figure'),
    Output('scatter-matrix', 'figure'),
    Input('selected-features', 'value'))
def update_plots(new_features):
    print("Updating parallel plot ********************************")
    print(new_features)
    parallel_fig, scatter_matrix = get_plots(new_features)
    parallel_fig = get_plot(new_features, px.parallel_coordinates)
    print("HI ********************************")
    gbi()
    print("BYEEE")
    print("Shouldve printed")
    return parallel_fig, scatter_matrix


if __name__ == '__main__':
    app.run_server(debug=True)




