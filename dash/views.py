import io, base64
from dash import Dash, dcc, html, no_update, Input, Output

from load_data import get_feature_names, get_image, get_cluster_names_pretty
from charts import embed_scatter, embed_scatter_heatmap, parallel_coordinates, scatter_matrix, correlation_matrix, state_transition

def exploratory_view():
# Feature Selection
    children = [
        html.H3("Features:"),
        dcc.Dropdown(get_feature_names(),
                value=['Experimental Condition', 'Area', 'Cluster Label'],
                multi=True,
                id='selected-features'),

        # Parallel Coordinates
        html.H3("Parallel Coordinates Plot"),
        dcc.Graph(id='parallel-coords'),

        # Scatter Matrix
        html.Hr(),
        html.H3("Scatter Matrix"),
        html.Div(
            className="container",
            children=[
                dcc.Graph(id='scatter-matrix'),
                dcc.Tooltip(id="scatter-matrix-tooltip", direction='bottom'),
            ])
    ]
    return children

def cluster_analysis_view():

    children = [
        html.H3('Embedded Features'),
        html.Div(
            className="container",
            children=[
                "Here is the provided embedding of the data. The color highlights the clustering of data in this space:",
                html.Center(dcc.Graph(id='embed-scatter', figure=embed_scatter())),
                dcc.Tooltip(id="embed-scatter-tooltip", direction='bottom'),
                "Analyze how features are distributed across the embedding space by selecting a feature to visualize from the dropdown:",
                html.Div(
                    children=[
                        dcc.Dropdown(get_feature_names(),
                            value='Experimental Condition',
                            id='embed-selected-feature',
                        ),
                        dcc.Graph(id='embed-scatter-heatmap'),
                        dcc.Tooltip(id="embed-scatter-tooltip-heatmap", direction='bottom'),],
                )
            ]   
        ),

        # Correlation Matrix
        html.H3("Expression Matrix"),
        html.Div(["Select Cluster:",
                dcc.Dropdown(get_cluster_names_pretty(), 
                                value=get_cluster_names_pretty()[0], 
                                id='selected-cluster'),
                ]),
        html.Center(dcc.Graph(id='correlation-matrix')),
        # State Transition Diagram
        html.Hr(),
        html.H3('State Transition Matrix'),
        html.Div(
        className="container",
        children=[
            html.Center(dcc.Graph(id='state-transition', figure=state_transition())),
            dcc.Tooltip(id="state-transition-tooltip", direction='bottom'),
        ]),
    ]
    return children
