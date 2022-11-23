import io, base64
from dash import Dash, dcc, html, no_update, Input, Output
import pandas as pd
from load_data import get_feature_names
from load_data import get_image, get_csv_df
from charts import embed_scatter_heatmap, parallel_coordinates, scatter_matrix, correlation_matrix , embed_scatter, state_transition

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
print("RENDERING APP, Loading DF")
DEFAULT_DF = get_csv_df()

app.layout = html.Div([
    # Header
    html.H1("Cluster Vis"),
    "Upload Data:",
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    "Graphs helpful for analyzing your dataset:",
    dcc.Checklist(["Parallel Coordinates", "Scatter Matrix"], ['Parallel Coordinates'], inline=True, id='exploratory-graphs'),
    "Graphs helpful for clustering analysis:",
    dcc.Checklist(['Embedded Clustering', 'Embedded Clustering Heatmap', 'Feature Expression Matrix', 'State Transition Diagram'],
                    ['Embedded Clustering', 'Embedded Clustering Heatmap'],
                    inline=True,
                    id='analysis-graphs'),
    html.Div(id='exploratory-feature-selection'),
    html.Div(id='parallel-coords-div'),
    html.Div(id='scatter-matrix-div'),
    html.Div(id='embed-cluster-div'),
    html.Div(id='embed-feature-div',style={'width': '50%'}),
    html.Div(id='feature-expression-div'),
    html.Div(id='state-transition-div')
])

def csv_to_df(contents, filename):
## Callback to process file updload
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = df.set_index('Unique ID')
    except Exception as e:
        print(e)
        return "Error getting dataframe from csv"
    return df

############################## PARALLEL/SCATTER FEATURE SELECTION ############################## 
@app.callback(
    Output('exploratory-feature-selection', 'children'),
    Input('exploratory-graphs', 'value')
)
def exploratory_view(graphs):
    children = []
    if 'Parallel Coordinates' in graphs or 'Scatter Matrix' in graphs:
        children.append(html.H3("Feature Selection:"))
        children.append(dcc.Dropdown(get_feature_names(df=DEFAULT_DF),
                value=['Area', 'Cluster'],
                multi=True,
                id='selected-features')
        )
    return children
############################## PARALLEL COORDINATES ############################## 
##  Callback to update exploratory view plots
@app.callback(
    Output('parallel-coords-div', 'children'),
    Input('exploratory-graphs', 'value'),
)
def show_parallel_coords(graphs):
    children =[]
    if 'Parallel Coordinates' in graphs:
        children.append(html.H3("Parallel Coordinates"))
        children.append(dcc.Graph(id='parallel-coords'))
    return children

@app.callback(
    Output('parallel-coords', 'figure'),
    Input('selected-features', 'value'),
)
def update_parallel_coords(features):
    df = DEFAULT_DF #if contents is None else csv_to_df(contents, filename)
    parallel_fig = parallel_coordinates(features, df.sample(250))
    return parallel_fig

############################## SCATTER MATRIX ##############################
##  Callback to update exploratory view plots
@app.callback(
    Output('scatter-matrix-div', 'children'),
    Input('exploratory-graphs', 'value'),
)
def show_scatter_matrix(graphs):
    children =[]
    if 'Scatter Matrix' in graphs:
        children.append(html.H3("Scatter Matrix"))
        children.append(dcc.Graph(id='scatter-matrix'))
        children.append(dcc.Tooltip(id="scatter-matrix-tooltip", direction='bottom'))
    return children

@app.callback(
    Output('scatter-matrix', 'figure'),
    Input('selected-features', 'value'),
)
def update_scatter_matrix(features):
    df = DEFAULT_DF.copy() #if contents is None else csv_to_df(contents, filename)
    scatter_fig = scatter_matrix(features, df.sample(250))
    return scatter_fig

# Callback to show images on hover of scatter matrix
@app.callback(
    Output('scatter-matrix-tooltip', "show"),
    Output('scatter-matrix-tooltip', "bbox"),
    Output('scatter-matrix-tooltip', "children"),
    Output('scatter-matrix-tooltip', "direction"),
    Input('scatter-matrix', 'hoverData')
)
def scatter_image_on_hover(hover_data):
    return image_on_hover(hover_data)

############################## EMBEDDED CLUSTERS ##############################
##  Callback to update exploratory view plots
@app.callback(
    Output('embed-cluster-div', 'children'),
    Input('analysis-graphs', 'value'),
)
def show_embed_clusters(graphs):
    children =[]
    if 'Embedded Clustering' in graphs:
        df = DEFAULT_DF.copy()
        figure= embed_scatter(df)
        children.append(html.H3("Embedded Clustering:"))
        children.append(dcc.Graph(id='embed-clusters', figure=figure))
        children.append(dcc.Tooltip(id="embed-clusters-tooltip", direction='bottom'))
    return children

# Callback to show images on hover of scatter matrix
@app.callback(
    Output('embed-clusters-tooltip', "show"),
    Output('embed-clusters-tooltip', "bbox"),
    Output('embed-clusters-tooltip', "children"),
    Output('embed-clusters-tooltip', "direction"),
    Input('embed-clusters', 'hoverData')
)
def scatter_image_on_hover(hover_data):
    return image_on_hover(hover_data)

############################## EMBEDDED FEATURE ##############################
##  Callback to update exploratory view plots
@app.callback(
    Output('embed-feature-div', 'children'),
    Input('analysis-graphs', 'value'),
)
def show_embed_feature(graphs):
    children =[]
    if 'Embedded Clustering Heatmap' in graphs:
        df = DEFAULT_DF.copy()
        children.append("Select a feature to visualize in the embedding space:")
        children.append(dcc.Dropdown(get_feature_names(df=df),
                            value='Cluster',
                            id='embed-feature-selection'
                        ))
        children.append(dcc.Graph(id='embed-feature'))
    return children

@app.callback(
    Output('embed-feature', 'figure'),
    Input('embed-feature-selection', 'value'),
)
def update_embed_feature(feature):
    df = DEFAULT_DF.copy().sample(10000)
    return embed_scatter_heatmap(feature, df)

############################## Feature Expression Matrix ##############################
##  Callback to update exploratory view plots
@app.callback(
    Output('feature-expression-div', 'children'),
    Input('analysis-graphs', 'value'),
)
def show_expression_matrix(graphs):
    children =[]
    if 'Feature Expression Matrix' in graphs:
        df = DEFAULT_DF.copy()
        children.append(html.H3("Expression Matrix"))
        children.append(html.Center(dcc.Graph(id='expression-matrix', figure=correlation_matrix(df))))
    return children

############################## State Transition Diagram ##############################
##  Callback to update exploratory view plots
@app.callback(
    Output('state-transition-div', 'children'),
    Input('analysis-graphs', 'value'),
)
def show_state_transition(graphs):
    children =[]
    if 'State Transition Diagram' in graphs:
        df = DEFAULT_DF.copy()
        children.append(html.H3("State Transition Diagram"))
        children.append(html.Center(dcc.Graph(id='state-transition', figure=state_transition(df))))
    return children

## Function to show images on hover
def image_on_hover(hover_data):
    if hover_data is None:
        return False, no_update, no_update, no_update

    id = hover_data['points'][0]['customdata'][0]
    im = get_image(id)
    # dump image to base64
    buffer = io.BytesIO()
    im.save(buffer, format='jpeg')
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image

    hover_data = hover_data["points"][0]
    bbox = hover_data['bbox']

    y = hover_data["y"]
    # If y is the cluster data, cast it to a string
    if type(y) is str:
        y = int(y[-1])
    direction = "bottom" if y > 1.5 else "top"

    caption = 'Unique ID: ' + str(id)
    children = [
        html.Img(
            src=im_url,
            style={"width": "150px"},
        ),
        html.P(caption),
    ]

    return True, bbox, children, direction

if __name__ == '__main__':
    app.run_server(debug=True)
