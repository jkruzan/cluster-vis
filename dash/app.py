import io, base64
from dash import Dash, dcc, html, no_update, Input, Output, State
import pandas as pd
from views import exploratory_view, cluster_analysis_view
from load_data import get_image, get_csv_df
from charts import embed_scatter_heatmap, parallel_coordinates, scatter_matrix, correlation_matrix

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
    html.Div(id='output-data'),
    "Select a view:",
    dcc.RadioItems(['Exploratory', 'Cluster Analysis'], 'Exploratory', id='view-selector'),
    html.Div(id='view'),
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

@app.callback(
    Output('view', 'children'),
    Input('view-selector', 'value'))
def parse_update(view):
    df = DEFAULT_DF
    if view == 'Exploratory':
        v = exploratory_view(df)
    else:
        v = cluster_analysis_view(df)
    return v

##  Callback to update exploratory view plots
@app.callback(
    Output('parallel-coords', 'figure'),
    Output('scatter-matrix', 'figure'),
    Input('selected-features', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def update_plots(new_features, contents, filename):
    print("UPDATE THE EXPLORATORY VIEW CALL")
    df = DEFAULT_DF if contents is None else csv_to_df(contents, filename)
    parallel_fig = parallel_coordinates(new_features, df.sample(250))
    scatter_mat = scatter_matrix(new_features, df.sample(250))
    return parallel_fig, scatter_mat

##  Callback to update features being shown in embed matrix
@app.callback(
    Output('embed-scatter-heatmap', 'figure'),
    Input('embed-selected-feature', 'value'))
def update_plots(feature):
    df = DEFAULT_DF.sample(10000)
    return embed_scatter_heatmap(feature, df)

@app.callback(
    Output('correlation-matrix', 'figure'),
    Input('selected-cluster', 'value'))
def get_matrix(cluster):
    df = DEFAULT_DF
    return correlation_matrix(cluster, df)

# Callback to show images on hover of embedding
@app.callback(
    Output('embed-scatter-tooltip', "show"),
    Output('embed-scatter-tooltip', "bbox"),
    Output('embed-scatter-tooltip', "children"),
    Output('embed-scatter-tooltip', "direction"),
    Input('embed-scatter', 'hoverData')
)
def scatter_embed_image_on_hover(hover_data):
    return image_on_hover(hover_data)


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
