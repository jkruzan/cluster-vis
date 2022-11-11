import io, base64
from dash import Dash, dcc, html, no_update, Input, Output
import plotly.express as px

from load_data import get_feature_names, get_image
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
    html.Div(
        className="container",
        children=[
            dcc.Graph(id='scatter-matrix'),
            dcc.Tooltip(id="scatter-matrix-tooltip", direction='bottom'),
        ]),
])


##  Callback to update features being shown in scatter matrix and
##  parallel coordinate plots
@app.callback(
    Output('parallel-coords', 'figure'),
    Output('scatter-matrix', 'figure'),
    Input('selected-features', 'value'))
def update_plots(new_features):
    parallel_fig, scatter_matrix = get_plots(new_features)
    parallel_fig = get_plot(new_features, px.parallel_coordinates)
    return parallel_fig, scatter_matrix

## Callback to show images on hover
@app.callback(
    Output('scatter-matrix-tooltip', "show"),
    Output('scatter-matrix-tooltip', "bbox"),
    Output('scatter-matrix-tooltip', "children"),
    Output('scatter-matrix-tooltip', "direction"),
    Input('scatter-matrix', 'hoverData')
)
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




