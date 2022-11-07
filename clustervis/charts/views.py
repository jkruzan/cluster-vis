from django.shortcuts import render
from plotly.offline import plot
from plotly.graph_objs import Scatter

def index(request):
    x_data = [0,1,2,3]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data,
                        mode='lines', name='test',
                        opacity=0.8, marker_color='green')],
               output_type='div',
               show_link=False,
               link_text="")
    return render(request, "charts/index.html", context={'plot_div': plot_div})
