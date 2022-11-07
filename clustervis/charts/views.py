from django.shortcuts import render
from .load_data import get_feature_names, get_data_frame
from .charts import get_basic_chart
from .charts import get_cell_parallel_coordinates_chart as gcpcc
def charts(request):
    # plot_div = get_basic_chart()
    plot_div = gcpcc()
    return render(request, "charts/index.html", context={'plot_div': plot_div})

from django.http import HttpResponse

def index(request):
    df = get_data_frame()
    out = ""
    for column in df.columns:
        out += column + "<br>"
    return HttpResponse(out)