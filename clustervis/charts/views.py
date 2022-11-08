from django.shortcuts import render
from django.http import HttpResponse
from .load_data import get_feature_names, get_data_frame
from .charts import get_basic_chart
from .charts import get_cell_parallel_coordinates_chart as gcpcc
from .forms import FeaturesForm
from django.http import HttpResponseRedirect
import numpy as np

FEATURE_NAMES = get_feature_names()

def charts(request):
    selected_features = None
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            selected_features = form.cleaned_data['features_field']
            selected_features = np.take(FEATURE_NAMES, selected_features)
    plot_div = gcpcc(selected_features)
    return render(request, "charts/index.html", context={'plot_div': plot_div})


def index(request):
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('DIRECT RESPONSE')
    else: 
        form = FeaturesForm()
    return render(request, 'charts/form.html', {'form': form})

def features_form(request):
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('/chart')
    else: 
        form = FeaturesForm()
    return render(request, 'charts/form.html', {form: form})

def test(request):
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            selected_features = form.cleaned_data['features_field']
            feat_names = np.take(FEATURE_NAMES, selected_features)
            feat_names = [name + '<br>' for name in feat_names]
            return HttpResponse(feat_names)
    else:
        return HttpResponse("Navigated to this page without posting")