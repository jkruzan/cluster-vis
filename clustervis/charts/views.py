from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from .load_data import get_feature_names, get_data_frame
from .charts import get_basic_chart
from .charts import get_cell_parallel_coordinates_chart as gcpcc
from .forms import FeaturesForm
from django.http import HttpResponseRedirect
import numpy as np

FEATURE_NAMES = get_feature_names()

class ParallelCoords(View):
    template_name = 'charts/index.html'
    selected_features = ['Experimental Condition', 'Velocity', 'Elongation']

    def post(self, request):
        plot_div = self.get_plot(request, True)
        return render(request, "charts/index.html", context={'plot_div': plot_div})

    def get(self, request):
        plot_div = self.get_plot(request, False)
        return render(request, "charts/index.html", context={'plot_div': plot_div})

    def get_plot(self, request, is_post):
        if is_post:
            form = FeaturesForm(request.POST)
            if form.is_valid():
                features = form.cleaned_data['features_field']
                self.selected_features = np.take(FEATURE_NAMES, features)
        return gcpcc(self.selected_features)

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