from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.http import HttpResponseRedirect
import plotly.express as px
from .load_data import get_feature_names
from .charts import get_plots
from .forms import FeaturesForm

import numpy as np

FEATURE_NAMES = get_feature_names()

class Charts(View):
    template_name = 'charts/index.html'
    feature_indexes = [2,4,31]

    def post(self, request):
        form = FeaturesForm(request.POST)
        if form.is_valid():
            self.feature_indexes = form.cleaned_data['features_field']
        return self.get_rendering(request)

    def get(self, request):
        return self.get_rendering(request)

    def get_rendering(self, request):
        parallel_coords, scatter_matrix = get_plots(self.feature_indexes)
        form_div = FeaturesForm(initial={'features_field': self.feature_indexes})
        context={'parallel_coords_plot': parallel_coords, 'scatter_matrix_plot': scatter_matrix, 'form': form_div}
        return render(request=request, template_name=self.template_name, context=context)

def index(request):
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('DIRECT RESPONSE')
    else: 
        form = FeaturesForm()
    return render(request, 'charts/form.html', {'form': form})