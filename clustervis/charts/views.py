from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from .load_data import get_feature_names
from .charts import get_cell_parallel_coordinates_chart as gcpcc
from .charts import get_cell_scatter_matrix_chart as gcsmc
from .forms import FeaturesForm
from django.http import HttpResponseRedirect
import numpy as np

FEATURE_NAMES = get_feature_names()

class Charts(View):
    template_name = 'charts/index.html'
    feature_indexes = [2,4,31]

    def post(self, request):
        parallel_coords, scatter_matrix = self.get_plots(request, True)
        form_div = self.get_form()
        return render(request, self.template_name, 
                context={'parallel_coords_plot': parallel_coords, 'scatter_matrix_plot': scatter_matrix, 'form': form_div})

    def get(self, request):
        parallel_coords, scatter_matrix = self.get_plots(request, False)
        form_div = self.get_form()
        return render(request, self.template_name, 
                context={'parallel_coords_plot': parallel_coords, 'scatter_matrix_plot': scatter_matrix, 'form': form_div})

    def get_plots(self, request, is_post):
        if is_post:
            form = FeaturesForm(request.POST)
            if form.is_valid():
                self.feature_indexes = form.cleaned_data['features_field']
        selected_features = np.take(FEATURE_NAMES, self.feature_indexes)
        parallel_coords = gcpcc(selected_features)
        scatter_matrix = gcsmc(selected_features)
        return parallel_coords, scatter_matrix

    def get_form(self):
        return FeaturesForm(initial={'features_field': self.feature_indexes})

def index(request):
    if request.method == 'POST':
        form = FeaturesForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('DIRECT RESPONSE')
    else: 
        form = FeaturesForm()
    return render(request, 'charts/form.html', {'form': form})


# # Scatter Matrix
# class ScatterMatrix(View):
    