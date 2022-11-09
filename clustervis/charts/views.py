from django.shortcuts import render
from django.views import View
from .charts import get_plots
from .forms import FeaturesForm

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