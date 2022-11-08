from django import forms
from .load_data import get_feature_names

class FeaturesForm(forms.Form):
    feature_names = get_feature_names()
    feature_choices = [(i, choice) for i, choice in enumerate(feature_names)][1:]
    features_field = forms.MultipleChoiceField(label='Features', choices=feature_choices, widget=forms.CheckboxSelectMultiple)