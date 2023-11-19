from django import forms
from .models import PlatNomor

class PlatNomorForm(forms.ModelForm):
    class Meta:
        model = PlatNomor
        fields = ['gambar']
