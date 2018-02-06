from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.forms.extras.widgets import SelectDateWidget
import datetime

class SignUpForm(UserCreationForm):
    birth_date = forms.DateField(help_text='Required. Format: YYYY-MM-DD')

    class Meta:
        model = User
        fields = ('username', 'birth_date', 'password1', 'password2', )

ASSET_CHOICES = [
    ('^NSEI', 'Nifty 50'),
    ('^GSPC', 'S&P 500'),
    ('^DJI', 'Dow Jones Industrial Average'),
    ('^RUT', 'Russell 2000'),
    ]

class PortfolioForm(forms.Form):
    start_date = forms.DateField(widget=SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day"), years=range(1990,2030)), initial=datetime.date.today)
    end_date = forms.DateField(widget=SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day"), years=range(1990,2030)), initial=datetime.date.today)
    asset1 = forms.CharField(label='Asset 1 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight1 = forms.FloatField(max_value=1, min_value=0)
    asset2 = forms.CharField(label='Asset 2 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight2 = forms.FloatField(max_value=1, min_value=0)
    asset3 = forms.CharField(label='Asset 3 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight3 = forms.FloatField(max_value=1, min_value=0)
    asset4 = forms.CharField(label='Asset 4 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight4 = forms.FloatField(max_value=1, min_value=0)