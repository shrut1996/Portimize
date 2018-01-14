from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    birth_date = forms.DateField(help_text='Required. Format: YYYY-MM-DD')

    class Meta:
        model = User
        fields = ('username', 'birth_date', 'password1', 'password2', )


ASSET_CHOICES = [
    ('orange', 'Oranges'),
    ('cantaloupe', 'Cantaloupes'),
    ('mango', 'Mangoes'),
    ('honeydew', 'Honeydews'),
    ]

class PortfolioForm(forms.Form):
    start_date = forms.DateField(help_text='Required. Format: YYYY-MM-DD')
    end_date = forms.DateField(help_text='Required. Format: YYYY-MM-DD')
    asset1 = forms.CharField(label='Asset 1 ', widget=forms.Select(choices=ASSET_CHOICES))
    asset2 = forms.CharField(label='Asset 2 ', widget=forms.Select(choices=ASSET_CHOICES))
    asset3 = forms.CharField(label='Asset 3 ', widget=forms.Select(choices=ASSET_CHOICES))
    asset4 = forms.CharField(label='Asset 4 ', widget=forms.Select(choices=ASSET_CHOICES))


class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)