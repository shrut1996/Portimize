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
    ('CRISIL.NS', 'Crisil'),
    ('SUNPHARMA.NS', 'Sun Pharma'),
    ('GOOGL', 'Alphabet'),
    ('AAPL', 'Apple'),
    ('FB', 'Facebook'),
    ('BABA', 'Alibaba Group'),
    ('AMZN', 'Amazon.com'),
    ('MSFT', 'Microsoft'),
    ('CSCO', 'Cisco'),
    ('WMT', 'Walmart'),
    ('ORCL', 'Oracle'),
    ('ADBE', 'Adobe Systems'),
    ('QCOM', 'Qualcomm'),
    ('NVDA', 'Nvidia'),
    ('TXN', 'Texas Instruments'),
    ('T', 'AT&T'),
    ('BRK-A', 'Berkshire Hathaway'),
    ('IOC.NS', 'Indian Oil'),
    ('RS', 'Reliance Steel'),
    ('JNJ', 'Johnson & Johnson'),
    ('VOD', 'Vodafone'),
    ('CHL', 'China Mobile'),
    ('HSBC', 'HSBC Holdings'),
    ('BCS', 'Barclays PLC'),
    ('ACN', 'Accenture PLC'),
    ('KO', 'Coca-Cola'),
    ('IBM', 'IBM'),
    ('INTC', 'Intel'),
    ('AXP', 'American Express'),
    ('DIS', 'Disney'),
    ('GS', 'Goldman Sachs'),
    ('JPM', 'JPMorgan Chase'),
    ('V', 'Visa'),
    ('NKE', 'Nike'),
    ('GE', 'General Electric'),
    ('FDX', 'FedEx Corporation'),
    ('SBUX', 'Starbucks'),
    ('GC=F', 'Gold'),
    ]

PERIOD_CHOICES = [
    ('5', '5-day period'),
    ('10', '10-day period'),
    ('15', '15-day period'),
    ('0', 'Long-term Holding')
]


class PortfolioForm(forms.Form):
    holding = forms.IntegerField(label='Holding Period ', widget=forms.Select(choices=PERIOD_CHOICES))
    start_date = forms.DateField(help_text="Only required for Long-term Holding", widget=SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day"), years=range(2008,2019)), initial=datetime.date.today)
    end_date = forms.DateField(help_text="Only required for Long-term Holding", widget=SelectDateWidget(empty_label=("Choose Year", "Choose Month", "Choose Day"), years=range(2008,2019)), initial=datetime.date.today)
    asset1 = forms.CharField(label='Asset 1 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight1 = forms.FloatField(max_value=1, min_value=0)
    asset2 = forms.CharField(label='Asset 2 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight2 = forms.FloatField(max_value=1, min_value=0)
    asset3 = forms.CharField(label='Asset 3 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight3 = forms.FloatField(max_value=1, min_value=0)
    asset4 = forms.CharField(label='Asset 4 ', widget=forms.Select(choices=ASSET_CHOICES))
    weight4 = forms.FloatField(max_value=1, min_value=0)