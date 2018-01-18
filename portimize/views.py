from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm
import pandas_datareader.data as web
import pandas as pd
from optimization import MarkowitzOptimize
from Predictions import *


class Home(generic.TemplateView):
    template_name = 'portimize/home.html'

    def get(self, request, *args, **kwargs):
        form = PortfolioForm
        return render(request, self.template_name, {'form' : form})

    def post(self, request):
        form = PortfolioForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']

            #asset1
            asset1 = form.cleaned_data['asset1']
            weight1 = form.cleaned_data['weight1']
            prices1 = web.DataReader(asset1, 'yahoo', start_date)
            df1 = pd.DataFrame({'Asset1': prices1['Close']})

            #asset2
            asset2 = form.cleaned_data['asset2']
            weight2 = form.cleaned_data['weight2']
            prices2 = web.DataReader(asset2, 'yahoo', start_date)
            df2 = pd.DataFrame({'Asset2': prices2['Close']})

            #asset3
            asset3 = form.cleaned_data['asset3']
            weight3 = form.cleaned_data['weight3']
            prices3 = web.DataReader(asset3, 'yahoo', start_date)
            df3 = pd.DataFrame({'Asset3': prices3['Close']})

            #asset4
            asset4 = form.cleaned_data['asset4']
            weight4 = form.cleaned_data['weight4']
            prices4 = web.DataReader(asset4, 'yahoo', start_date)
            df4 = pd.DataFrame({'Asset4': prices4['Close']})

            portfolio_prices = df1.merge(pd.DataFrame(df2), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df3), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df4), left_index=True, right_index=True)

            weights = []
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            weights.append(weight4)

            # new_portfolio_prices = lstm(portfolio_prices)

            opti_model = MarkowitzOptimize(portfolio_prices, weights)
            new_weights = opti_model.minimizeSharpeRatio()

        args = {'form' : form, 'start_date' : start_date,
                'end_date': end_date, 'asset1' : asset1,
                'asset2' : asset2, 'asset3' : asset3,
                'asset4' : asset4, 'new_weights1' : new_weights[0],
                'new_weights2' : new_weights[1],
                'new_weights3' : new_weights[2],
                'new_weights4' : new_weights[3]}

        return render(request, 'portimize/results.html', args)

def results(request):
    return render(request, 'portimize/results.html')

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.refresh_from_db()  # load the profile instance created by the signal
            user.profile.birth_date = form.cleaned_data.get('birth_date')
            user.save()
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=user.username, password=raw_password)
            login(request, user)
            return redirect('home')
    else:
        form = SignUpForm()
    return render(request, 'portimize/signup.html', {'form': form})


