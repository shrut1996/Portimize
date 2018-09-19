from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm

import pandas as pd
import pandas_datareader.data as web
from optimization import MarkowitzOptimize
from predictions import predict
from pandas.tseries.holiday import *


class Home(generic.TemplateView):
    template_name = 'portimize/home.html'

    def get(self, request, *args, **kwargs):
        form = PortfolioForm
        return render(request, self.template_name, {'form' : form})

    def post(self, request):
        form = PortfolioForm(request.POST)
        if form.is_valid():
            holding = form.cleaned_data['holding']
            if holding != 0:
                span = 'Short-Term'
            else:
                span = 'Long-Term'

            #asset1
            asset1 = form.cleaned_data['asset1']
            name1 = dict(form.fields['asset1'].widget.choices)[asset1]
            weight1 = form.cleaned_data['weight1']

            # asset2
            asset2 = form.cleaned_data['asset2']
            name2 = dict(form.fields['asset2'].widget.choices)[asset2]
            weight2 = form.cleaned_data['weight2']

            # asset3
            asset3 = form.cleaned_data['asset3']
            name3 = dict(form.fields['asset3'].widget.choices)[asset3]
            weight3 = form.cleaned_data['weight3']

            # asset4
            asset4 = form.cleaned_data['asset4']
            name4 = dict(form.fields['asset4'].widget.choices)[asset4]
            weight4 = form.cleaned_data['weight4']

            # Short-term holding
            if span == 'Short-Term':
                start_date = datetime.today()+timedelta(days=-100)
                end_date = datetime.today()
                #holidays = get_calendar('USFederalHolidayCalendar').holidays(start_date, end_date)
                #period = [x for x in period if x not in holidays.date]
                #period = pd.DatetimeIndex(period)
                prices1 = web.DataReader(asset1, 'yahoo', start_date, end_date)
                df1 = pd.DataFrame(predict(prices1, holding))
                df1.rename(columns={df1.columns[0]: 'Close1'}, inplace=True)

                prices2 = web.DataReader(asset2, 'yahoo', start_date, end_date)
                df2 = pd.DataFrame(predict(prices2, holding))
                df2.rename(columns={df2.columns[0]: 'Close2'}, inplace=True)

                prices3 = web.DataReader(asset3, 'yahoo', start_date, end_date)
                df3 = pd.DataFrame(predict(prices3, holding))
                df3.rename(columns={df3.columns[0]: 'Close3'}, inplace=True)

                prices4 = web.DataReader(asset4, 'yahoo', start_date, end_date)
                df4 = pd.DataFrame(predict(prices4, holding))
                df4.rename(columns={df4.columns[0]: 'Close4'}, inplace=True)


            #Long-term Holding
            else:
                start_date = form.cleaned_data['start_date']
                end_date = form.cleaned_data['end_date']
                df1 = web.DataReader(asset1, 'yahoo', start_date, end_date)
                df2 = web.DataReader(asset2, 'yahoo', start_date, end_date)
                df3 = web.DataReader(asset3, 'yahoo', start_date, end_date)
                df4 = web.DataReader(asset4, 'yahoo', start_date, end_date)

                df1 = pd.DataFrame({'Close1': df1['Close']})
                df2 = pd.DataFrame({'Close2': df2['Close']})
                df3 = pd.DataFrame({'Close3': df3['Close']})
                df4 = pd.DataFrame({'Close4': df4['Close']})


            portfolio_prices = df1.merge(df2, left_index=True, right_index=True) \
                .merge(df3, left_index=True, right_index=True) \
                .merge(df4, left_index=True, right_index=True)

            print portfolio_prices

            period = pd.bdate_range(start_date, end_date).date

            weights = []
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            weights.append(weight4)

            opti_model = MarkowitzOptimize(portfolio_prices, weights)
            new_weights = opti_model.minimizeSharpeRatio()

            attributes = list(portfolio_prices.columns.values)
            return_prices = portfolio_prices / portfolio_prices.iloc[0]
            return1 = return_prices[attributes].mul(weights).sum(1)

            portfolio = pd.DataFrame(portfolio_prices)
            attributes = list(portfolio.columns.values)
            portfolio = portfolio[attributes].sum(1)
            return2 = portfolio / portfolio.iloc[0]


            ts_list = period.tolist()
            date_string = [str(date) for date in ts_list]


        args = {'form' : form, 'start_date' : start_date,
                'end_date': end_date, 'name1' : name1,
                'name2' : name2, 'name3' : name3,
                'name4' : name4, 'new_weights1' : new_weights[0],
                'new_weights2' : new_weights[1],
                'new_weights3' : new_weights[2],
                'new_weights4' : new_weights[3],
                'values1': return1.values.tolist(),
                'values2': return2.values.tolist(),
                'dates': date_string}

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