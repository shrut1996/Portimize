from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm
import pandas_datareader.data as web
import pandas as pd
import matplotlib.pyplot as plt
from optimization import MarkowitzOptimize
from predictions import predict


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
            # period = pd.date_range(start=start_date, end=end_date)

            #asset1
            asset1 = form.cleaned_data['asset1']
            weight1 = form.cleaned_data['weight1']
            prices1 = web.DataReader(asset1, 'yahoo', start_date, end_date)
            df1 = pd.DataFrame(predict(prices1))
            df1.rename(columns={df1.columns[0]: 'Close1'}, inplace = True)

            #asset2
            asset2 = form.cleaned_data['asset2']
            weight2 = form.cleaned_data['weight2']
            prices2 = web.DataReader(asset2, 'yahoo', start_date, end_date)
            df2 = pd.DataFrame(predict(prices2))
            df2.rename(columns={df2.columns[0]: 'Close2'}, inplace=True)

            #asset3
            asset3 = form.cleaned_data['asset3']
            weight3 = form.cleaned_data['weight3']
            prices3 = web.DataReader(asset3, 'yahoo', start_date, end_date)
            df3 = pd.DataFrame(predict(prices3))
            df3.rename(columns={df3.columns[0]: 'Close3'}, inplace=True)

            #asset4
            asset4 = form.cleaned_data['asset4']
            weight4 = form.cleaned_data['weight4']
            prices4 = web.DataReader(asset4, 'yahoo', start_date, end_date)
            df4 = pd.DataFrame(predict(prices4))
            df4.rename(columns={df4.columns[0]: 'Close4'}, inplace=True)

            portfolio_prices = df1.merge(pd.DataFrame(df2), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df3), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df4), left_index=True, right_index=True)

            weights = []
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            weights.append(weight4)

            opti_model = MarkowitzOptimize(portfolio_prices, weights)
            new_weights = opti_model.minimizeSharpeRatio()

            portfolio_prices = portfolio_prices / portfolio_prices.iloc[0]
            attributes = list(portfolio_prices.columns.values)

            return1 = portfolio_prices[attributes].mul(weights).sum(1)
            return2 = portfolio_prices[attributes].mul(new_weights).sum(1)

            ts_list1 = return1.index.tolist()  # a list of Timestamp's
            ts_list2 = return2.index.tolist()

            date_list1 = [ts.date() for ts in ts_list1]  # a list of datetime.date's
            date_list2 = [ts.date() for ts in ts_list2]

            index1 = [str(date) for date in date_list1]  # a list of strings
            index2 = [str(date) for date in date_list2]

            
        args = {'form' : form, 'start_date' : start_date,
                'end_date': end_date, 'asset1' : asset1,
                'asset2' : asset2, 'asset3' : asset3,
                'asset4' : asset4, 'new_weights1' : new_weights[0],
                'new_weights2' : new_weights[1],
                'new_weights3' : new_weights[2],
                'new_weights4' : new_weights[3],
                'return1_values': return1.values.tolist(),
                'return1_index' : index1,
                'return2_values': return2.values.tolist(),
                'return2_index': index2}

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