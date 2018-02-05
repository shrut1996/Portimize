from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm
import pandas_datareader.data as web
import pandas as pd
from optimization import MarkowitzOptimize
from sklearn import preprocessing
from predictions import lstm


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
            prices1 = web.DataReader(asset1, 'yahoo', start_date, end_date)
            prices1.drop(['Volume'], 1, inplace=True)
            prices1.drop(['Close'], 1, inplace=True)
            # prices1 = prices1.dropna(inplace=True)
            # prices1 = prices1.fillna(lambda x: x.median())
            for i in prices1.columns:  # df.columns[w:] if you have w column of line description
                prices1[i] = prices1[i].fillna(prices1[i].median())
            min_max_scaler = preprocessing.MinMaxScaler()
            prices1['Open'] = min_max_scaler.fit_transform(prices1.Open.values.reshape(-1, 1))
            prices1['High'] = min_max_scaler.fit_transform(prices1.High.values.reshape(-1, 1))
            prices1['Low'] = min_max_scaler.fit_transform(prices1.Low.values.reshape(-1, 1))
            prices1['Adj Close'] = min_max_scaler.fit_transform(prices1['Adj Close'].values.reshape(-1, 1))
            prices1.to_csv('a.csv')
            df1 = pd.DataFrame(lstm(prices1))
            df1.rename(columns={df1.columns[0]: 'Close'}, inplace = True)
            # p = pd.date_range(start=start_date, end=end_date)
            # df1.set_index(p[:84], inplace=True)
            print df1.head()

            #asset2
            asset2 = form.cleaned_data['asset2']
            weight2 = form.cleaned_data['weight2']
            prices2 = web.DataReader(asset2, 'yahoo', start_date, end_date)
            prices2.drop(['Volume'], 1, inplace=True)
            prices2.drop(['Close'], 1, inplace=True)
            for i in prices2.columns:  # df.columns[w:] if you have w column of line description
                prices2[i] = prices2[i].fillna(prices2[i].median())
            min_max_scaler = preprocessing.MinMaxScaler()
            prices2['Open'] = min_max_scaler.fit_transform(prices2.Open.values.reshape(-1, 1))
            prices2['High'] = min_max_scaler.fit_transform(prices2.High.values.reshape(-1, 1))
            prices2['Low'] = min_max_scaler.fit_transform(prices2.Low.values.reshape(-1, 1))
            prices2['Adj Close'] = min_max_scaler.fit_transform(prices1['Adj Close'].values.reshape(-1, 1))
            prices2.to_csv('a.csv')
            df2 = pd.DataFrame(lstm(prices2))
            df2.rename(columns={df2.columns[0]: 'Close'}, inplace=True)
            # p = pd.date_range(start=start_date, end=end_date)
            # df2.set_index(p[:84], inplace=True)
            print df2.head()

            #asset3
            asset3 = form.cleaned_data['asset3']
            weight3 = form.cleaned_data['weight3']
            prices3 = web.DataReader(asset3, 'yahoo', start_date, end_date)
            prices3.drop(['Volume'], 1, inplace=True)
            prices3.drop(['Close'], 1, inplace=True)
            for i in prices3.columns:  # df.columns[w:] if you have w column of line description
                prices3[i] = prices3[i].fillna(prices3[i].median())
            min_max_scaler = preprocessing.MinMaxScaler()
            prices3['Open'] = min_max_scaler.fit_transform(prices3.Open.values.reshape(-1, 1))
            prices3['High'] = min_max_scaler.fit_transform(prices3.High.values.reshape(-1, 1))
            prices3['Low'] = min_max_scaler.fit_transform(prices3.Low.values.reshape(-1, 1))
            prices3['Adj Close'] = min_max_scaler.fit_transform(prices3['Adj Close'].values.reshape(-1, 1))
            prices3.to_csv('a.csv')
            df3 = pd.DataFrame(lstm(prices3))
            df3.rename(columns={df3.columns[0]: 'Close'}, inplace=True)
            # p = pd.date_range(start=start_date, end=end_date)
            # df3.set_index(p[:84], inplace=True)
            print df3.head()

            #asset4
            asset4 = form.cleaned_data['asset4']
            weight4 = form.cleaned_data['weight4']
            prices4 = web.DataReader(asset4, 'yahoo', start_date, end_date)
            prices4.drop(['Volume'], 1, inplace=True)
            prices4.drop(['Close'], 1, inplace=True)
            for i in prices4.columns:  # df.columns[w:] if you have w column of line description
                prices4[i] = prices4[i].fillna(prices4[i].median())
            min_max_scaler = preprocessing.MinMaxScaler()
            prices4['Open'] = min_max_scaler.fit_transform(prices4.Open.values.reshape(-1, 1))
            prices4['High'] = min_max_scaler.fit_transform(prices4.High.values.reshape(-1, 1))
            prices4['Low'] = min_max_scaler.fit_transform(prices4.Low.values.reshape(-1, 1))
            prices4['Adj Close'] = min_max_scaler.fit_transform(prices4['Adj Close'].values.reshape(-1, 1))
            prices4.to_csv('a.csv')
            df4 = pd.DataFrame(lstm(prices4))
            df4.rename(columns={df4.columns[0]: 'Close'}, inplace=True)
            # p = pd.date_range(start=start_date, end=end_date)
            # df4.set_index(p[:84], inplace=True)
            print df4.head()

            portfolio_prices = df1.merge(pd.DataFrame(df2), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df3), left_index=True, right_index=True) \
                .merge(pd.DataFrame(df4), left_index=True, right_index=True)

            weights = []
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            weights.append(weight4)

            print portfolio_prices
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


