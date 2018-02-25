from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm
import pandas_datareader.data as web
import pandas as pd
from optimization import MarkowitzOptimize
from predictions import predict
from datetime import timedelta


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
            name1 = dict(form.fields['asset1'].widget.choices)[asset1]
            weight1 = form.cleaned_data['weight1']
            prices1 = web.DataReader(asset1, 'yahoo', start_date, end_date)
            df1 = pd.DataFrame(predict(prices1))
            df1.rename(columns={df1.columns[0]: 'Close1'}, inplace = True)

            #asset2
            asset2 = form.cleaned_data['asset2']
            name2 = dict(form.fields['asset2'].widget.choices)[asset2]
            weight2 = form.cleaned_data['weight2']
            prices2 = web.DataReader(asset2, 'yahoo', start_date, end_date)
            df2 = pd.DataFrame(predict(prices2))
            df2.rename(columns={df2.columns[0]: 'Close2'}, inplace=True)

            #asset3
            asset3 = form.cleaned_data['asset3']
            name3 = dict(form.fields['asset3'].widget.choices)[asset3]
            weight3 = form.cleaned_data['weight3']
            prices3 = web.DataReader(asset3, 'yahoo', start_date, end_date)
            df3 = pd.DataFrame(predict(prices3))
            df3.rename(columns={df3.columns[0]: 'Close3'}, inplace=True)

            #asset4
            asset4 = form.cleaned_data['asset4']
            name4 = dict(form.fields['asset4'].widget.choices)[asset4]
            weight4 = form.cleaned_data['weight4']
            prices4 = web.DataReader(asset4, 'yahoo', start_date, end_date)
            df4 = pd.DataFrame(predict(prices4))
            df4.rename(columns={df4.columns[0]: 'Close4'}, inplace=True)

            weights = []
            weights.append(weight1)
            weights.append(weight2)
            weights.append(weight3)
            weights.append(weight4)

            new_weights = []
            portfolio= []
            for i in range((end_date - start_date).days+1):
                p1 = web.DataReader(asset1, 'yahoo', start_date + timedelta(days=i-50), start_date + timedelta(days=i))
                p2 = web.DataReader(asset2, 'yahoo', start_date + timedelta(days=i - 50),
                                         start_date + timedelta(days=i))
                p3 = web.DataReader(asset3, 'yahoo', start_date + timedelta(days=i - 50),
                                         start_date + timedelta(days=i))
                p4 = web.DataReader(asset4, 'yahoo', start_date + timedelta(days=i - 50),
                                         start_date + timedelta(days=i))
                df1 = pd.DataFrame(p1['Close'])
                df2 = pd.DataFrame(p2['Close'])
                df3 = pd.DataFrame(p3['Close'])
                df4 = pd.DataFrame(p4['Close'])
                p_prices = df1.merge(pd.DataFrame(df2), left_index=True, right_index=True) \
                    .merge(pd.DataFrame(df3), left_index=True, right_index=True) \
                    .merge(pd.DataFrame(df4), left_index=True, right_index=True)
                opti_model = MarkowitzOptimize(p_prices, weights)
                new_weights.append(opti_model.minimizeSharpeRatio())
                # print new_weights[i]
                # print p_prices.iloc[i].values
                portfolio.append(p_prices.iloc[i].values * new_weights[i])

            # opti_model = MarkowitzOptimize(portfolio_prices, weights)
            # new_weights = opti_model.minimizeSharpeRatio()
            new_weights = new_weights[(end_date - start_date).days-1]

            prices1 = pd.DataFrame(prices1['Close'])
            prices1.rename(columns={prices1.columns[0]: 'Close1'}, inplace = True)
            prices2 = pd.DataFrame(prices2['Close'])
            prices2.rename(columns={prices2.columns[0]: 'Close2'}, inplace = True)
            prices3 = pd.DataFrame(prices3['Close'])
            prices3.rename(columns={prices3.columns[0]: 'Close3'}, inplace = True)
            prices4 = pd.DataFrame(prices4['Close'])
            prices4.rename(columns={prices4.columns[0]: 'Close4'}, inplace = True)

            portfolio_prices = prices1.merge(prices2, left_index=True, right_index=True) \
                .merge(prices3, left_index=True, right_index=True) \
                .merge(prices4, left_index=True, right_index=True)

            attributes = list(portfolio_prices.columns.values)
            return_prices = (portfolio_prices - portfolio_prices.iloc[0]) / portfolio_prices.iloc[0]
            return1 = return_prices[attributes].mul(weights).sum(1)
            # print return1

            portfolio = pd.DataFrame(portfolio)
            attributes = list(portfolio.columns.values)
            portfolio = portfolio[attributes].sum(1)
            return2 = (portfolio - portfolio.iloc[0]) / portfolio.iloc[0]
            # print return2

            period = pd.date_range(start_date, end_date)
            ts_list = period.tolist()
            date_list = [ts.date() for ts in ts_list]
            date_string = [str(date) for date in date_list]


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