from django.views import generic
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from forms import SignUpForm, PortfolioForm



class Home(generic.TemplateView):
    template_name = 'portimize/home.html'

    def get(self, request):
        form = PortfolioForm
        return render(request, self.template_name, {'form' : form})

    def post(self, request):
        form = PortfolioForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            asset1 = form.cleaned_data['asset1']
            # port_obj = Portfolio(start_date = start_date, end_date = end_date)
            # port_obj.save()
            # return redirect('home')
        args = {'form': form, 'start_date': start_date,
                'end_date': end_date, 'asset1' : asset1,
                'asset2': asset2, 'asset3' : asset3,
                'asset4': asset4}
        return render(request, 'portimize/home.html', args)

def results(request):
    return render(request, 'portimize/home.html')

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


