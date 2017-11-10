import textwrap

from django.http import HttpResponse
from django.views import generic
from django.views.generic import View
from django.views.generic.base import TemplateView


class Home(generic.TemplateView):
    template_name = 'portimize/home.html'
