"""datamining URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.button,name='home'),
    path('output2013', views.output2013,name='script'),
    path('output2014', views.output2014,name='script1'),
    path('outputmg', views.outputmg,name='script2'),        
    path('train', views.train,name='training'),
    path('model', views.model,name='model'),
    path('dtc', views.dtc,name='dtc'),
    path('random', views.random,name='random'),
    path('testModel', views.testModel,name='testModel'),
    path('train1', views.upload,name='upload'),
    path('form', views.index),

]

urlpatterns += staticfiles_urlpatterns()
