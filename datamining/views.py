from django.shortcuts import render
import requests
import csv
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from info_gain import info_gain
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from datamining.settings import BASE_DIR
from .forms import UserForm


def button(request):
    return render(request,'home.html')

def random(request):
    return render(request,'random.html')

def dtc(request):
    return render(request,'dtc.html')

def train(request):
    data1 = pd.read_csv('datamining/wheat-2013-supervised.csv')    
    plt.subplots(figsize=(12,10))
    list1=[]
    for i in data1['CountyName']:
        list1.append(i)
    ax=pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('RdBu_r',10))
    for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 
        ax.text(.9, i, v,fontsize=12,color='white',weight='bold')
    ax.patches[9].set_facecolor('r')
    plt.title('Les Pays les plus étudiés') 
    plt.savefig('hist.png')
    return render(request,'train.html') 

def model(request):
    return render(request,'model.html') 

def testModel(request):
    return render(request,'testModel.html')            

def output2013(request):
    data1 = pd.read_csv('datamining/wheat-2013-supervised.csv')
    data=data1.head(5)
    data_html = data.to_html()
    context = {'loaded_data': data_html}
    return render(request, 'train.html', context)

def output2014(request):
    print(BASE_DIR)
    data1 = pd.read_csv('datamining/wheat-2014-supervised.csv')
    data=data1.head(5)
    data_html = data.to_html()
    context = {'loaded_data1': data_html}
    return render(request, 'train.html', context)

def outputmg(request):
    print(BASE_DIR)
    data1 = pd.read_csv('datamining/merged.csv')
    data=data1.head(5)
    data_html = data.to_html()
    context = {'loaded_data2': data_html}
    return render(request, 'train.html', context)  

def upload(request):
    context1 = {}
    if request.method == 'POST':
        upload_file = request.FILES['document']
        upload_file.v
        data = upload_file.head(5)
        data_html = data.to_html()
        print(data_html)
        context1 = {'loaded_dataup': data_html}        
    return render(request, 'train.html' ,context1)



def index(request):
    submitbutton= request.POST.get("submit")

    Latitude=0
    Longitude=0
    apparentTemperatureMax=0
    apparentTemperatureMin=0
    cloudCover=0
    dewPoint=0
    humidity=0
    precipIntensity=0
    precipIntensityMax=0
    precipProbability=0
    precipAccumulation=0
    precipTypeIsRain=0
    precipTypeIsSnow=0
    precipTypeIsOther=0
    pressure=0
    temperatureMax=0
    temperatureMin=0
    visibility=0
    windBearing=0
    windSpeed=0
    NDVI=0
    DayInSeason=0
    data = [22]

    form= UserForm(request.POST or None)
    if form.is_valid():
        Latitude= form.cleaned_data.get("Latitude")
        Longitude= form.cleaned_data.get("Longitude")
        apparentTemperatureMax= form.cleaned_data.get("apparentTemperatureMax")
        apparentTemperatureMin= form.cleaned_data.get("apparentTemperatureMin")
        cloudCover= form.cleaned_data.get("cloudCover")
        dewPoint= form.cleaned_data.get("dewPoint")
        humidity= form.cleaned_data.get("humidity")
        precipIntensity= form.cleaned_data.get("precipIntensity")
        precipIntensityMax= form.cleaned_data.get("precipIntensityMax")
        precipProbability= form.cleaned_data.get("precipProbability")
        precipAccumulation= form.cleaned_data.get("precipAccumulation")
        precipTypeIsRain= form.cleaned_data.get("precipTypeIsRain")
        precipTypeIsSnow= form.cleaned_data.get("precipTypeIsSnow")
        precipTypeIsOther= form.cleaned_data.get("precipTypeIsOther")
        pressure= form.cleaned_data.get("pressure")
        temperatureMax= form.cleaned_data.get("temperatureMax")
        temperatureMin= form.cleaned_data.get("temperatureMin")
        visibility= form.cleaned_data.get("visibility")
        windBearing= form.cleaned_data.get("windBearing")
        windSpeed= form.cleaned_data.get("windSpeed")
        NDVI= form.cleaned_data.get("NDVI")
        DayInSeason= form.cleaned_data.get("DayInSeason")

    data.insert(0,Latitude)
    data.insert(1,Longitude)
    data.insert(2,apparentTemperatureMax)
    data.insert(3,apparentTemperatureMin)
    data.insert(4,cloudCover)
    data.insert(5,dewPoint)
    data.insert(6,humidity)
    data.insert(7,precipIntensity)
    data.insert(8,precipIntensityMax)
    data.insert(9,precipProbability)
    data.insert(10,precipAccumulation)
    data.insert(11,precipTypeIsRain)
    data.insert(12,precipTypeIsSnow)
    data.insert(13,precipTypeIsOther)
    data.insert(14,pressure)
    data.insert(15,temperatureMax)
    data.insert(16,temperatureMin)
    data.insert(17,visibility)
    data.insert(18,windBearing)
    data.insert(19,windSpeed)
    data.insert(20,NDVI)
    data.insert(21,DayInSeason)

    print(data[0])
    #read data
    data1 = pd.read_csv('datamining/wheat-2013-supervised.csv')
    data2 = pd.read_csv('datamining/wheat-2014-supervised.csv')
    #merge two dataset for train the model later
    merged = data1.append(data2, ignore_index=True)
    merged = merged[["CountyName","State","Latitude","Longitude","Date","apparentTemperatureMax","apparentTemperatureMin","cloudCover","dewPoint","humidity","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow","precipTypeIsOther",	"pressure",	"temperatureMax","temperatureMin","visibility",	"windBearing","windSpeed","NDVI","DayInSeason","Yield" ]]
    merged.to_csv('datamining/merged.csv', index=None, header=True)
    mg = pd.read_csv('datamining/merged.csv')
    #Selection des attributs a traiter
    mg = mg[["Latitude","Longitude","apparentTemperatureMax","apparentTemperatureMin","cloudCover","dewPoint","humidity","precipIntensity","precipIntensityMax","precipProbability","precipAccumulation","precipTypeIsRain","precipTypeIsSnow","precipTypeIsOther",	"pressure",	"temperatureMax","temperatureMin","visibility",	"windBearing","windSpeed","NDVI","DayInSeason","Yield" ]]

    #Supprimer les champs vide dans la dataset
    mg.dropna(inplace=True)

    #Suppression du champs Yield
    Attribut = np.array(mg.drop(["Yield"],1))

    #Classe = np.asarray(mg['Yield'], dtype="|S6")

    classe1 = np.asarray(mg['Yield'])

    list2=[]

    for i in classe1:
        if i<50: 
            list2.append(0)
        else:
            list2.append(1)
    
    print(list2[1000])

    #Create a Gaussian Classifier
    rf=RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    rf.fit(Attribut[:999],list2[:999])

#    y_pred=rf.predict([data]) #el star hetha howa el ghalet kahaw


    context= {'form': form,  'submitbutton' : submitbutton }
        
    return render(request, 'form.html', context)



