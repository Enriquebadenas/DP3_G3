## TAREAS REUNIÓN JUEVES

#SACAR LA CIUDAD/PROVINCIA A TRAVES DE LA LONGITUD_LATITUD GPS

#enlaces
# detectar outliers:   https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/
# habilitar apis google: https://maplink.global/blog/es/como-obtener-google-maps-api-key/

import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype #For definition of custom categorical data types (ordinal if necesary)
import matplotlib.pyplot as plt
import seaborn as sns  # For hi level, Pandas oriented, graphics


# os.chdir('G:\Mi unidad\MDA\DataProject3')
os.chdir("D:/EDEM DataAnalytics/DataProject3_Grupo3/data")
os.getcwd()
os.listdir()

#Datasets
dem = pd.read_csv("train_datos_demograficos.csv")
prev = pd.read_csv("train_previous_loan.csv")
perf = pd.read_csv("train_performance.csv")


dem_diego = pd.read_csv ("demograficos_diego.csv")
prev_diego = pd.read_csv("previous_loan_diego.csv")


#----------------------------PREVIOUS_LOAN Dataset Description:
    
#CUSTOMERID - SYSTEMLOANID - Identificadores de clientes y préstamos
prev.describe()
prev.dtypes

# Only variable "referredby" has missing values = 17.157
prev.isna().sum()

#Variables Cuantitativas:

# LOANAMOUNT - Cantidad pedida en el préstamo
prev.loanamount.describe()
sns.boxplot(prev["loanamount"])

prev.loanamount.nlargest(10)
# Histograma  loanamount
res = prev.loanamount.describe()  
m = round(res['mean'],0) 
sd = round(res['std'],0)
n = round(res['count'],0)

x = prev.loanamount
plt.hist(x, edgecolor='black', bins=50)
plt.xticks(np.arange(0, 60001, step=10000))
plt.title('Figura X. loan amount)')
plt.ylabel('Frecuencia')
plt.xlabel('loan amount')
plt.axvline(x=m, linewidth=1, linestyle= 'solid', color="red", label='Media')
plt.axvline(x=m-sd, linewidth=1, linestyle= 'dashed', color="green", label='- 1 S.D.')
plt.axvline(x=m+sd, linewidth=1, linestyle= 'dashed', color="green", label='+ 1 S.D.')
props = dict(boxstyle='round', facecolor='white', lw=0.5)
plt.text(45000, 9000, f'Media: {m} \nS.D.:{sd} \nn: {n}', bbox=props)
plt.legend(loc='upper left', bbox_to_anchor=(0.73, 0.76))
plt.show()



# TOTALDUE - Cantidad total requerida para liquidar el préstamo; este es el valor del préstamo
# de capital desembolsado + intereses e impuestos
prev.totaldue.describe()
sns.boxplot(prev["totaldue"])

# TERMDAYS - Plazo del préstamo
prev.termdays.describe()
sns.boxplot(prev["termdays"])


sns.pairplot(prev,  diag_kind = 'hist', plot_kws = {'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
#fig.suptitle("Relación variables",  y=1.1)


#_____________

#Variables Cualitativas:
    
#APPROVEDDATE - Fecha en la que se aprobó el préstamo
prev.approveddate.describe()

#CREATIONDATE - Fecha en que se creó la solicitud del préstamo
prev.creationdate.describe()

#CLOSEDDATE - Fecha en la cual se liquidó el préstamo
prev.closeddate.describe()

#FIRSTDUEDATE - Fecha del primer pago en caso de que el plazo sea superior a 30 días.
#En el caso de que el plazo sea de más de 60 días, se deben realizar varios pagos mensuales,
#y esta fecha refleja la fecha del primer pago
prev.firstduedate.describe()

#FIRSTREPAIDDATE - Fecha real en la que se hizo el primer pago
prev.firstrepaiddate.describe()

#LOAN NUMBER - Categorías del préstamo:
prev.loannumber.describe()
prev["loannumber"].value_counts()


#--------------TRATAMIENTO
#Fechas a DateTime
prev[["approveddate","creationdate","closeddate","firstduedate","firstrepaiddate"]] = prev[["approveddate",
                                                                                             "creationdate",
                                                                                             "closeddate",
                                                                                             "firstduedate",
                                                                                             "firstrepaiddate"]].apply(pd.to_datetime)
# Restar CLOSED DATE - APPROVEDDAY para sacar los dias de diferencia

prev["diff_approv_closed_date"] = (prev["closeddate"] - prev["approveddate"]).dt.days

# Si los dias de diferencia son menores al plazo del préstamo:
prev.loc[prev["termdays"] >= prev["diff_approv_closed_date"], "Good_bad_flag"] = "Good"            
prev.loc[prev["termdays"] < prev["diff_approv_closed_date"], "Good_bad_flag"] = "Bad"   

prev.Good_bad_flag.describe()
prev.Good_bad_flag.hist()

# Num de préstamos por cliente:

prev["num_prest"] = prev.groupby("customerid")["systemloanid"].transform("size")    
prev.num_prest.describe()


#---------------------DATOS DEMOGRÁFICOS DataSet Description

#Acciones:    
    # BORRAR LEVEL OF EDUCACATION CLIENTS + BANK BRANCH CLIENTS
    # Añadir valores restantes de Employment_status_clients

#CUSTOMERID - Identificador cliente
dem.describe()
dem.dtypes

#Missing values 
dem.isna().sum()

#BIRTHDATE - Fecha de nacimiento del cliente
dem.birthdate.describe()
dem["birthdate"].value_counts()

#BANK_ACCOUNT_TYPE - Tipo de cuenta bancaria
dem.bank_account_type.describe()
dem["bank_account_type"].value_counts()

#BANK_NAME_CLIENTS - Nombre del banco
dem.bank_name_clients.describe()
dem["bank_name_clients"].value_counts()

#BANK_BRANCH_CLIENTS - Localización del Banco
dem.bank_branch_clients.describe()
dem["bank_branch_clients"].value_counts()

#EMPLOYMENT_STATUS_CLIENTS - Tipo de empleo del cliente
dem.employment_status_clients.describe()
dem.employment_status_clients.value_counts()

#LEVEL_OF_EDUCATION_CLIENTs - Nivel de estudios del cliente
dem.level_of_education_clients.describe()
dem["level_of_education_clients"].value_counts()

# Bar graph of level education clients
mytable = dem.groupby(['level_of_education_clients']).size() 
print(mytable)
mytable.sum() # obtenemos el tamaño de la muestra, los suma todos. mytable es un objeto que sabe sumarse a sí mismo

# porcentajes
n = mytable.sum()

mytable2 = (mytable/n)*100
print(mytable2)

mytable3 = round(mytable2,1) # le estoy indicando que me redondee la tabla entera a 1 decimal
mytable3

bar_list = ['Graduate', 'Secondary', 'Post-Graduate', 'Primary'] # lista de etiquetas el eje X
plt.bar(bar_list, mytable2, edgecolor='black') # primer argumento=nombre barras, segundo argumento=longitud de las barras
plt.title('Figure 1. Level of education')
plt.ylabel('Percentage')
plt.text(3, 60, f'n: {n}')
plt.text(1, 67, f'% sobre total clientes con nivel estudios \ninformado sobre total clientes = 4346')
plt.show()

# ojo que de los 4346 clientes solo tienen informado el campo level educaton 587


#--------------TRATAMIENTO

#Fechas a DateTime
dem[["birthdate"]] = dem[["birthdate"]].apply(pd.to_datetime)
#Coordenadas a Float
dem[["longitude_gps", "latitude_gps"]] = dem[["longitude_gps", "latitude_gps"]].astype(float)

# Obtener direcciones a partir de coordenadas:

## BUENO https://github.com/softhints/Pandas-Tutorials/blob/master/geocoding/2.reverse-geocoding-latitude-longitude-city-country-python-pandas.ipynb

import geocoder

# FUNCION PARA CONVERTIR LAS COORDENADAS EN PAIS
def geo_rev(x):
    g = geocoder.osm([x.latitude_gps, x.longitude_gps], method='reverse').json
    if g:
        return g.get('country')
    else:
        return 'no country'

#dem["Country"] = dem[['latitude_gps', 'longitude_gps']].tail().apply(geo_rev, axis=1)
dem["Country"] = dem[['latitude_gps', 'longitude_gps']].apply(geo_rev, axis=1)


dem.Country.describe()
dem.Country.hist()

dem["Country"].value_counts()
dem["Country"].value_counts(normalize=True)

# FUNCION PARA CONVERTIR LAS COORDENADAS EN CIUDAD
def geo_city(x):
    g = geocoder.osm([x.latitude_gps, x.longitude_gps], method='reverse').json
    if g:
        return g.get('city')
    else:
        return 'no city' 

dem["city"] = dem[['latitude_gps', 'longitude_gps']].apply(geo_city, axis=1)

dem["city"].value_counts()


"""
# GEOPY OPTION https://stackoverflow.com/questions/69409255/how-to-get-city-state-and-country-from-a-list-of-latitude-and-longitude-coordi

pip install geopy
pip install geocoder

from geopy.geocoders import Nominatim
import io
import geocoder

geolocator = Nominatim(user_agent="geoapiExercises")

def city_state_country(row):
    coord = f"{row['latitude_gps']}, {row['longitude_gps']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    row['city'] = city
    row['state'] = state
    row['country'] = country
    return row

geo = geo.apply(city_state_country, axis=1)
print(geo)
"""

#####option 2   https://stackoverflow.com/questions/63072917/run-the-location-latitude-and-longitude-function-to-get-the-city-and-country-on
"""
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="geoapiExercises")


def city_state_country(coord):
    try:
        location = geolocator.reverse(coord, exactly_one=True)
        address = location.raw['address']
        city = address.get('city', '')
        state = address.get('state', '')
        country = address.get('country', '')
        return city, state, country
#print(city_state_country("30.930508, 75.8419883"))

    except ValueError:
     return(0)

geo['Location'] = (geo[['latitude_gps', 'longitude_gps']].astype(str)
                   .apply(lambda row: city_state_country(', '.join(row)),
                          axis=1)
                 )
"""

"""
## GOOGLE MAPS API OPTION
# https://stackoverflow.com/questions/51645417/assigning-city-name-by-latitude-longitude-values-in-pandas-dataframe

pip install urllib3
pip install requests

import requests
import urllib3
import json

def location(lat, long):
    url = 'http://maps.googleapis.com/maps/api/geocode/json?latlng={0},{1}&sensor=false'.format(lat, long)
    r = requests.get(url)
    r_json = r.json()
    if len(r_json['results']) < 1: return None, None
    res = r_json['results'][0]['address_components']
    country  = next((c['long_name'] for c in res if 'country' in c['types']), None)
    locality = next((c['long_name'] for c in res if 'locality' in c['types']), None)
    return locality, country


location(30.314368, 76.384381)

geo["localidad"] = np.vectorize(location)(geo["latitude_gps"], geo["longitude_gps"] )
"""


#------------------PERFORMANCE DATASET DESCRIPTION

#CUSTOMERID - SYSTEMLOANID - Identificadores de clientes y préstamos
perf.describe()
perf.dtypes

# Only variable "referredby" has missing values = 17.157
perf.isna().sum()

#Variables Cuantitativas:

# LOANAMOUNT - Cantidad pedida en el préstamo
perf.loanamount.describe()
sns.boxplot(perf["loanamount"])

# TOTALDUE - Cantidad total requerida para liquidar el préstamo; este es el valor del préstamo
# de capital desembolsado + intereses e impuestos
perf.totaldue.describe()
sns.boxplot(perf["totaldue"])

# TERMDAYS - Plazo del préstamo
perf.termdays.describe()
sns.boxplot(perf["termdays"])
#_____________

#Variables Cualitativas:
    
#APPROVEDDATE - Fecha en la que se aprobó el préstamo
perf.approveddate.describe()

#CREATIONDATE - Fecha en que se creó la solicitud del préstamo
perf.creationdate.describe()

#CLOSEDDATE - Fecha en la cual se liquidó el préstamo
perf.closeddate.describe()

#FIRSTDUEDATE - Fecha del primer pago en caso de que el plazo sea superior a 30 días.
#En el caso de que el plazo sea de más de 60 días, se deben realizar varios pagos mensuales,
#y esta fecha refleja la fecha del primer pago
perf.firstduedate.describe()

#FIRSTREPAIDDATE - Fecha real en la que se hizo el primer pago
perf.firstrepaiddate.describe()


prev.to_csv("previous_loan_diego.csv")
prev.to_excel("previous_loan_diego.xlsx")

dem.to_csv("demograficos_diego.csv")
dem.to_excel("demograficos_diego.xlsx")




