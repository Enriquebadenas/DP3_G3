# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 08:41:02 2022

@author: quimi
"""

# Load basic libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns  
import datetime as dt
from datetime import timedelta
import scipy.stats as stats
from scipy.stats.stats import pearsonr


os.chdir("D:/EDEM DataAnalytics/DataProject3_Grupo3/data")
os.getcwd()

df = pd.read_csv(r'D:\EDEM DataAnalytics\DataProject3_Grupo3\data\train_previous_loan.csv', sep=',', decimal=',')

df.shape
df.head()
df.tail()
df.columns
# QC OK

df.describe()

df.dtypes # todos los campos de fechas son objetos

# conversión a formato datetime
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['creationdatetime'] = pd.to_datetime(df['creationdate'])
df['creationdatetime'].describe()

df['approveddate'].describe()
df['approveddatetime'] = pd.to_datetime(df['approveddate'])
df['approveddatetime'].describe()

# Creo nuevas columnas. Tiempo entre solicitud crédito y su aprobación
df['days_creation_approved']=df['creationdatetime'] - df['approveddatetime']
df['days_creation_approved'].describe()


# no funciona
df['days_creation_approved']=timedelta(df['creationdatetime'] - df['approveddatetime'])
df['days_creation_approved']=timedelta(df['creationdate'] - df['approveddate'])



# Histograma days creation-approved 
plt.hist(df.days_creation_approved)










    
    
    