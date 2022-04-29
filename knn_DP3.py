# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 17:48:21 2022

@author: quimi
"""

''' Resumo script sin ayudas '''
import os
import numpy as np;
import pandas as pd;
from sklearn.datasets import load_iris;
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# aux_functions
def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)

# os.chdir('G:\Mi unidad\MDA\DataProject3')
os.chdir("D:/EDEM DataAnalytics/DataProject3_Grupo3/data")

#Datasets
dem = pd.read_csv("train_datos_demograficos.csv")
prev = pd.read_csv("train_previous_loan.csv")
perf = pd.read_csv("train_performance.csv")


dem_diego = pd.read_csv ("demograficos_diego.csv")
prev_diego = pd.read_csv("previous_loan_diego.csv")

target = prev_diego.Good_bad_flag
target_names = ['Good', 'Bad']
dat = prev_diego

'''from sklearn.datasets import load_iris;
iris = load_iris();
dat = iris.data;
target = iris.target;
target_names = iris.target_names;'''

unique, counts = np.unique(target, return_counts=True);
print(np.asarray((unique, counts)).T);

target_names

dat.shape

dat[0:5]
dat.drop('customerid', axis=1, inplace=True)
dat.drop('Unnamed: 0', axis=1, inplace=True)
dat.columns
dat.drop('approveddate', axis=1, inplace=True)
dat.drop('creationdate', axis=1, inplace=True)
dat.drop('firstduedate', axis=1, inplace=True)
dat.drop('firstrepaiddate', axis=1, inplace=True)
dat.drop('closeddate', axis=1, inplace=True)
dat.drop('referredby', axis=1, inplace=True)
dat.shape

dat['Good_bad_flag'] = np.where(dat['Good_bad_flag']=='Good', 1, 0)


# 1) Importar modelo que se quiere emplear.
from sklearn.neighbors import KNeighborsClassifier as model

# 2) Importar métrica a emplear.
from sklearn.metrics import accuracy_score as metric

# 3) Definir modelo.
model = model(n_neighbors=1)

# 4) Llamar al método fit para entrenar el modelo.
model.fit(dat, target)

# 5) Llamar al método predict para generar las predicciones.
pred = model.predict(dat)
pred[0:5]

# 6) Calcular métrica usando las predicciones obtenidas en el paso anterior.
metric(target, pred)

# Por ejemplo si quiero otra métrica:
from sklearn.metrics import zero_one_loss as metric
metric(target, pred)

'''Validation '''
# ### 1.1 Cross-Validation
# - **Train**: 85%.
# - **Test**: 15%.

perc_values = [0.85, 0.15];

X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(dat, target, test_size = perc_values[1],\
                                                                            random_state=1);
# Veamos las dimensiones de los dos conjuntos
print('Train data size = ' + str(X_train_cross.shape))
print('Train target size = ' + str(y_train_cross.shape))
print('Test data size = ' + str(X_test_cross.shape))
print('Test target size = ' + str(y_test_cross.shape))


# ### 1.2 Conjunto de Validación Fijo
# - **Train**: 70%.
# - **Validación**: 15%.
# - **Test**: 15%.
perc_values = [0.7, 0.15, 0.15];

X_train, X_val, y_train, y_val = train_test_split(X_train_cross, y_train_cross, test_size=(perc_values[1] / (perc_values[0] + perc_values[1])), random_state=1);
X_test = X_test_cross
y_test = y_test_cross                          

# Veamos las dimensiones de los tres conjuntos
print('Train data size = ' + str(X_train.shape))
print('Train target size = ' + str(y_train.shape))
print('Validation data size = ' + str(X_val.shape))
print('Validation target size = ' + str(y_val.shape))
print('Test data size = ' + str(X_test.shape))
print('Test target size = ' + str(y_test.shape))


# ## 2. Validación Manual
# 
# Vamos a realizar ahora la validación de un modelo de KNN para varios valores distintos del hiperparámetro k de 
# forma manual, es decir, realizando explícitamente cada una de las llamadas. Para ello utilizaremos el método 
# del **conjunto de validación fijo**.

# ### Paso 1: Importar modelo que se quiere emplear
from sklearn.neighbors import KNeighborsClassifier

# ### Paso 2: Importar métrica a emplear.
from sklearn.metrics import accuracy_score as metric

# ### Paso 3: Definir modelo
# Creamos el modelo para k = 1
knn_1 = KNeighborsClassifier(n_neighbors=1)

# Creamos el modelo para k = 3
knn_3 = KNeighborsClassifier(n_neighbors=3)


# ### Paso 4: Llamar al método fit para entrenar el modelo
# Entrenamos ambos modelos usando el conjunto de train
# k = 1
knn_1.fit(X = X_train, y = y_train);

# k = 3
knn_3.fit(X = X_train, y = y_train);

# ### Paso 5: Llamar al método predict para generar las predicciones
# Calculamos las predicciones para train/val/test de cada uno de los modelos
# k = 1
pred_train_1 = knn_1.predict(X_train);
pred_val_1 = knn_1.predict(X_val);

# k = 3
pred_train_3 = knn_3.predict(X_train);
pred_val_3 = knn_3.predict(X_val);


# ### Paso 6: Calcular métrica usando las predicciones obtenidas en el paso anterior
# Y ahora sus correspondientes accuracies
# k = 1
acc_train_1 = metric(y_train, pred_train_1);
acc_val_1 = metric(y_val, pred_val_1);

# k = 3
acc_train_3 = metric(y_train, pred_train_3);
acc_val_3 = metric(y_val, pred_val_3);


# Veamos los resultados
# k = 1
print('k = 1 - accuracy train = ' + str(acc_train_1))
print('k = 1 - accuracy val = ' + str(acc_val_1))

# k = 3
print('k = 3 - accuracy train = ' + str(acc_train_3))
print('k = 3 - accuracy val = ' + str(acc_val_3))


# ## 3. Curvas de Validación
# 
# Vamos a construir ahora unos gráficos que nos permiten analizar el error de validación para toda una serie 
# de valores del hiperparámetro k. Para ello utilizaremos el método de **cross-validation**.

k_values = np.arange(1, 30);
train_scores, val_scores = validation_curve(KNeighborsClassifier(), X_train_cross, y_train_cross,
                                       'n_neighbors', k_values,
                                       scoring = make_scorer(metric))


plot_with_err(k_values, train_scores, label='training scores')
plot_with_err(k_values, val_scores, label='validation scores')
plt.xlabel('k'); plt.ylabel('accuracy')
plt.grid()
plt.legend();


# ## 4. Grid Search
# ### 4.1 Elegimos una familia de modelos
# En nuestro estamos trabajando con los modelos KNN.
# ### 4.2 Elegimos unos hiperparámetros a optimizar
# En nuestro caso optimizaremos el hiperparámetro k, o n_neighbors, el único relevante para los modelos de KNN.
# ### 4.3 Para cada hiperparámetro, elegimos una serie de valores a probar
# En nuestro caso vamos a probar todos los valores del 1 al 20, es decir:

k_values = np.arange(1, 30);


# ### 4.4 Entrenamos nuestro modelo sobre el conjunto de train con los diferentes hiperparámetros haciendo todas 
# las combinaciones posibles / 4.5 Hacemos la predicción de los diferentes modelos sobre el conjunto de validación y 
# calculamos el error con la métrica seleccionada
# Realizaremos estos dos pasos en un único bloque de código.
#
# **Conjunto de validación fijo**

grid_results = pd.DataFrame();
for k in k_values:
    # 4.4 Entrenar modelo
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X = X_train, y = y_train)
    
    # 4.5 Generar predicciones
    pred_train = knn.predict(X_train);
    pred_val = knn.predict(X_val);
    
    # 4.5 Calcular métricas de evaluación
    acc_train = metric(y_train, pred_train);
    acc_val = metric(y_val, pred_val);
    
    grid_results = grid_results.append(pd.DataFrame(data={'k':[k],'acc_train':[acc_train],'acc_val':[acc_val]}), ignore_index=True)


# Veamos los resultados obtenidos.
grid_results


# **Cross-validation**

help(GridSearchCV)

param_grid = [
  {'n_neighbors': k_values}
 ]

# Hacemos un 5-fold CV:

grid_results_cv = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)
grid_results_cv.fit(X_train_cross, y_train_cross)

print('Values validated for k hyperparameter: ' + str(grid_results_cv.cv_results_['param_n_neighbors']))
print('Validation accuracy: ' + str(grid_results_cv.cv_results_['mean_test_score']))
print('Best score ' + str(grid_results_cv.best_score_))
print('Best k: ' + str(grid_results_cv.best_estimator_.n_neighbors))
best_k = grid_results_cv.best_estimator_.n_neighbors

# ### 4.6 Escogemos el que mejor métrica obtenga y lo aplicamos sobre el conjunto de test para ver el error final 
# esperado de nuestro modelo
# Escogeremos en este el valor k = 8 (k = 9 también sería una opción válida) como valor óptimo para nuestro modelo final. 
# Entrenamos el modelo, para ello **juntamos los conjuntos de train y validación**.

print('Train data size = ' + str(X_train.shape))
print('Train target size = ' + str(y_train.shape))
print('Validation data size = ' + str(X_val.shape))
print('Validation target size = ' + str(y_val.shape))

# Combinar train y validación
X_train = np.concatenate((X_train,X_val), axis = 0)
y_train = np.concatenate((y_train, y_val), axis = 0)

del X_val, y_val

print('New train data size = ' + str(X_train.shape))
print('New train target size = ' + str(y_train.shape))


# Entrenar modelo
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(X = X_train, y = y_train)


# Obtenemos predicciones:    
# Generar predicciones
pred_train = knn.predict(X_train);
pred_test = knn.predict(X_test);    


# Calculamos las métricas de evaluación:
# Calcular métricas de evaluación
acc_train = metric(y_train, pred_train);
acc_test = metric(y_test, pred_test);


# Veamos los resultados finales:
print('accuracy train = ' + str(acc_train))
print('accuracy test = ' + str(acc_test))

