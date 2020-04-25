import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#Importando Datos de la Libreria
boston = datasets.load_boston()
print(boston)
print()

#Entendiendo la Data
print('Informacion en el DataSet')
print(boston.keys())
print()

#Verifico las Caracteristicas del dataset
print('Caracteristicas del dataset:')
print(boston.DESCR)

#Verifico la Cantidad de datos que hay en los dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verifico la informacion de las columnas
print('Nombres columnas:')
print(boston.feature_names)

#Preparar la data regresion lineal simple
X=boston.data[:,np.newaxis,5]
#ddefino los datos correspondientes a las etiquetas
y=boston.target

#Graficamos los datos correspondientes a las etiquetas
plt.scatter(X,y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()

#Implementacion de Regresion Lineal simple
from sklearn.model_selection import train_test_split

#separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

#Defino el algoritmo a utilizar
lr=linear_model.LinearRegression()

#Entrenar el Modelo
lr.fit(X_train,y_train)

#Realizo una Prediccion
Y_pred=lr.predict(X_test)

print('precision del modelo')
print(lr.score(X_train,y_train))
