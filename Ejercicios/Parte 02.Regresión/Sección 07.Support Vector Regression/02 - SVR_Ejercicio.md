#  Regresión con Soporte Vectorial

**Contexto**  
Este conjunto de datos consiste en una lista de puestos en una empresa junto con los niveles y su salario asociado.

**Contenido**  
El conjunto de datos incluye columnas para el Puesto con valores que van desde Analista de negocios, Consultor junior hasta CEO, Nivel que varía de 1 a 10 y, finalmente, el Salario asociado con cada puesto que varía de **45,000 a 1,000,000**.

**Planteamiento del problema**  
El enunciado del problema es que el candidato con nivel 6.5 tenía un salario anterior de 160000. Para contratar al candidato para un nuevo puesto, a la compañía le gustaría confirmar si está siendo honesto acerca de su último salario para que pueda tomar una decisión de contratación . Para hacer esto, haremos uso del método Soporte vectorial para predecir el salario exacto del empleado.

## Importar las librerías


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
```

## Importar el archivo Position_Salaries.csv


```python
dataset = pd.read_csv('Position_Salaries.csv')
```


```python
#Revisemos los datos
dataset.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Position</th>
      <th>Level</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Business Analyst</td>
      <td>1</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Junior Consultant</td>
      <td>2</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Senior Consultant</td>
      <td>3</td>
      <td>60000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Manager</td>
      <td>4</td>
      <td>80000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Country Manager</td>
      <td>5</td>
      <td>110000</td>
    </tr>
  </tbody>
</table>
</div>



## Crear las variables X (todas las filas, columna [1:2]) & y (todas las filas, columna 2)


```python
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
print(X)
print(y)
```

    [[ 1]
     [ 2]
     [ 3]
     [ 4]
     [ 5]
     [ 6]
     [ 7]
     [ 8]
     [ 9]
     [10]]
    [  45000   50000   60000   80000  110000  150000  200000  300000  500000
     1000000]
    

## Realiza el escalado de variables


```python

```

## Divide el conjunto de datos en conjunto de entrenamiento y conjunto de prueba


```python
from sklearn.model_selection import cross_val_score, train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
```

## Genera el modelo de SVR con diferentes kernels ("linear", ,"rbf",["poly",degree = 3])

#### Genera el modelo "Lineal"


```python
## Escribe tu código aqui

```


```python

```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
# Predicción de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[8.5]]))))
print("La predicción es:", y_pred)
# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
```

    La predicción es: [303706.02408055]
    


![png](../../imagenes/02%20-%20SVR_Ejercicio_17_1.png)


#### Genera el modelo "rbf"


```python
## Escribe tu código aqui

```


```python

```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
# Predicción de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
print("La predicción es:", y_pred)
# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
```

    La predicción es: [170370.0204065]
    


![png](../../imagenes/02%20-%20SVR_Ejercicio_21_1.png)


#### Genera el modelo "poly"


```python
## Escribe tu código aqui

```


```python

```




    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='scale',
        kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)




```python
# Predicción de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
print("La predicción es:", y_pred)
# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
```

    La predicción es: [197301.33007096]
    


![png](../../imagenes/02%20-%20SVR_Ejercicio_25_1.png)

