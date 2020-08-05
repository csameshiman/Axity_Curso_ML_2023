# Regresión Polinómica

**Contexto**  
Este conjunto de datos consiste en una lista de puestos en una empresa junto con los niveles y su salario asociado.

**Contenido**  
El conjunto de datos incluye columnas para el Puesto con valores que van desde Analista de negocios, Consultor junior hasta CEO, Nivel que varía de 1 a 10 y, finalmente, el Salario asociado con cada puesto que varía de **45,000 a 1,000,000**.

**Planteamiento del problema**  
El enunciado del problema es que el candidato con nivel 6.5 tenía un salario anterior de 160,000. Para contratar al candidato para un nuevo puesto, a la compañía le gustaría confirmar si está siendo honesto acerca de su último salario para que pueda tomar una decisión de contratación . Para hacer esto, haremos uso del método de Regresión Polinómicapara predecir el salario exacto del empleado.

```python
# Importamos las librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
%matplotlib inline
```

## Cargar Datos


```python
# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
```


```python
# Revisemos los datos
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
    


```python
type(X)
type(y)
```




    numpy.ndarray




```python
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




```python
dataset.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10 entries, 0 to 9
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype 
    ---  ------    --------------  ----- 
     0   Position  10 non-null     object
     1   Level     10 non-null     int64 
     2   Salary    10 non-null     int64 
    dtypes: int64(2), object(1)
    memory usage: 368.0+ bytes
    


```python
# Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
dataset.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Level</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.00000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.50000</td>
      <td>249500.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.02765</td>
      <td>299373.883668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>45000.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.25000</td>
      <td>65000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.50000</td>
      <td>130000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.75000</td>
      <td>275000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.00000</td>
      <td>1000000.000000</td>
    </tr>
  </tbody>
</table>
</div>



## EDA (Análisis Exploratorio de Datos)


```python
sns.set_style('whitegrid')
dataset['Salary'].hist()
plt.xlabel('Position')
```




    Text(0.5, 0, 'Position')




![png](../../imagenes/02-%20Polynomial%20Regression_10_1.png)



```python
sns.pairplot(dataset,palette='Set1')
```




    <seaborn.axisgrid.PairGrid at 0x245d2165e08>




![png](../../imagenes/02-%20Polynomial%20Regression_11_1.png)



```python
%matplotlib inline
pd.crosstab(dataset.Level, dataset.Position).plot(kind="bar")
plt.title("Frecuencia de salario por  posición")
plt.xlabel("Salario")
plt.ylabel("Frecuencia por posición")
```




    Text(0, 0.5, 'Frecuencia por posición')




![png](../../imagenes/02-%20Polynomial%20Regression_12_1.png)



```python
sns.distplot(dataset['Salary'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x245d2510b88>




![png](../../imagenes/02-%20Polynomial%20Regression_13_1.png)



```python
sns.heatmap(dataset.corr(),cmap="YlGnBu")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x245d25e9b88>




![png](../../imagenes/02-%20Polynomial%20Regression_14_1.png)


## Ajustar la regresión polinómica con el dataset


```python
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



## Visualización de los resultados del Modelo Lineal


```python
# Visualización de los resultados del Modelo Lineal
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de Regresión Lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
```


![png](../../imagenes/02-%20Polynomial%20Regression_18_0.png)



```python
# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de Regresión Polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
```


![png](../../imagenes/02-%20Polynomial%20Regression_19_0.png)



```python
# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
print('Regresión Lineal:',lin_reg.predict([[6.5]]))
print('Regresión Polinomial:',lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
```

    Regresión Lineal: [330378.78787879]
    Regresión Polinomial: [158862.45265153]
    
