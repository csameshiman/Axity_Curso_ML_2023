# Regresión Polinómica

El conjunto de datos de este modelo proviene del repositorio “UCI Machine Learning”. 
Estos datos se recopilaron en 1978, y cada una de las 506 entradas representan datos agregados de 14 características de casas en diversos barrios de Boston.
Las características son:
- CRIM: Índice de criminalidad per capita
- ZN: Proporción de superficie residencial establecida en lotes mayores de 25.000 sq.ft (equivalente a 2.223 metros cuadrados).
- INDUS: Proporción de superficie de negocio no minorista.
- CHAS: Es la variable ficticia “río Charles” (igual a 1 si el tramo considerado está en la ribera del río, 0 en el otro caso)
- NOX: Concentración de óxidos de nitrógeno (partes por 10 millones)
- RM: Número promedio de habitaciones por vivienda
- AGE: Proporción de viviendas en propiedad ocupadas, construidas antes de 1940
- DIS: Distancias ponderadas a cinco centros de empleo de Boston
- RAD: Índice de accesibilidad a las autopistas radiales
- TAX: Parte del impuesto de bienes inmuebles por cada 10.000 $ de propiedad.
- PTRATIO: Ratio de alumnos por profesor
- B: Se calcula como 1000(Bk — 0.63)², donde Bk es la proporción de personas de descendencia Afroamericana
- LSTAT: Porcentaje de población de “estatus de bajo nivel”
- MEDV: Mediana del valor de viviendas en propiedad (en miles de dólares)

El objetivo es predecir la mediana del valor de las viviendas (MEDV), basándose en las demás características

## Importar Librerias


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn import metrics
%matplotlib inline
```

## Cargar datos


```python
boston = load_boston()
df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
df_boston['MEDV'] = boston.target
```

## Revisar datos


```python
df_boston.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_boston.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    float64
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    float64
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  MEDV     506 non-null    float64
    dtypes: float64(14)
    memory usage: 55.5 KB
    


```python
df_boston.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>



## EDA


```python
sns.pairplot(df_boston)
```




    <seaborn.axisgrid.PairGrid at 0x22303fa4b88>




![png](../../imagenes/01-%20Polynomial%20Regression_11_1.png)



```python
sns.distplot(df_boston['MEDV'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230c2a4b88>




![png](../../imagenes/01-%20Polynomial%20Regression_12_1.png)



```python
sns.heatmap(df_boston.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230dbed1c8>




![png](../../imagenes/01-%20Polynomial%20Regression_13_1.png)



```python
X = pd.DataFrame(np.c_[df_boston['LSTAT']], columns = ['LSTAT'])
y = df_boston['MEDV']
```


```python
np.shape(df_boston['LSTAT'])
```




    (506,)




```python
np.shape(np.c_[df_boston['LSTAT']])
```




    (506, 1)



## Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de pruebas


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2019)
```

## Creando los modelos


```python
lr = LinearRegression()
lr1 = LinearRegression()
lr2 = LinearRegression()
```

### Modelo Lineal


```python
lr = lr.fit(X_train, y_train)
```


```python
lr_pred = lr.predict(X_test)

linear_r2 = r2_score(y_test, lr_pred)
linear_r2
```




    0.5387495541301137




```python
print(lr.intercept_)
```

    35.3675553512179
    

### Modelo Cuadrático


```python
quadratic = PolynomialFeatures(degree=2)
```


```python
X_quad = quadratic.fit_transform(X_train)
```


```python
pr_quad = lr1.fit(X_quad, y_train)
```


```python
pr_quad_pred = pr_quad.predict(quadratic.fit_transform(X_test))

quadratic_r2 = r2_score(y_test, pr_quad_pred)
quadratic_r2
```




    0.6249460055368756




```python
print(pr_quad.intercept_)
```

    44.37353534525711
    

### Modelo Cúbico


```python
cubic = PolynomialFeatures(degree=3)
```


```python
X_cubic = cubic.fit_transform(X_train)
```


```python
pr_cubic = lr2.fit(X_cubic, y_train)
```


```python
pr_cubic_pred = pr_cubic.predict(cubic.fit_transform(X_test))

cubic_r2 = r2_score(y_test, pr_cubic_pred)
cubic_r2
```




    0.6592992766576145




```python
print(pr_cubic.intercept_)
```

    49.63165763207688
    

Como podemos observar el valor de R cuadrado es mayor en el modelo cúbico


```python
#Evaluemos los resultados del modelo cúbico
ax1 = sns.distplot(y_test, hist=False, color="r", label="Valor real")
sns.distplot(pr_cubic_pred, hist=False, color="b", label="Predicción" , ax=ax1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2230ec5a448>




![png](../../imagenes/01-%20Polynomial%20Regression_38_1.png)



```python
print('MAE:', metrics.mean_absolute_error(y_test, pr_cubic_pred))
print('MSE:', metrics.mean_squared_error(y_test, pr_cubic_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pr_cubic_pred)))
```

    MAE: 3.813846378253311
    MSE: 23.97604290605164
    RMSE: 4.896533764414541
    

## Visualización:


```python
X_plot = np.linspace(2, 40, 50).reshape(-1, 1)
lr_pred1 = lr.predict(X_plot)
pr_quad_pred1 = pr_quad.predict(quadratic.fit_transform(X_plot))
pr_cubic_pred1 = pr_cubic.predict(cubic.fit_transform(X_plot))
```


```python
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Training points', color='white', edgecolor='black', s=60)
plt.plot(X_plot, lr_pred1, label='linear grade=1, $R^2=%.2f$' % linear_r2, color='blue', lw=3, linestyle='-')
plt.plot(X_plot, pr_quad_pred1, label='Quadratic grade=2, $R^2=%.2f$' % quadratic_r2,
        color='green', lw=3, linestyle='-')
plt.plot(X_plot, pr_cubic_pred1, label='Cubic grade=3, $R^2=%.2f$' % cubic_r2,
        color='red', lw=3, linestyle='-')

plt.xlabel('% Lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [Price]')
plt.legend(loc='best',fancybox=True)
```




    <matplotlib.legend.Legend at 0x2230ed02c88>




![png](../../imagenes/01-%20Polynomial%20Regression_42_1.png)


## Referencia:

http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
