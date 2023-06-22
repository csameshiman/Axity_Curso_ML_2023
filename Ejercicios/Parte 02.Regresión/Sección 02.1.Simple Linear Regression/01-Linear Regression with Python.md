# Regresión Lineal

Tu vecina es un agente de bienes raíces y requiere de tu ayuda, para predecir los precios de la vivienda en regiones de los Estados Unidos. Sería genial, si de alguna manera, pudieras crear un modelo para ella, que le permita indicar algunas características de una casa y le devuelva una estimación del precio en que se podría vender la casa.  

Ella te preguntó si podrías ayudarla con tus nuevas habilidades de ciencia de datos. ¡Dices sí y decides que la regresión lineal podría ser un buen camino para resolver este problema!  

Luego, tu vecina te brinda la información sobre un grupo de casas en regiones de los Estados Unidos; todo está en el conjunto de datos: USA_Housing.csv.

Los datos contienen las siguientes columnas:

* 'Avg. Area Income': Ingreso promedio de los residentes en la ciudad donde esta ubicada la casa.
* 'Avg. Area House Age': Edad promedio de las casas en la misma ciudad
* 'Avg. Area Number of Rooms': Número promedio de habitaciones para casas en la misma ciudad
* 'Avg. Area Number of Bedrooms': Número promedio de recámaras para casas en la misma ciudad
* 'Area Population': Población de la ciudad donde esta ubicada la casa
* 'Price': precio al que se vendió la casa
* 'Address': Dirección de la casa

## Importar Librerias


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Cargar Datos


```python
USAhousing = pd.read_csv('USA_Housing.csv')
```


```python
USAhousing.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
      <th>Address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79545.458574</td>
      <td>5.682861</td>
      <td>7.009188</td>
      <td>4.09</td>
      <td>23086.800503</td>
      <td>1.059034e+06</td>
      <td>208 Michael Ferry Apt. 674\nLaurabury, NE 3701...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79248.642455</td>
      <td>6.002900</td>
      <td>6.730821</td>
      <td>3.09</td>
      <td>40173.072174</td>
      <td>1.505891e+06</td>
      <td>188 Johnson Views Suite 079\nLake Kathleen, CA...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61287.067179</td>
      <td>5.865890</td>
      <td>8.512727</td>
      <td>5.13</td>
      <td>36882.159400</td>
      <td>1.058988e+06</td>
      <td>9127 Elizabeth Stravenue\nDanieltown, WI 06482...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>63345.240046</td>
      <td>7.188236</td>
      <td>5.586729</td>
      <td>3.26</td>
      <td>34310.242831</td>
      <td>1.260617e+06</td>
      <td>USS Barnett\nFPO AP 44820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59982.197226</td>
      <td>5.040555</td>
      <td>7.839388</td>
      <td>4.23</td>
      <td>26354.109472</td>
      <td>6.309435e+05</td>
      <td>USNS Raymond\nFPO AE 09386</td>
    </tr>
  </tbody>
</table>
</div>




```python
USAhousing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5000 entries, 0 to 4999
    Data columns (total 7 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   Avg. Area Income              5000 non-null   float64
     1   Avg. Area House Age           5000 non-null   float64
     2   Avg. Area Number of Rooms     5000 non-null   float64
     3   Avg. Area Number of Bedrooms  5000 non-null   float64
     4   Area Population               5000 non-null   float64
     5   Price                         5000 non-null   float64
     6   Address                       5000 non-null   object 
    dtypes: float64(6), object(1)
    memory usage: 273.6+ KB
    


```python
USAhousing.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Area Income</th>
      <th>Avg. Area House Age</th>
      <th>Avg. Area Number of Rooms</th>
      <th>Avg. Area Number of Bedrooms</th>
      <th>Area Population</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5.000000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>68583.108984</td>
      <td>5.977222</td>
      <td>6.987792</td>
      <td>3.981330</td>
      <td>36163.516039</td>
      <td>1.232073e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10657.991214</td>
      <td>0.991456</td>
      <td>1.005833</td>
      <td>1.234137</td>
      <td>9925.650114</td>
      <td>3.531176e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17796.631190</td>
      <td>2.644304</td>
      <td>3.236194</td>
      <td>2.000000</td>
      <td>172.610686</td>
      <td>1.593866e+04</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>61480.562388</td>
      <td>5.322283</td>
      <td>6.299250</td>
      <td>3.140000</td>
      <td>29403.928702</td>
      <td>9.975771e+05</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68804.286404</td>
      <td>5.970429</td>
      <td>7.002902</td>
      <td>4.050000</td>
      <td>36199.406689</td>
      <td>1.232669e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75783.338666</td>
      <td>6.650808</td>
      <td>7.665871</td>
      <td>4.490000</td>
      <td>42861.290769</td>
      <td>1.471210e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>107701.748378</td>
      <td>9.519088</td>
      <td>10.759588</td>
      <td>6.500000</td>
      <td>69621.713378</td>
      <td>2.469066e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
USAhousing.columns
```




    Index(['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
           'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address'],
          dtype='object')



## Análisis Exploratorio de Datos (EDA)




```python
sns.pairplot(USAhousing)
```




    <seaborn.axisgrid.PairGrid at 0x1f2f8edc208>




![png](../../imagenes/01-Linear%20Regression%20with%20Python_10_1.png)



```python
sns.distplot(USAhousing['Price'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f2fb04a978>




![png](../../imagenes/01-Linear%20Regression%20with%20Python_11_1.png)



```python
sns.heatmap(USAhousing.corr())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f2fb0432e8>




![png](../../imagenes//01-Linear%20Regression%20with%20Python_12_1.png)


## Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de prueba


```python
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
```


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```

## Crear el modelo


```python
from sklearn.linear_model import LinearRegression
```


```python
lm = LinearRegression()
```


```python
lm.fit(X_train,y_train)
```




    LinearRegression()



## Evaluar el Modelo

Vamos a evaluar el modelo comprobando sus coeficientes y cómo podemos interpretarlos.


```python
print(lm.intercept_)
```

    -2642239.251234695
    


```python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Area Income</th>
      <td>21.570413</td>
    </tr>
    <tr>
      <th>Avg. Area House Age</th>
      <td>166552.477670</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Rooms</th>
      <td>119512.534382</td>
    </tr>
    <tr>
      <th>Avg. Area Number of Bedrooms</th>
      <td>2758.951878</td>
    </tr>
    <tr>
      <th>Area Population</th>
      <td>15.296861</td>
    </tr>
  </tbody>
</table>
</div>




## Interpretando los coeficientes:

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Avg. Area Income** está asociado con un aumento de **$21.57**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Avg. Area House Age** está asociado con un aumento de **$166552.47**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Avg. Area Number of Rooms** está asociado con un aumento de **$119512.53**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Avg. Area Number of Bedrooms** está asociado con un aumento de **$2758.95**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Area Population** se asocia con un aumento de **$15.29**.  

¿Esto tiene sentido?



## Predicciones del Modelo




```python
predictions = lm.predict(X_test)
```


```python
ax1 = sns.distplot(y_test, hist=False, color="r", label="Valor real")
sns.distplot(predictions, hist=False, color="b", label="Predicción" , ax=ax1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f2fb971668>




![png](../../imagenes/01-Linear%20Regression%20with%20Python_27_1.png)



```python
ax = sns.scatterplot(x="total_bill", y="tip", style="time", data=tips)
```


```python
plt.scatter(y_test,predictions)
```




    <matplotlib.collections.PathCollection at 0x1f2fba7cbe0>




![png](../../imagenes/01-Linear%20Regression%20with%20Python_29_1.png)


**Histograma de Residuos**


```python
sns.distplot((y_test-predictions),bins=50);
```


![png](../../imagenes/01-Linear%20Regression%20with%20Python_31_0.png)


## Métricas de evaluación para problemas de regresión:
    
**Mean Absolute Error** (MAE) es la media del valor absoluto de los errores:

<img src="https://render.githubusercontent.com/render/math?math=\frac%201n\sum_{i=1}^n|y_i-\hat{y}_i|">

**Mean Squared Error** (MSE) es la media de los errores al cuadrado:

<img src="https://render.githubusercontent.com/render/math?math=\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2">

**Root Mean Squared Error** (RMSE) es la raíz cuadrada de la media de los errores al cuadrado:

<img src="https://render.githubusercontent.com/render/math?math=\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}">

Comparando estas métricas:

- **MAE** es el más fácil de entender, porque es el error promedio.
- **MSE** es más popular que MAE, porque MSE "castiga" errores más grandes, lo que tiende a ser útil en el mundo real.
- **RMSE** es aún más popular que MSE, porque RMSE es interpretable en las unidades "y".

Todas estas son **funciones de pérdida**, porque queremos minimizarlas.


```python
from sklearn import metrics
```


```python
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 81331.22699573652
    MSE: 10119734875.653336
    RMSE: 100596.89297216557
    


```python

```
