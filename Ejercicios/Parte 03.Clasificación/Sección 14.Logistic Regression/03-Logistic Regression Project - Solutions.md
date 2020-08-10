# Regresión Logística

En este proyecto, trabajaremos con un conjunto de datos publicitarios falsos, que indica si un usuario de Internet en particular hizo clic en un anuncio. Intentaremos crear un modelo que prediga si harán clic o no en un anuncio, en función de las características de ese usuario.

Este conjunto de datos contiene las siguientes características:

* 'Daily Time Spent on Site': tiempo del consumidor en el sitio en minutos
* 'Age': edad del cliente en años
* 'Area Income': Prom. Ingresos del área geográfica del consumidor
* 'Daily Internet Usage': Prom. minutos al día el consumidor está en internet
* 'Ad Topic Line': título del anuncio
* 'City': Ciudad del consumidor
* 'Male': si el consumidor era o no hombre
* 'Country': País del consumidor
* 'Timestamp': hora en que el consumidor hizo clic en el anuncio o en la ventana cerrada
* 'Clicked on Ad': 0 o 1 indicaron hacer clic en el anuncio

## Importar Librerías


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Cargar datos
**Lee el archivo advertising.csv y configúralo en un dataframe llamado ad_data.**


```python
ad_data = pd.read_csv('advertising.csv')
```

**Valide los datos con head()**


```python
ad_data.head(15)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Ad Topic Line</th>
      <th>City</th>
      <th>Male</th>
      <th>Country</th>
      <th>Timestamp</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>68.95</td>
      <td>35</td>
      <td>61833.90</td>
      <td>256.09</td>
      <td>Cloned 5thgeneration orchestration</td>
      <td>Wrightburgh</td>
      <td>0</td>
      <td>Tunisia</td>
      <td>2016-03-27 00:53:11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>80.23</td>
      <td>31</td>
      <td>68441.85</td>
      <td>193.77</td>
      <td>Monitored national standardization</td>
      <td>West Jodi</td>
      <td>1</td>
      <td>Nauru</td>
      <td>2016-04-04 01:39:02</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69.47</td>
      <td>26</td>
      <td>59785.94</td>
      <td>236.50</td>
      <td>Organic bottom-line service-desk</td>
      <td>Davidton</td>
      <td>0</td>
      <td>San Marino</td>
      <td>2016-03-13 20:35:42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.15</td>
      <td>29</td>
      <td>54806.18</td>
      <td>245.89</td>
      <td>Triple-buffered reciprocal time-frame</td>
      <td>West Terrifurt</td>
      <td>1</td>
      <td>Italy</td>
      <td>2016-01-10 02:31:19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68.37</td>
      <td>35</td>
      <td>73889.99</td>
      <td>225.58</td>
      <td>Robust logistical utilization</td>
      <td>South Manuel</td>
      <td>0</td>
      <td>Iceland</td>
      <td>2016-06-03 03:36:18</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>59.99</td>
      <td>23</td>
      <td>59761.56</td>
      <td>226.74</td>
      <td>Sharable client-driven software</td>
      <td>Jamieberg</td>
      <td>1</td>
      <td>Norway</td>
      <td>2016-05-19 14:30:17</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>88.91</td>
      <td>33</td>
      <td>53852.85</td>
      <td>208.36</td>
      <td>Enhanced dedicated support</td>
      <td>Brandonstad</td>
      <td>0</td>
      <td>Myanmar</td>
      <td>2016-01-28 20:59:32</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>66.00</td>
      <td>48</td>
      <td>24593.33</td>
      <td>131.76</td>
      <td>Reactive local challenge</td>
      <td>Port Jefferybury</td>
      <td>1</td>
      <td>Australia</td>
      <td>2016-03-07 01:40:15</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>74.53</td>
      <td>30</td>
      <td>68862.00</td>
      <td>221.51</td>
      <td>Configurable coherent function</td>
      <td>West Colin</td>
      <td>1</td>
      <td>Grenada</td>
      <td>2016-04-18 09:33:42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>69.88</td>
      <td>20</td>
      <td>55642.32</td>
      <td>183.82</td>
      <td>Mandatory homogeneous architecture</td>
      <td>Ramirezton</td>
      <td>1</td>
      <td>Ghana</td>
      <td>2016-07-11 01:42:51</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>47.64</td>
      <td>49</td>
      <td>45632.51</td>
      <td>122.02</td>
      <td>Centralized neutral neural-net</td>
      <td>West Brandonton</td>
      <td>0</td>
      <td>Qatar</td>
      <td>2016-03-16 20:19:01</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>83.07</td>
      <td>37</td>
      <td>62491.01</td>
      <td>230.87</td>
      <td>Team-oriented grid-enabled Local Area Network</td>
      <td>East Theresashire</td>
      <td>1</td>
      <td>Burundi</td>
      <td>2016-05-08 08:10:10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>69.57</td>
      <td>48</td>
      <td>51636.92</td>
      <td>113.12</td>
      <td>Centralized content-based focus group</td>
      <td>West Katiefurt</td>
      <td>1</td>
      <td>Egypt</td>
      <td>2016-06-03 01:14:41</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>79.52</td>
      <td>24</td>
      <td>51739.63</td>
      <td>214.23</td>
      <td>Synergistic fresh-thinking array</td>
      <td>North Tara</td>
      <td>0</td>
      <td>Bosnia and Herzegovina</td>
      <td>2016-04-20 21:49:22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>42.95</td>
      <td>33</td>
      <td>30976.00</td>
      <td>143.56</td>
      <td>Grass-roots coherent extranet</td>
      <td>West William</td>
      <td>0</td>
      <td>Barbados</td>
      <td>2016-03-24 09:31:49</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Identifica los valores únicos de country**


```python
count=0
a= ad_data['Country'].unique().tolist()
for dia in a:
    count += 1
count
```




    237



**Continua el análisis del dataframe con las funciones info y describe**


```python
ad_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Daily Time Spent on Site  1000 non-null   float64
     1   Age                       1000 non-null   int64  
     2   Area Income               1000 non-null   float64
     3   Daily Internet Usage      1000 non-null   float64
     4   Ad Topic Line             1000 non-null   object 
     5   City                      1000 non-null   object 
     6   Male                      1000 non-null   int64  
     7   Country                   1000 non-null   object 
     8   Timestamp                 1000 non-null   object 
     9   Clicked on Ad             1000 non-null   int64  
    dtypes: float64(3), int64(3), object(4)
    memory usage: 78.2+ KB
    


```python
ad_data.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Daily Time Spent on Site</th>
      <th>Age</th>
      <th>Area Income</th>
      <th>Daily Internet Usage</th>
      <th>Male</th>
      <th>Clicked on Ad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.000200</td>
      <td>36.009000</td>
      <td>55000.000080</td>
      <td>180.000100</td>
      <td>0.481000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.853615</td>
      <td>8.785562</td>
      <td>13414.634022</td>
      <td>43.902339</td>
      <td>0.499889</td>
      <td>0.50025</td>
    </tr>
    <tr>
      <th>min</th>
      <td>32.600000</td>
      <td>19.000000</td>
      <td>13996.500000</td>
      <td>104.780000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.360000</td>
      <td>29.000000</td>
      <td>47031.802500</td>
      <td>138.830000</td>
      <td>0.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68.215000</td>
      <td>35.000000</td>
      <td>57012.300000</td>
      <td>183.130000</td>
      <td>0.000000</td>
      <td>0.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>78.547500</td>
      <td>42.000000</td>
      <td>65470.635000</td>
      <td>218.792500</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>91.430000</td>
      <td>61.000000</td>
      <td>79484.800000</td>
      <td>269.960000</td>
      <td>1.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>
</div>



## Análisis Exploratorio de Datos (EDA)

¡Usemos Seaborn para explorar los datos!

¡Intenta recrear los grafícos que se muestran a continuación!

**Crea un histograma de "Age"**


```python
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
```




    Text(0.5, 0, 'Age')




![png](../../imagenes/03-Logistic%20Regression%20Project%20-%20Solutions_12_1.png)


**Crea un jointplot que muestre "Area Income" vs. "Age"**


```python
sns.jointplot(x='Age',y='Area Income',data=ad_data)
```




    <seaborn.axisgrid.JointGrid at 0x277fe996088>




![png](../../imagenes/03-Logistic%20Regression%20Project%20-%20Solutions_14_1.png)


**Cree un jointplot que muestre la distribución KDE de "Daily Time spent on site" vs. "Age".**


```python
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');
```


![png](../../imagenes/03-Logistic%20Regression%20Project%20-%20Solutions_16_0.png)


**Crea un jointplot que muestre "Daily Time Spent on Site" vs. "Daily Internet Usage"**


```python
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')
```




    <seaborn.axisgrid.JointGrid at 0x27780395a88>




![png](../../imagenes/03-Logistic%20Regression%20Project%20-%20Solutions_18_1.png)


**Finalmente, crea un pairplot con el parámetro 'hue' definido por la característica "Clicked on Ad".**


```python
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
```




    <seaborn.axisgrid.PairGrid at 0x277804d8748>




![png](../../imagenes/03-Logistic%20Regression%20Project%20-%20Solutions_20_1.png)


# Inicialicemos el modelo de regresión logística

¡Ahora es el momento de hacer una división del conjunto de datos en entrenamiento y pruebas para nuestro modelo!

¡Tendrás la libertad de elegir las columnas en las que quieras entrenar!

**Divide los datos en un conjunto de entrenamiento y pruebas usando train_test_split**


```python
# Importa la librería
from sklearn.model_selection import train_test_split
```


```python
# Genera la variable X con las columnas que quieras entrenar y la variable y con "Clicked on Ad"
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
```


```python
# Genera los conjuntos de entrenamiento y pruebas, con los parámetros test_size=0.33 y randomstate=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

**Genera el modelo de Regresión Logística con el conjunto de entrenamiento.**


```python
# Importa la librería
from sklearn.linear_model import LogisticRegression
```


```python
# Genera el modelo y entrénalo
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



## Predicción y evaluación

**Ahora realiza la predicción de los valores para los datos de prueba.**


```python
predictions = logmodel.predict(X_test)
```

**Crea un reporte del performance para la clasificación del modelo**


```python
# Importa las librerías
from sklearn.metrics import classification_report
```


```python
# Imprime el reporte de la clasificación
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.96      0.91       162
               1       0.96      0.85      0.90       168
    
        accuracy                           0.91       330
       macro avg       0.91      0.91      0.91       330
    weighted avg       0.91      0.91      0.91       330
    
    

La precision es la relación donde está el número de positivos verdaderos y el número de falsos positivos. La precision es la capacidad del clasificador de no etiquetar como positiva una muestra que es negativa.

tp / (tp + fp) tpfp

El recall es la relación donde está el número de positivos veraderos y el número de falsos negativos. El recall la capacidad del clasificador para encontrar todas las muestras positivas.

tp / (tp + fn) tpfn

El f1-score se puede interpretar como una media armónica ponderada de la precisión y el recall, donde un f1-score alcanza su mejor valor en 1 y el peor puntaje en 0.

El support es el número de ocurrencias de cada clase en y_true.

**Crea la matriz de confusión**


```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)
cm
```




    array([[156,   6],
           [ 25, 143]], dtype=int64)



## ¡Buen trabajo!
