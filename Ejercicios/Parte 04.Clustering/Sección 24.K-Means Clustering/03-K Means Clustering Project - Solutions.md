# K Means Clustering

Para este proyecto, intentaremos utilizar KMeans Clustering para agrupar universidades en dos grupos, privado y público.

___
Es muy importante tener en cuenta que, en realidad, tenemos las etiquetas para este conjunto de datos, pero NO las usaremos para el algoritmo de agrupación de KMeans, ya que es un algoritmo de aprendizaje no supervisado. **

Cuando se usa el algoritmo Kmeans en circunstancias normales, es porque no tiene etiquetas. En este caso, usaremos las etiquetas para tratar de tener una idea de qué tan bien funcionó el algoritmo, pero generalmente no lo hará para Kmeans, por lo que el informe de clasificación y la matriz de confusión al final de este proyecto, realmente no tiene sentido en un entorno del mundo real!
___

## Los datos

Utilizaremos un conjunto de datos con 777 observaciones en las siguientes 18 variables.

* Private: Un factor con niveles No y Sí que indican universidad privada o pública
* Apps: Número de aplicaciones recibidas
* Accept: Número de solicitudes aceptadas
* Enroll: Número de nuevos estudiantes matriculados
* Top10perc: Pct. nuevos estudiantes del 10% superior de H.S. clase
* Top25perc: Pct. nuevos estudiantes del 25% superior de H.S. clase
* F.Ungrado: Número de estudiantes universitarios a tiempo completo
* P.Undergrad: Número de estudiantes universitarios a tiempo parcial
* Outstate: fuera del estado fuera del estado
* Room: Alojamiento y comida.
* Books: Costo estimado del libro
* Personal: personales estimados personales
* PhD Pct.: de la facultad con doctorado
* Terminal: Pct. de facultad con título terminal
* S.F.Ratio: Relación estudiante / facultad
* perc.alumni: Pct. ex alumnos que donan
* Expend gastos de instrucción por alumno
* Grad.Rate Tasa de graduación

## Importar Librerías

** Importa las librerías que generalmente utilizas para el análisis de datos**


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Cargar los datos

*Lee el archivo College_Data.csv usando read_csv. Descubre cómo establecer la primera columna como índice.*


```python
df = pd.read_csv('College_Data.csv',index_col=0)
```

**Revisa los datos con head()**


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abilene Christian University</th>
      <td>Yes</td>
      <td>1660</td>
      <td>1232</td>
      <td>721</td>
      <td>23</td>
      <td>52</td>
      <td>2885</td>
      <td>537</td>
      <td>7440</td>
      <td>3300</td>
      <td>450</td>
      <td>2200</td>
      <td>70</td>
      <td>78</td>
      <td>18.1</td>
      <td>12</td>
      <td>7041</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Adelphi University</th>
      <td>Yes</td>
      <td>2186</td>
      <td>1924</td>
      <td>512</td>
      <td>16</td>
      <td>29</td>
      <td>2683</td>
      <td>1227</td>
      <td>12280</td>
      <td>6450</td>
      <td>750</td>
      <td>1500</td>
      <td>29</td>
      <td>30</td>
      <td>12.2</td>
      <td>16</td>
      <td>10527</td>
      <td>56</td>
    </tr>
    <tr>
      <th>Adrian College</th>
      <td>Yes</td>
      <td>1428</td>
      <td>1097</td>
      <td>336</td>
      <td>22</td>
      <td>50</td>
      <td>1036</td>
      <td>99</td>
      <td>11250</td>
      <td>3750</td>
      <td>400</td>
      <td>1165</td>
      <td>53</td>
      <td>66</td>
      <td>12.9</td>
      <td>30</td>
      <td>8735</td>
      <td>54</td>
    </tr>
    <tr>
      <th>Agnes Scott College</th>
      <td>Yes</td>
      <td>417</td>
      <td>349</td>
      <td>137</td>
      <td>60</td>
      <td>89</td>
      <td>510</td>
      <td>63</td>
      <td>12960</td>
      <td>5450</td>
      <td>450</td>
      <td>875</td>
      <td>92</td>
      <td>97</td>
      <td>7.7</td>
      <td>37</td>
      <td>19016</td>
      <td>59</td>
    </tr>
    <tr>
      <th>Alaska Pacific University</th>
      <td>Yes</td>
      <td>193</td>
      <td>146</td>
      <td>55</td>
      <td>16</td>
      <td>44</td>
      <td>249</td>
      <td>869</td>
      <td>7560</td>
      <td>4120</td>
      <td>800</td>
      <td>1500</td>
      <td>76</td>
      <td>72</td>
      <td>11.9</td>
      <td>2</td>
      <td>10922</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



**Usa los metodos info() y describe() sobre los datos.**


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 777 entries, Abilene Christian University to York College of Pennsylvania
    Data columns (total 18 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Private      777 non-null    object 
     1   Apps         777 non-null    int64  
     2   Accept       777 non-null    int64  
     3   Enroll       777 non-null    int64  
     4   Top10perc    777 non-null    int64  
     5   Top25perc    777 non-null    int64  
     6   F.Undergrad  777 non-null    int64  
     7   P.Undergrad  777 non-null    int64  
     8   Outstate     777 non-null    int64  
     9   Room.Board   777 non-null    int64  
     10  Books        777 non-null    int64  
     11  Personal     777 non-null    int64  
     12  PhD          777 non-null    int64  
     13  Terminal     777 non-null    int64  
     14  S.F.Ratio    777 non-null    float64
     15  perc.alumni  777 non-null    int64  
     16  Expend       777 non-null    int64  
     17  Grad.Rate    777 non-null    int64  
    dtypes: float64(1), int64(16), object(1)
    memory usage: 115.3+ KB
    


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.000000</td>
      <td>777.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3001.638353</td>
      <td>2018.804376</td>
      <td>779.972973</td>
      <td>27.558559</td>
      <td>55.796654</td>
      <td>3699.907336</td>
      <td>855.298584</td>
      <td>10440.669241</td>
      <td>4357.526384</td>
      <td>549.380952</td>
      <td>1340.642214</td>
      <td>72.660232</td>
      <td>79.702703</td>
      <td>14.089704</td>
      <td>22.743887</td>
      <td>9660.171171</td>
      <td>65.46332</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3870.201484</td>
      <td>2451.113971</td>
      <td>929.176190</td>
      <td>17.640364</td>
      <td>19.804778</td>
      <td>4850.420531</td>
      <td>1522.431887</td>
      <td>4023.016484</td>
      <td>1096.696416</td>
      <td>165.105360</td>
      <td>677.071454</td>
      <td>16.328155</td>
      <td>14.722359</td>
      <td>3.958349</td>
      <td>12.391801</td>
      <td>5221.768440</td>
      <td>17.17771</td>
    </tr>
    <tr>
      <th>min</th>
      <td>81.000000</td>
      <td>72.000000</td>
      <td>35.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>139.000000</td>
      <td>1.000000</td>
      <td>2340.000000</td>
      <td>1780.000000</td>
      <td>96.000000</td>
      <td>250.000000</td>
      <td>8.000000</td>
      <td>24.000000</td>
      <td>2.500000</td>
      <td>0.000000</td>
      <td>3186.000000</td>
      <td>10.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>776.000000</td>
      <td>604.000000</td>
      <td>242.000000</td>
      <td>15.000000</td>
      <td>41.000000</td>
      <td>992.000000</td>
      <td>95.000000</td>
      <td>7320.000000</td>
      <td>3597.000000</td>
      <td>470.000000</td>
      <td>850.000000</td>
      <td>62.000000</td>
      <td>71.000000</td>
      <td>11.500000</td>
      <td>13.000000</td>
      <td>6751.000000</td>
      <td>53.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1558.000000</td>
      <td>1110.000000</td>
      <td>434.000000</td>
      <td>23.000000</td>
      <td>54.000000</td>
      <td>1707.000000</td>
      <td>353.000000</td>
      <td>9990.000000</td>
      <td>4200.000000</td>
      <td>500.000000</td>
      <td>1200.000000</td>
      <td>75.000000</td>
      <td>82.000000</td>
      <td>13.600000</td>
      <td>21.000000</td>
      <td>8377.000000</td>
      <td>65.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3624.000000</td>
      <td>2424.000000</td>
      <td>902.000000</td>
      <td>35.000000</td>
      <td>69.000000</td>
      <td>4005.000000</td>
      <td>967.000000</td>
      <td>12925.000000</td>
      <td>5050.000000</td>
      <td>600.000000</td>
      <td>1700.000000</td>
      <td>85.000000</td>
      <td>92.000000</td>
      <td>16.500000</td>
      <td>31.000000</td>
      <td>10830.000000</td>
      <td>78.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>48094.000000</td>
      <td>26330.000000</td>
      <td>6392.000000</td>
      <td>96.000000</td>
      <td>100.000000</td>
      <td>31643.000000</td>
      <td>21836.000000</td>
      <td>21700.000000</td>
      <td>8124.000000</td>
      <td>2340.000000</td>
      <td>6800.000000</td>
      <td>103.000000</td>
      <td>100.000000</td>
      <td>39.800000</td>
      <td>64.000000</td>
      <td>56233.000000</td>
      <td>118.00000</td>
    </tr>
  </tbody>
</table>
</div>



## EDA

¡Es hora de crear algunas visualizaciones de datos!

**Crea un diagrama de dispersión de 'Grad.Rate' vs. 'Room.Board', donde los puntos están coloreados por la columna 'Private'.**


```python
sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
```



    <seaborn.axisgrid.FacetGrid at 0x2208c369c88>




![png](../../imagenes/03-K%20Means%20Clustering%20Project%20-%20Solutions_12_2.png)


**Crea un diagrama de dispersión de 'F.Undergrad' vs. 'Outstate', donde los puntos están coloreados por la columna 'Private'.**


```python
sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)
```




    <seaborn.axisgrid.FacetGrid at 0x2208c6bcd68>




![png](../../imagenes/03-K%20Means%20Clustering%20Project%20-%20Solutions_14_1.png)


**Crea un histograma apilado que muestre la matrícula 'outstate' basada en la columna 'Private'. Intenta hacer esto usando [sns.FacetGrid] (https://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.FacetGrid.html). Si eso es demasiado complicado, ve si puedes hacerlo simplemente usando dos instancias de pandas.plot (kind = 'hist').**


```python
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)
```



![png](../../imagenes/03-K%20Means%20Clustering%20Project%20-%20Solutions_16_1.png)


**Crea un diagrama similar al anterior pero utilizando como hue la columna 'Grad.RateCreate'**


```python
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
```


![png](../../imagenes/03-K%20Means%20Clustering%20Project%20-%20Solutions_18_0.png)


**Observa cómo parece haber una escuela privada con una tasa de graduación superior al 100%. ¿Cuál es el nombre de esa escuela?**


```python
df[df['Grad.Rate'] > 100]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cazenovia College</th>
      <td>Yes</td>
      <td>3847</td>
      <td>3433</td>
      <td>527</td>
      <td>9</td>
      <td>35</td>
      <td>1010</td>
      <td>12</td>
      <td>9384</td>
      <td>4840</td>
      <td>600</td>
      <td>500</td>
      <td>22</td>
      <td>47</td>
      <td>14.3</td>
      <td>20</td>
      <td>7697</td>
      <td>118</td>
    </tr>
  </tbody>
</table>
</div>



**Establece la tasa de graduación de esa escuela en 100 para que tenga sentido. Es posible que recibas una advertencia, (no un error) al realizar esta operación, por lo tanto, utiliza las operaciones de dataframe o simplemente vuelve a hacer la visualización del histograma para asegurarte de que realmente se realizó.**


```python
df['Grad.Rate']['Cazenovia College'] = 100
```

   


```python
df[df['Grad.Rate'] > 100]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',height=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)
```


![png](../../imagenes/03-K%20Means%20Clustering%20Project%20-%20Solutions_24_0.png)


## Creación del modelo de K Means



**Importa KMeans desde SciKit Learn.**


```python
from sklearn.cluster import KMeans
```

**Crea una instancia de un modelo de K Means con 2 clusters.**


```python
kmeans = KMeans(n_clusters=2)
```

**Entrena el modelo con todos los datos, excepto la etiqueta private.**


```python
kmeans.fit(df.drop('Private',axis=1))
```




    KMeans(n_clusters=2)



**¿Cuales son los centroides?**


```python
kmeans.cluster_centers_
```




    array([[1.03631389e+04, 6.55089815e+03, 2.56972222e+03, 4.14907407e+01,
            7.02037037e+01, 1.30619352e+04, 2.46486111e+03, 1.07191759e+04,
            4.64347222e+03, 5.95212963e+02, 1.71420370e+03, 8.63981481e+01,
            9.13333333e+01, 1.40277778e+01, 2.00740741e+01, 1.41705000e+04,
            6.75925926e+01],
           [1.81323468e+03, 1.28716592e+03, 4.91044843e+02, 2.53094170e+01,
            5.34708520e+01, 2.18854858e+03, 5.95458894e+02, 1.03957085e+04,
            4.31136472e+03, 5.41982063e+02, 1.28033632e+03, 7.04424514e+01,
            7.78251121e+01, 1.40997010e+01, 2.31748879e+01, 8.93204634e+03,
            6.50926756e+01]])



## Evaluación

No hay una manera perfecta de evaluar la agrupación si no tenemos las etiquetas; sin embargo, dado que esto es solo un ejercicio, tenemos las etiquetas, por lo que aprovechamos esto para evaluar nuestros grupos. Ten en cuenta que generalmente no tenemos este lujo en el mundo real.

**Crea una nueva columna para df llamada 'Cluster', que es un 1 para una escuela privada y un 0 para una escuela pública.**


```python
def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0
```


```python
df['Cluster'] = df['Private'].apply(converter)
```


```python
#Verifica los datos con el método head()
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Private</th>
      <th>Apps</th>
      <th>Accept</th>
      <th>Enroll</th>
      <th>Top10perc</th>
      <th>Top25perc</th>
      <th>F.Undergrad</th>
      <th>P.Undergrad</th>
      <th>Outstate</th>
      <th>Room.Board</th>
      <th>Books</th>
      <th>Personal</th>
      <th>PhD</th>
      <th>Terminal</th>
      <th>S.F.Ratio</th>
      <th>perc.alumni</th>
      <th>Expend</th>
      <th>Grad.Rate</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Abilene Christian University</th>
      <td>Yes</td>
      <td>1660</td>
      <td>1232</td>
      <td>721</td>
      <td>23</td>
      <td>52</td>
      <td>2885</td>
      <td>537</td>
      <td>7440</td>
      <td>3300</td>
      <td>450</td>
      <td>2200</td>
      <td>70</td>
      <td>78</td>
      <td>18.1</td>
      <td>12</td>
      <td>7041</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Adelphi University</th>
      <td>Yes</td>
      <td>2186</td>
      <td>1924</td>
      <td>512</td>
      <td>16</td>
      <td>29</td>
      <td>2683</td>
      <td>1227</td>
      <td>12280</td>
      <td>6450</td>
      <td>750</td>
      <td>1500</td>
      <td>29</td>
      <td>30</td>
      <td>12.2</td>
      <td>16</td>
      <td>10527</td>
      <td>56</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Adrian College</th>
      <td>Yes</td>
      <td>1428</td>
      <td>1097</td>
      <td>336</td>
      <td>22</td>
      <td>50</td>
      <td>1036</td>
      <td>99</td>
      <td>11250</td>
      <td>3750</td>
      <td>400</td>
      <td>1165</td>
      <td>53</td>
      <td>66</td>
      <td>12.9</td>
      <td>30</td>
      <td>8735</td>
      <td>54</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Agnes Scott College</th>
      <td>Yes</td>
      <td>417</td>
      <td>349</td>
      <td>137</td>
      <td>60</td>
      <td>89</td>
      <td>510</td>
      <td>63</td>
      <td>12960</td>
      <td>5450</td>
      <td>450</td>
      <td>875</td>
      <td>92</td>
      <td>97</td>
      <td>7.7</td>
      <td>37</td>
      <td>19016</td>
      <td>59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Alaska Pacific University</th>
      <td>Yes</td>
      <td>193</td>
      <td>146</td>
      <td>55</td>
      <td>16</td>
      <td>44</td>
      <td>249</td>
      <td>869</td>
      <td>7560</td>
      <td>4120</td>
      <td>800</td>
      <td>1500</td>
      <td>76</td>
      <td>72</td>
      <td>11.9</td>
      <td>2</td>
      <td>10922</td>
      <td>15</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**Crea una matriz de confusión y un informe de clasificación para ver qué tan bien funcionó la agrupación de Kmeans sin recibir ninguna etiqueta.**


```python
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))
```

    [[ 74 138]
     [ 34 531]]
                  precision    recall  f1-score   support
    
               0       0.69      0.35      0.46       212
               1       0.79      0.94      0.86       565
    
        accuracy                           0.78       777
       macro avg       0.74      0.64      0.66       777
    weighted avg       0.76      0.78      0.75       777
    
    


¡No está tan mal teniendo en cuenta que el algoritmo está usando puramente las características para agrupar las universidades en 2 grupos distintos! ¡Esperemos que puedas comenzar a ver cómo K Means es útil para agrupar datos sin etiquetar!

