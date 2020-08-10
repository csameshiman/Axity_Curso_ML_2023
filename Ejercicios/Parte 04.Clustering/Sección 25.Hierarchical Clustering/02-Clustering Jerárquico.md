# Clustering Jerárquico

## Los datos

Utilizaremos un conjunto de datos de UCI [Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality). Los dos conjuntos de datos corresponden a las variantes de vino tinto y blanco del vino portugués "Vinho Verde".  

Contiene características psicoquímicas y sensoriales de 1599 muestras de vino tinto y 4898 muestras de vino blanco.

* fixed acidity
* volatile acidity
* citric acid
* residual sugar
* chlorides
* free sulfur dioxide
* total sulfur dioxide
* density
* pH
* sulfates
* alcohol
* quality: calificación entre 0 y 10

Usaremos los datos correspondientes al vino tinto

### Importar las librerías


```python

```

### Importar el archivo winequality-red.csv


```python

```

### Verificar los datos con head(), info(), describe()


```python
#Escribe tu código aqui

```


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Escribe tu código aqui

```


```python

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB
    


```python
#Escribe tu código aqui

```


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Realizar un histograma sobre quality


```python
#Escribe tu código aqui

```


```python

```




    (array([ 10.,   0.,  53.,   0., 681.,   0., 638.,   0., 199.,  18.]),
     array([3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. , 7.5, 8. ]),
     <a list of 10 Patch objects>)




![png](../../imagenes/02-Clustering%20Jer%C3%A1rquico_14_1.png)


### Realizar un groupby por la columna quality


```python
#Escribe tu código aqui

```


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
    <tr>
      <th>quality</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>8.360000</td>
      <td>0.884500</td>
      <td>0.171000</td>
      <td>2.635000</td>
      <td>0.122500</td>
      <td>11.000000</td>
      <td>24.900000</td>
      <td>0.997464</td>
      <td>3.398000</td>
      <td>0.570000</td>
      <td>9.955000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.779245</td>
      <td>0.693962</td>
      <td>0.174151</td>
      <td>2.694340</td>
      <td>0.090679</td>
      <td>12.264151</td>
      <td>36.245283</td>
      <td>0.996542</td>
      <td>3.381509</td>
      <td>0.596415</td>
      <td>10.265094</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.167254</td>
      <td>0.577041</td>
      <td>0.243686</td>
      <td>2.528855</td>
      <td>0.092736</td>
      <td>16.983847</td>
      <td>56.513950</td>
      <td>0.997104</td>
      <td>3.304949</td>
      <td>0.620969</td>
      <td>9.899706</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.347179</td>
      <td>0.497484</td>
      <td>0.273824</td>
      <td>2.477194</td>
      <td>0.084956</td>
      <td>15.711599</td>
      <td>40.869906</td>
      <td>0.996615</td>
      <td>3.318072</td>
      <td>0.675329</td>
      <td>10.629519</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.872362</td>
      <td>0.403920</td>
      <td>0.375176</td>
      <td>2.720603</td>
      <td>0.076588</td>
      <td>14.045226</td>
      <td>35.020101</td>
      <td>0.996104</td>
      <td>3.290754</td>
      <td>0.741256</td>
      <td>11.465913</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.566667</td>
      <td>0.423333</td>
      <td>0.391111</td>
      <td>2.577778</td>
      <td>0.068444</td>
      <td>13.277778</td>
      <td>33.444444</td>
      <td>0.995212</td>
      <td>3.267222</td>
      <td>0.767778</td>
      <td>12.094444</td>
    </tr>
  </tbody>
</table>
</div>



### Normaliza los datos mediante la formula (df-df.min())/(df.max()-df.min()) y visualiza la información con un head()


```python
#Escribe tu código aqui

```


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.247788</td>
      <td>0.397260</td>
      <td>0.00</td>
      <td>0.068493</td>
      <td>0.106845</td>
      <td>0.140845</td>
      <td>0.098940</td>
      <td>0.567548</td>
      <td>0.606299</td>
      <td>0.137725</td>
      <td>0.153846</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.283186</td>
      <td>0.520548</td>
      <td>0.00</td>
      <td>0.116438</td>
      <td>0.143573</td>
      <td>0.338028</td>
      <td>0.215548</td>
      <td>0.494126</td>
      <td>0.362205</td>
      <td>0.209581</td>
      <td>0.215385</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.283186</td>
      <td>0.438356</td>
      <td>0.04</td>
      <td>0.095890</td>
      <td>0.133556</td>
      <td>0.197183</td>
      <td>0.169611</td>
      <td>0.508811</td>
      <td>0.409449</td>
      <td>0.191617</td>
      <td>0.215385</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.584071</td>
      <td>0.109589</td>
      <td>0.56</td>
      <td>0.068493</td>
      <td>0.105175</td>
      <td>0.225352</td>
      <td>0.190813</td>
      <td>0.582232</td>
      <td>0.330709</td>
      <td>0.149701</td>
      <td>0.215385</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.247788</td>
      <td>0.397260</td>
      <td>0.00</td>
      <td>0.068493</td>
      <td>0.106845</td>
      <td>0.140845</td>
      <td>0.098940</td>
      <td>0.567548</td>
      <td>0.606299</td>
      <td>0.137725</td>
      <td>0.153846</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>



## Crea el modelo de Clustering jerárquico con sklearn.cluster y AgglomerativeClustering


```python
#Importa la librería

```


```python
clus= AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df_norm)
```


```python
md_h = pd.Series(clus.labels_)
```

### Realiza un histograma de md_h


```python
#Escribe tu código aqui

```


```python

```




    Text(0, 0.5, 'Número de vinos del cluster')




![png](../../imagenes/02-Clustering%20Jer%C3%A1rquico_27_1.png)


### Importa scipy.cluster.hierarchy con dendrogram, linkage. Genera el dendograma, con metodo ward


```python

```


```python

```

### Genera una gráfica de dendograma


```python
#Escribe tu código aqui

```


```python

```


![png](../../imagenes/02-Clustering%20Jer%C3%A1rquico_33_0.png)


## K-means

### Importa kmeans y datasets de sklearn


```python

```

### Cree el modelo de Kmeans llamado model, 6 clusters


```python
#Escribe tu código aqui

```


```python

```




    KMeans(n_clusters=6)




```python
md_k = pd.Series(model.labels_)
```


```python
df_norm["clust_h"] = md_h
df_norm["clust_k"] = md_k
```


```python
df_norm.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>clust_h</th>
      <th>clust_k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.247788</td>
      <td>0.397260</td>
      <td>0.00</td>
      <td>0.068493</td>
      <td>0.106845</td>
      <td>0.140845</td>
      <td>0.098940</td>
      <td>0.567548</td>
      <td>0.606299</td>
      <td>0.137725</td>
      <td>0.153846</td>
      <td>0.4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.283186</td>
      <td>0.520548</td>
      <td>0.00</td>
      <td>0.116438</td>
      <td>0.143573</td>
      <td>0.338028</td>
      <td>0.215548</td>
      <td>0.494126</td>
      <td>0.362205</td>
      <td>0.209581</td>
      <td>0.215385</td>
      <td>0.4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.283186</td>
      <td>0.438356</td>
      <td>0.04</td>
      <td>0.095890</td>
      <td>0.133556</td>
      <td>0.197183</td>
      <td>0.169611</td>
      <td>0.508811</td>
      <td>0.409449</td>
      <td>0.191617</td>
      <td>0.215385</td>
      <td>0.4</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.584071</td>
      <td>0.109589</td>
      <td>0.56</td>
      <td>0.068493</td>
      <td>0.105175</td>
      <td>0.225352</td>
      <td>0.190813</td>
      <td>0.582232</td>
      <td>0.330709</td>
      <td>0.149701</td>
      <td>0.215385</td>
      <td>0.6</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.247788</td>
      <td>0.397260</td>
      <td>0.00</td>
      <td>0.068493</td>
      <td>0.106845</td>
      <td>0.140845</td>
      <td>0.098940</td>
      <td>0.567548</td>
      <td>0.606299</td>
      <td>0.137725</td>
      <td>0.153846</td>
      <td>0.4</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



### Construye un histograma sobre md_k


```python
#Escribe tu código aqui

```


```python

```




    (array([280.,   0., 220.,   0., 246.,   0., 315.,   0., 508.,  30.]),
     array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ]),
     <a list of 10 Patch objects>)




![png](../../imagenes/02-Clustering%20Jer%C3%A1rquico_45_1.png)



```python
model.cluster_centers_
```




    array([[0.36166245, 0.15994374, 0.4195    , 0.10391389, 0.10674934,
            0.16272636, 0.08533569, 0.41656965, 0.42244094, 0.23588537,
            0.47774725, 0.68857143],
           [0.17304907, 0.32334994, 0.09159091, 0.09327522, 0.09518136,
            0.25496159, 0.12690331, 0.32130557, 0.58031496, 0.1860098 ,
            0.49296037, 0.61272727],
           [0.58263184, 0.2063565 , 0.51191057, 0.13459183, 0.13061742,
            0.13838314, 0.09148496, 0.65695773, 0.33118878, 0.22216543,
            0.29034813, 0.54715447],
           [0.3179941 , 0.28097412, 0.30292063, 0.15262013, 0.12723852,
            0.3782249 , 0.29718997, 0.54121856, 0.43444569, 0.17787283,
            0.21102157, 0.45269841],
           [0.26259494, 0.35962275, 0.1178937 , 0.0913669 , 0.12258949,
            0.14543363, 0.10707827, 0.49014592, 0.48775498, 0.15520534,
            0.22161821, 0.44133858],
           [0.33716814, 0.29223744, 0.48133333, 0.07557078, 0.53789649,
            0.20938967, 0.21071849, 0.5143906 , 0.2335958 , 0.59001996,
            0.16512821, 0.46666667]])




```python
model.inertia_
```




    186.54820577586554



## Interpretación final, utiliza groupby por la columna clust_k


```python
#Escribe tu código aqui

```


```python

```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>clust_h</th>
    </tr>
    <tr>
      <th>clust_k</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.361662</td>
      <td>0.159944</td>
      <td>0.419500</td>
      <td>0.103914</td>
      <td>0.106749</td>
      <td>0.162726</td>
      <td>0.085336</td>
      <td>0.416570</td>
      <td>0.422441</td>
      <td>0.235885</td>
      <td>0.477747</td>
      <td>0.688571</td>
      <td>0.142857</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.173049</td>
      <td>0.323350</td>
      <td>0.091591</td>
      <td>0.093275</td>
      <td>0.095181</td>
      <td>0.254962</td>
      <td>0.126903</td>
      <td>0.321306</td>
      <td>0.580315</td>
      <td>0.186010</td>
      <td>0.492960</td>
      <td>0.612727</td>
      <td>3.459091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.582632</td>
      <td>0.206356</td>
      <td>0.511911</td>
      <td>0.134592</td>
      <td>0.130617</td>
      <td>0.138383</td>
      <td>0.091485</td>
      <td>0.656958</td>
      <td>0.331189</td>
      <td>0.222165</td>
      <td>0.290348</td>
      <td>0.547154</td>
      <td>2.052846</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.317994</td>
      <td>0.280974</td>
      <td>0.302921</td>
      <td>0.152620</td>
      <td>0.127239</td>
      <td>0.378225</td>
      <td>0.297190</td>
      <td>0.541219</td>
      <td>0.434446</td>
      <td>0.177873</td>
      <td>0.211022</td>
      <td>0.452698</td>
      <td>1.057143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.262595</td>
      <td>0.359623</td>
      <td>0.117894</td>
      <td>0.091367</td>
      <td>0.122589</td>
      <td>0.145434</td>
      <td>0.107078</td>
      <td>0.490146</td>
      <td>0.487755</td>
      <td>0.155205</td>
      <td>0.221618</td>
      <td>0.441339</td>
      <td>2.027559</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.337168</td>
      <td>0.292237</td>
      <td>0.481333</td>
      <td>0.075571</td>
      <td>0.537896</td>
      <td>0.209390</td>
      <td>0.210718</td>
      <td>0.514391</td>
      <td>0.233596</td>
      <td>0.590020</td>
      <td>0.165128</td>
      <td>0.466667</td>
      <td>4.900000</td>
    </tr>
  </tbody>
</table>
</div>


