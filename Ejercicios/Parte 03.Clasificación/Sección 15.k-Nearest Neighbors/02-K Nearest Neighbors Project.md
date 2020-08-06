# K Nearest Neighbors


¡Bienvenido al Proyecto KNN! Este será un proyecto simple, muy similar al ejemplo anterior, excepto que se te dará otro conjunto de datos. Sigue adelante y solo sigue las instrucciones a continuación.

## Importa las librerías


```python

```

## Cargar los datos
**Cargue los datos del archivo'KNN_Project_Data.csv en un dataframe  lamado df**


```python

```

**Valida el head del dataframe.**


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
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1636.670614</td>
      <td>817.988525</td>
      <td>2565.995189</td>
      <td>358.347163</td>
      <td>550.417491</td>
      <td>1618.870897</td>
      <td>2147.641254</td>
      <td>330.727893</td>
      <td>1494.878631</td>
      <td>845.136088</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1013.402760</td>
      <td>577.587332</td>
      <td>2644.141273</td>
      <td>280.428203</td>
      <td>1161.873391</td>
      <td>2084.107872</td>
      <td>853.404981</td>
      <td>447.157619</td>
      <td>1193.032521</td>
      <td>861.081809</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1300.035501</td>
      <td>820.518697</td>
      <td>2025.854469</td>
      <td>525.562292</td>
      <td>922.206261</td>
      <td>2552.355407</td>
      <td>818.676686</td>
      <td>845.491492</td>
      <td>1968.367513</td>
      <td>1647.186291</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1059.347542</td>
      <td>1066.866418</td>
      <td>612.000041</td>
      <td>480.827789</td>
      <td>419.467495</td>
      <td>685.666983</td>
      <td>852.867810</td>
      <td>341.664784</td>
      <td>1154.391368</td>
      <td>1450.935357</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1018.340526</td>
      <td>1313.679056</td>
      <td>950.622661</td>
      <td>724.742174</td>
      <td>843.065903</td>
      <td>1370.554164</td>
      <td>905.469453</td>
      <td>658.118202</td>
      <td>539.459350</td>
      <td>1899.850792</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

Dato que estos son datos inventados, solo hagamos un pairplot con seaborn.

**Usa seaborn sobre el dataframe para crear un pairplot con el hue indicado por la columna TARGET CLASS.**


```python
#Escribe tu código aqui

```


```python
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')
```




    <seaborn.axisgrid.PairGrid at 0x1d97dc20c48>




![png](../../imagenes/02-K%20Nearest%20Neighbors%20Project_9_1.png)


# Estandarización de variables

Es tiempo de estandarizar las variables

**Importe StandardScaler de Scikit learn.**


```python

```

**Crea un StandardScaler() llamado scaler**


```python

```

**Elimina "TARGET CLASS" de las características.**


```python
#Escribe tu código aqui

```


```python

```




    StandardScaler(copy=True, with_mean=True, with_std=True)



**Usa el metodo .transform() para transformar las características a una versión escalada.**


```python

```

**Convierte las funciones escaladas en un dataframe y verifica el encabezado de este dataframe para asegurar que la escala funcionó.**


```python
#Escribe tu código aqui

```


```python
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>XVPM</th>
      <th>GWYH</th>
      <th>TRAT</th>
      <th>TLLZ</th>
      <th>IGGA</th>
      <th>HYKR</th>
      <th>EDFS</th>
      <th>GUUB</th>
      <th>MGJM</th>
      <th>JHZC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.568522</td>
      <td>-0.443435</td>
      <td>1.619808</td>
      <td>-0.958255</td>
      <td>-1.128481</td>
      <td>0.138336</td>
      <td>0.980493</td>
      <td>-0.932794</td>
      <td>1.008313</td>
      <td>-1.069627</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.112376</td>
      <td>-1.056574</td>
      <td>1.741918</td>
      <td>-1.504220</td>
      <td>0.640009</td>
      <td>1.081552</td>
      <td>-1.182663</td>
      <td>-0.461864</td>
      <td>0.258321</td>
      <td>-1.041546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.660647</td>
      <td>-0.436981</td>
      <td>0.775793</td>
      <td>0.213394</td>
      <td>-0.053171</td>
      <td>2.030872</td>
      <td>-1.240707</td>
      <td>1.149298</td>
      <td>2.184784</td>
      <td>0.342811</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.011533</td>
      <td>0.191324</td>
      <td>-1.433473</td>
      <td>-0.100053</td>
      <td>-1.507223</td>
      <td>-1.753632</td>
      <td>-1.183561</td>
      <td>-0.888557</td>
      <td>0.162310</td>
      <td>-0.002793</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.099059</td>
      <td>0.820815</td>
      <td>-0.904346</td>
      <td>1.609015</td>
      <td>-0.282065</td>
      <td>-0.365099</td>
      <td>-1.095644</td>
      <td>0.391419</td>
      <td>-1.365603</td>
      <td>0.787762</td>
    </tr>
  </tbody>
</table>
</div>



# Dividir el conjunto de datos en entrenamiento y pruebas

**Use train_test_split para dividir los datos en un conjunto de tentrenamiento y pruebas**


```python
#Importa la librería

```


```python
#Genera los conjuntos de datos, con el parámetro test_size=0.30

```

# Usando KNN

**Importa KNeighborsClassifier de scikit learn.**


```python

```

**Crea un modelo KNN con n_neighbors=1**


```python

```

**Entrena el modelo de KNN con los datos de entrenamiento.**


```python
#Escribe tu código aqui

```


```python
knn.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                         weights='uniform')



# Predicción y evaluación del modelo


**Usa el metodo predict method para realizar la predicción de los valores usando tu modelo KNN y X_test.**


```python

```

**Crea una matriz de confusión y un reporte de clasificación.**


```python
#Importa la librería

```


```python
#Genera la matriz de confusión

```


```python

```

    [[125  33]
     [ 42 100]]
    


```python
#Genera el reporte de clasificación

```


```python

```

                  precision    recall  f1-score   support
    
               0       0.75      0.79      0.77       158
               1       0.75      0.70      0.73       142
    
        accuracy                           0.75       300
       macro avg       0.75      0.75      0.75       300
    weighted avg       0.75      0.75      0.75       300
    
    

# Eligiendo un valor para K

¡Sigamos y usemos el método del codo para elegir un buen valor K!

**Crea un bucle "for" que entrene varios modelos KNN con diferentes valores de k, luego revisa la tasa de error para cada uno de estos modelos con una lista.**


```python

```

**Ahora crea la siguiente gráfica usando la información del ciclo.**


```python
#Escribe tu código aqui

```


```python

```




    Text(0, 0.5, 'Error Rate')




![png](../../imagenes/02-K%20Nearest%20Neighbors%20Project_45_1.png)


## Intenta con nuevo valor de K

** Vuelve a entrenar el modelo con el mejor valor K (depende de ti decidir qué quieres) y vuelve a hacer el informe de clasificación y la matriz de confusión.**


```python
#Escribe tu código aqui

```


```python

```

    WITH K=30
    
    
    [[138  20]
     [ 26 116]]
    
    
                  precision    recall  f1-score   support
    
               0       0.84      0.87      0.86       158
               1       0.85      0.82      0.83       142
    
        accuracy                           0.85       300
       macro avg       0.85      0.85      0.85       300
    weighted avg       0.85      0.85      0.85       300
    
    

# ¡Buen trabajo!
