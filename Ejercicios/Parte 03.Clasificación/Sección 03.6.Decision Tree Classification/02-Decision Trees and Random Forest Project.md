___
# Bosques aleatorios


Para este proyecto, exploraremos los datos disponibles públicamente de [LendingClub.com] (www.lendingclub.com). Lending Club conecta a las personas que necesitan dinero (prestatarios) con las personas que tienen dinero (inversores). Afortunadamente, como inversor, querrás invertir en personas que mostraron un perfil de tener una alta probabilidad de devolver el dinero. Intentaremos crear un modelo que ayude a predecir esto.

El club de préstamos tuvo un [año muy interesante en 2016] (https://en.wikipedia.org/wiki/Lending_Club#2016), así que revisemos algunos de sus datos y tengamos en cuenta el contexto. Estos datos son anteriores incluso a que se hicieran públicos.

Utilizaremos datos de préstamos de 2007-2010 e intentaremos clasificar y predecir si el prestatario pagó o no el préstamo en su totalidad. Puedes descargar los datos desde [aquí] (https://www.lendingclub.com/info/download-data.action) o simplemente usar el csv ya proporcionado. Se recomienda que uses el csv proporcionado, ya que se ha limpiado de los valores de NA.

Esto es lo que representan las columnas:
* credit.policy: 1 si el cliente cumple con los criterios de suscripción de crédito de LendingClub.com, y 0 en caso contrario.
* purpose: El propósito del préstamo (toma los valores "tarjeta de crédito", "consolidación de deuda", "educativo", "compra_principal", "negocio pequeño" y "todo_otro").
* int rate .: la tasa de interés del préstamo, como una proporción (una tasa del 11% se almacenaría como 0,11). A los prestatarios que LendingClub.com consideran más riesgosos se les asignan tasas de interés más altas.
* installment: las cuotas mensuales adeudadas por el prestatario si se financia el préstamo.
* log.annual.inc: el registro natural del ingreso anual autoinformado del prestatario.
* dti: la relación deuda / ingreso del prestatario (monto de la deuda dividido por el ingreso anual).
* fico: El puntaje de crédito FICO del prestatario.
* days.with.cr.line: la cantidad de días que el prestatario ha tenido una línea de crédito.
* revol.bal: saldo rotatorio del prestatario (monto no pagado al final del ciclo de facturación de la tarjeta de crédito).
* revol.util: la tasa de utilización de la línea rotatoria del prestatario (el monto de la línea de crédito utilizada en relación con el crédito total disponible).
* inq.last.6mths: el número de consultas del prestatario por parte de los acreedores en los últimos 6 meses.
* delinq.2yrs: El número de veces que el prestatario ha estado atrasado más de 30 días en un pago en los últimos 2 años.
* pub.rec: el número de registros públicos despectivos del prestatario (declaraciones de quiebra, gravámenes fiscales o sentencias).

## Importar librerías


```python

```

## Cargar los datos

**Usa pandas para leer el archivo "loan_data.csv" y carga la información en un dataframe llamado loans.**


```python
loans = pd.read_csv('loan_data.csv')
```

**Usa los métodos info (), head () y describe () en loans.**


```python
#Escribe tu código aqui

```


```python

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9578 entries, 0 to 9577
    Data columns (total 14 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   credit.policy      9578 non-null   int64  
     1   purpose            9578 non-null   object 
     2   int.rate           9578 non-null   float64
     3   installment        9578 non-null   float64
     4   log.annual.inc     9578 non-null   float64
     5   dti                9578 non-null   float64
     6   fico               9578 non-null   int64  
     7   days.with.cr.line  9578 non-null   float64
     8   revol.bal          9578 non-null   int64  
     9   revol.util         9578 non-null   float64
     10  inq.last.6mths     9578 non-null   int64  
     11  delinq.2yrs        9578 non-null   int64  
     12  pub.rec            9578 non-null   int64  
     13  not.fully.paid     9578 non-null   int64  
    dtypes: float64(6), int64(7), object(1)
    memory usage: 1.0+ MB
    


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
      <th>credit.policy</th>
      <th>int.rate</th>
      <th>installment</th>
      <th>log.annual.inc</th>
      <th>dti</th>
      <th>fico</th>
      <th>days.with.cr.line</th>
      <th>revol.bal</th>
      <th>revol.util</th>
      <th>inq.last.6mths</th>
      <th>delinq.2yrs</th>
      <th>pub.rec</th>
      <th>not.fully.paid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9.578000e+03</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
      <td>9578.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.804970</td>
      <td>0.122640</td>
      <td>319.089413</td>
      <td>10.932117</td>
      <td>12.606679</td>
      <td>710.846314</td>
      <td>4560.767197</td>
      <td>1.691396e+04</td>
      <td>46.799236</td>
      <td>1.577469</td>
      <td>0.163708</td>
      <td>0.062122</td>
      <td>0.160054</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.396245</td>
      <td>0.026847</td>
      <td>207.071301</td>
      <td>0.614813</td>
      <td>6.883970</td>
      <td>37.970537</td>
      <td>2496.930377</td>
      <td>3.375619e+04</td>
      <td>29.014417</td>
      <td>2.200245</td>
      <td>0.546215</td>
      <td>0.262126</td>
      <td>0.366676</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.060000</td>
      <td>15.670000</td>
      <td>7.547502</td>
      <td>0.000000</td>
      <td>612.000000</td>
      <td>178.958333</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>0.103900</td>
      <td>163.770000</td>
      <td>10.558414</td>
      <td>7.212500</td>
      <td>682.000000</td>
      <td>2820.000000</td>
      <td>3.187000e+03</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>0.122100</td>
      <td>268.950000</td>
      <td>10.928884</td>
      <td>12.665000</td>
      <td>707.000000</td>
      <td>4139.958333</td>
      <td>8.596000e+03</td>
      <td>46.300000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.140700</td>
      <td>432.762500</td>
      <td>11.291293</td>
      <td>17.950000</td>
      <td>737.000000</td>
      <td>5730.000000</td>
      <td>1.824950e+04</td>
      <td>70.900000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>0.216400</td>
      <td>940.140000</td>
      <td>14.528354</td>
      <td>29.960000</td>
      <td>827.000000</td>
      <td>17639.958330</td>
      <td>1.207359e+06</td>
      <td>119.000000</td>
      <td>33.000000</td>
      <td>13.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Escribe tu código aqui

```


```python
loans.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>credit.policy</th>
      <th>purpose</th>
      <th>int.rate</th>
      <th>installment</th>
      <th>log.annual.inc</th>
      <th>dti</th>
      <th>fico</th>
      <th>days.with.cr.line</th>
      <th>revol.bal</th>
      <th>revol.util</th>
      <th>inq.last.6mths</th>
      <th>delinq.2yrs</th>
      <th>pub.rec</th>
      <th>not.fully.paid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1189</td>
      <td>829.10</td>
      <td>11.350407</td>
      <td>19.48</td>
      <td>737</td>
      <td>5639.958333</td>
      <td>28854</td>
      <td>52.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>credit_card</td>
      <td>0.1071</td>
      <td>228.22</td>
      <td>11.082143</td>
      <td>14.29</td>
      <td>707</td>
      <td>2760.000000</td>
      <td>33623</td>
      <td>76.7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1357</td>
      <td>366.86</td>
      <td>10.373491</td>
      <td>11.63</td>
      <td>682</td>
      <td>4710.000000</td>
      <td>3511</td>
      <td>25.6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>debt_consolidation</td>
      <td>0.1008</td>
      <td>162.34</td>
      <td>11.350407</td>
      <td>8.10</td>
      <td>712</td>
      <td>2699.958333</td>
      <td>33667</td>
      <td>73.2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>credit_card</td>
      <td>0.1426</td>
      <td>102.92</td>
      <td>11.299732</td>
      <td>14.97</td>
      <td>667</td>
      <td>4066.000000</td>
      <td>4740</td>
      <td>39.5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# EDA

¡Hagamos un poco de visualización de datos! Utilizaremos capacidades de trazado incorporadas de seaborn y pandas, pero siéntete libre de usar la biblioteca que desees. No te preocupes por la coincidencia de colores de las gráficas, solo preocúpate por tener la idea principal de lo que se busca con la visualización.  

**Crea un histograma de dos distribuciones de la columna FICO, una para cada crédito (credit.policy). Las distribuciones deberán estar una encima de la otra**  

* Nota: Esto es bastante complicado, no dudes en consultar las soluciones. Probablemente necesitarás una línea de código para cada histograma *


```python
#Escribe tu código aqui

```


```python

```




    Text(0.5, 0, 'FICO')




![png](../../imagenes/02-Decision%20Trees%20and%20Random%20Forest%20Project_14_1.png)


**Crea una figura similar, excepto que esta vez selecciona la columna not.fully.paid.**


```python
#Escribe tu código aqui

```


```python

```




    Text(0.5, 0, 'FICO')




![png](../../imagenes/02-Decision%20Trees%20and%20Random%20Forest%20Project_17_1.png)


**Crea un diagrama utilizando seaborn que muestre el número de préstamos por propósito (purpose), con el tono de color definido por not.fully.paid.**


```python
#Escribe tu código aqui

```


```python

```




    <matplotlib.axes._subplots.AxesSubplot at 0x2e2b56a1688>




![png](../../imagenes/02-Decision%20Trees%20and%20Random%20Forest%20Project_20_1.png)


**Veamos la tendencia entre el puntaje FICO y la tasa de interés. Recree la siguiente gráfica.**


```python
#Escribe tu código aqui

```


```python

```




    <seaborn.axisgrid.JointGrid at 0x2e2b56b9cc8>




![png](../../imagenes/02-Decision%20Trees%20and%20Random%20Forest%20Project_23_1.png)


**Crea los siguientes lmplots, para ver si la tendencia difiere entre not.fully.paid y credit.policy. Consulta la documentación de lmplot () si no puedes encontrar la manera de separarlo en columnas.**


```python
#Escribe tu cpodigo aqui

```


```python

```




    <seaborn.axisgrid.FacetGrid at 0x2e2b5b124c8>




    <Figure size 792x504 with 0 Axes>



![png](../../imagenes/02-Decision%20Trees%20and%20Random%20Forest%20Project_26_2.png)


# Configurando los datos

¡Preparémonos para configurar nuestros datos para nuestro modelo de clasificación de bosque aleatorio!

**Verifiquemos los datos nuevamente, con la función info()**


```python
#Escribe tu código aqui

```


```python

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9578 entries, 0 to 9577
    Data columns (total 14 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   credit.policy      9578 non-null   int64  
     1   purpose            9578 non-null   object 
     2   int.rate           9578 non-null   float64
     3   installment        9578 non-null   float64
     4   log.annual.inc     9578 non-null   float64
     5   dti                9578 non-null   float64
     6   fico               9578 non-null   int64  
     7   days.with.cr.line  9578 non-null   float64
     8   revol.bal          9578 non-null   int64  
     9   revol.util         9578 non-null   float64
     10  inq.last.6mths     9578 non-null   int64  
     11  delinq.2yrs        9578 non-null   int64  
     12  pub.rec            9578 non-null   int64  
     13  not.fully.paid     9578 non-null   int64  
    dtypes: float64(6), int64(7), object(1)
    memory usage: 1.0+ MB
    

## Características Categóricas

Como podemos observar, **purpose** es categórica.  

Esto significa que necesitamos transformarla, usando variables dummy, para que sklearn sea capaz de entenderla. Hagamos esto en un paso, usando 'pd.get_dummies'.  

Veamos una manera de manejar estas columnas, que puede extenderse a varias categorías categóricas, en caso de ser necesario.  

**Crea una lista de 1 elemento, que contenga la cadena 'purpose'. Llama esta lista con el nombre 'cat_feats'.**


```python
cat_feats = ['purpose']
```

**Usa pd.get_dummies(loans,columns=cat_feats,drop_first=True) para crear un dataframe fijo mayor, que contenga las nuevas columnas de las características con la variable dummy. Llama a este dataframe con el nombre 'final_data?.**


```python

```


```python

```


```python

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9578 entries, 0 to 9577
    Data columns (total 19 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   credit.policy               9578 non-null   int64  
     1   int.rate                    9578 non-null   float64
     2   installment                 9578 non-null   float64
     3   log.annual.inc              9578 non-null   float64
     4   dti                         9578 non-null   float64
     5   fico                        9578 non-null   int64  
     6   days.with.cr.line           9578 non-null   float64
     7   revol.bal                   9578 non-null   int64  
     8   revol.util                  9578 non-null   float64
     9   inq.last.6mths              9578 non-null   int64  
     10  delinq.2yrs                 9578 non-null   int64  
     11  pub.rec                     9578 non-null   int64  
     12  not.fully.paid              9578 non-null   int64  
     13  purpose_credit_card         9578 non-null   uint8  
     14  purpose_debt_consolidation  9578 non-null   uint8  
     15  purpose_educational         9578 non-null   uint8  
     16  purpose_home_improvement    9578 non-null   uint8  
     17  purpose_major_purchase      9578 non-null   uint8  
     18  purpose_small_business      9578 non-null   uint8  
    dtypes: float64(6), int64(7), uint8(6)
    memory usage: 1.0 MB
    

## Dividir los datos

¡Ahora es el momento de dividir nuestros datos en un conjunto de entrenamiento y un conjunto de prueba!

**Usa sklearn para dividir los datos en un conjunto de entrenamiento y un conjunto de pruebas, como lo hemos hecho en el pasado.**


```python

```


```python

```

## Entrenemos el modelo


¡Comencemos entrenando primero un árbol de decisión único!

**Importar DecisionTreeClassifier**


```python

```

**Crea una instancia de DecisionTreeClassifier () llamada dtree y entrénala con los datos de entrenamiento.**


```python

```


```python

```


```python

```




    DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                           max_depth=None, max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort='deprecated',
                           random_state=None, splitter='best')



## Predicción y evaluación

**Crea predicciones a partir del conjunto de pruebas y crea un informe de clasificación y una matriz de confusión**


```python

```


```python

```


```python

```


```python

```

                  precision    recall  f1-score   support
    
               0       0.85      0.82      0.84      2431
               1       0.17      0.21      0.19       443
    
        accuracy                           0.73      2874
       macro avg       0.51      0.51      0.51      2874
    weighted avg       0.75      0.73      0.74      2874
    
    


```python

```


```python

```

    [[1994  437]
     [ 351   92]]
    

## Entrenar el modelo de bosques aleatorios


¡Ahora es el momento de entrenar a nuestro modelo!

**Crea una instancia de la clase RandomForestClassifier y entrénala con los datos de entrenamiento del paso anterior.**


```python

```


```python
rfc = RandomForestClassifier(n_estimators=600)
```


```python

```


```python

```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=600,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)



## Predicción y evaluación

Vamos a predecir los valores de y_test y evaluar nuestro modelo.

**Realiza una predicción para los datos de X_test.**


```python

```

**Ahora crea un informe de clasificación a partir de los resultados. ¿Hay algo extraño o algún tipo de advertencia?**


```python

```


```python

```


```python

```

                  precision    recall  f1-score   support
    
               0       0.85      1.00      0.92      2431
               1       0.43      0.01      0.03       443
    
        accuracy                           0.85      2874
       macro avg       0.64      0.51      0.47      2874
    weighted avg       0.78      0.85      0.78      2874
    
    

**Muestra la matriz de confusión para las predicciones.**


```python

```


```python

```

    [[2423    8]
     [ 437    6]]
    

**¿Qué funcionó mejor, el bosque aleatorio o el árbol de decisión?**


```python
  
```

# ¡Buen trabajo!
