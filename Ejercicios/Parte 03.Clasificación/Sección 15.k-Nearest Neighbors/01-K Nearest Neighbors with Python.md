# K vecinos cercanos (K Nearest Neighbors)

¡Le han dado un conjunto de datos clasificados de una empresa! Han ocultado los nombres de las columnas de las características, pero le han proporcionado los datos y las clases a predecir.

Intentaremos usar KNN para crear un modelo que prediga directamente una clase para un nuevo punto de datos basado en las características.

## Importemos la librerías




```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```

## Carguemos los datos

Establece index_col = 0 para usar la primera columna como índice.


```python
df = pd.read_csv("Classified Data.csv",index_col=0)
```


```python
#Revisemos los datos
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
      <th>TARGET CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.913917</td>
      <td>1.162073</td>
      <td>0.567946</td>
      <td>0.755464</td>
      <td>0.780862</td>
      <td>0.352608</td>
      <td>0.759697</td>
      <td>0.643798</td>
      <td>0.879422</td>
      <td>1.231409</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.635632</td>
      <td>1.003722</td>
      <td>0.535342</td>
      <td>0.825645</td>
      <td>0.924109</td>
      <td>0.648450</td>
      <td>0.675334</td>
      <td>1.013546</td>
      <td>0.621552</td>
      <td>1.492702</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.721360</td>
      <td>1.201493</td>
      <td>0.921990</td>
      <td>0.855595</td>
      <td>1.526629</td>
      <td>0.720781</td>
      <td>1.626351</td>
      <td>1.154483</td>
      <td>0.957877</td>
      <td>1.285597</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.234204</td>
      <td>1.386726</td>
      <td>0.653046</td>
      <td>0.825624</td>
      <td>1.142504</td>
      <td>0.875128</td>
      <td>1.409708</td>
      <td>1.380003</td>
      <td>1.522692</td>
      <td>1.153093</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.279491</td>
      <td>0.949750</td>
      <td>0.627280</td>
      <td>0.668976</td>
      <td>1.232537</td>
      <td>0.703727</td>
      <td>1.115596</td>
      <td>0.646691</td>
      <td>1.463812</td>
      <td>1.419167</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Estandarizar las variables

Debido a que el clasificador KNN predice la clase de una observación de prueba dada, al identificar las observaciones más cercanas a ella, la escala de las variables es importante. Cualquier variable que esté a gran escala tendrá un efecto mucho mayor en la distancia entre las observaciones y, por lo tanto, en el clasificador KNN, que las variables que están en pequeña escala.


```python
#Importemos la librería
from sklearn.preprocessing import StandardScaler
```


```python
#Generemos el modelo
scaler = StandardScaler()
```


```python
#Eliminemos la clase a predecir
scaler.fit(df.drop('TARGET CLASS',axis=1))
```




    StandardScaler(copy=True, with_mean=True, with_std=True)




```python
#Escalemos las características
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
```


```python
#Revisemos el resultado
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTT</th>
      <th>PTI</th>
      <th>EQW</th>
      <th>SBI</th>
      <th>LQE</th>
      <th>QWG</th>
      <th>FDJ</th>
      <th>PJF</th>
      <th>HQE</th>
      <th>NXJ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.123542</td>
      <td>0.185907</td>
      <td>-0.913431</td>
      <td>0.319629</td>
      <td>-1.033637</td>
      <td>-2.308375</td>
      <td>-0.798951</td>
      <td>-1.482368</td>
      <td>-0.949719</td>
      <td>-0.643314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.084836</td>
      <td>-0.430348</td>
      <td>-1.025313</td>
      <td>0.625388</td>
      <td>-0.444847</td>
      <td>-1.152706</td>
      <td>-1.129797</td>
      <td>-0.202240</td>
      <td>-1.828051</td>
      <td>0.636759</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.788702</td>
      <td>0.339318</td>
      <td>0.301511</td>
      <td>0.755873</td>
      <td>2.031693</td>
      <td>-0.870156</td>
      <td>2.599818</td>
      <td>0.285707</td>
      <td>-0.682494</td>
      <td>-0.377850</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.982841</td>
      <td>1.060193</td>
      <td>-0.621399</td>
      <td>0.625299</td>
      <td>0.452820</td>
      <td>-0.267220</td>
      <td>1.750208</td>
      <td>1.066491</td>
      <td>1.241325</td>
      <td>-1.026987</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.139275</td>
      <td>-0.640392</td>
      <td>-0.709819</td>
      <td>-0.057175</td>
      <td>0.822886</td>
      <td>-0.936773</td>
      <td>0.596782</td>
      <td>-1.472352</td>
      <td>1.040772</td>
      <td>0.276510</td>
    </tr>
  </tbody>
</table>
</div>



## Dividir los datos en entrenamiento y pruebas.


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30,random_state=2020)
```

## Usando KNN
Recuerda que estamos tratando de generar un modelo que prediga, si cualquira elemento esta en la clase objetivo o no. Comenzaremos con k = 1.


```python
#Importemos la librería
from sklearn.neighbors import KNeighborsClassifier
```


```python
#Generemos el modelo
knn = KNeighborsClassifier(n_neighbors=1)
```


```python
#Entrenemos el modelo
knn.fit(X_train,y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                         weights='uniform')




```python
#Realicemos la predicción
pred = knn.predict(X_test)
```

## Predicción y Evaluación del modelo

¡Evaluemos el modelo!


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
#Veamos la matriz de confusión
print(confusion_matrix(y_test,pred))
```

    [[138  19]
     [ 12 131]]
    


```python
#Veamos las métricas de evaluación
print(classification_report(y_test,pred))
```

                  precision    recall  f1-score   support
    
               0       0.92      0.88      0.90       157
               1       0.87      0.92      0.89       143
    
        accuracy                           0.90       300
       macro avg       0.90      0.90      0.90       300
    weighted avg       0.90      0.90      0.90       300
    
    

## Encontrando el K adecuado

Sigamos y usemos el método del codo para elegir un buen valor K:


```python
error_rate = []

# esto tomara un tiempo
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
```


```python
#Veamos los resultados
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
```




    Text(0, 0.5, 'Error Rate')




![png](../../imagenes/01-K%20Nearest%20Neighbors%20with%20Python_26_1.png)


Aquí podemos ver que después de alrededor de K> 23, la tasa de error tiende a oscilar entre 0.06-0.05. ¡Volvamos a entrenar el modelo con ese parámetro y verifiquemos el informe de clasificación!


```python
# Comparemos con K=1
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=1')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

    WITH K=1
    
    
    [[138  19]
     [ 12 131]]
    
    
                  precision    recall  f1-score   support
    
               0       0.92      0.88      0.90       157
               1       0.87      0.92      0.89       143
    
        accuracy                           0.90       300
       macro avg       0.90      0.90      0.90       300
    weighted avg       0.90      0.90      0.90       300
    
    


```python
# Ahora con K=23
knn = KNeighborsClassifier(n_neighbors=22)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=23')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```

    WITH K=23
    
    
    [[143  14]
     [  3 140]]
    
    
                  precision    recall  f1-score   support
    
               0       0.98      0.91      0.94       157
               1       0.91      0.98      0.94       143
    
        accuracy                           0.94       300
       macro avg       0.94      0.94      0.94       300
    weighted avg       0.95      0.94      0.94       300
    
    

# ¡Buen trabajo!
