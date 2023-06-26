# Support Vector Machines

## Importar librerías


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Carguemos los datos

Utilizaremos el conjunto de datos integrado de cáncer de seno de Scikit Learn.

Se utilizan 30 características, por ejemplo:

  - radius (media de distancias desde el centro a puntos en el perímetro)
  - texture (desviación estándar de valores de escala de grises)
  - perimeter
  - area
  - smoothness (variación local en longitudes de radio)
  - compactness (perímetro ^ 2 / área - 1.0)
  - concavity (severidad de las porciones cóncavas del contorno)
  - concave points (número de porciones cóncavas del contorno)
  - symmetry
  - fractal dimension ("aproximación de la costa" - 1)

Los conjuntos de datos son linealmente separables, usando las 30 características de entrada

Número de instancias: 569
Distribución de la clase: 212 maligno, 357 benigno
Clase objetivo:
   - maligno
   - benigno

Podemos obtenelo con la función de carga:


```python
from sklearn.datasets import load_breast_cancer
```


```python
cancer = load_breast_cancer()
```


El conjunto de datos se presenta en forma de diccionario:


```python
cancer.keys()
```




    dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])



Podemos obtener información y matrices de este diccionario para configurar nuestro dataframe y comprender las características:


```python
print(cancer['DESCR'])
```

    .. _breast_cancer_dataset:
    
    Breast cancer wisconsin (diagnostic) dataset
    --------------------------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 569
    
        :Number of Attributes: 30 numeric, predictive attributes and the class
    
        :Attribute Information:
            - radius (mean of distances from center to points on the perimeter)
            - texture (standard deviation of gray-scale values)
            - perimeter
            - area
            - smoothness (local variation in radius lengths)
            - compactness (perimeter^2 / area - 1.0)
            - concavity (severity of concave portions of the contour)
            - concave points (number of concave portions of the contour)
            - symmetry 
            - fractal dimension ("coastline approximation" - 1)
    
            The mean, standard error, and "worst" or largest (mean of the three
            largest values) of these features were computed for each image,
            resulting in 30 features.  For instance, field 3 is Mean Radius, field
            13 is Radius SE, field 23 is Worst Radius.
    
            - class:
                    - WDBC-Malignant
                    - WDBC-Benign
    
        :Summary Statistics:
    
        ===================================== ====== ======
                                               Min    Max
        ===================================== ====== ======
        radius (mean):                        6.981  28.11
        texture (mean):                       9.71   39.28
        perimeter (mean):                     43.79  188.5
        area (mean):                          143.5  2501.0
        smoothness (mean):                    0.053  0.163
        compactness (mean):                   0.019  0.345
        concavity (mean):                     0.0    0.427
        concave points (mean):                0.0    0.201
        symmetry (mean):                      0.106  0.304
        fractal dimension (mean):             0.05   0.097
        radius (standard error):              0.112  2.873
        texture (standard error):             0.36   4.885
        perimeter (standard error):           0.757  21.98
        area (standard error):                6.802  542.2
        smoothness (standard error):          0.002  0.031
        compactness (standard error):         0.002  0.135
        concavity (standard error):           0.0    0.396
        concave points (standard error):      0.0    0.053
        symmetry (standard error):            0.008  0.079
        fractal dimension (standard error):   0.001  0.03
        radius (worst):                       7.93   36.04
        texture (worst):                      12.02  49.54
        perimeter (worst):                    50.41  251.2
        area (worst):                         185.2  4254.0
        smoothness (worst):                   0.071  0.223
        compactness (worst):                  0.027  1.058
        concavity (worst):                    0.0    1.252
        concave points (worst):               0.0    0.291
        symmetry (worst):                     0.156  0.664
        fractal dimension (worst):            0.055  0.208
        ===================================== ====== ======
    
        :Missing Attribute Values: None
    
        :Class Distribution: 212 - Malignant, 357 - Benign
    
        :Creator:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
    
        :Donor: Nick Street
    
        :Date: November, 1995
    
    This is a copy of UCI ML Breast Cancer Wisconsin (Diagnostic) datasets.
    https://goo.gl/U2Uwz2
    
    Features are computed from a digitized image of a fine needle
    aspirate (FNA) of a breast mass.  They describe
    characteristics of the cell nuclei present in the image.
    
    Separating plane described above was obtained using
    Multisurface Method-Tree (MSM-T) [K. P. Bennett, "Decision Tree
    Construction Via Linear Programming." Proceedings of the 4th
    Midwest Artificial Intelligence and Cognitive Science Society,
    pp. 97-101, 1992], a classification method which uses linear
    programming to construct a decision tree.  Relevant features
    were selected using an exhaustive search in the space of 1-4
    features and 1-3 separating planes.
    
    The actual linear program used to obtain the separating plane
    in the 3-dimensional space is that described in:
    [K. P. Bennett and O. L. Mangasarian: "Robust Linear
    Programming Discrimination of Two Linearly Inseparable Sets",
    Optimization Methods and Software 1, 1992, 23-34].
    
    This database is also available through the UW CS ftp server:
    
    ftp ftp.cs.wisc.edu
    cd math-prog/cpo-dataset/machine-learn/WDBC/
    
    .. topic:: References
    
       - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
         for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
         Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
         San Jose, CA, 1993.
       - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
         prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
         July-August 1995.
       - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
         to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
         163-171.
    


```python
cancer['feature_names']
```




    array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error',
           'fractal dimension error', 'worst radius', 'worst texture',
           'worst perimeter', 'worst area', 'worst smoothness',
           'worst compactness', 'worst concavity', 'worst concave points',
           'worst symmetry', 'worst fractal dimension'], dtype='<U23')



## Configurar el dataframe


```python
df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df_feat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 569 entries, 0 to 568
    Data columns (total 30 columns):
     #   Column                   Non-Null Count  Dtype  
    ---  ------                   --------------  -----  
     0   mean radius              569 non-null    float64
     1   mean texture             569 non-null    float64
     2   mean perimeter           569 non-null    float64
     3   mean area                569 non-null    float64
     4   mean smoothness          569 non-null    float64
     5   mean compactness         569 non-null    float64
     6   mean concavity           569 non-null    float64
     7   mean concave points      569 non-null    float64
     8   mean symmetry            569 non-null    float64
     9   mean fractal dimension   569 non-null    float64
     10  radius error             569 non-null    float64
     11  texture error            569 non-null    float64
     12  perimeter error          569 non-null    float64
     13  area error               569 non-null    float64
     14  smoothness error         569 non-null    float64
     15  compactness error        569 non-null    float64
     16  concavity error          569 non-null    float64
     17  concave points error     569 non-null    float64
     18  symmetry error           569 non-null    float64
     19  fractal dimension error  569 non-null    float64
     20  worst radius             569 non-null    float64
     21  worst texture            569 non-null    float64
     22  worst perimeter          569 non-null    float64
     23  worst area               569 non-null    float64
     24  worst smoothness         569 non-null    float64
     25  worst compactness        569 non-null    float64
     26  worst concavity          569 non-null    float64
     27  worst concave points     569 non-null    float64
     28  worst symmetry           569 non-null    float64
     29  worst fractal dimension  569 non-null    float64
    dtypes: float64(30)
    memory usage: 133.5 KB
    


```python
cancer['target']
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1,
           1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0,
           0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
           1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
           1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
           1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1])




```python
df_target = pd.DataFrame(cancer['target'],columns=['Cancer'])
```

Ahora veamos el dataframe


```python
df_target
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cancer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>564</th>
      <td>0</td>
    </tr>
    <tr>
      <th>565</th>
      <td>0</td>
    </tr>
    <tr>
      <th>566</th>
      <td>0</td>
    </tr>
    <tr>
      <th>567</th>
      <td>0</td>
    </tr>
    <tr>
      <th>568</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>569 rows × 1 columns</p>
</div>



# EDA



Omitiremos la parte del EDA para este ejemplo, ya que hay muchas características que son difíciles de interpretar si no se tiene el conocimiento de dominio del cáncer o las células tumorales.

## Dividir los datos en entrenamiento y pruebas


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
```

# Entrenar el clasificador Support Vector Classifier


```python
from sklearn.svm import SVC
```


```python
model = SVC()
```


```python
model.fit(X_train,y_train)
```




    SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)



## Predicción y evaluación

Ahora pronostiquemos usando el modelo entrenado.


```python
predictions = model.predict(X_test)
```


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print(confusion_matrix(y_test,predictions))
```

    [[ 56  10]
     [  3 102]]
    


```python
print(classification_report(y_test,predictions))
```

                  precision    recall  f1-score   support
    
               0       0.95      0.85      0.90        66
               1       0.91      0.97      0.94       105
    
        accuracy                           0.92       171
       macro avg       0.93      0.91      0.92       171
    weighted avg       0.93      0.92      0.92       171
    
    

¡Woah! ¡Ten en cuenta que estamos clasificando todo en una sola clase! Esto significa que nuestro modelo necesita tener sus parámetros ajustados (también puede ayudar la normalización de los datos).

¡Podemos buscar parámetros usando un GridSearch!

# Gridsearch


¡Encontrar los parámetros correctos (como qué valores de C o gamma usar) es una tarea difícil! Pero afortunadamente, podemos ser un poco perezosos y probar varias combinaciones y ver qué funciona mejor. Esta idea de crear un 'grid' de parámetros y simplemente probar todas las combinaciones posibles, se llama Gridsearch. Este método es lo suficientemente común, como para que Scikit-learn tenga esta funcionalidad incorporada, con GridSearchCV. El CV significa validación cruzada.

GridSearchCV toma un diccionario, que describe los parámetros que deben probarse y un modelo para entrenar. La cuadrícula de parámetros se define como un diccionario, donde las claves son los parámetros y los valores son las configuraciones a probar.


```python
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
```


```python
from sklearn.model_selection import GridSearchCV
```

Una de las mejores cosas de GridSearchCV es que es un metaestimulador. Toma un estimador como SVC, y crea un nuevo estimador, que se comporta exactamente igual, en este caso, como un clasificador. Se debe agregar refit = True y elegir el detalle a producir (verbose) a cualquier número que se desee; cuanto mayor sea el número, más detallado (detallado solo significa la salida de texto que describe el proceso).


```python
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
```

Lo que ajusta es un poco más complicado de lo habitual. Primero, ejecuta el mismo ciclo con validación cruzada, para encontrar la mejor combinación de parámetros. Una vez que tiene la mejor combinación, se ejecuta nuevamente en todos los datos pasados para ajustar (sin validación cruzada), para construir un solo modelo nuevo, utilizando la mejor configuración de parámetros.


```python
grid.fit(X_train,y_train)
```

    Fitting 5 folds for each of 25 candidates, totalling 125 fits
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=1, kernel=rbf ......................................
    [CV] .......... C=0.1, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=0.1, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
    

    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=0.1, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=0.1, gamma=0.001, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.887, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.938, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.963, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.962, total=   0.0s
    [CV] C=0.1, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=0.1, gamma=0.0001, kernel=rbf, score=0.886, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=1, kernel=rbf ........................................
    [CV] ............ C=1, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=0.1, kernel=rbf ......................................
    [CV] .......... C=1, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=0.01, kernel=rbf .....................................
    [CV] ......... C=1, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.900, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.925, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.962, total=   0.0s
    [CV] C=1, gamma=0.001, kernel=rbf ....................................
    [CV] ........ C=1, gamma=0.001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.950, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.975, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.962, total=   0.0s
    [CV] C=1, gamma=0.0001, kernel=rbf ...................................
    [CV] ....... C=1, gamma=0.0001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=1, kernel=rbf .......................................
    [CV] ........... C=10, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=0.1, kernel=rbf .....................................
    [CV] ......... C=10, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.613, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=0.01, kernel=rbf ....................................
    [CV] ........ C=10, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.887, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.900, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=10, gamma=0.001, kernel=rbf ...................................
    [CV] ....... C=10, gamma=0.001, kernel=rbf, score=0.924, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.950, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.975, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.949, total=   0.0s
    [CV] C=10, gamma=0.0001, kernel=rbf ..................................
    [CV] ...... C=10, gamma=0.0001, kernel=rbf, score=0.949, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=1, kernel=rbf ......................................
    [CV] .......... C=100, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=0.1, kernel=rbf ....................................
    [CV] ........ C=100, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.613, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=0.01, kernel=rbf ...................................
    [CV] ....... C=100, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.887, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.900, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=100, gamma=0.001, kernel=rbf ..................................
    [CV] ...... C=100, gamma=0.001, kernel=rbf, score=0.924, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.925, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.975, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=100, gamma=0.0001, kernel=rbf .................................
    [CV] ..... C=100, gamma=0.0001, kernel=rbf, score=0.949, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=1, kernel=rbf .....................................
    [CV] ......... C=1000, gamma=1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.625, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=0.1, kernel=rbf ...................................
    [CV] ....... C=1000, gamma=0.1, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.637, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.613, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=0.01, kernel=rbf ..................................
    [CV] ...... C=1000, gamma=0.01, kernel=rbf, score=0.633, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.887, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.900, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.937, total=   0.0s
    [CV] C=1000, gamma=0.001, kernel=rbf .................................
    [CV] ..... C=1000, gamma=0.001, kernel=rbf, score=0.924, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.938, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.912, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.963, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.924, total=   0.0s
    [CV] C=1000, gamma=0.0001, kernel=rbf ................................
    [CV] .... C=1000, gamma=0.0001, kernel=rbf, score=0.962, total=   0.0s
    

    [Parallel(n_jobs=1)]: Done 125 out of 125 | elapsed:    1.4s finished
    




    GridSearchCV(cv=None, error_score=nan,
                 estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                               class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='scale', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='deprecated', n_jobs=None,
                 param_grid={'C': [0.1, 1, 10, 100, 1000],
                             'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                             'kernel': ['rbf']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=3)



Se pueden revisar los mejores parámetros encontrados por GridSearchCV en el atributo best_params_ y el mejor estimador en el atributo best_estimator_:


```python
grid.best_params_
```




    {'C': 1, 'gamma': 0.0001, 'kernel': 'rbf'}




```python
grid.best_estimator_
```




    SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)



Se pueden volver a ejecutar predicciones en este objeto, tal como lo haría con un modelo normal.


```python
grid_predictions = grid.predict(X_test)
```


```python
print(confusion_matrix(y_test,grid_predictions))
```

    [[ 59   7]
     [  4 101]]
    


```python
print(classification_report(y_test,grid_predictions))
```

                  precision    recall  f1-score   support
    
               0       0.94      0.89      0.91        66
               1       0.94      0.96      0.95       105
    
        accuracy                           0.94       171
       macro avg       0.94      0.93      0.93       171
    weighted avg       0.94      0.94      0.94       171
    
    

# ¡Buen trabajo!
