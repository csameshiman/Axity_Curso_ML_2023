# Apriori

## Los datos

Utilizaremos el conjunto de datos [Market Basket Optimisation de Kaggle](https://www.kaggle.com/roshansharma/market-basket-optimization).

El archivo contiene información sobre las compras de diferentes artículos que los clientes realizaron en un centro comercial. Contiene 7501 transacciones, cada una con la lista de artículos vendidos en dicha transacción.


```python
# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Importar el data set
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
print(dataset.head())
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0,20)])
```

                  0          1           2                 3             4   \
    0         shrimp    almonds     avocado    vegetables mix  green grapes   
    1        burgers  meatballs        eggs               NaN           NaN   
    2        chutney        NaN         NaN               NaN           NaN   
    3         turkey    avocado         NaN               NaN           NaN   
    4  mineral water       milk  energy bar  whole wheat rice     green tea   
    
                     5     6               7             8             9   \
    0  whole weat flour  yams  cottage cheese  energy drink  tomato juice   
    1               NaN   NaN             NaN           NaN           NaN   
    2               NaN   NaN             NaN           NaN           NaN   
    3               NaN   NaN             NaN           NaN           NaN   
    4               NaN   NaN             NaN           NaN           NaN   
    
                   10         11     12     13             14      15  \
    0  low fat yogurt  green tea  honey  salad  mineral water  salmon   
    1             NaN        NaN    NaN    NaN            NaN     NaN   
    2             NaN        NaN    NaN    NaN            NaN     NaN   
    3             NaN        NaN    NaN    NaN            NaN     NaN   
    4             NaN        NaN    NaN    NaN            NaN     NaN   
    
                      16               17       18         19  
    0  antioxydant juice  frozen smoothie  spinach  olive oil  
    1                NaN              NaN      NaN        NaN  
    2                NaN              NaN      NaN        NaN  
    3                NaN              NaN      NaN        NaN  
    4                NaN              NaN      NaN        NaN  
    


```python
transactions[10]
```




    ['eggs',
     'pet food',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan',
     'nan']




```python
# Entrenar el algoritmo de Apriori
from apyori import apriori
rules = apriori(transactions, min_support = 0.003 , min_confidence = 0.3,
                min_lift = 3, min_length = 2)

# rules = apriori(transactions, min_support = 0.004 , min_length = 2)
# Visualización de los resultados
results = list(rules)
```


```python
def inspect(results):
    rh          = [tuple(result[2][0][0]) for result in results]
    lh          = [tuple(result[2][0][1]) for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(rh, lh, supports, confidences, lifts))
```


```python
# Este comando crea un frame para ver los resultados
resultDataFrame=pd.DataFrame(inspect(results),
                columns=['rhs','lhs','support','confidence','lift'])
```


```python
#Imprimimos el dataframe con las reglas
resultDataFrame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rhs</th>
      <th>lhs</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(mushroom cream sauce,)</td>
      <td>(escalope,)</td>
      <td>0.005733</td>
      <td>0.300699</td>
      <td>3.790833</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(pasta,)</td>
      <td>(escalope,)</td>
      <td>0.005866</td>
      <td>0.372881</td>
      <td>4.700812</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(herb &amp; pepper,)</td>
      <td>(ground beef,)</td>
      <td>0.015998</td>
      <td>0.323450</td>
      <td>3.291994</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(tomato sauce,)</td>
      <td>(ground beef,)</td>
      <td>0.005333</td>
      <td>0.377358</td>
      <td>3.840659</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(pasta,)</td>
      <td>(shrimp,)</td>
      <td>0.005066</td>
      <td>0.322034</td>
      <td>4.506672</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>(frozen vegetables, milk, mineral water)</td>
      <td>(olive oil, nan)</td>
      <td>0.003333</td>
      <td>0.301205</td>
      <td>4.582834</td>
    </tr>
    <tr>
      <th>98</th>
      <td>(soup, frozen vegetables)</td>
      <td>(milk, mineral water, nan)</td>
      <td>0.003066</td>
      <td>0.383333</td>
      <td>7.987176</td>
    </tr>
    <tr>
      <th>99</th>
      <td>(mineral water, spaghetti, shrimp)</td>
      <td>(frozen vegetables, nan)</td>
      <td>0.003333</td>
      <td>0.390625</td>
      <td>4.098011</td>
    </tr>
    <tr>
      <th>100</th>
      <td>(frozen vegetables, mineral water, tomatoes)</td>
      <td>(spaghetti, nan)</td>
      <td>0.003066</td>
      <td>0.522727</td>
      <td>3.002280</td>
    </tr>
    <tr>
      <th>101</th>
      <td>(ground beef, mineral water, tomatoes)</td>
      <td>(spaghetti, nan)</td>
      <td>0.003066</td>
      <td>0.560976</td>
      <td>3.221959</td>
    </tr>
  </tbody>
</table>
<p>102 rows × 5 columns</p>
</div>




```python
res.sort_values(by = ['support'],axis=0,ascending=False).head(30)
# , ascending=False
```
