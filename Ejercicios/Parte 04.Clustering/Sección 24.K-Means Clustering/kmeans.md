# K-Means

## Los datos

Utilizaremos un conjunto de datos con las siguientes variables.

* CustomerID: número de cliente
* Genre: género
* Age: edad
* Annual Income (k$): ingreso anual en miles de USD
* Spending Score (1-100): calificación del cliente


```python
# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# Cargamos los datos con pandas
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values
print(dataset)
```

         CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
    0             1    Male   19                  15                      39
    1             2    Male   21                  15                      81
    2             3  Female   20                  16                       6
    3             4  Female   23                  16                      77
    4             5  Female   31                  17                      40
    ..          ...     ...  ...                 ...                     ...
    195         196  Female   35                 120                      79
    196         197  Female   45                 126                      28
    197         198    Male   32                 126                      74
    198         199    Male   32                 137                      18
    199         200    Male   30                 137                      83
    
    [200 rows x 5 columns]
    


```python
# Método del codo para averiguar el número óptimo de clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title("Método del codo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS(k)")
plt.show()
```


![png](../../imagenes/kmeans_3_0.png)



```python
# Aplicar el método de k-means para segmentar el data set
kmeans = KMeans(n_clusters = 5, init="k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
```


```python
# Visualización de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = "yellow", label = "Baricentros")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()
```


![png](../../imagenes/kmeans_5_0.png)

