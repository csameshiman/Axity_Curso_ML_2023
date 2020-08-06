# K Means Clustering

## Metodo usado

K Means Clustering es un algoritmo de aprendizaje no supervisado, que intenta agrupar datos en función de su similitud. El aprendizaje no supervisado significa que no hay resultados para predecir y el algoritmo solo trata de encontrar patrones en los datos. En k means tenemos que especificar el número de agrupaciones en las que queremos que se agrupen los datos. El algoritmo asigna aleatoriamente cada observación a un grupo y encuentra el centroide de cada grupo. Luego, el algoritmo itera a través de dos pasos:  
1. Asigna puntos de datos al grupo cuyo centroide es el más cercano. 
2. Calcula el nuevo centroide de cada grupo.  
Estos dos pasos se repiten, hasta que la variación dentro del clúster no se pueda reducir más. La variación dentro del grupo se calcula como la suma de la distancia euclidiana entre los puntos de datos y sus respectivos centroides del grupo.

## Importar librerias


```python
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## Creamos un conjunto de datos aleatorio


```python
from sklearn.datasets import make_blobs
```


```python
# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=42)
```

## Visualizar los datos generados


```python
plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```




    <matplotlib.collections.PathCollection at 0x1f63a6c8748>




![png](../../imagenes/01-K%20Means%20Clustering%20with%20Python_7_1.png)


## Creación de los clusters


```python
from sklearn.cluster import KMeans
```


```python
kmeans = KMeans(n_clusters=5)
```


```python
kmeans.fit(data[0])
```




    KMeans(n_clusters=5)




```python
kmeans.cluster_centers_
```




    array([[ 4.5394351 ,  2.27912418],
           [-7.43226031, -6.5249557 ],
           [-2.85095191,  8.66676102],
           [-8.80180093,  7.6007331 ],
           [-3.89239907, -8.21570277]])




```python
kmeans.labels_
```




    array([0, 3, 3, 3, 1, 1, 0, 3, 0, 1, 1, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 1,
           3, 2, 3, 2, 2, 1, 2, 0, 1, 4, 3, 3, 2, 0, 3, 0, 3, 2, 1, 2, 1, 4,
           3, 0, 0, 1, 0, 2, 2, 2, 3, 0, 2, 2, 4, 1, 2, 0, 3, 0, 1, 3, 3, 1,
           0, 2, 3, 2, 2, 3, 2, 1, 0, 1, 0, 2, 1, 2, 2, 0, 1, 3, 3, 3, 3, 2,
           0, 3, 1, 2, 0, 0, 0, 3, 2, 0, 1, 2, 3, 3, 2, 1, 2, 0, 3, 1, 4, 3,
           0, 1, 2, 3, 2, 3, 3, 2, 2, 2, 2, 4, 0, 3, 3, 0, 2, 0, 0, 2, 1, 1,
           2, 3, 3, 0, 1, 4, 2, 4, 0, 2, 3, 0, 0, 2, 0, 3, 1, 1, 2, 3, 0, 3,
           1, 3, 3, 0, 0, 0, 2, 0, 0, 3, 2, 1, 0, 0, 1, 0, 3, 2, 1, 1, 0, 4,
           0, 2, 2, 1, 2, 1, 3, 3, 3, 2, 0, 0, 0, 2, 2, 1, 3, 3, 2, 3, 0, 2,
           1, 0])




```python
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
```




    <matplotlib.collections.PathCollection at 0x1f63b10c470>




![png](../../imagenes/01-K%20Means%20Clustering%20with%20Python_14_1.png)


Debes tener en cuenta que los colores no tienen relación entre los dos gráficos.
