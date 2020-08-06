# Ejercicio Regresión Lineal


¡Felicidades! Acaba de obtener un contrato de trabajo con una empresa de comercio electrónico, con sede en la ciudad de Nueva York, que vende ropa en línea, pero también tiene sesiones de asesoramiento de estilo y ropa en la tienda. Los clientes entran a la tienda, tienen sesiones / reuniones con un estilista personal, luego pueden ir a casa y pedir la ropa que desean, ya sea en una aplicación móvil o en un sitio web.

La compañía está tratando de decidir si debe enfocar sus esfuerzos en su experiencia de aplicación móvil o en su sitio web. ¡Te han contratado para ayudarles a resolverlo! ¡Empecemos!

Simplemente sigue los pasos que se indican a continuación, para analizar los datos del cliente (los datos son falsos, no te preocupes, no se proporcionan números de tarjeta de crédito o correos electrónicos reales).

## Importar las librerias


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Cargar los datos

Trabajaremos con el archivo csv "Ecommerce Customers" de la compañía. Tiene información del cliente, como correo electrónico, dirección y color de su Avatar. También tiene columnas de valor numérico:

* Avg. Session Length: duración promedio de las sesiones de asesoramiento de estilo en la tienda.
* Time on App: tiempo promedio empleado en la aplicación en minutos
* Time on Website: tiempo promedio de permanencia en el sitio web en minutos
* Length of Membership: cuántos años ha sido miembro el cliente.

**Lee el archivo csv de "Ecommerce Customers" como un DataFrame llamado customers.**


```python
customers = pd.read_csv('Ecommerce Customers.csv')
```

**Verifica los datos de customers mediante las funciones head(), describe() e info()**


```python
customers.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Email</th>
      <th>Address</th>
      <th>Avatar</th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mstephenson@fernandez.com</td>
      <td>835 Frank Tunnel\nWrightmouth, MI 82180-9605</td>
      <td>Violet</td>
      <td>34.497268</td>
      <td>12.655651</td>
      <td>39.577668</td>
      <td>4.082621</td>
      <td>587.951054</td>
    </tr>
    <tr>
      <th>1</th>
      <td>hduke@hotmail.com</td>
      <td>4547 Archer Common\nDiazchester, CA 06566-8576</td>
      <td>DarkGreen</td>
      <td>31.926272</td>
      <td>11.109461</td>
      <td>37.268959</td>
      <td>2.664034</td>
      <td>392.204933</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pallen@yahoo.com</td>
      <td>24645 Valerie Unions Suite 582\nCobbborough, D...</td>
      <td>Bisque</td>
      <td>33.000915</td>
      <td>11.330278</td>
      <td>37.110597</td>
      <td>4.104543</td>
      <td>487.547505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>riverarebecca@gmail.com</td>
      <td>1414 David Throughway\nPort Jason, OH 22070-1220</td>
      <td>SaddleBrown</td>
      <td>34.305557</td>
      <td>13.717514</td>
      <td>36.721283</td>
      <td>3.120179</td>
      <td>581.852344</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mstephens@davidson-herman.com</td>
      <td>14023 Rodriguez Passage\nPort Jacobville, PR 3...</td>
      <td>MediumAquaMarine</td>
      <td>33.330673</td>
      <td>12.795189</td>
      <td>37.536653</td>
      <td>4.446308</td>
      <td>599.406092</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Avg. Session Length</th>
      <th>Time on App</th>
      <th>Time on Website</th>
      <th>Length of Membership</th>
      <th>Yearly Amount Spent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33.053194</td>
      <td>12.052488</td>
      <td>37.060445</td>
      <td>3.533462</td>
      <td>499.314038</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.992563</td>
      <td>0.994216</td>
      <td>1.010489</td>
      <td>0.999278</td>
      <td>79.314782</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.532429</td>
      <td>8.508152</td>
      <td>33.913847</td>
      <td>0.269901</td>
      <td>256.670582</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>32.341822</td>
      <td>11.388153</td>
      <td>36.349257</td>
      <td>2.930450</td>
      <td>445.038277</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>33.082008</td>
      <td>11.983231</td>
      <td>37.069367</td>
      <td>3.533975</td>
      <td>498.887875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>33.711985</td>
      <td>12.753850</td>
      <td>37.716432</td>
      <td>4.126502</td>
      <td>549.313828</td>
    </tr>
    <tr>
      <th>max</th>
      <td>36.139662</td>
      <td>15.126994</td>
      <td>40.005182</td>
      <td>6.922689</td>
      <td>765.518462</td>
    </tr>
  </tbody>
</table>
</div>




```python
customers.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 8 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Email                 500 non-null    object 
     1   Address               500 non-null    object 
     2   Avatar                500 non-null    object 
     3   Avg. Session Length   500 non-null    float64
     4   Time on App           500 non-null    float64
     5   Time on Website       500 non-null    float64
     6   Length of Membership  500 non-null    float64
     7   Yearly Amount Spent   500 non-null    float64
    dtypes: float64(5), object(3)
    memory usage: 31.4+ KB
    

## EDA

**Usa seaborn para crear un gráfico jointplot para comparar las columnas del tiempo en el sitio web (Time on Website) y la cantidad anual gastada (Yearly Amount Spent). ¿Tiene sentido la correolación?**

**¡Vamos a explorar los datos!**

Para el resto del ejercicio, solo utilizaremos los datos numéricos del archivo csv.


```python
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')
```


```python
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x1d3103ed3c8>




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_11_1.png)


**Lo mismo pero ahora con la columna de tiempo en la aplicación (Time on App)**


```python
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
```




    <seaborn.axisgrid.JointGrid at 0x1d3109b5fc8>




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_13_1.png)


**Exploremos las relaciones que existen en todo el conjunto de datos. Usa [pairplot] (https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) para recrear el diagrama siguiente**


```python
sns.pairplot(customers)
```




    <seaborn.axisgrid.PairGrid at 0x1d310b36788>




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_15_1.png)


**Basado en esta gráfica, ¿cuál parece ser la característica más correlacionada con la cantidad anual gastada (Yearly Amount Spent)?**


```python
# Length of Membership 
```

**Crea un gráfico lineal (usando lmplot de seaborn) de la cantidad anual gastada (Yearly Amount Spent) vs. el tiempo de la membresía (Length of Membership).**


```python
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
```




    <seaborn.axisgrid.FacetGrid at 0x1d312c0e408>




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_19_1.png)


## Dividir el conjunto de datos en conjunto de entrenamiento y conjunto de pruebas

**Establece una variable "X" igual a las características numéricas de los clientes y una variable "y" igual a la columna cantidad anual gastada (Yearly Amount Spent).**


```python
y = customers['Yearly Amount Spent']
```


```python
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
```

**Usa la libraría model_selection.train_test_split de sklearn para dividir los datos en conjuntos de entrenamiento y prueba. Establece test_size = 0.3 y random_state = 101 **


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

## Entrenando el modelo

Ahora entrenemos el modelo con nuestros datos de entrenamiento!

**Importa la libreria LinearRegression de sklearn.linear_model**


```python
from sklearn.linear_model import LinearRegression
```

**Crea una instancia de LinearRegression() llamada "lm".**


```python
lm = LinearRegression()
```


```python
lm.fit(X_train,y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



**Imprime los coeficientes resultantes del modelo**


```python
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Coefficients: \n', lm.coef_)
```

    Coefficients: 
     [25.98154972 38.59015875  0.19040528 61.27909654]
    

## Predicción

**Usa lm.predict() para predecir los valores con los datos de prueba (X_test) del conjunto de datos.**


```python
predictions = lm.predict( X_test)
```

**Crea un gráfico de los valores reales contra los de la predicción**


```python
ax1 = sns.distplot(y_test, hist=False, color="r", label="Valor real")
sns.distplot(predictions, hist=False, color="b", label="Predicción" , ax=ax1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d3118eed08>




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_37_1.png)



```python
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
```




    Text(0, 0.5, 'Predicted Y')




![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_38_1.png)


## Evaluando el modelo


Evaluemos el rendimiento de nuestro modelo calculando la suma residual de cuadrados y el coeficiente de determinación (R ^ 2).

**Calcule el error absoluto medio, el error cuadrático medio y la raiz del error cuadrático medio. Consulte la conferencia o Wikipedia para las fórmulas**


```python
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

    MAE: 7.228148653430838
    MSE: 79.81305165097461
    RMSE: 8.933815066978642
    

## Resultados

Deberías haber conseguido un muy buen modelo con un buen ajuste. Exploremos rápidamente los resultados para asegurarnos de que todo esté bien con nuestros datos.

**Traza un histograma de los residuos y asegúrate de que tenga una distribución normal. Use seaborn distplot o simplemente plt.hist ().**


```python
sns.distplot((y_test-predictions),bins=50);
```


![png](../../imagenes/02-Linear%20Regression%20Project%20-%20Soluci%C3%B3n_42_0.png)


## Conclusiones
Todavía necesitamos encontrar la respuesta a la pregunta original, ¿enfocamos nuestro esfuerzo en el desarrollo de aplicaciones móviles o sitios web? O tal vez eso ni siquiera importa, y el tiempo de membresía es lo realmente importante. Veamos si podemos interpretar los coeficientes para tener una idea.

**Recrea el dataframe de abajo.**


```python
coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
coefficients
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Avg. Session Length</th>
      <td>25.981550</td>
    </tr>
    <tr>
      <th>Time on App</th>
      <td>38.590159</td>
    </tr>
    <tr>
      <th>Time on Website</th>
      <td>0.190405</td>
    </tr>
    <tr>
      <th>Length of Membership</th>
      <td>61.279097</td>
    </tr>
  </tbody>
</table>
</div>



** ¿Como podemos interpretar los resultados? **

## Interpretando los coeficientes:

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Avg. Session Length** está asociado con un aumento de **$25.98 dolares gastados**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Time on App** está asociado con un aumento de **$38.59 dolares gastados**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Time on Website** está asociado con un aumento de **$0.19 dolares gastados**.  

- Manteniendo todas las demás características fijas, un aumento de 1 unidad en **Length of Membership** está asociado con un aumento de **$61.27 dolares gastados**.   

¿Esto tiene sentido?

**¿Dónde crees que la empresa debe enfocarse: en la aplicación móbil o en su website?**

Esto es complicado; hay dos formas de pensar en esto: Desarrollar el sitio web para igualar el rendimiento con el de la aplicación móvil, o desarrollar más la aplicación, ya que eso es lo que funciona mejor. Este tipo de respuesta realmente depende de los otros factores que suceden en la empresa, ¡probablemente desees explorar la relación entre la duración de la membresía y la aplicación o el sitio web, antes de llegar a una conclusión!

## ¡Buen trabajo!
