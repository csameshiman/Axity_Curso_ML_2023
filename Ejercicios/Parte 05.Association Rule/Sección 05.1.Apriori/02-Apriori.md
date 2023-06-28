# Algoritmo Apriori - Mining Movie Choices

Las reglas de asociación se utilizan para identificar las relaciones subyacentes entre diferentes elementos. Tomemos un ejemplo de una Plataforma de películas donde los clientes pueden alquilar o comprar películas. Por lo general, hay un patrón en lo que compran los clientes. Hay patrones claros, por ejemplo, el tema Superhéroe o la categoría Niños.

Se pueden generar más ganancias si se puede identificar la relación entre las películas.

Si las películas A y B se compran con frecuencia juntas, este patrón puede explotarse para aumentar las ganancias

Las personas que compran o alquilan una de estas dos películas pueden ser empujadas a alquilar o comprar la otra, a través de campañas o sugerencias dentro de la plataforma.

Hoy estamos muy familiarizados con estos motores de recomendación en Netflix, Amazon, por nombrar los más destacados.

El Algotitmo de Apriori cae en la categoría de Regla de Asociación.

### Implementando el algoritmo Apriori

En esta sección, utilizaremos el algoritmo Apriori para encontrar reglas que describan las asociaciones entre diferentes productos a las que se les dan 7500 transacciones en el transcurso de un mes. El conjunto de datos de las películas se selecciona al azar, estos no son datos reales.

Para instalar La biblioteca apyori, usa el siguiente comando en tu entorno: pip install apyori

#### Importar las librerías



```python

```

#### Importa el conjunto de datos movie_dataset.csv y muestra el número de renglones


```python
#Escribe tu código aqui

```


```python

```

    7501
    


Ahora usaremos el algoritmo Apriori para descubrir qué artículos se venden comúnmente juntos, de modo que el propietario de la tienda pueda tomar medidas para colocar los artículos relacionados juntos o publicitarlos juntos para obtener mayores ganancias.

#### EDA

La biblioteca Apriori que vamos a utilizar requiere que nuestro conjunto de datos tenga la forma de una lista de listas, donde todo el conjunto de datos es una lista grande y cada transacción en el conjunto de datos es una lista interna dentro de la lista grande externa. Actualmente tenemos datos en forma de un dataframe de pandas. Para convertir nuestro dataframe de pandas en una lista de listas, ejecuta el siguiente script:


```python
records = []  
for i in range(0, num_records):  
    records.append([str(movie_data.values[i,j]) for j in range(0, 20)])
```

#### Generando el modelo Apriori


Ahora podemos especificar los parámetros de la clase a priori.

- La lista
- min_support
- min_confidence
- min_lift
- min_length (la cantidad mínima de elementos que desea en sus reglas, generalmente 2)

Supongamos que solo queremos películas que se compren al menos 40 veces en un mes. El soporte para esos artículos se puede calcular como 40/7500 = 0.0053. La confianza mínima para las reglas es del 20% o 0.2. De manera similar, especificamos el valor para lift como 3 y finalmente min_length es 2 ya que queremos al menos dos productos en nuestras reglas. Estos valores se seleccionan principalmente de forma arbitraria y deben ajustarse empíricamente.

**Genera el algoritmo apriori con los siguientes parametros min_support 0.0053, min_confidence 0.20, min_lift 3, min_length 2**


```python

```

**Valida la longitud de la lista resultante**


```python
#Escribe tu código aqui

```


```python

```

    32
    

**Visualiza los resultados del modelo**


```python
#Escribe tu código aqui

```


```python

```

    RelationRecord(items=frozenset({'Green Lantern', 'Red Sparrow'}), support=0.005732568990801226, ordered_statistics=[OrderedStatistic(items_base=frozenset({'Red Sparrow'}), items_add=frozenset({'Green Lantern'}), confidence=0.3006993006993007, lift=3.790832696715049)])
    

**El siguiente script muestra la regla en un dataframe de una manera mucho más legible:**


```python
results = []
for item in association_results:
    
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    
    value0 = str(items[0])
    value1 = str(items[1])

    #second index of the inner list
    value2 = str(item[1])[:7]

    #third index of the list located at 0th
    #of the third index of the inner list

    value3 = str(item[2][0][2])[:7]
    value4 = str(item[2][0][3])[:7]
    
    rows = (value0, value1,value2,value3,value4)
    results.append(rows)
    
labels = ['Title 1','Title 2','Support','Confidence','Lift']
movie_suggestion = pd.DataFrame.from_records(results, columns = labels)

print(movie_suggestion)
```

              Title 1                Title 2  Support Confidence     Lift
    0   Green Lantern            Red Sparrow  0.00573    0.30069  3.79083
    1   Green Lantern              Star Wars  0.00586    0.37288  4.70081
    2   Kung Fu Panda                Jumanji  0.01599    0.32345  3.29199
    3    Wonder Woman                Jumanji  0.00533    0.37735  3.84065
    4     Spiderman 3  The Spy Who Dumped Me  0.00799    0.27149  4.12241
    5            Coco                 Intern  0.00533    0.23255  3.25451
    6   Green Lantern            Red Sparrow  0.00573    0.30069  3.79083
    7   Green Lantern                    nan  0.00586    0.37288  4.70081
    8          Intern             Tomb Rider  0.00866    0.31100  3.16532
    9          Intern           The Revenant  0.00719    0.30508  3.20061
    10         Intern            Spiderman 3  0.00573    0.20574  3.12402
    11         Intern           The Revenant  0.00599    0.21531  3.01314
    12         Intern            World War Z  0.00666    0.23923  3.49804
    13       Iron Man             Tomb Rider  0.00533    0.32258  3.28314
    14  Kung Fu Panda          Ninja Turtles  0.00666    0.39062  3.97568
    15  Kung Fu Panda             Tomb Rider  0.00639    0.39344  4.00435
    16  Kung Fu Panda                    nan  0.01599    0.32345  3.29199
    17     Tomb Rider           The Revenant  0.00599    0.52325  3.00531
    18            nan           Wonder Woman  0.00533    0.37735  3.84065
    19     Tomb Rider                  Moana  0.00719    0.20300  3.08250
    20    Spiderman 3                    nan  0.00799    0.27149  4.13077
    21           Coco                 Intern  0.00533    0.23255  3.26059
    22         Intern             Tomb Rider  0.00866    0.31100  3.16532
    23         Intern                    nan  0.00719    0.30508  3.20061
    24         Intern            Spiderman 3  0.00573    0.20574  3.13036
    25         Intern                    nan  0.00599    0.21531  3.01878
    26         Intern                    nan  0.00666    0.23923  3.49804
    27       Iron Man             Tomb Rider  0.00533    0.32258  3.28314
    28  Kung Fu Panda                    nan  0.00666    0.39062  3.97568
    29  Kung Fu Panda             Tomb Rider  0.00639    0.39344  4.00435
    30     Tomb Rider                    nan  0.00599    0.52325  3.00531
    31     Tomb Rider                  Moana  0.00719    0.20300  3.08876
    

## Conclusión

Los algoritmos de minería de reglas de asociación como Apriori son muy útiles para encontrar asociaciones simples entre nuestros elementos de datos. Son fáciles de implementar y fáciles de explicar. Google, Amazon, Netflix, Spotify utilizan algoritmos más complejos para su motor de recomendaciones.
