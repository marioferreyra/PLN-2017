Trabajo Práctico 3 - Análisis Sintáctico
========================================

Mario E. Ferreyra
=================


Ejercicio 1: Evaluación de Parsers
----------------------------------

Se implemento el script *eval.py*, el cual permite evaluar parsers.  
Se entreno y evaluó (con los scripts *train.py* y *eval.py* respectivamente) los modelos "baseline" para todas las oraciones de largo menor o igual a 20.  
Los resultados fueron los siguientes:

* Cantidad de oraciones de largo menor o igual a 20 = 1444

#### Parser Flat

|           | Labeled | Unlabeled |
|:---------:|:-------:|:---------:|
| Precision | 99.93 % | 100.00 %  |
|  Recall   | 14.58 % |  14.59 %  |
|    F1     | 25.44 % |  25.46 %  |


#### Parser Lbranch

|           | Labeled | Unlabeled |
|:---------:|:-------:|:---------:|
| Precision | 8.81 %  |  14.71 %  |
|  Recall   | 14.58 % |  24.35 %  |
|    F1     | 10.98 % |  18.34 %  |

#### Parser Rbranch

|           | Labeled | Unlabeled |
|:---------:|:-------:|:---------:|
| Precision | 8.81 %  |  8.88 %   |
|  Recall   | 14.58 % |  14.69 %  |
|    F1     | 10.98 % |  11.07 %  |


Ejercicio 2: Algoritmo CKY
--------------------------
Se implemento el algoritmo CKY en un módulo *cky_parser.py*, podemos encontrar un pseudocódigo de este algoritmo en las [Notas de Michael Collins], específicamente en la pagina 14.

Se agrego a los tests del modulo *cky_parser.py* un test con una gramática y una oración tal que la oración tenga más de un análisis posible (es decir, sintácticamente ambigua).

#### Gramática ambigua

![Gramática][1]



**Oración ambigua:** _the man saw the dog with the telescope_

**Traducida al español nos queda:** _El hombre vio al perro con el telescopio_

Como podemos apreciar, esta oración tiene dos posibles interpretaciones:
1) El hombre usa el telescopio para ver a un perro
2) El hombre observo a un perro con un telescopio, es decir, el perro tenia el telescopio


#### Análisis posibles

###### Árbol 1

![Árbol 1][3]

###### Árbol 2

![Árbol 2 - Mayor Probabilidad][2]

El árbol al cual se termina llegando por medio del *CKY Parser* es el **Árbol 2**, porque este es el que tiene mayor probabilidad.

Ejercicio 3: PCFGs No Lexicalizadas
-----------------------------------
__**Notación:**__
* UPCFG: Unlexicalized Probabilistic Context-Free Grammars.
* PCFG: Probabilistic Context-Free Grammars.
* Deslexicalizar PCFG: En las reglas, reemplazar todas las entradas léxicas por su POS tag.

Se implemento una UPCFG, es decir una PCFG cuyas reglas y probabilidades se obtienen a partir de un corpus de entrenamiento y luego se deslexicaliza completamente la PCFG.  
Luego para parsear una oración taggeada, nuestro parser utiliza el algoritmo de CKY (anteriormente implementado), utilizando la oración de POS tags para parsear e ignorando las entradas léxicas.

#### Resultados usando Parser CKY

|                     | Labeled | Unlabeled |
|:-------------------:|:-------:|:---------:|
|      Precision      | 72.52 % |  74.69 %  |
|       Recall        | 72.37 % |  74.54 %  |
|         F1          | 72.44 % |  74.62 %  |

##### Tiempo de Evaluación: 3 min 35 seg


Ejercicio 4: Markovización Horizontal
-------------------------------------

Se modifico la clase UPCFG del archivo *upcfg.py* para poder admitir el uso de Markovización Horizontal de orden *n*.
También se agrego al script de entrenamiento *train.py* una opción de línea de comandos que habilita esta funcionalidad, así pudiendo elegir el orden de la markovización.


#### Resultados de Evaluación para distintos ordenes de Markovización Horizontal

##### Markovización Horizontal de orden 0

|                     | Labeled | Unlabeled |
|:-------------------:|:-------:|:---------:|
|      Precision      | 69.75 % |  71.65 %  |
|       Recall        | 69.83 % |  71.72 %  |
|         F1          | 69.79 % |  71.69 %  |

>**Tiempo de Evaluación:** 2 min 1 seg


##### Markovización Horizontal de orden 1

|                     | Labeled | Unlabeled |
|:-------------------:|:-------:|:---------:|
|      Precision      | 74.26 % |  76.32 %  |
|       Recall        | 74.27 % |  76.33 %  |
|         F1          | 74.27 % |  76.33 %  |

>**Tiempo de Evaluación** 2 min 10 seg


##### Markovización Horizontal de orden 2

|                     | Labeled | Unlabeled |
|:-------------------:|:-------:|:---------:|
|      Precision      | 74.66 % |  76.67 %  |
|       Recall        | 74.21 % |  76.21 %  |
|         F1          | 74.44 % |  76.44 %  |

>**Tiempo de Evaluación:** 3 min


##### Markovización Horizontal de orden 3

|                     | Labeled | Unlabeled |
|:-------------------:|:-------:|:---------:|
|      Precision      | 73.75 % |  75.89 %  |
|       Recall        | 73.12 % |  75.24 %  |
|         F1          | 73.43 % |  75.56 %  |

>**Tiempo de Evaluación:** 3 min 25 seg


Como podemos apreciar, los mejores resultados se obtienen al hacer uso de una Markovización Horizontal de orden 2 ó 3.

<!-- Enlaces a documento e imágenes -->

[Notas de Michael Collins]: http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf

[1]: images/tabla_ambigua.png
[2]: images/tree01.png
[3]: images/tree02.png
