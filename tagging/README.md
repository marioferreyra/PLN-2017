Trabajo Práctico 2 - Etiquetado de Secuencias
=============================================

Mario E. Ferreyra
=================


Ejercicio 1: Corpus AnCora - Estadísticas de etiquetas POS
----------------------------------------------------------
Se implemento el script *stats.py*, que se encarga de mostrar las siguientes estadísticas del corpus AnCora:


#### Estadísticas Básicas

* Cantidad de oraciones = 17378
* Cantidad de tags (vocabulario de tags) = 85
* Cantidad de ocurrencias de palabras = 517194
* Cantidad de palabras (vocabulario) = 46501


#### Etiquetas más frecuentes

|   Tag    | Frecuencia | Porcentaje |       5 Palabras mas frecuentes       |
|:--------:|:----------:|:----------:|:-------------------------------------:|
|  sp000   |   79884    |   15.45 %  |            de en a del con            |
| nc0s000  |   63452    |   12.27 %  |  presidente equipo partido país año   |
|  da0000  |   54549    |   10.55 %  |           la el los las El            |
|  aq0000  |   33906    |   6.56 %   |    pasado gran mayor nuevo próximo    |
|    fc    |   30147    |   5.83 %   |                   ,                   |
| np00000  |   29111    |   5.63 %   |  Gobierno España PP Barcelona Madrid  |
| nc0p000  |   27736    |   5.36 %   |  años millones personas países días   |
|    fp    |   17512    |   3.39 %   |                   .                   |
|    rg    |   15336    |   2.97 %   |        más hoy también ayer ya        |
|    cc    |   15023    |    2.9 %   |            y pero o Pero e            |




#### Descripción de cada Etiqueta

|   Tag   |         Descripción         |
|:-------:|:---------------------------:|
|  sp000  |         Preposición         |
| nc0s000 | Sustantivo común (singular) |
| da0000  |     Artículo (definido)     |
| aq0000  |   Adjetivo (descriptivo)    |
|   fc    |            Coma             |
| np00000 |      Sustantivo propio      |
| nc0p000 |  Sustantivo común (plural)  |
|   fp    |            Punto            |
|   rg    |     Adverbio (general)      |
|   cc    |   Conjunción (coordinada)   |


#### Niveles de ambigüedad de las palabras

| Nivel de Ambigüedad  | #Palabras | Porcentaje |  5 Palabras mas frecuentes   |
|:--------------------:|:---------:|:----------:|:----------------------------:|
|          1           |   43972   |   94.56 %  |       , con por su El        |
|          2           |   2318    |   4.98 %   |        el en y " los         |
|          3           |    180    |   0.39 %   |        de la . un no         |
|          4           |    23     |   0.05 %   |      que a dos este fue      |
|          5           |     5     |   0.01 %   | mismo cinco medio ocho vista |
|          6           |     3     |   0.01 %   |         una como uno         |
|          7           |     0     |    0.0 %   |                              |
|          8           |     0     |    0.0 %   |                              |
|          9           |     0     |    0.0 %   |                              |


Ejercicio 2: Baseline Tagger
----------------------------
Se implemento en el archivo *baseline.py* un "Etiquetador Baseline", el cual se encarga de etiquetar cada palabra con su etiqueta más frecuente observada en entrenamiento y a las palabras desconocidas, es decir, aquellas no vistas en el entrenamiento, las etiquetamos con **nc0s000**.


Ejercicio 3: Entrenamiento y Evaluación de Taggers
--------------------------------------------------
Se implemento el script *train.py*, el cual nos permitirá entrenar un "Etiquetador Baseline".  

Se implemento el script *eval.py*, el cual nos permitirá evaluar un modelo de tagging.  
Este script calcula lo siguiente:

* Accuracy sobre todas las palabras.
* Accuracy sobre las palabras conocidas.
* Accuracy sobre las palabras desconocidas.
* Matriz de confusión.

__**Nota:**__ *Accuracy* es el porcentaje de etiquetas correctas, es decir, la cantidad de aciertos del modelo de tagging sobre el tagging original.


### Evaluación del "Etiquetador Baseline"
* Accuracy sobre todas las palabras = 87.61 %
* Accuracy sobre las palabras conocidas = 95.30 %
* Accuracy sobre las palabras desconocidas = 18.01 %

#### Matriz de confusión para los 10 tags más frecuentes

Muestra el porcentaje de que una palabra con tag x se haya etiquetado incorrectamente con tag y.

|         |  sp000  | nc0s000 | da0000  | aq0000  |   fc    | nc0p000 |   rg    | np00000 |   fp    |   cc    |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  sp000  |  14.28  |  0.05   |    -    |    -    |    -    |    -    |  0.01   |    -    |    -    |    -    |
| nc0s000 |    -    |  12.24  |    -    |  0.23   |    -    |    -    |  0.03   |    -    |    -    |    -    |
| da0000  |    -    |  0.15   |  9.54   |    -    |    -    |    -    |    -    |    -    |    -    |    -    |
| aq0000  |  0.01   |  2.06   |    -    |  4.81   |    -    |  0.14   |    -    |    -    |    -    |    -    |
|   fc    |    -    |    -    |    -    |    -    |  5.85   |    -    |    -    |    -    |    -    |    -    |
| nc0p000 |    -    |  1.24   |    -    |  0.18   |    -    |   4.1   |    -    |    -    |    -    |    -    |
|   rg    |  0.02   |  0.31   |    -    |  0.03   |    -    |    -    |  3.29   |    -    |    -    |  0.02   |
| np00000 |    -    |  2.05   |    -    |    -    |    -    |    -    |    -    |  1.52   |    -    |    -    |
|   fp    |    -    |    -    |    -    |    -    |    -    |    -    |    -    |    -    |  3.55   |    -    |
|   cc    |    -    |  0.01   |    -    |    -    |    -    |    -    |  0.05   |    -    |    -    |  3.34   |


Ejercicio 4: Hidden Markov Models y Algoritmo de Viterbi
--------------------------------------------------------
Se implemento en el archivo *hmm.py* la clase *HMM* la cual es una implementacion de los Hidden Markov Model (modelo Generativo), cuyos parámetros son:
* Las probabilidades de transición entre estados (las etiquetas).
* Las probabilidades de emisión de símbolos (las palabras).

Este modelo se encarga de calcular la probabilidad de que una oración sea etiquetada con una secuencia de tags.  
Para esto, utilizamos la supocición de Markov, la cual calcula lo siguiente:

* Probabilidad de que ocurra un tag, dado que ocurrieron una cierta cantidad de tags previos.
* Probabilidad de que ocurra una palabra dado un tag (es decir, la palabra observada emparejada con el tag).

Tambien se implemento la clase *ViterbiTagger* la cual es una implementacion del Algoritmo de Viterbi el cual calcula el
etiquetado más probable de una oración.

Para las implentaciones se siguo las [Notas de Michael Collins].


Ejercicio 5: HMM POS Tagger
---------------------------
Se implemento en el archivo *hmm.py* la clase *MLHMM* la cual es una implementacion de los Hidden Markov Model cuyos parámetros se estiman usando Maximum Likelihood sobre un corpus de oraciones etiquetado.  
Estas estimaciones se hacen por medio de *counts* sobre el corpus de oraciones etiquetado.

Para dicha implementación tambien se siguo las [Notas de Michael Collins].

Tambien se le agrego al script *train.py*, la opcion de entrenar un "Maximum Likelihood Markov Model" de parametro *N* (Por los n-gramas).

#### Accuracy de los modelos entrenados con MLHMM, con su tiempo de evaluación


Ejercicio 6: Features para Etiquetado de Secuencias
---------------------------------------------------
Se implemento en el archivo *features.py* lo siguiente:

#### Features Básicos:

* word_lower: La palabra actual en minúsculas.
* word_istitle: La palabra actual empieza en mayúsculas.
* word_isupper: La palabra actual está en mayúsculas.
* word_isdigit: La palabra actual es un número.

#### Features Paramétricos:

* NPrevTags(n): La tupla de los últimos n tags.
* PrevWord(f): Dado un feature *f*, aplicarlo sobre la palabra anterior en lugar de la actual.


Ejercicio 7: Maximum Entropy Markov Models
------------------------------------------




[Notas de Michael Collins]: http://www.cs.columbia.edu/~mcollins/hmms-spring2013.pdf
