Trabajo Práctico 1 - Modelado de Lenguaje
=========================================

Mario E. Ferreyra
=================


Ejercicio 1: Evaluación de Parsers
----------------------------------

Se implemento el script *eval.py*, el cual permite evaluar parsers.  
Se entreno y evaluo (con los scripts *train.py* y *eval.py* respectivamente) los modelos "baseline" para todas las oraciones de largo menor o igual a 20.  
Los resultados fueron los siguientes:

* Cantidad de oraciones de largo menor o igual a 20 = 1444

### Parser Flat
##### Labeled
* Precision = 99.93%
* Recall = 14.58%
* F1 = 25.44%

##### Unlabeled
* Precision = 100.00%
* Recall = 14.59%
* F1 = 25.46%


### Parser Lbranch
##### Labeled
* Precision = 8.81%
* Recall = 14.58%
* F1 = 10.98%

##### Unlabeled
* Precision = 14.71%
* Recall = 24.35%
* F1 = 18.34%


### Parser Rbranch
##### Labeled
* Precision = 8.81%
* Recall = 14.58%
* F1 = 10.98%

##### Unlabeled
* Precision = 8.88%
* Recall = 14.69%
* F1 = 11.07%


Ejercicio 2: Algoritmo CKY
--------------------------
Se implemento el algoritmo CKY en un módulo *cky_parser.py*, podemos encotrar una pseudocódigo de este en estas [Notas de Michael Collins] especificamente en la pagina 14.

Se agrego a los tests un test con una gramática y una oración tal que la oración tenga más de un análisis posible (sintácticamente ambigua).


### Gramática ambigua

|   Producción    | Probabilidad |
|:----------------|:------------:|
| S -> NP VP      |     1.0      |
|-----------------|--------------|
| VP -> Vi        |     0.3      |
| VP -> Vt NP     |     0.5      |
| VP -> VP PP     |     0.2      |
|-----------------|--------------|
| NP -> DT NN     |     0.8      |
| NP -> NP PP     |     0.2      |
|-----------------|--------------|
| PP -> IN NP     |     1.0      |
|-----------------|--------------|
| Vi -> sleeps    |     1.0      |
|-----------------|--------------|
| Vt -> saw       |     1.0      |
|-----------------|--------------|
| NN -> man       |     0.1      |
| NN -> woman     |     0.1      |
| NN -> telescope |     0.3      |
| NN -> dog       |     0.5      |
|-----------------|--------------|
| DT -> the       |     1.0      |
|-----------------|--------------|
| IN -> with      |     0.6      |
| IN -> in        |     0.4      |


<!-- Terminar bien la parte de la ambiguedad -->
<!-- Tanto en el test como en el README -->

Ejercicio 3: PCFGs No Lexicalizadas
-----------------------------------
__**Notacion:**__
* UPCFG: Unlexicalized Probabilistic Context-Free Grammars.
* PCFG: Probabilistic Context-Free Grammars.
* Deslexicalizar PCFG: En las reglas, reemplazar todas las entradas léxicas por su POS tag.

Se implemento una UPCFG, es decir una PCFG cuyas reglas y probabilidades se obtienen a partir de un corpus de entrenamiento y luego se deslexicaliza completamente la PCFG.  
Luego para parsear una oracion taggeada, nuestro parser utiliza el algoritmo de CKY (anteriormente implementado), utilizando la oración de POS tags para parsear e ignorando las entradas léxicas.



Ejercicio 4: Markovización Horizontal
-------------------------------------




[Notas de Michael Collins]: http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf
