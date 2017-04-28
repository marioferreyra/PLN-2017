Trabajo Práctico 2 - Etiquetado de Secuencias
=============================================

Mario E. Ferreyra
=================


Ejercicio 1: Corpus AnCora - Estadísticas de etiquetas POS
----------------------------------------------------------
Se implemento un script *stats.py*, que se encarga de mostrar las siguientes estadisticas del corpus AnCora:

Estadisticas
============
Cantidad de oraciones =               17378
Cantidad de tags =                    85
Cantidad de ocurrencias de palabras = 517194
Cantidad de palabras distintas =      46501

-----------------------------------------------------------
 Tag | Frecuencia | Porcentaje | 5 Palabras mas frecuentes 
-----------------------------------------------------------
 sp000   | 79884 | 93981.18% | de en a del con
nc0s000  | 63452 | 74649.41% | presidente equipo partido país año
 da0000  | 54549 | 64175.29% | la el los las El
 aq0000  | 33906 | 39889.41% | pasado gran mayor nuevo próximo
   fc    | 30147 | 35467.06% | ,
np00000  | 29111 | 34248.24% | Gobierno España PP Barcelona Madrid
nc0p000  | 27736 | 32630.59% | años millones personas países días
   fp    | 17512 | 20602.35% | .
   rg    | 15336 | 18042.35% | más hoy también ayer ya
   cc    | 15023 | 17674.12% | y pero o Pero e


--------------------------------------------------------------------------
 Nivel de Ambigüedad | #Palabras | Porcentaje | 5 Palabras mas frecuentes 
--------------------------------------------------------------------------
         1           |   43972   |   94.56  % | , con por su El
         2           |   2318    |   4.98   % | el en y " los
         3           |    180    |   0.39   % | de la . un no
         4           |    23     |   0.05   % | que a dos este fue
         5           |     5     |   0.01   % | mismo cinco medio ocho vista
         6           |     3     |   0.01   % | una como uno
         7           |     0     |    0.0   % | 
         8           |     0     |    0.0   % | 
         9           |     0     |    0.0   % | 


Ejercicio 2: Baseline Tagger
----------------------------

Ejercicio 3: Entrenamiento y Evaluación de Taggers
--------------------------------------------------

Ejercicio 4: Hidden Markov Models y Algoritmo de Viterbi
--------------------------------------------------------

Ejercicio 5: HMM POS Tagger
---------------------------

Ejercicio 7: Maximum Entropy Markov Models
------------------------------------------

| n-gram | Perplexity |
|:------:|:----------:|
|   1    |    1364    |
|   2    |    508     |
|   3    |    476     |
|   4    |    472     |


Ejercicio 7: Suavizado por Back-Off con Discounting
---------------------------------------------------
