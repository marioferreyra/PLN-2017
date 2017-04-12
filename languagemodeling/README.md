Trabajo Práctico 1 - Modelado de Lenguaje
=========================================

Ejercicio 1: Corpus
-------------------

El corpus que se eligio para este Trabajo Práctico concatenacion de todos las novelas de Gabriel Garcia Marquez y algunos de sus cuentos.
En este ejercicio se trabajo sobre el script train.py
Para cargar el corpus se utilizo el "corpus reader" de NLTK.
Luego utilizamos un tokenizador de NLTK con un pattern provisto por la documentacion de NLTK (al cual le agregamos un par de abreviaciones), para luego aplicarlo a nuestro corpus.


Ejercicio 2: Modelo de n-gramas
-------------------------------

En este ejercicio se trabajo sobre el script ngram.py.
En este archivo tuvimos que definir la clase NGram con los siguientes metodos:

    *  __init__
    *  count
    *  cond_prob
    *  sent_prob
    *  sent_log_prob

#### Metodo __init__
Este metodo toma un n correspodiente de n-grama y una lista de oraciones.

Adentro del metodo lo que se hace es agregarle a cada oracion sus correspodientes delimitadores de inicio y fin de oracion, pero hay una particularidad y es que a cada oracion le agregamos n-1 marcadores de inicio para olvidarnos de discriminar casos de (1-grama, 2-grama, 3-grama, ...) y tambien para evitar los problemas de borde.

counts ...

#### Metodo count
    + Input:
        - tokens: Tupla de palabras
    + Output:
        - Cantidad de veces que aparece tokens en el n-grama


#### Metodo cond_prob
    + Input:
        - token: Palabra
        - prev_tokens: Lista de palabras previas a token
    + Output:
        - Probabilidad condicional del token: P(token | prev_tokens)
          Denotemos:
                    * token: wi
                    * prev_tokens: w1 w2 ... wi-1
          Entonces:
                P(token | prev_tokens) = P(wi | w1 w2 ... wi-1)

Como esto son demasiadas sentencias, se usa una Suposicion de Markov de orden k.
Es decir:

    P(wi | wi-k ... wi-1)

Ya que:

    P(wi | w1 w2 ... wi-1) ≈ P(wi | wi-k ... wi-1)

Cabe destacar que si tenemos un n-grama de orden n, entonces vamos a tener una Suposicion de Markov de orden n-1.
Para poder estimar estas probabilidades, se usa lo siguiente:

                             count(wi-k ... wi-1 wi)
    P(wi | wi-k ... wi-1) = -------------------------
                              count(wi-k ... wi-1)

AGREGAR PROBLEMA CON LA DIVISION POR 0

#### Metodo sent_prob
    + Input:
        - sent: Lista de palabras (oracion)
    + Output:
        - Probabilidad de la oracion

Recordamos que cada oracion tiene n-1 marcadores de inicio y uno de fin.
Como sabemos una oracion es una lista de palabras:

    sent = w1 w2 ... wm                     Donde: wi es una palabra

Por Regla de la Cadena, tenemos que:

    P(w1 w2 ... wm) = productoria(i=1, m) P(wi | w1 w2 ... wi-1)

Por lo visto anteriormente en el Metodo count_prob, gracias a la Suposicion de Markov de orden k, tenemos que:

    P(wi | w1 w2 ... wi-1) ≈ P(wi | wi-k ... wi-1)

Por lo que:

    P(w1 w2 ... wm) = productoria(i=1, m) P(wi | wi-k ... wi-1)


#### Metodo sent_prob_log
    + Input:
        - sent: Lista de palabras (oracion)
    + Output:
        - Probabilidad logaritmica de la oracion

Este metodo es muy parecido al sent_prob, salvo que tiene unas cuestiones practicas como trabajar en el espacio logaritmico para calcular la probabilidad de una oracion:
Tiene las siguientes caracteristicas:

* Sumar es mas rapido que multiplicar.
* Hay que tener cuidado con el underflow.

    productoria(i=1, m) pi = sumatoria(i=1, m) log2(pi)           Donde pi = probabilidad i-esima

Por lo que:

    P(w1 w2 ... wm) = productoria(i=1, m) P(wi | wi-k ... wi-1)

                    = sumatoria(i=1, m) log2( P(wi | wi-k ... wi-1) )

AGREGAR PROBLEMA CON UNDERFLOW (-inf)


Ejercicio 3: Generación de Texto
--------------------------------

En este ejercicio se trabajo sobre el script ngram.py.
En este archivo tuvimos que definir la clase NGramGenerator con los siguentes metodos:

    *  __init__
    *  generate_token
    *  generate_sent


