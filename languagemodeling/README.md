Trabajo Práctico 1 - Modelado de Lenguaje
=========================================

Mario E. Ferreyra
=================


Ejercicio 1: Corpus
-------------------

El corpus que se eligió para este Trabajo Práctico es la concatenación de todas las novelas de Gabriel García Márquez y algunos de sus cuentos.  
En este ejercicio se trabajo sobre el script *train.py*.  
Para cargar el corpus se utilizo el "corpus reader" de NLTK *"PlaintextCorpusReader"*.  
Luego utilizamos un tokenizador de NLTK *"RegexpTokenizer"* con un pattern provisto por la documentación de NLTK (al cual le agregamos un par de abreviaciones), para luego aplicarlo a nuestro corpus.

#### train.py
Este script carga un corpus, lo tokeniza y nos devuelve un modelo entrenado.


Ejercicio 2: Modelo de n-gramas
-------------------------------

En este ejercicio se trabajo sobre el archivo *ngram.py*.  
En este archivo tuvimos que definir la clase NGram, esta clase tiene como input lo siguiente:

    - n: el n correspondiente a n-grama
    - sents: lista de oraciones

Lo que se hace en esta clase es agregarle a cada oración sus correspondientes marcadores de inicio \<s> y fin de oración \</s>, pero hay una particularidad la cual y es que a cada oración le agregamos n-1 marcadores de inicio (y un marcador de fin) para olvidarnos de discriminar los distintos casos (1-grama, 2-grama, 3-grama, ...) y también para evitar los problemas de borde.  
Luego, gracias a esta representación, podemos guardar la cantidad de veces que ocurre de cada n-grama y (n-1)-grama, es decir, los tokens de largo n y sus prev_tokens de largo n-1.

Se implementaron los siguientes métodos:

    * count
    * cond_prob
    * sent_prob
    * sent_log_prob

#### Método count
    + Input:
        - tokens: Tupla de palabras
    + Output:
        - Cantidad de veces que aparece tokens en el n-grama

Solamente retorna el count correspondiente al tokens ingresado.

#### Método cond_prob
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

Como esto son demasiadas sentencias, se usa una Suposición de Markov de orden k.
Es decir:

    P(wi | wi-k ... wi-1)

Ya que:

    P(wi | w1 w2 ... wi-1) ≈ P(wi | wi-k ... wi-1)

Notar que si tenemos un n-grama de orden n, entonces vamos a tener una Suposición de Markov de orden n-1.  
Para poder estimar estas probabilidades, se usa lo siguiente:

                                 count(wi-(n-1)) ... wi-1 wi)
    P(wi | wi-(n-1) ... wi-1) = ------------------------------
                                   count(wi-(n-1) ... wi-1)

##### Problema surgido mientras se implementaba el método:
* __División por 0__: Tuvimos que manejar la situación cuando el "count" del denominador daba 0. Para solucionar dicho problema se puso un condicional, el cual detectaba cuando dicho "count" era igual a 0 entonces se devuelve una probabilidad de 0.

#### Método sent_prob
    + Input:
        - sent: Lista de palabras (oración)
    + Output:
        - Probabilidad de la oración

Recordamos que cada oración tiene n-1 marcadores de inicio \<s> y uno de fin \</s>.  
Como sabemos una oración es una lista de palabras:

    sent = w1 w2 ... wm                     Donde: wi es una palabra

Por Regla de la Cadena, tenemos que:

    P(w1 w2 ... wm) = productoria(i=1...m) P(wi | w1 w2 ... wi-1)

Por lo visto anteriormente en el método *count_prob*, gracias a la Suposición de Markov de orden n-1 (por el n-grama de orden n), tenemos que:

    P(wi | w1 w2 ... wi-1) ≈ P(wi | wi-(n-1) ... wi-1)

Por lo que:

    P(w1 w2 ... wm) = productoria(i=1...m) P(wi | wi-(n-1) ... wi-1)

Notar que esto puede ocasionar problemas de underflow, ya que al multiplicar por probabilidades muy pequeñas los valores se hacen cada vez mas pequeños.

#### Método sent_prob_log
    + Input:
        - sent: Lista de palabras (oración)
    + Output:
        - Probabilidad logarítmica de la oración

Este método es muy parecido al *sent_prob*, salvo que tiene unas cuestiones practicas como trabajar en el espacio logarítmico para calcular la probabilidad de una oración.

* Notación: log2(x) = logaritmo en base 2 de x.

También se aprovecho la cuestión de que sumar es mas rápido que multiplicar:

    - productoria(i=1, m) Pi = sumatoria(i=1, m) log2(Pi)           Donde Pi = probabilidad i-esima

Por lo que:

    P(w1 w2 ... wm) = productoria(i=1...m) P(wi | wi-(n-1) ... wi-1)

                    = sumatoria(i=1...m) log2( P(wi | wi-(n-1) ... wi-1) )

##### Problema surgido mientras se implementaba el método:
* __Logaritmo de 0__: Tuvimos que manejar la situación cuando la probabilidad condicional daba 0. Para solucionar dicho problema nos basamos en la Matemática, dado que el limite del logaritmo de x cuando x --> 0 es igual -∞, entonces lo que hicimos fue devolver -∞ cuando x = 0 y calcular log2(x) cuando x > 0.


Ejercicio 3: Generación de Texto
--------------------------------

En este ejercicio se trabajo sobre el archivo *ngram.py* y se programo un script *generate.py*.  
En el archivo *ngram.py* tuvimos que definir la clase NGramGenerator, esta clase tiene como input lo siguiente:

    - model: modelo de n-grama

Lo que se hace al principio de esta clase es guardar las probabilidades condicionales de cada palabra en un diccionario *probs*.  
También lo que se hace es guardar las mismas probabilidades condicionales en otro diccionario *sorted_probs*, con la ventaja de en este nuevo diccionario se las tiene ordenadas, haciendo mucho mas rápido el método *generate_token* que describiremos a continuación.

Se implementaron los siguientes métodos:

    * generate_token
    * generate_sent

#### Método generate_token
    + Input:
        - prev_tokens: Lista de palabras
    + Output:
        - Token aleatorio, teniendo en cuenta los prev_tokens

Para poder generar el token se utilizo el método de la Transformada inversa visto en la materia Modelos y Simulación.  
Para mas información ver el siguiente enlace de Wikipedia: [Método de la transformada inversa]

#### Método generate_sent
    + Output:
        - Oración aleatoria

Para poder generar una oración de utilizo el método *generate_token* tomando como prev_tokens a los n-1 marcadores de inicio \<s> obteniendo así un token nuevo, luego se toma este token nuevo como *prev_tokens* y obteniendo así otro token nuevo; de esta forma se sigue iterando hasta que nos topamos con el marcador de fin \</s> y ahí devolvemos la oración generada.

#### generate.py

Este script se encarga de generar oraciones del lenguaje natural basándose en un modelo de n-grama.  
Pero antes de generar oraciones hay que entrenar nuestro corpus por medio del script *train.py*, para obtener un modelo de n-grama.

## Oraciones generadas

| n-gram | Oraciones                                                                                           |
|:------:|-----------------------------------------------------------------------------------------------------|
|   1    | La aparejos plaza . de agua , de su funerales renunció Fue ojos se la también Fermina , título la se cuando de estimaba . muestra las , general gritó y los fogón le , par presentó al no le veteranos la Segundo madre lectura : los nunca . , a hasta escritor albahaca el , al como se del dando llegaba que que extendida lo Marqués volví los majestad antes negras del la los escribí ,                                                                              |
|        | conde volvió encontré Solo las desde que . a .                                                      |
|        | con reconocido almuerzo en de nadie patio noticia , suspiró tres de Sus                             |
|        | metiera parecía y ballena de la las cuellos durante abre lo en repartía vez saña ideas amistad les levantar , cuatro . cama ella que el daba un corre turno se dormido Dios . en de 20 haberla vigilias ya embajador un una lo condición pública por , la le y Por la régimen el la animal por , lo desde lo del dinastía : , cuando las el , sirio ponían , zapatos                                                                                               |
|   2    | La hojarasca revuelta por supuesto , a través del pueblo no había dado demasiadas incomprensiones recíprocas , la hace muy mujer , desde hace cuando la situación de la costa cuando entró de sudor helado resbaló por franjas lívidas que lo vio el paquete .                                                                                        |
|        | Lo mejor de un reflejo incontrolable más plácidas , dijo el mas identidad de su atadito de su último se normalizara el fondo del fondo de dormir una soga .                                                            |
|        | Una noche con el sillón , empeñaron sus Victorias e inapelable de pilas .                           |
|        | Sea lo hacía una de Pilar , pensando madre entró poco se vino con la tragedia , fue un instante de paloma para acá por pura esencia , donde no encontró el ojo izquierdo , en tres días , ofuscado , no se le hacían pruebas de la impresión de familia con una hora de Navidad de repugnancia y sangre y lanzo aun así que tenían la oscuridad . |
|   3    | Camille no alcanzó siquiera a mirarlo .                                                             |
|        | Pero cuando Meme vino a sentarse en el sitio de muerte , la hija .                                  |
|        | Hasta cierto punto ya no era consciente de la pianola , y recayo en el vacío : se preguntó de pronto una música de cuerdas que le causó ninguna preocupación , salvo la noche le equivocaba la memoria , sin embargo sabíamos que los incendios en la cubierta inferior , y que las buenas y en todo el día anterior hasta las dos manos tratando de que hay un árbol gigantesco                                                                                        |
|        | El padre Ángel no pudo creer ni entonces ni antes ningún indicio de premeditación , sin embargo fue una contrariedad .                                                                                                 |
|   4    | Los dientes estaban completos , sanos y bien alineados .                                            |
|        | Al contrario : se complació en verlo a plena luz tal como lo había visto , por supuesto , era lo único que estaban seguros , después de escuchar los informes de sus agentes , sino todo lo contrario : nunca más volví a sentir , y que murió desangrado después de haber conocido el prodigio de haberlo concebido sin concurso de varón y de haber recibido en un sueño incómodo y pegajoso , pensando , mientras dormía , alguien que nunca fu capturado entró una noche al cuartel revolucionario de Manaure y asesinó a puñaladas a su intimo amigo , el señor Carmichael , vestido de dril blanco con corbata , y con sus armas de reglamento bien visibles en el cinto .                                            |
|        | Sobrevivían de milagro .                                                                            |
|        | Lo único que le había asignado un rincón del palco en penumbra desde donde vio sin ser visto al minotauro espeso cuya voz de centella marina lo sacó en vilo de su sitio y se protegieron bajo el alar .                 |

Se puede apreciar que a medida que el n se hace más grande, las oraciones van cobrando mucho más sentido.


Ejercicio 4: Suavizado "Add-One"
--------------------------------

En este ejercicio se trabajo sobre el archivo *ngram.py* y ademas se le agrego al script *train.py* una opción de linea de comandos que le permita utilizar Add-One en lugar de los n-gramas clásicos.  
En el archivo *ngram.py* tuvimos que definir la clase AddOneNGram, esta clase tiene como input lo siguiente:

    - n: el n correspondiente a n-grama
    - sents: lista de oraciones

Al principio de esta clase usamos el método *countWordTypes* para calcular el tamaño del alfabeto.

Se implementaron los siguientes métodos:

    * countWordTypes
    * V
    * cond_prob

#### Método countWordTypes
    + Input:
        - sents: Lista de oraciones
    + Output:
        - Calcula el tamaño del alfabeto

Se encarga de calcular el tamaño del alfabeto (cantidad de words type) incluyendo el marcador de fin \</s>.

#### Método V
    + Output:
        - Tamaño del vocabulario

Solamente se devuelve el valor del tamaño del alfabeto calculado al principio de la clase.

#### Método cond_prob
    + Input:
        - token: Palabra
        - prev_tokens: Lista de palabras previas a token
    + Output:
        - Probabilidad condicional con Add-One del token: P*(token | prev_tokens)

Este método fue una re-implementación del hecho en la clase NGram, por lo cual nos basamos en los mismo fundamentos.  
Veamos como se calcula la nueva probabilidad condicional:

                                  count(wi-(n-1)) ... wi-1 wi) + 1
    P*(wi | wi-(n-1) ... wi-1) = ----------------------------------
                                    count(wi-(n-1) ... wi-1) + V


Ejercicio 5: Evaluación de Modelos de Lenguaje
----------------------------------------------

En este ejercicio se separo nuestro corpus en dos partes:
* corpus_train: para entrenamiento (90%)
* corpus_test: para pruebas (10%)

Ademas se programo un script *eval.py* para cargar un modelo de lenguajes y evaluarlo sobre el conjunto de test.

También lo que hicimos fue extender la clase NGram, con nuevos métodos los cuales son:

    * log-probability
    * cross-entropy
    * perplexity

Antes de explicar cada método, tengamos en cuenta lo siguiente:

    Supongamos que tenemos m oraciones: s1, s2, ..., sm

#### Método log-probability
    + Input:
        - sents: Lista de oraciones
    + Output:
        - Calculo de log-probability

En este método se encarga del calculo de log-probability de una lista de oraciones.  
El calculo es el siguiente:

    log-probability(s1, s2, ..., sm) = sumatoria(i=1...m) log2(P(si))

#### Método cross-entropy
    + Input:
        - sents: Lista de oraciones
    + Output:
        - Calculo de cross-entropy

En este método se encarga del calculo de cross-entropy de una lista de oraciones.  
El calculo es como sigue:

                                      log-probability(s1, s2, ..., sm)
    cross-entropy(s1, s2, ..., sm) = ----------------------------------
                                                      M

    Donde: M es la cantidad de palabras en bruto de las oraciones (se cuentan las palabras repetidas).

#### Método perplexity
    + Input:
        - sents: Lista de oraciones
    + Output:
        - Calculo de perplexity

En este método se encarga del calculo de la perplexity de una lista de oraciones.  
El calculo es como sigue:

    perplexity(s1, s2, ..., sm) = 2^(- cross-entropy(s1, s2, ..., sm))

#### eval.py

Este script carga un modelo de lenguaje natural entrenado, el cual se obtiene aplicando el *train.py* sobre el corpus_train y luego se calcula la perplexity de un conjunto de oraciones, es decir, el corpus_test.

### Perplejidad de los modelos entrenados con el Suavizado Add-One

| n-gram | Perplexity |
|:------:|:----------:|
|   1    |    1317    |
|   2    |    4175    |
|   3    |   26179    |
|   4    |   42331    |

Al ver estos resultados, vemos que mientras el n se hace más grande, la perplexity aumenta.  
Por lo que podemos decir que el suavizado Add-One no es bueno, porque mientras más chica sea la perplexity mejor.


Ejercicio 6: Suavizado por Interpolación
----------------------------------------

En este ejercicio se trabajo sobre el archivo *ngram.py* y ademas de agregarle al script *train.py* una opción de linea de comandos que le permita utilizar Interpolación en lugar de los n-gramas clásicos y Add-One.  
En el archivo *ngram.py* tuvimos que definir la clase InterpolatedNGram, esta clase tiene como input lo siguiente:

    - n: el n correspondiente a n-grama
    - sents: lista de oraciones
    - gamma: parámetro para el calculo de la probabilidad condicional
    - addone: parámetro para saber si el primer 1-grama es suavizado con Add-One

Lo que hacemos al principio en esta clase es analizar si el *gamma* es parámetro, en el caso de que no sea parámetro, vamos a necesitar *Datos Held Out* para su posterior calculo.  
Los *Datos Held Out* son un porcentaje de las *sents* (en este caso un 10%), es decir, a las *sents* les sacamos un 10%, para los *Datos Held Out*.  
Luego lo que hacemos es generar: 1-grama, 2-grama, ..., n-grama con el método *getModels*. Esto cobrara mas sentido cuando expliquemos el método *cond_prob*

Se implementaron los siguientes métodos:

    * getModels
    * getHeldOut
    * getGamma
    * count
    * getLambdas
    * cond_prob

#### Método getModels
    + Input:
        - n: orden del modelo
        - sents: Listas de oraciones
        - is_addone: Booleano para saber si el primer n-grama es suavizado con Add-One
    + Output:
        - Lista de modelos

Genera una lista de *n* modelos donde el primero puede ser suavizado con Add-One si es que el parámetro *is_addone* es True caso contrario se usan los N-gramas clásicos, el resto de modelos son N-Gramas clásicos.

#### Método getHeldOut
    + Input:
        - sents: Listas de oraciones
        - percentage: porcentaje de oraciones a tomar
    + Output:
        - Datos Held Out y las nuevas sents

Solamente nos genera los *Datos Held Out* a partir de las oraciones dadas y las nuevas "sents", es decir, las *sents* menos los *Datos Held Out*.

#### Método getGamma
    + Input:
        - held_out: Datos Held Out
    + Output:
        - Calculo de un *gamma*

Para el calculo del gamma se realiza un "barrido", usando los *Datos Held Out*.  
En las siguientes notas [Notas de Michael Collins] nos explica lo siguiente:  
1. El valor para Gamma puede ser elegido de nuevo maximizando la log-probability de un held-out data

2. El valor para Gamma puede ser elegido de nuevo minimizar la perplexity de un held-out data

Notar que (1) y (2) son equivalentes, pero hacer (1) lleva menos operaciones, por ello se implemento (1).  

Básicamente el "barrido" es ir probando valores de *gamma*:  
A. Ponerle a *gamma* un valor inicial  
B. Calcular *log_probability* con los *Datos Held Out*.  
C. Aumentar *gamma* y calcular *log_probability*.  
D. Aumentar *gamma* y calcular *log_probability*.  
E. Así sucesivamente.  
F. Quedarse con el *gamma* que mejora la *log_probability*.  

#### Método count
    + Input:
        - tokens: Tupla de palabras
    + Output:
        - Cantidad de veces que aparece tokens en alguno de los n-gramas

Este método count a diferencias de los otros *count* tiene la particularidad de que el *tokens* puede pertenecer a cualquiera de los n-gramas, por lo que para solucionar este problema analizamos el largo de *tokens* para poder determinar a que n-grama pertenece.

#### Método getLambdas
    + Input:
        - tokens: lista de palabras
    + Output:
        - Calculo de lambdas.

Para el calculo de los lambdas, implementamos las formulas que están en las notas: [Modelado de Lenguaje: Notas Complementarias].

#### Método cond_prob
    + Input:
        - token: Palabra
        - prev_tokens: Lista de palabras previas a token
    + Output:
        - Probabilidad condicional.

Para poder realizar el calculo de de la probabilidad condicional, se implemento las formulas de las siguientes notas [Modelado de Lenguaje: Notas Complementarias].  
Los n-gramas que se generaron al principio son para el calculo de la probabilidades condicionales de los distintos n-gramas (*qML* en las notas).


### Perplejidad de los modelos entrenados con el Suavizado por Interpolación

| n-gram | Perplexity |
|:------:|:----------:|
|   1    |    1364    |
|   2    |    508     |
|   3    |    476     |
|   4    |    472     |

Calculamos la perplexity usando como modelo el Suavizado por Interpolación y como se ven en los datos expuestos en la tabla de arriba, vemos que mientras el n se hace más grande, la perplexity disminuye y eso es bueno.  
Haciendo una comparación con el Suavizado Add-One, concluimos que el Suavizado por Interpolación funciona mucho mejor.  


Ejercicio 7: Suavizado por Back-Off con Discounting
---------------------------------------------------

En este ejercicio se trabajo sobre el archivo *ngram.py* y ademas de agregarle al script *train.py* una opción de linea de comandos que le permita utilizar Back-Off con Discounting en lugar de los n-gramas clásicos, Add-One y Interpolación.  
En el archivo *ngram.py* tuvimos que definir la clase BackOffNGram, esta clase tiene como input lo siguiente:

    - n: el n correspondiente a n-grama
    - sents: lista de oraciones
    - beta: parámetro que afecta el calculo de la probabilidad condicional
    - addone: parámetro para saber si el primer 1-grama es suavizado con Add-One

Lo que hacemos al principio en esta clase es analizar si el *beta* es parámetro, en el caso de que no sea parámetro, vamos a necesitar *Datos Held Out* para su posterior calculo.  
Los *Datos Held Out* son un porcentaje de las *sents* (en este caso un 10%), es decir, a las *sents* les sacamos un 10%, para los *Datos Held Out*.  
Luego lo que hacemos es generar: 1-grama, 2-grama, ..., n-grama con el método *getModels*.  
Ademas se crean dos diccionarios *dict_denom* y *dict_alpha*, esto es para que el calculo de la probabilidad condicional en el método *cond_prob* sea mucho mas rápido y no tener que re-calcular siempre lo mismo en cada llamada al método.

Se implementaron los siguientes métodos:

    * getModels
    * getHeldOut
    * getBeta
    * generateSetA
    * count
    * A
    * generateDictAlpha
    * generateDictDenom
    * alpha
    * denom
    * cond_prob

#### Método getModels
    + Input:
        - n: orden del modelo
        - sents: Listas de oraciones
        - is_addone: Booleano para saber si el primer n-grama es suavizado con Add-One
    + Output:
        - Lista de modelos

Genera una lista de *n* modelos donde el primero puede ser suavizado con Add-One si es que el parámetro *is_addone* es True caso contrario se usan los N-gramas clásicos, el resto de modelos son N-Gramas clásicos.

#### Método getHeldOut
    + Input:
        - sents: Listas de oraciones
        - percentage: porcentaje de oraciones a tomar
    + Output:
        - Datos Held Out y las nuevas sents

Solamente nos genera los *Datos Held Out* a partir de las oraciones dadas y las nuevas "sents", es decir, las *sents* menos los *Datos Held Out*.

#### Método getBeta
    + Input:
        - held_out: Datos Held Out
    + Output:
        - Calculo de un *beta*

Para el calculo del gamma se realiza un "barrido", usando los *Datos Held Out*.  
En las siguientes notas [Notas de Michael Collins] nos explica lo siguiente:  

    El valor para Beta puede ser elegido de nuevo maximizando la log-probability de un held-out data

Básicamente el "barrido" es ir probando valores de *beta* en un rango de 0 a 1, es decir, 0 <= *beta* <= 1:  
A. Ponerle a *beta* un valor inicial  
B. Calcular los diccionario *dict_alpha* y *dict_denom*.  
C. Calcular *log_probability* con los *Datos Held Out*.  
D. Aumentar *beta*, calcular nuevamente los diccionario *dict_alpha* y *dict_denom* y calcular *log_probability*.  
E. Así sucesivamente.  
F. Quedarse con el *beta* que mejora la *log_probability*.  

#### generateSetA
    + Input:
        - n: Orden del modelos
        - models: Lista de modelos de n-gramas
    + Output:
        - Calculo de un *beta*

Se encarga de generar el conjunto A(x1 ... xi) = {x : count(x1 ... xi x) > 0} expuesto en las [Notas de Michael Collins].  
Para ellos se hace uso de los modelos generados por el método *getModels*.

#### Método count
    + Input:
        - tokens: Tupla de palabras
    + Output:
        - Cantidad de veces que aparece tokens en alguno de los n-gramas

Este método count a diferencias de los otros *count* tiene la particularidad de que el *tokens* puede pertenecer a cualquiera de los n-gramas, por lo que para solucionar este problema analizamos el largo de *tokens* para poder determinar a que n-grama pertenece.  
Tuvimos la dificultad en los casos de las tuplas que era de la forma (\<s>, \<s>, \<s>, ...) ya que siempre se mandaban a un n-grama incorrecto, para solucionar dicho problema, se analizo si el tamaño de la tupla ingresada es igual a la cantidad de elementos \<s> de la tupla, es decir, que la tupla contenga solamente \<s>, luego se las pudo mandar a los n-gramas correspondientes y así calcular el *count* correcto de la tupla.

#### A
    + Input:
        - tokens: Lista de palabras
    + Output:
        - Conjunto de palabras

Solamente retorna el siguiente conjunto de palabras {x : count(tokens x) > 0}

#### generateDictAlpha

Carga el diccionario *dict_alpha* con las palabras pertenecientes al conjunto A y un valor numérico que se corresponde con el Calculo de Alfa de las notas [Modelado de Lenguaje: Notas Complementarias].

#### generateDictDenom

Carga el diccionario *dict_denom* con las palabras pertenecientes al conjunto A y un valor numérico que se corresponde con el Calculo del Denominador Normalizador de las notas [Modelado de Lenguaje: Notas Complementarias].

#### alpha
    + Input:
        - tokens: Lista de palabras
    + Output:
        - Calculo de Alfa

Solamente retorna el valor almacenado en el diccionario *dict_alpha* correspondiente al *tokens* ingresado.

#### denom
    + Input:
        - tokens: Lista de palabras
    + Output:
        - Calculo del Denominador Normalizador

Solamente retorna el valor almacenado en el diccionario *dict_denom* correspondiente al *tokens* ingresado.

#### Método cond_prob
    + Input:
        - token: Palabra
        - prev_tokens: Lista de palabras previas a token
    + Output:
        - Probabilidad condicional.

Para el calculo de de la probabilidad condicional, se implemento las formulas de las notas [Modelado de Lenguaje: Notas Complementarias].  
Tuvimos la dificultad en el caso i=1, ya que la formula expresada en las notas en nuestro caso podría ser para un n-grama clásico o uno suavizado con Add-One (por el parámetro *addone* explicado en el método *getModels*), el problema se soluciono fácilmente con un condicional, en donde en el caso de que el n-grama fuera suavizado con Add-One se usaba su calculo correspondiente, sino se usaba el calculo para los n-gramas clásicos.


### Perplejidad de los modelos entrenados con el Suavizado por Back-Off con Discounting

| n-gram | Perplexity |
|:------:|:----------:|
|   1    |    1364    |
|   2    |    384     |
|   3    |    361     |
|   4    |    365     |

Calculamos la perplexity usando como modelo el Suavizado por Back-Off con Discounting y como se ven en los datos expuestos en la tabla de arriba, vemos que mientras el n aumenta la perplexity disminuye (excepto en el caso de n=4) y eso es bueno.  
Haciendo una comparación con el Suavizado Add-One y el Suavizado por Interpolación, concluimos que el Suavizado por Back-Off funciona mucho mejor en comparación con Add-One y levemente mejor que con Interpolación.  

Para ver esto un poco mas claro comparemos la perplexity con todos los suavizados vistos:

### Comparación final de la Perplexity con todos los Suavizados

|       Suavizado          | n=1  | n=2  |  n=3  |  n=4  |
|:------------------------:|:----:|:----:|:-----:|:-----:|
|        Add-one           | 1317 | 4175 | 26179 | 42331 |
|     Interpolación        | 1364 | 508  |  476  |  472  |
| Back-Off con Discounting | 1364 | 384  |  361  |  365  |


### Test para Log-Probability, Cross-Entropy, Perplexity

Se implementaron un total de 6 tests nuevo para probar la funcionalidad de *log_probability*, *cross-entropy* y *perplexity*, se distribuyeron de la siguiente manera:
* 2 tests para *log_probability* (1-grama y 2-grama respectivamente)
* 2 tests para *cross_entropy* (1-grama y 2-grama respectivamente)
* 2 tests para *perplexity* (1-grama y 2-grama respectivamente)

#### Test Log-Probability
Para su implementación nos basamos en las oraciones ya provistas en los test y los tests ya hechos para el método de *sent_log_prob*, del cual se tomaron las probabilidades ya calculadas de cada oración, las cuales se sumaron para así obtener la *log_probability* en cada uno de los casos de los n-gramas.

#### Test Cross-Entropy
Una vez obtenido la *log_probability* se las dividió por la cantidad de palabras de las oraciones, así obteniendo los valores de *cross_entropy*.

#### Test Perplexity
Ya obtenido la *cross_entropy* en los test anteriores se realizo el calculo de la perplexity mostrado anteriormente, así obteniendo los valores de la *perplexity*.


[Método de la transformada inversa]: https://es.wikipedia.org/wiki/M%C3%A9todo_de_la_transformada_inversa
[Notas de Michael Collins]: http://www.cs.columbia.edu/~mcollins/lm-spring2013.pdf
[Modelado de Lenguaje: Notas Complementarias]: https://cs.famaf.unc.edu.ar/~francolq/lm-notas.pdf
