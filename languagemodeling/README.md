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

En este ejercicio se trabajo sobre el archivo *ngram.py* y ademas se agrego al script *train.py* una opción de linea de comandos que le permita utilizar Add-One en lugar de los n-gramas clásicos.  
En el archivo *ngram.py* tuvimos que definir la clase AddOneNGram, esta clase tiene como input lo siguiente:

    - n: el n correspondiente a n-grama
    - sents: lista de oraciones

Al principio de esta clase se calcula el tamaño del alfabeto (cantidad de words type) incluyendo el marcador def fin \</s>.

Se implementaron los siguientes métodos:

    * V
    * cond_prob

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

## Perplejidad de los modelos entrenados con el Suavizado Add-One

| n-gram | Perplexity |
|:------:|:----------:|
|   1    |    1317    |
|   2    |    4175    |
|   3    |   26179    |
|   4    |   42331    |

Al ver estos resultados, vemos que mientras el n se hace más grande, la perplexity aumenta.  
Por lo que podemos decir que el suavizado Add-One no es bueno, porque mientras más chica sea la perplexity mejor.

[Método de la transformada inversa]: https://es.wikipedia.org/wiki/M%C3%A9todo_de_la_transformada_inversa
