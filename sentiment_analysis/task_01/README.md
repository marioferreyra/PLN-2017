Task 1: Sentiment Analysis at Tweet
===================================

En esta tarea lo que se propone es la evaluación de sistemas de clasificación de polaridad de tweets en español.
Para ello se utilizaran los [Corpus de la TASS-2017] correspondientes a la Task 1.

El sentimiento de los tweets del corpus tienen 4 niveles de polaridad:
* NONE: No se sabe la polaridad del tweet.
* N: La polaridad del tweet Negativa.
* NEU: La polaridad del tweet es Neutra.
* P: La polaridad del tweet es Positiva.


## Descripción General
Primero se hizo una Preclasificación de los tweets basándonos en los emoticones.

Lo que se hizo para esta tarea es hacer un preprocesamiento de los contenidos de los tweets en donde básicamente se hace lo siguiente:
* Reemplazar las letras con tildes por sus respectivas versiones sin tildes.
* Convertir el texto a minúscula.
* Eliminar URL's.
* Eliminar signos de puntuación.
* Eliminar nombres de usuarios (i.e. los que comienzan con @. e.g. @user).
* Eliminar caracteres no alfanuméricos.
* Reducción de elongaciones (i.e. "hooollaaaa" -> "hola", "jjjaaajjaaaja" -> "jajaja")
* Cambiar las expresiones de risas por la palabra "risas" (i.e. "jajaja" -> "risas", "jeje" -> "risas")
* Remover las stopwords del texto.
* Aplicar a cada palabra el Stemming.

Después tokenizamos el texto usando [NLTK], para luego probar distintas combinaciones de vectorizadores y clasificadores para entrenar nuestro modelo con el Corpus de Entrenamiento y ver los resultados que entregan estas combinaciones.

Los vectorizadores que se usaron fueron los siguientes:
* [CountVectorizer]
* [TfidfVectorizer]

Los clasificadores que se usaron fueron los siguientes:
* [LinearSVC]
* [LogisticRegression]
* [RandomForestClassifier]

Luego de obtener la clasificación se implemento una idea leída en uno de los [Papers_2016], la cual consiste en definir la polaridad por la siguiente regla:
* Mantener la polaridad que se obtuvo en la Preclasificación si el tweet es marcado como "P" o "N" de lo contrario tomamos el valor estimado por el clasificador.


## Resultados

* *Tweets analizados =* 1899

### Preclasificación usando Emoticones
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        1897        |
|     N     |         1          |
|    NEU    |         0          |
|     P     |         1          |

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"LinearSVC"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        179         |
|     N     |        910         |
|    NEU    |        229         |
|     P     |        581         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        179         |
|     N     |        910         |
|    NEU    |        228         |
|     P     |        582         |

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"LogisticRegression"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |         74         |
|     N     |        1105        |
|    NEU    |         88         |
|     P     |        632         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |         74         |
|     N     |        1105        |
|    NEU    |         87         |
|     P     |        633         |

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"RandomForestClassifier"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        109         |
|     N     |        1255        |
|    NEU    |         60         |
|     P     |        475         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        109         |
|     N     |        1254        |
|    NEU    |         60         |
|     P     |        476         |

<!-- ###################################################################### -->
<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"LinearSVC"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        113         |
|     N     |        988         |
|    NEU    |        151         |
|     P     |        647         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        113         |
|     N     |        988         |
|    NEU    |        150         |
|     P     |        648         |

<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"LogisticRegression"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |         3          |
|     N     |        1361        |
|    NEU    |         5          |
|     P     |        530         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:----------------- :|
|   NONE    |         3          |
|     N     |        1360        |
|    NEU    |         5          |
|     P     |        531         |

<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"RandomForestClassifier"*
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        155         |
|     N     |        1128        |
|    NEU    |         94         |
|     P     |        522         |

##### Clasificación en base a Heuristica
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |        155         |
|     N     |        1128        |
|    NEU    |         93         |
|     P     |        523         |

<!-- ###################################################################### -->

[Corpus de la TASS-2017]: http://www.sepln.org/workshops/tass/2017/#datasets
[NLTK]: http://www.nltk.org/
[Papers_2016]: http://ceur-ws.org/Vol-1702/

<!-- Vectorizadores -->

[CountVectorizer]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
[TfidfVectorizer]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

<!-- Clasificadores -->

[LinearSVC]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
[LogisticRegression]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
[RandomForestClassifier]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
