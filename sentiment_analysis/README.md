Trabajo Práctico 4 - Sentiment Analysis at Tweet
================================================

Mario E. Ferreyra
=================

Lo que se propone es la evaluación de sistemas de clasificación de polaridad de tweets en español.
Para ello se utilizaran los [Corpus] de la [TASS-2017] correspondientes a la Task 1.

Para esta tarea nos basamos en los [Papers de la TASS-2016].

El sentimiento de los tweets del corpus tienen 4 niveles de polaridad:
* **NONE**: No se sabe la polaridad del tweet.
* **N**: La polaridad del tweet Negativa.
* **NEU**: La polaridad del tweet es Neutra.
* **P**: La polaridad del tweet es Positiva.

Para la tarea se utilizaron los siguientes Corpus:
* **Corpus de Entrenamiento**: compuesto de 1008 tweets
* **Corpus de Desarrollo**: compuesto de 506 tweets (para la evaluación del modelo)

## Descripción General
Lo primero que se hace es un pre-procesamiento de los tweets, es decir, una limpieza del contenido de los mismos:
* Separar el contenido del tweet por palabras y expresiones (tokenizar).
* Reemplazar las letras con tildes por sus respectivas versiones sin tildes.
* Convertir el texto a minúscula.
* Cambiar todas las palabras positivas, palabras negativas, emoticones positivos y emoticones negativos por "*positiveword*", "*negativeword*", "*positiveemoticon*" y "*negativeemoticon*" respectivamente.
* Eliminar URL’s, signos de puntuación, nombres de usuarios (i.e. los que comienzan con @. e.g. @user) y caracteres no alfanuméricos.
* Transformar los hashtags en palabras, eliminando su primer carácter (i.e. el sı́mbolo #)
* Reducción de elongaciones, por ejemplo:
    * "*hooollaaaa*" --> "*hola*"
    * "*jjjaaajjaaaja*" --> "*jajaja*"
* Cambiar las expresiones de risas por la palabra "*risas*", por ejemplo:
    * "*jajaja*" --> "*risas*"
    * "*jeje*" --> "*risas*"
* Remover las Stopwords (palabras que son muy frecuentes pero que no aportan gran valor semántico, como artı́culos, pronombres, preposiciones, etc. e.g. de, po r, con).
* Aplicar a cada palabra el Stemming (proceso por el cual transformamos cada palabra en su raı́z), por ejemplo:
    * *maravilloso* --> *maravill*
    * *maravilla* --> *maravill*
    * *maravillarse* --> *maravill*

Después tokenizamos el texto usando [NLTK], para luego entrenar nuestro modelo usando el **Corpus de Entrenamiento** probando con distintas combinaciones de vectorizadores y clasificadores para luego evaluarlo usando el **Corpus de Desarrollo**.

Los vectorizadores que se usaron fueron los siguientes:
* [CountVectorizer]
* [TfidfVectorizer]

Los clasificadores que se usaron fueron los siguientes:
* [LinearSVC]
* [LogisticRegression]
* [RandomForestClassifier]


## Resultados
Una vez que los tweets fueron clasificados, evaluamos nuestro modelo usando el Corpus de Desarrollo, utilizando las siguientes métricas:
* Accuracy
* Macro-Precision
* Macro-Recall
* Macro-F1


Se analizaron un total de **506 Tweets**.

### Polaridades Golden
| Polaridad | Cantidad de Tweets |
|:---------:|:------------------:|
|   NONE    |         62         |
|     N     |        219         |
|    NEU    |         69         |
|     P     |        156         |

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"LinearSVC"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         56         | 32.14% (18/56)   | 29.03% (18/62)   | 30.51% |
|     N     |        228         | 57.02% (130/228) | 59.36% (130/219) | 58.17% |
|    NEU    |         48         | 18.75% (9/48)    | 13.04% (9/69)    | 15.38% |
|     P     |        174         | 48.85% (85/174)  | 54.49% (85/156)  | 51.52% |

* Accuracy: 47.83% (242/506)
* Macro-Precision: 39.19%
* Macro-Recall: 38.98%
* Macro-F1: 39.09%

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"LogisticRegression"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         33         | 36.36% (12/33)   | 19.35% (12/62)   | 25.26% |
|     N     |        263         | 55.13% (145/263) | 66.21% (145/219) | 60.17% |
|    NEU    |         26         | 15.38% (4/26)    | 5.80% (4/69)     | 8.42%  |
|     P     |        184         | 49.46% (91/184)  | 58.33% (91/156)  | 53.53% |

* Accuracy: 49.80% (252/506)
* Macro-Precision: 39.08%
* Macro-Recall: 37.42%
* Macro-F1: 38.24%

<!-- ###################################################################### -->

### Vectorizador *"CountVectorizer"* y Clasificador *"RandomForestClassifier"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         16         | 25.00% (4/16)    | 6.45% (4/62)     | 10.26% |
|     N     |        261         | 52.11% (136/261) | 62.10% (136/219) | 56.67% |
|    NEU    |         49         | 14.29% (7/49)    | 10.14% (7/69)    | 11.86% |
|     P     |        180         | 50.56% (91/180)  | 58.33% (91/156)  | 54.17% |

* Accuracy: 47.04% (238/506)
* Macro-Precision: 35.49%
* Macro-Recall: 34.26%
* Macro-F1: 34.86%

<!-- ###################################################################### -->
<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"LinearSVC"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         35         | 37.14% (13/35)   | 20.97% (13/62)   | 26.80% |
|     N     |        260         | 55.00% (143/260) | 65.30% (143/219) | 59.71% |
|    NEU    |         32         | 12.50% (4/32)    | 5.80% (4/69)     | 7.92%  |
|     P     |        179         | 48.60% (87/179)  | 55.77% (87/156)  | 51.94% |

* Accuracy: 48.81% (247/506)
* Macro-Precision: 38.31%
* Macro-Recall: 36.96%
* Macro-F1: 37.62%

<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"LogisticRegression"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         1          | 0.00% (0/1)      | 0.00% (0/62)     | 0.00%  |
|     N     |        324         | 52.78% (171/324) | 78.08% (171/219) | 62.98% |
|    NEU    |         1          | 100.00% (1/1)    | 1.45% (1/69)     | 2.86%  |
|     P     |        180         | 52.78% (95/180)  | 60.90% (95/156)  | 56.55% |

* Accuracy: 52.77% (267/506)
* Macro-Precision: 51.39%
* Macro-Recall: 35.11%
* Macro-F1: 41.72%

<!-- ###################################################################### -->

### Vectorizador *"TfidfVectorizer"* y Clasificador *"RandomForestClassifier"*
| Polaridad | Cantidad de Tweets |    Precision     |      Recall      |   F1   |
|:---------:|:------------------:|:----------------:|:----------------:|:------:|
|   NONE    |         30         | 16.67% (5/30)    | 8.06% (5/62)     | 10.87% |
|     N     |        286         | 53.15% (152/286) | 69.41% (152/219) | 60.20% |
|    NEU    |         18         | 16.67% (3/18)    | 4.35% (3/69)     | 6.90%  |
|     P     |        172         | 50.58% (87/172)  | 55.77% (87/156)  | 53.05% |

* Accuracy: 48.81% (247/506)
* Macro-Precision: 34.27%
* Macro-Recall: 34.40%
* Macro-F1: 34.33%


##### [Filminas de la Presentación de Sentiment Analysis]

<!-- ###################################################################### -->

[Corpus]: http://www.sepln.org/workshops/tass/2017/#datasets
[TASS-2017]: http://www.sepln.org/workshops/tass/2017/
[Papers de la TASS-2016]: http://ceur-ws.org/Vol-1702/
[NLTK]: http://www.nltk.org/
[Filminas de la Presentación de Sentiment Analysis]: https://github.com/marioferreyra/PLN-2017/blob/practico04/sentiment_analysis/filminas_sentiment_analysis_pln.pdf

<!-- Vectorizadores -->

[CountVectorizer]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
[TfidfVectorizer]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

<!-- Clasificadores -->

[LinearSVC]: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
[LogisticRegression]: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
[RandomForestClassifier]: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
