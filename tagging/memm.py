# Vectorizador
# http://feature-forge.readthedocs.io/en/latest/feature_evaluation.html
from featureforge.vectorizer import Vectorizer

# Clasificador de máxima entropía
# http://scikit-learn.org/stable/modules/generated/
# sklearn.linear_model.LogisticRegression.html
from sklearn.linear_model import LogisticRegression

# Pipeline
# http://scikit-learn.org/stable/modules/generated/
# sklearn.pipeline.Pipeline.html
from sklearn.pipeline import Pipeline

# Features
from tagging.features import History
# Features básicos
from tagging.features import word_lower, word_istitle, word_isupper
from tagging.features import word_isdigit, prev_tags
# Features paramétricos
from tagging.features import NPrevTags, PrevWord


class MEMM:

    def __init__(self, n, tagged_sents):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.vocabulary = vocabulary = set()  # Conjunto de palabras conocidas

        print("-->Computar vocabulario")
        # Formamos el conjunto de tags
        for sent_tagging in tagged_sents:
            for word, tag in sent_tagging:
                vocabulary.add(word)
        print("Vocabulario computado")

        # Formamos el vector de features
        print("-->Cargar features")
        basic_features = [word_lower, word_istitle, word_isupper]
        basic_features += [word_isdigit, prev_tags]

        parametric_features = [NPrevTags(i) for i in range(n)]
        parametric_features += [PrevWord(f) for f in basic_features]

        features = basic_features + parametric_features
        print("Features cargados")

        print("-->Comienzo del Pipeline")
        print("    -->Cargar Vector de features y Clasificador")
        v = Vectorizer(features)  # Se instancia con una lista de features
        c = LogisticRegression()  # Clasificador de máxima entropía
        print("    Vector de features y Clasificador cargados")
        print("    -->Cargar Pipeline")
        pipe = Pipeline([("vectorizador", v), ("clasificador", c)])
        print("    Pipeline cargado")

        # Datos de entrenamiento (training_data)
        # Los features trabajan sobre las History
        print("    -->Cargar datos de entrenamiento")
        X = self.sents_histories(tagged_sents)
        print("    Datos de entrenamiento cargados")

        # Objetivos de entrenamiento (training_targets)
        print("    -->Cargar objetivos de entrenamiento")
        y = self.sents_tags(tagged_sents)  # Lista de tags
        print("    Objetivos de entrenamiento cargados")

        print("    -->Comienzo del fit")
        # CUANDO HAGO EL TRAIN SE QUEDA COLGADO EN EL FIT
        self.pipeline = pipe.fit(X, y) # TODAVIA NOSE PORQUE ANDA, NO ENTIENDO LO DEL y
        print("    Fit finalizado")
        print("Pipeline Finalizado")

    def unknown(self, w):
        """
        Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.vocabulary

    def sent_histories(self, tagged_sent):
        """
        Iterator over the histories of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        # Recordar:
        # History = namedtuple('History', 'sent prev_tags i')
                    # sent -- the whole sentence.
                    # prev_tags -- a tuple with the n previous tags.
                    # i -- the position to be tagged.
        n = self.n
        words = [word for word, tag in tagged_sent]  # W[1:n] = sent
        tags = [tag for word, tag in tagged_sent] # lista de tags

        tags = ["<s>"]*(n-1) + tags # Lista de tags con para casos de borde

        m = len(words) # Largo de la lista

        my_histories = []
        for i in range(m):
            prev_tags = tuple(tags[i: i+n-1]) # n tags previos a la posicion i
            # print(i, prev_tags)
            my_histories += [History(words, prev_tags, i)]

        return my_histories

    def sents_histories(self, tagged_sents):
        """
        Iterator over the histories of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        my_histories = []
        for sent_tagging in tagged_sents:
            my_histories += self.sent_histories(sent_tagging)

        return my_histories

    def sent_tags(self, tagged_sent):
        """
        Iterator over the tags of a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        return [tag for word, tag in tagged_sent]

    def sents_tags(self, tagged_sents):
        """
        Iterator over the tags of a corpus.

        tagged_sents -- the corpus (a list of sentences)
        """
        my_iter = []
        for sent_tagging in tagged_sents:
            my_iter += self.sent_tags(sent_tagging)

        return my_iter

    def tag_history(self, h):
        """
        Tag a history.

        h -- the history.
        """
        X = [h]  # Tiene que ser iterable
        # self.pipeline.predict(X) Me devuelve una lista de un elemento
        return self.pipeline.predict(X)[0]


    def tag(self, sent):
        """
        Tag a sentence.

        sent -- the sentence.
        """
        n = self.n
        m = len(sent) # Largo de la oracion

        prev_tags = ("<s>",)*(n-1)
        history = History(sent, prev_tags, 0)
        my_tagging = [self.tag_history(history)]

        for i in range(1, m):
            prev_tags = prev_tags[1:] + tuple(my_tagging)
            history = History(sent, prev_tags, i)
            my_tagging += [self.tag_history(history)]

        return my_tagging
