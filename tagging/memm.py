# http://feature-forge.readthedocs.io/en/latest/feature_evaluation.html
from featureforge.vectorizer import Vectorizer  # Vectorizador

# http://scikit-learn.org/stable/modules/generated/
# sklearn.pipeline.Pipeline.html
from sklearn.pipeline import Pipeline  # Pipeline

from sklearn.svm import LinearSVC

# --> Features <--
from tagging.features import History
# Features básicos
from tagging.features import word_lower, word_istitle
from tagging.features import word_isupper, word_isdigit
# Features paramétricos
from tagging.features import NPrevTags, PrevWord


class MEMM:

    def __init__(self, n, tagged_sents, classifier=LinearSVC()):
        """
        n -- order of the model.
        tagged_sents -- list of sentences, each one being a list of pairs.
        """
        self.n = n
        self.vocabulary = vocabulary = set()  # Conjunto de palabras conocidas

        # Formamos el conjunto de tags
        for sent_tagging in tagged_sents:
            for word, tag in sent_tagging:
                vocabulary.add(word)

        # Formamos el vector de features
        basic_features = [word_lower, word_istitle, word_isupper, word_isdigit]

        parametric_features = [NPrevTags(i) for i in range(1, n)]
        parametric_features += [PrevWord(f) for f in basic_features]

        features = basic_features + parametric_features

        v = Vectorizer(features)  # Se instancia con una lista de features
        c = classifier  # Clasificador
        pipe = Pipeline([("vectorizador", v), ("clasificador", c)])

        # Datos de entrenamiento (training_data)
        # Los features trabajan sobre las History
        histories = self.sents_histories(tagged_sents)  # Lista de histories

        # Objetivos de entrenamiento (training_targets)
        tags = self.sents_tags(tagged_sents)  # Lista de tags

        pipe.fit(histories, tags)
        self.pipeline = pipe

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
        tags = [tag for word, tag in tagged_sent]  # lista de tags

        tags = ["<s>"]*(n-1) + tags  # Lista de tags para los casos de borde

        m = len(words)  # Largo de la lista

        my_histories = []
        for i in range(m):
            prev_tags = tuple(tags[i: i+n-1])  # n tags previos a la posicion i
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
        # self.pipeline.predict(X) --> me devuelve una lista de un elemento
        return self.pipeline.predict(X)[0]

    def tag(self, sent):
        """
        Tag a sentence.

        sent -- the sentence.
        """
        n = self.n
        m = len(sent)  # Largo de la oracion

        prev_tags = ("<s>",)*(n-1)
        history = History(sent, prev_tags, 0)
        my_tagging = [self.tag_history(history)]

        for i in range(1, m):
            prev_tags = (prev_tags + (my_tagging[i-1],))[1:]
            history = History(sent, prev_tags, i)
            my_tagging += [self.tag_history(history)]

        return my_tagging
