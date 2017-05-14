from math import log
from collections import defaultdict


def addMarkers(tagging, n):
    """
    Agrega a un tagging:
            * n-1 marcadores <s> al comienzo y
            * 1 marcadores </s> al final.
    """
    # Añadimos marcadores de comienzo y fin de tagging.
    return ["<s>"]*(n-1) + tagging + ["</s>"]


def log2Extended(x):
    """
    Funcion que calcula logaritmo en base 2.
        Si x = 0 ==> return -infinito
        Si x != 0 ==> return log2(x)
    """
    # Al calcular log2(0) daria error al no estar definido
    # Entonces le sumamos -inf, porque: lim log(x) x-->0 = -inf
    if x == 0:
        return float("-inf")
    else:
        return log(x, 2)


class HMM:

    def __init__(self, n, tagset, trans, out):
        """
        n -- n-gram size.
        tagset -- set of tags.
        trans -- transition probabilities dictionary.
        out -- output probabilities dictionary.
        """
        self.n = n
        self.tagset = tagset
        self.trans = trans
        self.out = out

    def tagset(self):
        """
        Returns the set of tags.
        """
        return self.tagset

    def trans_prob(self, tag, prev_tags):
        """
        Probability of a tag.

            trans_prob(tag, prev_tags) = q(tag | prev_tags)

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).
        """
        # return self.trans[prev_tags][tag]
        return self.trans.get(prev_tags, {}).get(tag, 0.0)

    def out_prob(self, word, tag):
        """
        Probability of a word given a tag.

            out_prob(word, tag) = e(word | tag)

        word -- the word.
        tag -- the tag.
        """
        # return self.out[tag][word]
        return self.out.get(tag, {}).get(word, 0.0)

    def tag_prob(self, y):
        """
        Probability of a tagging.
        Warning: subject to underflow problems.

        y -- tagging.
        """
        n = self.n
        y = addMarkers(y, n)
        m = len(y)  # Logitud del tagging

        probability = 1
        for i in range(n-1, m):
            tag = y[i]
            prev_tags = y[i-n+1:i]
            probability *= self.trans_prob(tag, tuple(prev_tags))

        return probability

    def prob(self, x, y):
        """
        Joint probability of a sentence and its tagging.
        Warning: subject to underflow problems.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        q_probability = self.tag_prob(y)

        e_probability = 1
        for word, tag in zip(x, y):
            e_probability *= self.out_prob(word, tag)

        probability = q_probability * e_probability

        return probability

    def tag_log_prob(self, y):
        """
        Log-probability of a tagging.

        y -- tagging.
        """
        n = self.n
        y = addMarkers(y, n)
        m = len(y)  # Logitud del tagging

        probability = 0  # Se inicializa en 0 por la sumatoria de logaritmos
        for i in range(n-1, m):
            tag = y[i]  # wi : primera palabra
            prev_tags = y[i-n+1:i]  # Markov Assumption: wi-k ... wi-1
            probability += log2Extended(self.trans_prob(tag, tuple(prev_tags)))

        return probability

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        assert len(x) == len(y)

        q_probability = self.tag_log_prob(y)

        e_probability = 0
        for word, tag in zip(x, y):
            e_probability += log2Extended(self.out_prob(word, tag))

        probability = q_probability + e_probability

        return probability

    def tag(self, sent):
        """
        Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        return ViterbiTagger(self).tag(sent)


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        self.hmm = hmm

    def tag(self, sent):
        """
        Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        hmm = self.hmm  # Hidden Markov Models
        n = hmm.n
        m = len(sent) # Tamaño de la oracion

        tagset = hmm.tagset # Conjunto de tags

        # pi = { key : { prev_tags : (log_prob, list_tags) } }
        self._pi = pi = defaultdict(lambda: defaultdict(tuple))

        # Inicializacion
        pi[0][("<s>",)*(n-1)] = (log2Extended(1.0), [])

        # Recursion
        for k in range(1, m+1):  # 1 ... m
            word = sent[k-1]
            for tag in tagset:
                # e(word | tag)
                e_probability = hmm.out_prob(word, tag)
                for prev_tags, (log_prob, list_tags) in pi[k-1].items():
                    # q(tag | prev_tags)
                    q_probability = hmm.trans_prob(tag, prev_tags)
                    # Analizo los No-Zeros
                    if q_probability * e_probability > 0.0:
                        # new_log_prob = PI(k-1, prev_tags) *
                        #                q(tag | prev_tags) *
                        #                e(word | tag)
                        new_log_prob = log_prob + log2Extended(q_probability) + log2Extended(e_probability)
                        new_list_tags = list_tags + [tag]
                        new_prev_tags = (prev_tags + (tag,))[1:]

                        # Bucamos el tag, que de el maximo
                        # Con k-1 salta el assert del eval
                        if (new_prev_tags not in pi[k]) or (new_log_prob > pi[k][new_prev_tags][0]):
                            pi[k][new_prev_tags] = (new_log_prob, new_list_tags)

        # Devolver
        max_log_prob = float("-inf")
        my_tagging = []
        for prev_tags, (log_prob, list_tags) in pi[m].items():
            # q(STOP | prev_tags)
            q_probability = hmm.trans_prob("</s>", prev_tags)
            # new_log_prob = PI(m, prev_tags) * q(tag | prev_tags)
            new_log_prob = log_prob + log2Extended(q_probability)
            if new_log_prob > max_log_prob:
                max_log_prob = new_log_prob
                my_tagging = list_tags

        # Convertimos todos los defaultdict a dict para solucionar el problema
        # de que pickle.dump no puede guardar funciones lambda
        self._pi = dict(pi)

        return my_tagging

# FALLA: El train para n > 2 ->_<- SOLUCION: Arreglado con el condicional de abajo
# FALLA: El eval con n>=1, salta el assert
# El problema esta aca
class MLHMM(HMM):
    """
    Heredamos de HMM para poder usar todos sus metodos.
    """
    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.addone = addone

        self.tagset = tagset = set()  # Conjuntos de tags
        self.vocabulary = vocabulary = set()  # Conjunto de palabras conocidas

        # { tag : count } --> tag es una tupla
        self.tag_counts = tag_counts = defaultdict(int)
        # { tag : {word : apariciones} }
        self.count_paired = count_paired = defaultdict(lambda: defaultdict(int))
        # { prev_tags : {tag : prob} } --> prev_tags es una tupla
        self.trans = trans = defaultdict(lambda: defaultdict(float))
        # { tag : {word : prob} }
        self.out = out = defaultdict(lambda: defaultdict(float))

        # Formamos el conjunto de tags y count_paired
        for tag_sent in tagged_sents:
            for word, tag in tag_sent:
                tagset.add(tag)
                vocabulary.add(word)
                count_paired[tag][word] += 1

        # Iteramos sobre cada oracion taggeada del conjunto de oraciones taggeada
        for tag_sent in tagged_sents:
            # words, tags = zip(*tag_sent)
            # Comentamos la linea de arriba porque en el train.py me tira error
            # -->ValueError: not enough values to unpack (expected 2, got 0)<--
            # Que significa que python esperaba que hubiera dos valores de
            # retorno de zip (), pero no había ninguno.

            words = [word for word, tag in tag_sent]
            tags = [tag for word, tag in tag_sent]
            words = addMarkers(words, n)
            tags = addMarkers(tags, n)

            for i in range(len(tags)-n+1):
                ngram = tuple(tags[i : i+n])
                tag_counts[ngram] += 1
                tag_counts[ngram[:-1]] += 1  # Todos menos el ultimo

        # Calculamos trans_prob:
        #                       count(prev_tags tag)
        # q(tag | prev_tags) = ----------------------
        #                         count(prev_tags)
        for tags in tag_counts.keys():
            if len(tags) == n:
                tag = tags[-1]  # El ultimo tag
                prev_tags = tags[:-1]  # Todos los tags previos a tag
                c_tags = tag_counts[tags]
                c_prevtags = tag_counts[prev_tags]
                trans[prev_tags][tag] = float(c_tags) / c_prevtags

        # Calculamos out_prob:
        #                  count(tag --> word)
        # e(word | tag) = ---------------------
        #                      count(tag)
        for tag_sent in tagged_sents:
            for word, tag in tag_sent:
                c_tagword = count_paired[tag][word]  # count(tag --> word)
                c_tag = tag_counts[(tag,)]
                if c_tag == 0:
                    out[tag][word] = 0.0
                else:
                    out[tag][word] = float(c_tagword) / c_tag

        # Convertimos todos los defaultdict a dict para solucionar el problema
        # de que pickle.dump no puede guardar funciones lambda
        self.count_paired = dict(count_paired)
        self.trans = dict(trans)
        self.out = dict(out)

    def tcount(self, tokens):
        """
        Count for an n-gram or (n-1)-gram of tags.

        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        return self.tag_counts.get(tokens, 0)

    def unknown(self, w):
        """
        Check if a word is unknown for the model.

        w -- the word.
        """
        return w not in self.vocabulary

    def V(self):
        """
        Size of the vocabulary of words.
        """
        return len(self.vocabulary)

    def T(self):
        """
        Size of the vocabulary of tags.
        """
        return len(self.tagset) + 1  # El "+ 1" es por el marcador </s>

    def trans_prob(self, tag, prev_tags):
        """
        Probability of a tag.

        tag -- the tag.
        prev_tags -- tuple with the previous n-1 tags (optional only if n = 1).

                    trans_prob(tag, prev_tags) = q(tag | prev_tags)

        si es addone:
                                      count(prev_tags tag) + 1
                q(tag | prev_tags) = --------------------------
                                       count(prev_tags) + V

                Donde V = |tagset|
        """
        if self.addone:
            tags = prev_tags + (tag,)
            c_tags_1 = float(self.tcount(tags) + 1)
            c_prevtags_T = self.tcount(prev_tags) + self.T()
            probability = c_tags_1 / c_prevtags_T
        else:
            probability = self.trans.get(prev_tags, {}).get(tag, 0.0)

        return probability

    def out_prob(self, word, tag):
        """
        Probability of a word given a tag.

        word -- the word.
        tag -- the tag.

                    out_prob(word, tag) = e(word | tag)

        Para P(x|y) hacer lo siguiente:
        1. Si la palabra es desconocida, devolver 1 / V a donde V es el
           tamaño del vocabulario.
        2. Si la palabra es conocida, devolver la Maximum Likelihood
           (sin addone ni nada, hasta puede dar cero).
        """
        if self.addone and self.unknown(word):
            probability = 1.0 / self.V()
        else:
            probability = self.out.get(tag, {}).get(word, 0.0)

        return probability
