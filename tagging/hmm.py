from math import log
from collections import defaultdict


def addMarkers(tagging, n):
    """
    Agrega a un tagging:
            * n-1 marcadores <s> al comienzo y
            * 1 marcadores </s> al final.
    """
    # Añadimos marcadores de comienzo y fin de tagging.
    tagging = ["<s>"]*(n-1) + tagging + ["</s>"]

    return tagging


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
                e_probability = hmm.out_prob(word, tag)
                for prev_tags, (log_prob, list_tags) in pi[k-1].items():
                    q_probability = hmm.trans_prob(tag, prev_tags)
                    # Analizo los No-Zeros
                    if q_probability > 0.0:
                        log_prob += log2Extended(q_probability) + log2Extended(e_probability)
                        new_list_tags = list_tags + [tag]
                        prev_tags = (prev_tags + (tag,))[1:]

                        # Bucamos el tag, que de el maximo
                        if prev_tags not in pi[k-1] or log_prob > pi[k-1][prev_tags][0]:
                            pi[k][prev_tags] = (log_prob, new_list_tags)

        # Devolver
        max_log_prob = float("-inf")
        my_tagging = []
        for prev_tags, (log_prob, list_tags) in pi[m].items():
            q_probability = hmm.trans_prob("</s>", prev_tags)
            log_prob += log2Extended(q_probability)
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                my_tagging = list_tags

        return my_tagging


class MLHMM(HMM):

    def __init__(self, n, tagged_sents, addone=True):
        """
        n -- order of the model.
        tagged_sents -- training sentences, each one being a list of pairs.
        addone -- whether to use addone smoothing (default: True).
        """
        self.n = n
        self.tagged_sents = tagged_sents
        self.addone = addone

    def tcount(self, tokens):
        """
        Count for an n-gram or (n-1)-gram of tags.

        tokens -- the n-gram or (n-1)-gram tuple of tags.
        """
        pass

    def unknown(self, w):
        """
        Check if a word is unknown for the model.

        w -- the word.
        """

    """
       Todos los métodos de HMM.
    """
    pass