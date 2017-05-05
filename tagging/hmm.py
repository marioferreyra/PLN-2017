from math import log


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
        return self.trans[prev_tags][tag]

    def out_prob(self, word, tag):
        """
        Probability of a word given a tag.

        word -- the word.
        tag -- the tag.
        """
        return self.out[tag][word]

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
        n = self.n
        # m = len(x)

        # assert len(x) == len(y)

        # probability = 1
        # for i in range(0, m):
        #     probability *= self.out_prob(x[i], y[i])

        # return probability
        probability = 1

        tag_prob = self.tag_prob(y)

        for i in range(0, len(x)):
            probability *= self.out_prob(x[i], y[i])

        return probability * tag_prob

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
            value = self.trans_prob(tag, tuple(prev_tags))
            probability += log2Extended(value)

        return probability

    def log_prob(self, x, y):
        """
        Joint log-probability of a sentence and its tagging.

        x -- sentence.
        y -- tagging.
        """
        n = self.n
        m = len(x)

        tag_prob = self.tag_prob(y)

        probability = 0
        for i in range(0, m):
            value = self.out_prob(x[i], y[i])
            probability += log2Extended(value)

        return probability
        


    def tag(self, sent):
        """
        Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        n = self.n

        # probability = 1
        # for i in


class ViterbiTagger:

    def __init__(self, hmm):
        """
        hmm -- the HMM.
        """
        pass

    def tag(self, sent):
        """
        Returns the most probable tagging for a sentence.

        sent -- the sentence.
        """
        pass
