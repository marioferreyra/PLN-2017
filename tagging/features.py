from collections import namedtuple
# http://stackoverflow.com/questions/2970608/what-are-named-tuples-in-python

from featureforge.feature import Feature


# sent -- the whole sentence.
# prev_tags -- a tuple with the n previous tags.
# i -- the position to be tagged.
History = namedtuple('History', 'sent prev_tags i')


def word_lower(h):
    """
    Feature: current lowercased word.
    Feature: La palabra actual en minúsculas.

    h -- a history.
    """
    sent = h.sent
    i = h.i

    return sent[i].lower()


def word_istitle(h):
    """
    Feature: La palabra actual empieza en mayúsculas.

    h -- a history.
    """
    sent = h.sent
    i = h.i

    return sent[i].istitle()


def word_isupper(h):
    """
    Feature: La palabra actual está en mayúsculas.

    h -- a history.
    """
    sent = h.sent
    i = h.i

    return sent[i].isupper()


def word_isdigit(h):
    """
    Feature: La palabra actual es un número.

    h -- a history.
    """
    sent = h.sent
    i = h.i

    return sent[i].isdigit()


def prev_tags(h):
    """
    Feature: La tupla de los tags previos.

    h -- a history.
    """
    prev_tags = h.prev_tags

    return prev_tags


class NPrevTags(Feature):

    def __init__(self, n):
        """
        Feature: n previous tags tuple.
        Feature: La tupla de los últimos n tags.

        n -- number of previous tags to consider.
        """
        self.n = n

    def _evaluate(self, h):
        """
        n previous tags tuple.

        h -- a history.
        """
        n = self.n
        prev_tags = h.prev_tags

        return prev_tags[-n:]


class PrevWord(Feature):

    def __init__(self, f):
        """
        Feature: the feature f applied to the previous word.
        Feature: Dado un feature f, aplicarlo sobre la palabra anterior
                 en lugar de la actual.

        f -- the feature.
        """
        self.f = f

    def _evaluate(self, h):
        """
        Apply the feature to the previous word in the history.

        h -- the history.
        """
        sent = h.sent
        prev_tags = h.prev_tags
        i = h.i

        result = "BOS"  # Comienzo de oracion
        if i != 0:
            result = str(self.f(History(sent, prev_tags, i-1)))

        return result
