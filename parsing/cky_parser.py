from collections import defaultdict
from nltk.tree import Tree # http://www.nltk.org/_modules/nltk/tree.html
from math import log

import pprint

# Para imprimir de forma legible
# https://docs.python.org/3/library/pprint.html
# >>> import pprint
# >>> stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
# >>> pprint.pprint(stuff)

# https://stackoverflow.com/questions/34882246/nltk-grammar-from-productions
# http://www.nltk.org/_modules/nltk/grammar.html


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


class CKYParser:

    def __init__(self, grammar):
        """
        grammar -- a binarised NLTK PCFG.
        """
        self.grammar = grammar
        # Ej: {Noun : {(pescado,) : 0.1, (gato,) : 0.9}}
        self.productions = defaultdict(lambda: defaultdict(float))
        # Ej: { (pescado,) :  {'Noun': 0.1} }
        #     { (gato,) : {'Noun': 0.9} }

        self.productions_check = defaultdict(lambda: defaultdict(float))
        for prod in self.grammar.productions():
            # nt: non-terminal
            left_hand_side = lhs = str(prod.lhs())  # type(prod.lhs()) = nt
            right_hand_side = rhs = tuple([str(nt) for nt in prod.rhs()])
            probability = prod.prob()

            # self.productions[left_hand_side][right_hand_side] = probability
            self.productions[lhs][rhs] = probability
            self.productions_check[rhs][lhs] = probability

        # pprint.pprint(self.productions)
        # pprint.pprint(self.productions_check)

    def parse(self, sent):
        """
        Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        productions = self.productions
        productions_check = self.productions_check
        n = len(sent)  # sent = x_1, x_2, x_3, ..., x_n
                       #         0    1    2   ...   n-1

        # { (i, j) : {"Non-terminal" : prob} }
        self._pi = pi = defaultdict(lambda: defaultdict())
        # { (i, j) : {"Non-terminal" : Tree.fromstring(...)} }
        self._bp = bp = defaultdict(lambda: defaultdict())

        # Inicializacion
        for i in range(1, n+1):  # [1 ... n]
            x_i = (sent[i-1],)
            for X, q in productions_check[x_i].items():
                pi[(i, i)][X] = log2Extended(q)
                bp[(i, i)][X] = Tree(X, [x_i])

        # pprint.pprint(pi)
        # pprint.pprint(bp)

        # Algoritmo
        for l in range(1, n):  # [1 ... n-1]
            for i in range(1, n-l+1):  # [1 ... n-l]
                j = i + l
                



# PARA TEST
from nltk.grammar import PCFG

grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]
                NP -> Det Noun          [0.6]
                NP -> Noun Adj          [0.4]
                VP -> Verb NP           [1.0]
                Det -> 'el'             [1.0]
                Noun -> 'gato'          [0.9]
                Noun -> 'pescado'       [0.1]
                Verb -> 'come'          [1.0]
                Adj -> 'crudo'          [1.0]
            """)

cky = CKYParser(grammar)

sent = ['el', 'gato', 'come', 'pescado', 'crudo']

cky.parse(sent)