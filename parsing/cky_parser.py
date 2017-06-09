from collections import defaultdict
from nltk.tree import Tree  # http://www.nltk.org/_modules/nltk/tree.html
from math import log

# import pprint

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
        # type(grammar) = nltk.grammar.PCFG
        self.start = grammar.start()

        # Ej: { (pescado,) :  {'Noun': 0.1} }
        #     { (gato,) : {'Noun': 0.9} }
        productions_check = defaultdict(lambda: defaultdict(float))

        for prod in grammar.productions():
            # type(prod): <class 'nltk.grammar.ProbabilisticProduction'>
            # type(prod.lhs): <class 'nltk.grammar.Nonterminal'>
            # type(prod.rhs): <class 'tuple'>
            #   Cada elemento de prod.rhs() es del tipo:
            #       <class 'nltk.grammar.Nonterminal'>
            # type(prod.prob()): <class 'float'>
            lhs = str(prod.lhs())  # Left hand side
            # prod.rhs() es una tupla de non-terminals => Las pasamos a str
            # porque las palabras son str, para su posterior check.
            rhs = tuple([str(nt) for nt in prod.rhs()])  # Right hand side
            probability = prod.prob()

            productions_check[rhs][lhs] = probability

        self.productions_check = dict(productions_check)

    def parse(self, sent):
        """
        Parse a sequence of terminals.

        sent -- the sequence of terminals.
        """
        productions_check = self.productions_check
        n = len(sent)  # sent = x_1, x_2, x_3, ..., x_n

        # { (i, j) : {str : prob} }
        self._pi = pi = {}
        # { (i, j) : {str : Tree.fromstring(...)} }
        self._bp = bp = {}

        # Inicializacion
        for i in range(1, n+1):  # [1 ... n]
            x_i = sent[i-1]
            pi[i, i] = defaultdict()
            bp[i, i] = defaultdict()
            for X, probability in productions_check[(x_i,)].items():
                pi[i, i][X] = log2Extended(probability)
                bp[i, i][X] = Tree(X, [x_i])

        # Algoritmo
        for l in range(1, n):  # [1 ... n-1]
            for i in range(1, n-l+1):  # [1 ... n-l]
                j = i + l
                pi[i, j] = defaultdict()
                bp[i, j] = defaultdict()
                for s in range(i, j):  # [i ... j-1]
                    for Y, prob_Y in pi[i, s].items():
                        for Z, prob_Z in pi[s+1, j].items():
                            if (Y, Z) in productions_check.keys():
                                for X, qXYZ in productions_check[Y, Z].items():
                                    new_log_prob = log2Extended(qXYZ)
                                    new_log_prob += prob_Y
                                    new_log_prob += prob_Z

                                    # new_lp : new_log_prob
                                    # Si pi[i, j][X] no esta => poner new_lp
                                    # Si pi[i, j][X] no esta y new_lp es mejor
                                    #                   => poner new_lp
                                    # (y tambien bp)
                                    check_val = pi[i, j].get(X, float('-inf'))
                                    if check_val < new_log_prob:
                                        pi[i, j][X] = new_log_prob
                                        t1 = bp[i, s][Y]
                                        t2 = bp[s+1, j][Z]
                                        bp[i, j][X] = Tree(X, [t1, t2])

        # Output
        output_pi = pi[1, n].get(str(self.start), float("-inf"))
        output_bp = bp[1, n].get(str(self.start), None)

        return output_pi, output_bp
