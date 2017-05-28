# http://www.nltk.org/_modules/nltk/grammar.html
# http://www.nltk.org/_modules/nltk/tree.html
from collections import defaultdict
from nltk.tree import Tree
from nltk.grammar import Nonterminal, ProbabilisticProduction
from parsing.util import unlexicalize, lexicalize

import pprint

# ENUNCIADO
# Implementar una UPCFG, una PCFG cuyas reglas y probabilidades se obtienen a
# partir de un corpus de entrenamiento.

# Deslexicalizar completamente la PCFG: en las reglas, reemplazar todas las
# entradas léxicas por su POS tag. Luego, el parser también debe ignorar las
# entradas léxicas y usar la oración de POS tags para parsear.

# Entrenar y evaluar la UPCFG para todas las oraciones de largo menor o igual a 20.
# Reportar resultados y tiempos de evaluación en el README.


class UPCFG:
    """
    Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence'):
        """
        parsed_sents -- list of training trees.
        """
        # { A -> B : count(A -> B) }
        productions_counts = productions_counts = defaultdict(int)
        # { A : count(A) }
        left_hand_side_count = lhs_count = defaultdict(int)
        # {A -> B : count(A -> B) / count(A)}
        probability_ML = defaultdict(float)

        self.prods = [] # Lista de producciones

        # Hacemos una copia de t porque al hacer el unlexicalize, este me
        # modifica el arbol
        # Original: unlexicalize_tree = [unlexicalize(t) for t in parsed_sents]
        unlex_sents = [unlexicalize(t.copy(deep=True)) for t in parsed_sents]

        for t in unlex_sents:
            for prod in t.productions():
                # type(prod): <class 'nltk.grammar.Production'>
                # type(prod.lhs): <class 'nltk.grammar.Nonterminal'>
                # type(prod.rhs): <class 'tuple'>
                #   Cada elemento de prod.rhs() es del tipo:
                #       <class 'nltk.grammar.Nonterminal'>
                productions_counts[prod] += 1
                lhs_count[prod.lhs()] += 1

        for prod, count in productions_counts.items():
            # type(prod): <class 'nltk.grammar.Production'>
            # type(count): int
            # type(prod.lhs): <class 'nltk.grammar.Nonterminal'>
            probability_ML[prod] = float(count) / lhs_count.get(prod.lhs(), 0)

        for production, q_ML in probability_ML.items():
            # type(production): <class 'nltk.grammar.Production'>
            # type(q_ML): float
            # type(prod.lhs): <class 'nltk.grammar.Nonterminal'>
            # type(prod.rhs): <class 'tuple'>
            self.prods += [ProbabilisticProduction(production.lhs(),
                                                   production.rhs(),
                                                   prob=q_ML)]
            # Cada elemento de self.prods es del tipo:
            #     <class 'nltk.grammar.ProbabilisticProduction'>

    def productions(self):
        """
        Returns the list of UPCFG probabilistic productions.
        """
        return self.prods

    def parse(self, tagged_sent):
        """
        Parse a tagged sentence.

        tagged_sent -- the tagged sentence (a list of pairs (word, tag)).
        """
        pass




# t = Tree.fromstring(
#             """
#                 (S
#                     (NP (Det el) (Noun gato))
#                     (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
#                 )
#             """)

# t2 = t.copy(deep=True)

# model = UPCFG([t])