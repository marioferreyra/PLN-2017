# http://www.nltk.org/_modules/nltk/grammar.html
# http://www.nltk.org/_modules/nltk/tree.html
from collections import defaultdict
from nltk.tree import Tree
from nltk.grammar import Nonterminal, ProbabilisticProduction, PCFG
from parsing.util import unlexicalize, lexicalize
from parsing.cky_parser import CKYParser


class UPCFG:
    """
    Unlexicalized PCFG.
    """

    def __init__(self, parsed_sents, start='sentence', horzMarkov=None):
        """
        parsed_sents -- list of training trees.
        """
        # { A -> B : count(A -> B) }
        productions_counts = defaultdict(int)
        # { A : count(A) }
        lhs_count = defaultdict(int)  # left_hand_side_count
        # {A -> B : count(A -> B) / count(A)}
        probability_ML = defaultdict(float)

        self.start = start  # Para la gramatica del parser CKY
        self.prods = []  # Lista de producciones

        # Hacemos una copia de t porque al hacer el unlexicalize, este me
        # modifica el arbol
        # Original: unlexicalize_tree = [unlexicalize(t) for t in parsed_sents]
        unlex_sents = [unlexicalize(t.copy(deep=True)) for t in parsed_sents]

        for t in unlex_sents:
            t.chomsky_normal_form(horzMarkov=horzMarkov)
            t.collapse_unary(collapsePOS=True, collapseRoot=True)
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

        # type(PCFG(...)) = <class 'nltk.grammar.PCFG'>
        # PCFG(start, productions)
        #       type(start): Nonterminal
        #       type(productions): list(Production)
        grammar = PCFG(Nonterminal(start), self.prods)
        self.my_parser = CKYParser(grammar)

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
        words, tags = zip(*tagged_sent)

        log_probability, tree = self.my_parser.parse(tags)

        # Si no se puede parsear con CKY, entonces devolvemos el Flat
        if tree is None:
            return Tree(self.start, [Tree(t, [w]) for w, t in tagged_sent])

        tree.un_chomsky_normal_form()

        return lexicalize(tree, words)
