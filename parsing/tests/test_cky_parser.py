# https://docs.python.org/3/library/unittest.html
from unittest import TestCase
from math import log2

from nltk.tree import Tree
from nltk.grammar import PCFG
# from nltk.draw import draw_trees

from parsing.cky_parser import CKYParser


class TestCKYParser(TestCase):

    def test_parse(self):
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

        parser = CKYParser(grammar)

        lp, t = parser.parse('el gato come pescado crudo'.split())

        # check chart
        pi = {
            (1, 1): {'Det': log2(1.0)},
            (2, 2): {'Noun': log2(0.9)},
            (3, 3): {'Verb': log2(1.0)},
            (4, 4): {'Noun': log2(0.1)},
            (5, 5): {'Adj': log2(1.0)},

            (1, 2): {'NP': log2(0.6 * 1.0 * 0.9)},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': log2(0.4 * 0.1 * 1.0)},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S':
                     log2(1.0) +  # rule S -> NP VP
                     log2(0.6 * 1.0 * 0.9) +  # left part
                     log2(1.0) + log2(1.0) + log2(0.4 * 0.1 * 1.0)},  # right part
        }

        self.assertEqualPi(parser._pi, pi)

        # # check partial results
        bp = {
            (1, 1): {'Det': Tree.fromstring("(Det el)")},
            (2, 2): {'Noun': Tree.fromstring("(Noun gato)")},
            (3, 3): {'Verb': Tree.fromstring("(Verb come)")},
            (4, 4): {'Noun': Tree.fromstring("(Noun pescado)")},
            (5, 5): {'Adj': Tree.fromstring("(Adj crudo)")},

            (1, 2): {'NP': Tree.fromstring("(NP (Det el) (Noun gato))")},
            (2, 3): {},
            (3, 4): {},
            (4, 5): {'NP': Tree.fromstring("(NP (Noun pescado) (Adj crudo))")},

            (1, 3): {},
            (2, 4): {},
            (3, 5): {'VP': Tree.fromstring(
                "(VP (Verb come) (NP (Noun pescado) (Adj crudo)))")},

            (1, 4): {},
            (2, 5): {},

            (1, 5): {'S': Tree.fromstring(
                """(S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                   )
                """)},
        }
        self.assertEqual(parser._bp, bp)

        # check tree
        t2 = Tree.fromstring(
            """
                (S
                    (NP (Det el) (Noun gato))
                    (VP (Verb come) (NP (Noun pescado) (Adj crudo)))
                )
            """)
        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.6 * 1.0 * 0.9 * 1.0 * 1.0 * 0.4 * 0.1 * 1.0)
        self.assertAlmostEqual(lp, lp2)

    def assertEqualPi(self, pi1, pi2):
        self.assertEqual(set(pi1.keys()), set(pi2.keys()))

        for k in pi1.keys():
            d1, d2 = pi1[k], pi2[k]
            self.assertEqual(d1.keys(), d2.keys(), k)
            for k2 in d1.keys():
                prob1 = d1[k2]
                prob2 = d2[k2]
                self.assertAlmostEqual(prob1, prob2)

    def test_parse_ambiguity(self):
        # Ejemplo tomado de las paginas 4, 5, 8 de las notas de Michael Collins
        # Probabilistic Context-Free Grammars (PCFGs)
        grammar = PCFG.fromstring(
            """
                S -> NP VP              [1.0]

                VP -> Vt NP             [0.65]
                VP -> VP PP             [0.35]

                NP -> DT NN             [0.8]
                NP -> NP PP             [0.2]

                PP -> IN NP             [1.0]

                Vt -> saw               [1.0]

                NN -> man               [0.2]
                NN -> telescope         [0.3]
                NN -> dog               [0.5]

                DT -> the               [1.0]

                IN -> with              [1.0]
            """)

        # Cambiando esto:
        # VP -> Vt NP             [0.85]
        # VP -> VP PP             [0.15]
        # Obtengo el otro arbol

        parser = CKYParser(grammar)

        lp, t = parser.parse('the man saw the dog with the telescope'.split())

        # draw_trees(t)

        # check tree
        t2 = Tree.fromstring(
                """
                    (S
                        (NP
                            (DT the)
                            (NN man)
                        )
                        (VP
                            (VP
                                (Vt saw)
                                (NP
                                    (DT the)
                                    (NN dog)
                                )
                            )
                            (PP
                                (IN with)
                                (NP
                                    (DT the)
                                    (NN telescope)
                                )
                            )
                        )
                    )
                """)

        self.assertEqual(t, t2)

        # check log probability
        lp2 = log2(1.0 * 0.8 * 1.0 * 0.2 * 0.35 * 0.65 * 1.0 * 0.8 * 1.0 *
                   0.5 * 1.0 * 1.0 * 0.8 * 1.0 * 0.3)

        self.assertAlmostEqual(lp, lp2)
