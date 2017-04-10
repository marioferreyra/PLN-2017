# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log
from random import random


def addDelimitator(sent, n):
    """
    Agrega a una oracion n-1 delimitadores <s> al comienzo y
    un delimitador </s> al final.
    """
    sent = ["<s>"] * (n-1) + sent  # Marcador de comienzo de oracion
    sent += ["</s>"]  # Marcador de final de oracion

    return sent


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


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        # Iteramos sobre cada oracion del conjunto de oraciones
        for sent in sents:
            sent = addDelimitator(sent, n)
            # Iteramos sobre cada palabra de la oracion
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1

    def count(self, tokens):
        """
        Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """
        Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # Calcula lo siguiente:
        # P(token | prev_tokens) = count(prev_tokens, token)/count(prev_tokens)
        n = self.n

        if not prev_tokens:
            prev_tokens = []

        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]  # (prev_tokens, token)

        # count(prev_tokens, token)
        count_tokens = float(self.counts[tuple(tokens)])
        # count(prev_tokens)
        count_prev_tokens = self.counts[tuple(prev_tokens)]

        probability = 0

        # En el caso de que count(prev_tokens) = 0
        # Tomamos la probabilidad como 0 (por la division por 0)

        if count_prev_tokens != 0:
            probability = count_tokens / count_prev_tokens

        return probability

    def sent_prob(self, sent):
        """
        Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self.n
        sent = addDelimitator(sent, n)
        probability = 1  # Se inicializa en 1 por la productoria
        m = len(sent)  # Tamaño de la oracion

        # RECORDAR: N-grama <==> N-1 orden de Markov Assumption
        #           (N+1)-grama <==> N orden de Markov Assumption
        #
        # Markov Assumption de orden k (N+1)
        # P(w1 w2 ... wm) ~= Productoria(i, m) P(wi | wi-k ... wi-1)
        #                       <==>
        # P(w1 w2 ... wm) ~= Productoria(i, m) P(wi | wi-N+1 ... wi-1)

        for i in range(n-1, m):
            # El i empieza en la posicion n-1 (para no tomar los delimitadores)
            token = sent[i]  # wi
            # n = 1 ==> i = 0 ... m
            # n = 2 ==> i = 1 ... m
            # n = 3 ==> i = 2 ... m
            prev_tokens = sent[i-n+1:i]
            # prev_tokens = [wi-n+1 ... wi-1]
            # n = 1 ==> [wi+0 ... wi-1] = []
            # n = 2 ==> [wi-1 ... wi-1] = [wi-1]
            # n = 3 ==> [wi-2 ... wi-1] = [wi-2 wi-1]
            probability *= self.cond_prob(token, prev_tokens)
            # Markov Assumption = P(wi | wi-n+1 ... wi-1)

        return probability

    def sent_log_prob(self, sent):
        """
        Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        n = self.n
        sent = addDelimitator(sent, n)
        probability = 0  # Se inicializa en 0 por la sumatoria de logaritmos

        # Iteramos sobre cada oracion del conjunto de oraciones
        for i in range(n-1, len(sent)):
            token = sent[i]  # wi : primera palabra
            prev_tokens = sent[i-n+1:i]  # Markov Assumption: wi-k ... wi-1
            x = self.cond_prob(token, prev_tokens)
            probability += log2Extended(x)

        return probability


class NGramGenerator:

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.n = n = model.n  # Tamaño del N-grama

        counts = model.counts
        self.probs = probs = dict()
        self.sorted_probs = sorted_probs = dict()

        for tokens in counts.keys():
            if len(tokens) == n:
                token = tokens[n-1]  # Tomamos el ultimo token
                prev_tokens = tokens[:-1]  # Tomamos todos los tokens anteriores a tokens
                probability = model.cond_prob(token, list(prev_tokens))
                
                # Si el prev_tokens no esta en el diccionario lo cargamos
                if prev_tokens not in probs:
                    probs[prev_tokens] = dict()

                probs[prev_tokens][token] = probability

        for key in probs:
            # print("\n")
            # print("Comun =", probs[key].items())
            # print("Sorted =", sorted(probs[key].items()))
            # print(key, "||", list(probs[key].items()))
            # if sorted(probs[key].items()) == list(probs[key].items()):
            #     print("True")
            sorted_probs[key] = sorted(probs[key].items())

        # print(sorted_probs)

    def generate_token(self, prev_tokens=None):
        """
        Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        U = random()  # Numero aleatorio entre 0 y 1

        n = self.n
        if not prev_tokens:
            prev_tokens = ()

        assert len(prev_tokens) == n - 1

        # print(self.probs[()])
        next_tokens = self.probs[prev_tokens]  # Posibles tokens
        list_probs = list(next_tokens.values())  # Lista de probabilidades
        
        # Transformada Inversa
        index = 0
        accumulator = list_probs[index]
        # Mientras la acumulada sea menor a U, sigo acumulando valores hasta
        # que la acumulada sea mayor a la U
        while U > accumulator:
            index += 1
            accumulator += list_probs[index]

        random_token = self.sorted_probs[prev_tokens][index][0]

        return random_token


    def generate_sent(self):
        """
        Randomly generate a sentence.
        """
        pass








sents = [
            'el gato come pescado .'.split(),
            'la gata come salmón .'.split(),
        ]

ngram = NGram(1, sents)
generator = NGramGenerator(ngram)
# print(generator.generate_token(()))
# generator.generate_sent()



# Metodo de la Transformada Inversa
# u = random.random()
# j = 1
# f = 0.5*((float(2)/3)**j) # f = p1
# # Mientras la acumulada sea menor a la U, sigo acumulando valores
# # hasta que la acumulada sea mayor a la U
# while u > f:
#     j += 1
#     f += 0.5*((float(2)/3)**j) # f = f + pj
# x = j