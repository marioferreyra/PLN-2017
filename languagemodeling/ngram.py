# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log
from random import random


def addDelimiters(sent, n):
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
            sent = addDelimiters(sent, n)
            # Iteramos sobre cada palabra de la oracion
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                counts[ngram] += 1
                counts[ngram[:-1]] += 1  # Todos menos el ultimo

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
        sent = addDelimiters(sent, n)
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
        sent = addDelimiters(sent, n)
        probability = 0  # Se inicializa en 0 por la sumatoria de logaritmos

        # Iteramos sobre cada oracion del conjunto de oraciones
        for i in range(n-1, len(sent)):
            token = sent[i]  # wi : primera palabra
            prev_tokens = sent[i-n+1:i]  # Markov Assumption: wi-k ... wi-1
            x = self.cond_prob(token, prev_tokens)
            probability += log2Extended(x)

        return probability

    def log_probability(self, sents):
        """
        Calcula el Log Probability de una serie de oraciones.

        sents -- lista de oraciones.
        """
        prob = 0
        for sent in sents:
            prob += self.sent_log_prob(sent)

        return prob

    def cross_entropy(self, sents):
        """
        Calcula el Cross Entropy de una serie de oraciones.

        sents -- lista de oraciones.
        """
        # Numero total de palabras en las oraciones (se pueden repetir)
        count_words = 0
        for sent in sents:
            count_words += len(sent)

        return self.log_probability(sents) / float(count_words)

    def perplexity(self, sents):
        """
        Calcula la Perplexity de una serie de oraciones.

        sents -- lista de oraciones.
        """
        return pow(2, -self.cross_entropy(sents))


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
                token = tokens[n-1]  # El ultimo token
                prev_tokens = tokens[:-1]  # Todos los tokens previos a token
                # Probabilidad condicional: P(token | prev_tokens)
                probability = model.cond_prob(token, list(prev_tokens))

                # Si el prev_tokens no esta en el diccionario lo cargamos
                if prev_tokens not in probs:
                    probs[prev_tokens] = dict()

                # Cargamos en el diccionario prev_tokens:
                # {token : probabilidad}
                # Donde: * token es talque: "prev_tokens, token"
                #        * probabilidad es talque: P(token | prev_tokens)
                probs[prev_tokens][token] = probability

        # Formamos el sorted_probs
        # Es similar a probs, salvo que:
        # {token : probabilidad} -->  (token, probabilidad)
        for key in probs:
            sorted_probs[key] = sorted(probs[key].items())

    def generate_token(self, prev_tokens=None):
        """
        Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n

        if not prev_tokens:
            prev_tokens = ()

        assert len(prev_tokens) == n - 1

        # Obtenemos los prev_tokens con sus probabilidades
        probs_prev_tokens = self.sorted_probs[prev_tokens]

        # *** Transformada Inversa ***
        U = random()  # Numero aleatorio entre 0 y 1
        index = 0  # Indice de los tokens
        accumulator = probs_prev_tokens[index][1]  # Acumula las probabilidades

        # Mientras la acumulada sea menor a U, sigo acumulando valores hasta
        # que la acumulada sea mayor a la U
        while accumulator < U:
            index += 1
            accumulator += probs_prev_tokens[index][1]

        # Me quedo con una de las palabra
        chosen_token = probs_prev_tokens[index][0]

        return chosen_token

    def generate_sent(self):
        """
        Randomly generate a sentence.
        """
        n = self.n

        prev_tokens = ["<s>"]*(n-1)  # Delimitadores iniciales
        final_delimiter = "</s>"  # Delimitador final

        my_sent = []  # La oracion que voy a formar
        my_sent += prev_tokens  # Le agregamos los delimitadores iniciales

        # Generamos un posible token a partir de los prev_tokens
        next_token = self.generate_token(tuple(prev_tokens))

        # Mientras el token sea distinto de </s> generero mas oraciones
        while next_token != final_delimiter:
            my_sent += list((next_token, ))  # Vamos armando la oracion
            prev_tokens += list((next_token, ))  # Tokens previos
            prev_tokens = prev_tokens[1:]  # Me quedo con todos menos el 1°
            next_token = self.generate_token(tuple(prev_tokens))

        # Nos quedamos con la oracion sin los delimitadores iniciales
        sent = my_sent[(n-1):]

        return sent


def countWordsType(sents):
    """
    Calcula el tamaño del vocabulario (words type) de una lista de oraciones.
    """
    words_type = []
    for sent in sents:
        for token in sent:
            if token not in words_type:
                words_type.append(token)

    return len(words_type)


class AddOneNGram(NGram):
    """
    Heredamos de NGram para poder usar todos sus metodos.
    """
    # super(), es una función build-in que sirve para acceder a atributos que
    # pertenecen a una clase superior.
    def __init__(self, n, sents):
        super().__init__(n, sents)  # Para poder usar las variables del init

        # Le sumamos 1 porque se incluye el marcador </s>.
        self.count_words_type = countWordsType(sents) + 1

        # print(self.counts) #--> Ejemplo de el uso de super

    def V(self):
        """
        Size of the vocabulary.
        """
        return self.count_words_type

    def cond_prob(self, token, prev_tokens=None):
        """
        Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # Calcula el Add-one estimation:
        #                           count(prev_tokens, token) + 1
        # P(token | prev_tokens) = -------------------------------
        #                              count(prev_tokens) + V
        n = self.n

        if not prev_tokens:
            prev_tokens = []

        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]  # (prev_tokens, token)

        # count(prev_tokens, token) + 1
        count_tokens = float(self.counts[tuple(tokens)]) + 1
        # count(prev_tokens) + V
        count_prev_tokens = self.counts[tuple(prev_tokens)] + self.V()

        probability = 0

        # En el caso de que count(prev_tokens) = 0
        # Tomamos la probabilidad como 0 (por la division por 0)

        if count_prev_tokens != 0:
            probability = count_tokens / count_prev_tokens

        return probability
