# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log
from random import random
import numpy as np  # Para generar lista de floats

def addMarkers(sent, n):
    """
    Agrega a una oracion:
                        * n-1 marcadores <s> al comienzo y
                        * 1 marcadores </s> al final.
    """
    # Añadimos marcadores de comienzo y fin de oracion.
    sent = ["<s>"]*(n-1) + sent + ["</s>"]

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
            sent = addMarkers(sent, n)
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
        sent = addMarkers(sent, n)
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
        sent = addMarkers(sent, n)
        probability = 0  # Se inicializa en 0 por la sumatoria de logaritmos

        # Iteramos sobre cada oracion del conjunto de oraciones
        for i in range(n-1, len(sent)):
            token = sent[i]  # wi : primera palabra
            prev_tokens = sent[i-n+1:i]  # Markov Assumption: wi-k ... wi-1
            x = self.cond_prob(token, prev_tokens)
            # if x == 0.0:
            #     import pdb; pdb.set_trace()
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
        # Para poder usar los parametros del init de la clase NGram
        super().__init__(n, sents)

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


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        # Para poder usar los parametros del init de la clase NGram
        super().__init__(n, sents)

        self.gamma = gamma

        # Ponemos esta linea aca porque en caso de que no den el gamma,
        # separamos el held_out, y calculamos los modelos sin el held_out
        # porque si se los dejamos falla el ultimo test
        if gamma is None:
            sents, held_out = self.getHeldOut(sents)

        # Listas de modelos, para la interpolacion
        self.models = self.getModels(n, sents, addone)

        # Revertimos los modelos porque empezamos de los modelos mas altos a
        # a los mas bajos, es decir:
        #       reversed_models = [n-grama, (n-1)-grama, ..., 1-grama]
        self.reversed_models = list(reversed(self.models))

        # Ponemos esta linea aca porque para obtener el gamma necesitamos
        # calcular la log-probability, que usa nuestro cond_prob, que esto a la
        # vez usa nuestros modelos, entonces necesitamos los modelos
        print("Computando Gammas\n")
        if gamma is None:
            self.gamma = self.getGamma(held_out)

    def getModels(self, n, sents, is_addone):
        """
        Calcula la lista de modelos talque:
                models = [1-grama, 2-grama, ... , n-grama]

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        addone -- whether to use addone smoothing.
        """
        models = []
        if is_addone:
            models.append(AddOneNGram(1, sents))
        else:
            models.append(NGram(1, sents))

        for i in range(2, n+1):
            # [2, 3, ..., n]
            models.append(NGram(i, sents))

        return models

    def getHeldOut(self, sents, percentage=0.1):
        """
        Obtine los datos Held Out de una lista de oraciones
        segun un porcentaje dado.

        sents -- listas oraciones
        percentage -- porcentaje a tomar de las oraciones (default: 10%)
        """
        # Lo que no quiero para el held_out
        rest = int((1 - percentage) * len(sents))

        # El resto de oraciones
        new_sents = sents[:rest]
        # El porcentaje de oraciones que quiero (tomo de las ultimas oraciones)
        held_out = sents[rest:]

        return new_sents, held_out

    def getGamma(self, held_out):
        """
        Calculamos un gamma a partir de un held_out de datos.

        held_out -- datos para calcular gamma.
        """
        # gammas = [(i+1)*10 for i in range(10)]
        self.gamma = 1
        max_log_prob = self.log_probability(held_out)  # Candidato a maximo

        # Rango de gammas a probar de 100 en 100 hasta 1000
        gammas = [i for i in range(100, 1100, 100)]

        print("Gamma =", self.gamma, "==> Log-Prob =", max_log_prob)
        best_gamma = self.gamma
        for gamma in gammas:
            self.gamma = gamma
            my_log_prob = self.log_probability(held_out)

            print("Gamma =", self.gamma, "==> Log-Prob =", my_log_prob)
            # Si my_log_prob es mas grande que mi candidato, lo seteo como maximo
            if max_log_prob < my_log_prob:
                max_log_prob = my_log_prob
                best_gamma = self.gamma

        print("\nMejor Gamma =", best_gamma)

        return best_gamma

    def count(self, tokens):
        """
        Count for an k-gram with 0 < k < n+1

        tokens -- the k-gram tuple.
        """
        # Tengo que analizar a que n-grama pertenece el tokens,
        # para ellos usamos su largo
        # Ejemplo: ["el", "gato", "come"] es talque len = 3 --> 3-grama
        #          ["el", "gato"] len = 2 --> 2-grama
        #          ["el"] len = 1 --> 1-grama
        length_token = len(tokens)

        # Si es "vacio" pertenece a un 1-grama
        if tokens == ():
            length_token = 1

        model = self.models[length_token-1]

        return model.count(tokens)

    def getLambdas(self, tokens):
        """
        Calculamos los lambdas implementando la formula de la siguiente nota:
            https://cs.famaf.unc.edu.ar/~francolq/lm-notas.pdf

        tokens -- tokens para el calculo de los lambdas
        """
        gamma = self.gamma

        lambdas = []  # Lista de lambdas
        for i in range(len(tokens)):  # [0, 1, 2, ..., len(tokens)-1]
            # [0, 1, ..., i-1]
            sumatoria = sum(lambdas[j] for j in range(0, i))
            # c = float(models[i].count(tuple(tokens[i:])))
            c = float(self.count(tuple(tokens[i:])))
            lambda_i = (1 - sumatoria) * (c/(c + gamma))

            assert lambda_i >= 0

            lambdas.append(lambda_i)

        lambdas.append(1 - sum(lambdas))

        assert sum(lambdas) == 1

        return lambdas

    def cond_prob(self, token, prev_tokens=None):
        """
        Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # token = xn
        # prev_tokens = x1 ... xn-1
        n = self.n

        if not prev_tokens:
            prev_tokens = []

        assert len(prev_tokens) == n - 1

        tokens = prev_tokens + [token]  # (prev_tokens, token)

        # Calculamos los lambdas:
        # [lambda_1, lambda_2, ... ,lambda_n]
        lambdas = self.getLambdas(prev_tokens)

        probability = 0
        for i in range(len(tokens)):
            # q_ML(xn | xi ... xn-1)
            q_ML = self.reversed_models[i].cond_prob(token, prev_tokens[i:])
            probability += lambdas[i] * q_ML

        return probability


class BackOffNGram(NGram):

    def __init__(self, n, sents, beta=None, addone=True):
        """
        Back-off NGram model with discounting as described by Michael Collins.

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        beta -- discounting hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        # Para poder usar los parametros del init de la clase NGram
        super().__init__(n, sents)

        self.beta = beta

        # Ponemos esta linea aca porque en caso de que no den el gamma,
        # separamos el held_out, y calculamos los modelos sin el held_out
        # porque si se los dejamos falla el ultimo test
        # print("Computando Held-Out")
        if beta is None:
            sents, held_out = self.getHeldOut(sents)
        # print("Termine de computar el Held-Out")

        # Listas de modelos, para la obtencion del conjunto A
        print("Computando Modelos")
        self.models = self.getModels(n, sents, addone)
        print("Termine de computar los Modelos")

        # Diccionario de conjuntos
        print("Computando conjunto A")
        self.set_A = self.generateSetA(n, self.models)
        print("Termine de computar el conjunto A")

        # Diccionarios para los valores de alpha y denom
        self.dict_denom = defaultdict(float)
        self.dict_alpha = defaultdict(float)

        # Ponemos esta linea aca porque para obtener el gamma necesitamos
        # calcular la log-probability, que usa nuestro cond_prob, que esto a la
        # vez usa nuestros modelos, entonces necesitamos los modelos
        print("Compuntando el mejor Beta\n")
        if beta is None:
            self.beta = self.getBeta(held_out)
        print("Termine de computar el mejor Beta")

        self.generateDictAlpha()
        self.generateDictDenom()

        # for i in self.set_A.items():
        #     print(i)

    def getModels(self, n, sents, is_addone):
        """
        Calcula la lista de modelos talque:
                models = [1-grama, 2-grama, ... , n-grama]

        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        addone -- whether to use addone smoothing.
        """
        models = []
        if is_addone:
            print("------> Estoy usando Add-One")
            models.append(AddOneNGram(1, sents))
            print("------> Modelo =", models[0])
        else:
            print("------> Estoy usando N-Gram comun")
            models.append(NGram(1, sents))
            print("------> Modelo =", models[0])

        for i in range(2, n+1):
            # [2, 3, ..., n]
            models.append(NGram(i, sents))

        return models

    def getHeldOut(self, sents, percentage=0.1):
        """
        Obtine los datos Held Out de una lista de oraciones
        segun un porcentaje dado.

        sents -- listas oraciones
        percentage -- porcentaje a tomar de las oraciones (default: 10%)
        """
        # Lo que no quiero para el held_out
        rest = int((1 - percentage) * len(sents))

        # El resto de oraciones
        new_sents = sents[:rest]
        # El porcentaje de oraciones que quiero (tomo de las ultimas oraciones)
        held_out = sents[rest:]

        return new_sents, held_out

    def getBeta(self, held_out):
        """
        Calculamos un beta a partir de un held_out de datos.

        held_out -- datos para calcular beta.
        """
        self.beta = 0.0
        self.generateDictAlpha()
        self.generateDictDenom()
        max_log_prob = self.log_probability(held_out)  # Candidato a maximo

        # Rango de betas a probar de 0,1 en 0,1 hasta 1
        betas = [i for i in np.arange(0.1, 1.1, 0.1)]

        print("Beta =", self.beta, "==> Log-Prob =", max_log_prob)
        best_beta = self.beta
        for beta in betas:
            self.beta = beta
            self.generateDictAlpha()
            self.generateDictDenom()
            my_log_prob = self.log_probability(held_out)
            print("Beta =", self.beta, "==> Log-Prob =", my_log_prob)
            # Si my_log_prob es mas grande que mi candidato, lo seteo como maximo
            if max_log_prob < my_log_prob:
                max_log_prob = my_log_prob
                best_beta = self.beta

        # beta tiene que estar en el rango [0, 1]
        assert 0 <= best_beta and best_beta <= 1

        print("\nMejor Beta =", best_beta, "\n")

        return best_beta

    def generateSetA(self, n, models):
        """
        Generamos los siguente conjuntos:
            set_A(x1 ... xi) = {x : count(x1 ... xi x) > 0}

        en total son n conjuntos

        Ejemplos:
            set_A(x1) = {x : count(x1 x) > 0}
            set_A(x1 x2) = {x : count(x1 x2 x) > 0}
            set_A(x1 x2 x3) = {x : count(x1 x2 x3 x) > 0}
            set_A(x1 x2 x3 x4) = {x : count(x1 x2 x3 x4 x) > 0}

        n -- order of the model
        models -- modelos usados para la generacion de los conjuntos
        """
        # Diccionario de conjuntos
        # set_A = dict()
        set_A = defaultdict(set)

        for i in range(1, n):
            # [1 ... n-1] => [2-grama, 3-grama,..., n-grama]
            for tokens, value in models[i].counts.items():
                # (x1 ... xi x) ∈ (i+1)-grama y count(x1 ... xi x)>0
                if len(tokens) == i+1 and value > 0:
                    # Ultimo elemento = x
                    x = tokens[-1]
                    # Todos menos el ultimo elemento = x1 ... xi
                    x_i = tokens[:-1]
                    if x_i not in set_A:
                        set_A[x_i] = set()
                    set_A[x_i].add(x)

        return set_A

    def count(self, tokens):
        """
        Count for an k-gram with 0 < k < n+1

        tokens -- the k-gram tuple.
        """
        # Tengo que analizar a que n-grama pertenece el tokens,
        # para ellos usamos su largo
        # Ejemplos: ("el", "gato", "come") es tq len = 3 --> 3-grama
        #           ("el", "gato") es tq len = 2 --> 2-grama
        #           ("el") es tq len = 1 --> 1-grama
        length_token = len(tokens)

        # Si es () pertenece a un 1-grama
        if tokens == ():
            length_token += 1

        # Si me como la tupla del tipo (<s>, <s>, ...), por como tenemos
        # definido la inclusion de los marcadores (n-1 <s> al inicio donde el
        # el n es el correspondiente al de n-grama), tenemos que pasarle dicho
        # tokens al de n-grama correspondienrte, es decir:
        #       (<s>,) corresponde a 2-grama
        #       (<s>, <s>) corresponde a 3-grama
        #       (<s>, <s>, <s>) corresponde a 4-grama
        # y asi sucesivamente.

        # tokens.count("<s>") == length_token : quiere decir que la tupla
        #  contiene solamente palabras del tipo <s>
        if tokens.count("<s>") == length_token:
            length_token += 1

        model = self.models[length_token-1]

        return model.count(tokens)

    def A(self, tokens):
        """
        Set of words with counts > 0 for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self.set_A[tuple(tokens)]

    def generateDictAlpha(self):
        """
        Generamos el diccionario con los valores de alfa.
        """
        # for tokens in self.count.key(): # DUDA
        for tokens in self.set_A.keys(): # DUDA
            alpha = 1
            cardinal_A = len(self.A(tokens))
            
            # Si A(x1 ... xi) != Vacio
            if cardinal_A != 0:
                beta = self.beta
                c = self.count(tuple(tokens))
                alpha = (beta * cardinal_A) / float(c)

                self.dict_alpha[tuple(tokens)] = alpha

    def generateDictDenom(self):
        """
        Normalization factor for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        # for tokens in self.count.key(): # DUDA
        for tokens in self.set_A.keys(): # DUDA
            s = sum(self.cond_prob(x, tokens[1:]) for x in self.A(tokens))
            denom = 1 - s
            self.dict_denom[tokens] = denom

    def alpha(self, tokens):
        """
        Missing probability mass for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self.dict_alpha.get(tokens, 1.0)

    def denom(self, tokens):
        """
        Normalization factor for a k-gram with 0 < k < n.

        tokens -- the k-gram tuple.
        """
        return self.dict_denom.get(tokens, 1.0)

    def cond_prob(self, token, prev_tokens=None):
        """
        Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # token = xi
        # prev_tokens = x1 ... xi-1

        probability = 0

        # Analizamos los casos expuestos en las notas
        # ===========================================

        # Caso i = 1
        if not prev_tokens:
            # probability = (self.count(tuple([token])) + 1) / ( float(self.count(())) + self.models[0].V())
            probability = self.count(tuple([token])) / float(self.count(()))

        # Casos i > 1, es decir, i >= 2
        else:
            # Si xi ∈ A(x1 ... xi-1)
            if token in self.A(prev_tokens):
                # print(type(prev_tokens), type([token]))
                # print(prev_tokens, [token])
                my_token = list(prev_tokens) + [token]
                c_estrella = self.count(tuple(my_token)) - self.beta
                c = self.count(tuple(prev_tokens))
                probability = c_estrella / float(c)
            # Si xi ∈ B(x1 ... xi-1)
            # Como no pertenece a las palabras talque count(palabra) > 0
            # Entonces pertenece a las palabras talque count(palabra) = 0
            else:
                new_prev_tokens = prev_tokens[1:]  # x2 ... xi-1
                alpha = self.alpha(tuple(prev_tokens))
                q_D = self.cond_prob(token, new_prev_tokens)

                # Caso en que el denominador q_D es 0
                if q_D != 0:
                    denom = self.denom(tuple(prev_tokens))
                    probability = alpha * (q_D / denom)

        return probability
