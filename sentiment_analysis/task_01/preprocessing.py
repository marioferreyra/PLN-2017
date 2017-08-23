import re
import csv
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from nltk.tokenize import TweetTokenizer

# Emoticones Positivos y Negativos
from sentiment_analysis.task_01.emoticons import positive_emoticons
from sentiment_analysis.task_01.emoticons import negative_emoticons


class PreprocessingTweet():

    def __init__(self):
        # Stopwords del español
        self.spanish_stopwords = stopwords.words('spanish')
        # Descargar las stopwords usando el comando nltk.download()

        # Tokenizador de tweets
        self.tweet_tokenizer = TweetTokenizer()

        self.positive_words = self.load_ISOL_words("isol",
                                                   "positivas_mejorada.csv")
        self.negative_words = self.load_ISOL_words("isol",
                                                   "negativas_mejorada.csv")

        self.positive_emoticons = positive_emoticons
        self.negative_emoticons = negative_emoticons

    def load_ISOL_words(self, path, file):
        """
        Carga un archivo CSV y retorna un conjunto con los elementos de dicho
        archivo.
        """
        word_set = set()
        filepath = path + "/" + file

        with open(filepath, encoding='latin-1') as csvfile:
            reader_csv = csv.reader(csvfile)

            for row in reader_csv:
                # word = row[0]
                word = self.delete_tildes(row[0])
                word_set.add(word)

        return word_set

    def get_spanish_stopwords(self):
        """
        Retorna las Stopwords del Español.
        """
        return self.spanish_stopwords

    def get_positive_emoticons(self):
        """
        Retorna el conjunto de emoticones positivos.
        """
        return self.positive_emoticons

    def get_negative_emoticons(self):
        """
        Retorna el conjunto de emoticones negativos.
        """
        return self.negative_emoticons

    def delete_tildes(self, content):
        """
        Remplazamos todos los caracteres de una cadena por su version sin
        acentuar (sin tilde).

        Ejemplo:
            >>> s = "hóla mùndó, ¿Cómo está la people?"
            >>> delete_tildes(s)
            >>> hola mundo, ¿Como esta la people?
        """
        new_list = []
        for c in unicodedata.normalize('NFD', content):
            if unicodedata.category(c) != 'Mn':
                new_list += [c]

        return ''.join(new_list)

    def change_pos_neg_words_emojis(self, content):
        """
        Reemplaza todas las palabras positivas y negativas presentes en los
        conjuntos "self.positive_words" y "self.negative_words" por las
        palabras "positiveword" y "negativeword" respectivamente.

        Reemplaza todas los emoticones positivos y negativos presentes en los
        conjuntos "positive_emoticons" y "negative_emoticons" por las
        palabras "positiveemoticon" y "negativeemoticon" respectivamente.
        """
        content_tokenized = self.tweet_tokenizer.tokenize(content)

        new_content_tokenized = []
        for word in content_tokenized:
            if word in self.positive_words:
                new_word = "positiveword"
            elif word in self.negative_words:
                new_word = "negativeword"
            elif word in positive_emoticons:
                new_word = "positiveemoticon"
            elif word in negative_emoticons:
                new_word = "negativeemoticon"
            else:
                new_word = word

            new_content_tokenized.append(new_word)

        new_content = " ".join(new_content_tokenized)

        return new_content

    def tweet_cleaner(self, content):
        """
        Limpiamos el contenido de un tweet de la siguiente forma:
            * Removemos URL.
            * Removemos signos de puntuacion.
            * Removemos nombres de usuarios (ie. @user).
            * Removemos caracteres no alfanumericos.

        Finalmente retornamos una cadena, es decir, el tweet limpio.
        """
        pattern = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
        repl = " "
        new_content = ' '.join(re.sub(pattern, repl, content).split())

        return new_content

    def remove_repeated_word(self, word):
        """
        Remueve de una palabra los caracteres repetidos.
        """
        repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        repl = r'\1\2\3'

        repl_word = repeat_regexp.sub(repl, word)
        if repl_word != word:
            return self.remove_repeated(repl_word)
        else:
            return repl_word

    def remove_repeated(self, content):
        """
        Remueve los caracteres repetidos de un contenido
        """
        content_tokenized = self.tweet_tokenizer.tokenize(content)

        new_content_tokenized = []
        for word in content_tokenized:
            new_word = self.remove_repeated_word(word)
            new_content_tokenized.append(new_word)

        new_content = " ".join(new_content_tokenized)

        return new_content

    def reduce_laught_expression(self, word):
        """
        Busca patrones repetidos de una palabra y devuelve la parte que se
        repite, si no hay ninguna parte que se repita devuelve la palabra que
        se recibio de entrada.
        """
        regex = r"(\b([a-z]{2,}?)\2+\b)"
        pattern = re.findall(regex, word, re.X | re.I)

        if pattern != []:
            return pattern[0][1]

        return word

    def change_to_risas(self, content):
        """
        Las expresiones que denotan risas por ejemplo "jajaja", "jejeje" entre
        otras son reemplazadas por la palabra "risas".
        """
        content_tokenized = self.tweet_tokenizer.tokenize(content)

        new_content_tokenized = []
        for word in content_tokenized:
            # Convertimos la expresion de risas a su "menor" forma:
            # Por ejemplo: "jajajajaja" --> "ja"
            reduce_laught = self.reduce_laught_expression(word)

            # Si la expresion esta en el conjunto de expresiones de risas,
            # reemplazamos la palabra por "risas" sino devolvemos la palabra
            # que se recibio de entrada
            laught_expression = {"ja", "je", "ji", "jo", "ju", "ah", "ha"}
            if reduce_laught in laught_expression:
                new_word = "risas"
            else:
                new_word = reduce_laught

            new_content_tokenized.append(new_word)

        new_content = " ".join(new_content_tokenized)

        return new_content

    def remove_stopwords(self, content_list):
        """
        Removemos de una lista de palabras, aquellas que son muy frecuentes
        pero que no aportan gran valor sintáctico, es decir las stopwords.
        """
        important_words = []
        for word in content_list:
            if word not in self.spanish_stopwords:
                important_words.append(word)

        return important_words

    def tweet_stemming(self, content_list):
        """
        Dada una lista de palabras, a cada una de ellas se le realiza el
        Stemming.
        Stemming es el proceso por el cual transformamos cada palabra en su
        raiz.

        Ejemplo:
            maravilloso     |
            maravilla       |-> maravill
            maravillarse    |
        """
        stemmer = SnowballStemmer("spanish")  # Lematizador del español
        result = []
        for word in content_list:
            result.append(stemmer.stem(word))

        return result
